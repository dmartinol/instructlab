# Standard
from unittest.mock import patch
import os
import shutil
import tempfile

# Third Party
from haystack import Document, component  # type: ignore
from pymilvus import MilvusClient  # type: ignore
import pytest

# First Party
from instructlab.rag.document_store import DocumentStoreIngestor, DocumentStoreRetriever
from instructlab.rag.document_store_factory import (
    create_document_retriever,
    create_document_store_ingestor,
)
from instructlab.rag.rag_configuration import (
    DocumentStoreConfig,
    EmbeddingModelConfig,
    RetrieverConfig,
)


@component
class DocumentEmbedderMock:
    @component.output_types(embedding=list[Document])
    def run(self, documents: list[Document]):
        for doc in documents:
            doc.embedding = [float(v * 0.5) for v in range(10)]
        return {"embedding": documents}


@component
class TextEmbedderMock:
    @component.output_types(embedding=list[float])
    def run(self, text: str):  # pylint: disable=unused-argument
        embedding = [float(v * 0.5) for v in range(10)]
        return {"embedding": embedding}


@component
class DocumentSplitterMock:
    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]):
        return {"documents": documents}


@pytest.fixture(name="mock_create_splitter")
def fixture_mock_create_splitter():
    with patch(
        "instructlab.rag.haystack.component_factory.create_splitter"
    ) as mock_function:
        mock_function.side_effect = lambda embedding_config: DocumentSplitterMock()
        yield mock_function


@pytest.fixture(name="mock_create_document_embedder")
def fixture_mock_create_document_embedder():
    with patch(
        "instructlab.rag.haystack.component_factory.create_document_embedder"
    ) as mock_function:
        mock_function.side_effect = lambda embedding_config: DocumentEmbedderMock()
        yield mock_function


@pytest.fixture(name="mock_create_text_embedder")
def fixture_mock_create_text_embedder():
    with patch(
        "instructlab.rag.haystack.component_factory.create_text_embedder"
    ) as mock_function:
        mock_function.side_effect = lambda embedding_config: TextEmbedderMock()
        yield mock_function


def test_document_store_ingest_and_retrieve_for_milvus(
    mock_create_splitter, mock_create_document_embedder, mock_create_text_embedder
):
    with tempfile.TemporaryDirectory() as temp_dir:
        # Ingest docs from a test folder
        document_store_config = DocumentStoreConfig(
            uri=os.path.join(temp_dir, "ingest.db"),
            collection_name="default",
        )
        embedding_config = EmbeddingModelConfig()
        ingestor: DocumentStoreIngestor = create_document_store_ingestor(
            document_store_config=document_store_config,
            embedding_config=embedding_config,
        )
        mock_create_splitter.assert_called_once()
        mock_create_document_embedder.assert_called_once()

        assert ingestor is not None
        assert isinstance(ingestor, DocumentStoreIngestor) is True
        assert (
            type(ingestor).__module__
            == "instructlab.rag.haystack.document_store_ingestor"
        )
        assert type(ingestor).__name__ == "HaystackDocumentStoreIngestor"

        input_dir = "tests/testdata/temp_datasets_documents"
        result, count = ingestor.ingest_documents(input_dir)

        assert result is True
        assert count > 0

        # Validate document store collection
        client = MilvusClient(document_store_config.uri)
        collections = client.list_collections()
        assert len(collections) == 1
        assert collections[0] == document_store_config.collection_name
        stats = client.get_collection_stats(document_store_config.collection_name)
        assert stats is not None
        assert isinstance(stats, dict)
        assert "row_count" in stats
        assert stats["row_count"] == count
        client.close()

        # Run a retriever session
        # Copy db file to avoid concurrent access issues
        new_file = os.path.join(temp_dir, "query.db")
        shutil.copy(document_store_config.uri, new_file)
        document_store_config = DocumentStoreConfig(
            uri=new_file,
            collection_name="default",
        )
        retriever_config = RetrieverConfig(embedding_config=embedding_config)
        retriever: DocumentStoreRetriever = create_document_retriever(
            document_store_config=document_store_config,
            retriever_config=retriever_config,
        )
        mock_create_text_embedder.assert_called_once()
        assert retriever is not None
        assert isinstance(retriever, DocumentStoreRetriever) is True
        assert (
            type(retriever).__module__
            == "instructlab.rag.haystack.document_store_retriever"
        )
        assert type(retriever).__name__ == "HaystackDocumentStoreRetriever"

        context = retriever.augmented_context(user_query="What is knowledge")

        assert context is not None
        assert len(context) > 0
        assert "familiarity with individuals" in context
