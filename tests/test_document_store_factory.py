# Standard
from unittest.mock import patch
import os
import shutil
import tempfile

# Third Party
from haystack import Document, component  # type: ignore
from haystack.document_stores.in_memory import InMemoryDocumentStore  # type: ignore
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


def test_document_store_ingest_and_retrieve_for_in_memory_store(
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
        document_store = InMemoryDocumentStore.load_from_disk(document_store_config.uri)
        documents = document_store.filter_documents()
        assert len(documents) == 1
        documents_count = document_store.count_documents()
        assert documents_count == 1

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
