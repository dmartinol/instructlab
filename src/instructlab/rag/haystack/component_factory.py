# Standard

# Third Party
from haystack.components.converters import TextFileToDocument  # type: ignore
from haystack.components.embedders import (  # type: ignore
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.components.retrievers import InMemoryEmbeddingRetriever  # type: ignore
from haystack.components.writers import DocumentWriter  # type: ignore
from haystack.document_stores.in_memory import InMemoryDocumentStore  # type: ignore

# First Party
from instructlab.rag.haystack.components.document_splitter import (
    DoclingDocumentSplitter,
)
from instructlab.rag.rag_configuration import (
    DocumentStoreConfig,
    EmbeddingModelConfig,
    RetrieverConfig,
)


def create_document_writer(
    document_store_config: DocumentStoreConfig,
) -> DocumentWriter:
    return DocumentWriter(
        create_document_store(
            document_store_config=document_store_config, drop_old=True
        )
    )


def create_document_store(document_store_config: DocumentStoreConfig, drop_old: bool):
    if not drop_old:
        # Retrieve use case: load file first
        return InMemoryDocumentStore.load_from_disk(document_store_config.uri)
    return InMemoryDocumentStore()


def create_retriever(
    document_store_config: DocumentStoreConfig,
    retriever_config: RetrieverConfig,
    document_store: InMemoryDocumentStore,
):
    return InMemoryEmbeddingRetriever(
        document_store=document_store,
        top_k=retriever_config.top_k,
    )


def create_document_embedder(embedding_config: EmbeddingModelConfig):
    if embedding_config is None:
        raise ValueError("Missing value for field embedding_model")

    return SentenceTransformersDocumentEmbedder(
        model=embedding_config.local_model_path()
    )


def create_text_embedder(embedding_config: EmbeddingModelConfig):
    if embedding_config is None:
        raise ValueError("Missing value for field embedding_model")

    return SentenceTransformersTextEmbedder(model=embedding_config.local_model_path())


def create_converter():
    return TextFileToDocument()


def create_splitter(embedding_config: EmbeddingModelConfig):
    return DoclingDocumentSplitter(
        embedding_model_id=embedding_config.local_model_path(),
        content_format="json",
        max_tokens=150,
    )
