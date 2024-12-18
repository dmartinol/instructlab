# Standard
import logging

# First Party
from instructlab.rag.document_store import DocumentStoreIngestor, DocumentStoreRetriever
from instructlab.rag.rag_configuration import (
    DocumentStoreConfig,
    EmbeddingModelConfig,
    RetrieverConfig,
)

logger = logging.getLogger(__name__)


def create_document_retriever(
    document_store_config: DocumentStoreConfig,
    retriever_config: RetrieverConfig,
) -> DocumentStoreRetriever:
    # First Party
    from instructlab.rag.haystack.document_store_factory import (
        create_milvus_document_retriever,
    )

    return create_milvus_document_retriever(
        document_store_config=document_store_config,
        retriever_config=retriever_config,
    )


def create_document_store_ingestor(
    document_store_config: DocumentStoreConfig,
    embedding_config: EmbeddingModelConfig,
) -> DocumentStoreIngestor:
    # First Party
    from instructlab.rag.haystack.document_store_factory import (
        create_milvus_document_store,
    )

    return create_milvus_document_store(
        document_store_config=document_store_config,
        embedding_config=embedding_config,
    )
