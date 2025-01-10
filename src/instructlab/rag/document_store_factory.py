# Standard
import logging

# First Party
from instructlab.rag.document_store import DocumentStoreIngestor, DocumentStoreRetriever

logger = logging.getLogger(__name__)


def create_document_retriever(
    document_store_uri: str,
    document_store_collection_name: str,
    top_k: int,
    embedding_model_path: str,
) -> DocumentStoreRetriever:
    # First Party
    from instructlab.rag.haystack.document_store_factory import (
        create_in_memory_document_retriever,
    )

    return create_in_memory_document_retriever(
        document_store_uri=document_store_uri,
        document_store_collection_name=document_store_collection_name,
        top_k=top_k,
        embedding_model_path=embedding_model_path,
    )


def create_document_store_ingestor(
    document_store_uri: str,
    document_store_collection_name: str,
    embedding_model_path: str,
) -> DocumentStoreIngestor:
    # First Party
    from instructlab.rag.haystack.document_store_factory import (
        create_in_memory_document_store,
    )

    return create_in_memory_document_store(
        document_store_uri=document_store_uri,
        document_store_collection_name=document_store_collection_name,
        embedding_model_path=embedding_model_path,
    )
