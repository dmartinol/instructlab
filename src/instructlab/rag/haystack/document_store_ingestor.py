# Standard
from pathlib import Path
import glob
import logging
import os

# Third Party
from haystack import Pipeline  # type: ignore
from haystack.components.preprocessors import DocumentCleaner  # type: ignore

# First Party
from instructlab.rag.document_store import DocumentStoreIngestor
from instructlab.rag.haystack.component_factory import (
    create_converter,
    create_document_embedder,
    create_document_writer,
    create_splitter,
)
from instructlab.rag.rag_configuration import DocumentStoreConfig, EmbeddingModelConfig

logger = logging.getLogger(__name__)


class HaystackDocumentStoreIngestor(DocumentStoreIngestor):
    def __init__(
        self,
        document_store_config: DocumentStoreConfig,
        embedding_config: EmbeddingModelConfig,
    ):
        super().__init__()
        self.document_store_config = document_store_config
        self.embedding_config = embedding_config
        self._pipeline = _create_pipeline(
            document_store_config=document_store_config,
            embedding_config=embedding_config,
        )
        _connect_components(self._pipeline)

    def ingest_documents(self, input_dir) -> tuple[bool, int]:
        pattern = "*.json"
        if Path(os.path.join(input_dir, "docling-artifacts")).exists():
            pattern = "docling-artifacts/" + pattern

        try:
            self._pipeline.run(
                {"converter": {"sources": glob.glob(os.path.join(input_dir, pattern))}}
            )
            document_store = self._pipeline.get_component(
                "document_writer"
            ).document_store
            logger.info(f"count_documents: {document_store.count_documents()}")

            # Final step required for InMemory document store
            document_store.save_to_disk(self.document_store_config.uri)
            logger.info(f"Saved document store as: {self.document_store_config.uri}")
            return True, document_store.count_documents()
        except Exception as e:
            logger.error(f"Ingestion attempt failed: {e}")
            return False, -1


def _create_pipeline(
    document_store_config: DocumentStoreConfig, embedding_config: EmbeddingModelConfig
) -> Pipeline:
    pipeline = Pipeline()
    pipeline.add_component(instance=create_converter(), name="converter")
    pipeline.add_component(instance=DocumentCleaner(), name="document_cleaner")
    # TODO make the params configurable
    pipeline.add_component(
        instance=create_splitter(embedding_config=embedding_config),
        name="document_splitter",
    )
    # TODO make this more generic
    pipeline.add_component(
        instance=create_document_embedder(embedding_config=embedding_config),
        name="document_embedder",
    )
    pipeline.add_component(
        instance=create_document_writer(document_store_config=document_store_config),
        name="document_writer",
    )
    return pipeline


def _connect_components(pipeline: Pipeline):
    pipeline.connect("converter", "document_cleaner")
    pipeline.connect("document_cleaner", "document_splitter")
    pipeline.connect("document_splitter", "document_embedder")
    pipeline.connect("document_embedder", "document_writer")
