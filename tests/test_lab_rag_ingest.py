# Standard
from pathlib import Path
from unittest.mock import patch
import os
import shutil

# Third Party
from click.testing import CliRunner
import pytest

# First Party
from instructlab import lab
from instructlab.rag.document_store import DocumentStoreIngestor


class MockDocumentStoreIngestor(DocumentStoreIngestor):
    def __init__(self, document_store_uri: str):
        self.document_store_uri = document_store_uri

    def ingest_documents(self, input_dir: str) -> tuple[bool, int]:
        with open(self.document_store_uri, "w", encoding="utf-8") as _:
            pass


@pytest.fixture(name="input_dir")
def fixture_input_dir() -> str:
    return "tests/testdata/temp_datasets_documents"


@pytest.fixture(name="copy_test_data")
def fixture_copy_test_data(input_dir: str, tmp_path_home, cli_runner: CliRunner):  # pylint: disable=unused-argument
    tmp_dir = os.path.join(tmp_path_home, os.listdir(tmp_path_home)[0])
    destination = os.path.join(tmp_dir, input_dir)
    os.makedirs(destination, exist_ok=True)
    shutil.copytree(input_dir, destination, dirs_exist_ok=True)


def test_ingestor(input_dir: str, tmp_path_home, copy_test_data, cli_runner: CliRunner):  # pylint: disable=unused-argument
    with (
        patch(
            "instructlab.rag.document_store_factory.create_document_store_ingestor"
        ) as mock_function,
    ):
        mock_function.side_effect = (
            lambda document_store_uri,
            document_store_collection_name,
            embedding_model_path: MockDocumentStoreIngestor(document_store_uri)
        )

        document_store_config_uri = os.path.join(tmp_path_home, "any.db")
        assert not Path(document_store_config_uri).exists()

        result = cli_runner.invoke(
            lab.ilab,
            [
                "--config=DEFAULT",
                "rag",
                "ingest",
                "--document-store-uri",
                document_store_config_uri,
                "--input",
                input_dir,
            ],
        )

        mock_function.assert_called_once()

        assert Path(document_store_config_uri).exists()
        assert result.exit_code == 0
