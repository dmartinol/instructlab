# SPDX-License-Identifier: Apache-2.0

# Standard
import tempfile
import os
from pathlib import Path

# Third Party
from click.testing import CliRunner

# First Party
from instructlab import lab


def test_convert_pdf_from_directory():
    """
    Tests converting from the sample PDF in tests/testdata/temp_datasets_documents/pdf.
    Verifies that it says that it is processing and finished that sample PDF.
    Also verifies that the expected JSON file exists in the temp directory at the end.
    """
    test_input_dir = "tests/testdata/documents/pdf"
    with tempfile.TemporaryDirectory() as temp_dir:
        test_output_dir = os.path.join(temp_dir, 'convert-outputs')
        expected_output_file = os.path.join(test_output_dir, "How to use YAML with InstructLab Page 1.json")

        runner = CliRunner()
        result = runner.invoke(lab.ilab, ["rag", "convert", "--input-dir", test_input_dir, "--output-dir", test_output_dir])
        assert result.exit_code == 0
        assert "Processing document How to use YAML with InstructLab Page 1.pdf" in result.output
        assert "Finished converting document How to use YAML with InstructLab Page 1.pdf" in result.output
        assert Path(expected_output_file).exists()


def test_convert_md_from_directory():
    """
    Tests converting from the sample Markdown file in tests/testdata/temp_datasets_documents/md.
    Verifies that the expected JSON file exists in the temp directory at the end.
    """
    test_input_dir = "tests/testdata/documents/md"
    with tempfile.TemporaryDirectory() as temp_dir:
        test_output_dir = os.path.join(temp_dir, 'convert-outputs')
        expected_output_file = os.path.join(test_output_dir, "hello.json")

        runner = CliRunner()
        result = runner.invoke(lab.ilab, ["rag", "convert", "--input-dir", test_input_dir, "--output-dir", test_output_dir])
        assert result.exit_code == 0
        assert Path(expected_output_file).exists()

# TODO: Tests for processing documents from a taxonomy.
