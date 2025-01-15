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

        # The sample directory has a file called 'How to use YAML with InstructLab Page 1.pdf', which Docling converts to 'YAML with InstructLab Page 1.json'
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

        # The sample directory has a file called hello.md, which Docling converts to hello.json
        expected_output_file = os.path.join(test_output_dir, "hello.json")

        runner = CliRunner()
        result = runner.invoke(lab.ilab, ["rag", "convert", "--input-dir", test_input_dir, "--output-dir", test_output_dir])
        assert result.exit_code == 0
        assert Path(expected_output_file).exists()


# Note: This test uses a local taxonomy that references a file in a remote git repository.  For that reason, this
# test won't pass when run on a machine with no connection to the internet.  It wil also fail if the repository
# is not working or if it ever gets deleted.  That's not ideal, but we do need to test these capabilities.
# TODO: Consider re-working this with a mock for the github server.
def test_convert_md_from_taxonomy():
    """
    Tests converting from the sample Markdown file in a github repo referenced in tests/testdata/sample_taxonomy.
    The specific file referenced is phoenix.md in https://github.com/RedHatOfficial/rhelai-taxonomy-data,
    because that is the one in
    https://github.com/instructlab/sdg/blob/main/tests/testdata/test_valid_knowledge_skill.yaml.
    Verifies that the expected JSON file exists in the temp directory at the end.
    """
    test_input_taxonomy = "tests/testdata/sample_taxonomy"
    with tempfile.TemporaryDirectory() as temp_dir:
        test_output_dir = os.path.join(temp_dir, 'convert-outputs')

        # The sample qna.yaml points to a file called phoenix.md, which Docling converts to phoenix.json
        expected_output_file = os.path.join(test_output_dir, "phoenix.json")

        runner = CliRunner()
        result = runner.invoke(lab.ilab, ["rag", "convert", "--taxonomy-path", test_input_taxonomy, "--taxonomy-base", "empty", "--output-dir", test_output_dir])
        assert result.exit_code == 0
        assert "Transforming source files ['phoenix.md']" in result.output
        assert Path(expected_output_file).exists()

# Note that there is no test for converting pdf from taxonomy.  The tests above verify that you can convert
# both PDF and md and that you can convert from both a directory and a taxonomy.  Testing a PDF from a
# taxonomy too seems redundant, and these tests are already taking a lot of time so I don't want to add
# another expensive one for that purpose.
