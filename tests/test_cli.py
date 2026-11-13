"""Unit tests for the pipeline CLI commands."""

import pytest
from unittest.mock import patch, MagicMock

from src.cli.pipeline_cli import main


class TestPipelineCLI:
    @patch("src.cli.pipeline_cli.cmd_download")
    def test_cli_download_args(self, mock_cmd):
        mock_cmd.return_value = 0
        with patch("sys.argv", ["pipeline_cli.py", "download", "--start", "2024-05-01", "--end", "2024-05-02"]):
            code = main()
            assert code == 0
            assert mock_cmd.called

    @patch("src.cli.pipeline_cli.cmd_validate")
    def test_cli_validate_args(self, mock_cmd):
        mock_cmd.return_value = 0
        with patch("sys.argv", ["pipeline_cli.py", "validate", "--timeframe", "m5"]):
            code = main()
            assert code == 0
            assert mock_cmd.called

    @patch("src.cli.pipeline_cli.cmd_duckdb_summary")
    def test_cli_duckdb_summary_args(self, mock_cmd):
        mock_cmd.return_value = 0
        with patch("sys.argv", ["pipeline_cli.py", "duckdb-summary", "--timeframe", "m5"]):
            code = main()
            assert code == 0
            assert mock_cmd.called

    @patch("src.cli.pipeline_cli.cmd_query")
    def test_cli_query_args(self, mock_cmd):
        mock_cmd.return_value = 0
        with patch("sys.argv", ["pipeline_cli.py", "query", "--sql", "SELECT 1"]):
            code = main()
            assert code == 0
            assert mock_cmd.called
