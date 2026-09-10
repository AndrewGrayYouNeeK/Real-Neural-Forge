"""Tests for the local inference CLI helpers."""

from pathlib import Path

import pytest

from src.predict import parse_sequence, sequence_from_csv


class TestParseSequence:
    def test_comma_separated(self):
        assert parse_sequence("0.1,0.2,0.3") == [[0.1], [0.2], [0.3]]

    def test_json_flat_list(self):
        assert parse_sequence("[1, 2, 3]") == [[1.0], [2.0], [3.0]]

    def test_json_nested_list(self):
        assert parse_sequence("[[1.0], [2.0]]") == [[1.0], [2.0]]

    def test_invalid_token_rejected(self):
        with pytest.raises(ValueError):
            parse_sequence("1,nope,3")


class TestSequenceFromCsv:
    def test_uses_last_seq_len_rows(self, tmp_path: Path):
        csv_path = tmp_path / "series.csv"
        csv_path.write_text("value\n1\n2\n3\n4\n")
        assert sequence_from_csv(str(csv_path), 2) == [[3.0], [4.0]]
