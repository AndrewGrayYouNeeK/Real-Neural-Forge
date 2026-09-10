"""Tests for the local inference CLI helpers."""

import json

import pytest

from src.predict import parse_sequence


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
