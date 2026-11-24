# Copyright 2026 Nuri ML contributors
import json

import pytest
from pydantic import ValidationError

from prepare_corpus import extract_sentences


def test_corpus_extraction_uses_only_sentence_text() -> None:
    payload = json.dumps(
        {
            "named_entity": [
                {
                    "title": [{"sentence": "TITLE", "labels": [{"text": "SECRET"}]}],
                    "subtitle": [],
                    "content": [{"sentence": " \u1100\u1161\tABC __label__2\n"}, {"sentence": " "}],
                    "source_site": "METADATA",
                }
            ],
        }
    ).encode()
    assert extract_sentences(payload) == ("title", "\uac00 abc label2")


def test_corpus_rejects_wrong_schema_instead_of_silently_skipping() -> None:
    with pytest.raises(ValidationError):
        _ = extract_sentences(b'{"SJML":{"text":[]}}')
