import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import stream_data


def test_conversation_skips_malformed_rows():
    value = [None, {}, {"from": "user"}, {"from": "user", "value": " hello "}, {"value": "world"}]
    assert stream_data._conversation(value) == "user: hello\nworld"


def test_stream_validation_rejects_bad_bounds():
    try:
        list(stream_data.stream_dataset("hindi", min_chars=10, max_chars=5))
    except ValueError:
        return
    raise AssertionError("invalid character bounds must fail")
