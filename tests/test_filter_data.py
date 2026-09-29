import io
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from filter_data import _coerce_str, filter_record, filter_text, main, normalize_text


# A comfortably varied English sentence, so the diversity and length gates pass
# and each test can turn on the one thing it is about.
GOOD = "The quick brown fox jumps over the lazy dog while birds sing."
HINDI = "भारत एक विशाल देश है जहाँ अनेक भाषाएँ बोली जाती हैं और लोग धान खाते हैं।"


def _all_thresholds_off(**over):
    """Every tunable gate lowered, so only unconditional rules remain."""
    kw = dict(min_chars=0, max_chars=10**6, min_unique=0, min_dev=0,
              min_latin=0, dev_ratio=0.0, latin_ratio=0.0)
    kw.update(over)
    return kw


# --- the bug: an empty record is never worth training on -------------------

@pytest.mark.parametrize("empty", ["", "   ", "\n\n", " \t \n ", "‌‍﻿",
                                   "\r\n\r\n", "\x00\x01\x02"])
def test_empty_text_is_rejected_with_every_gate_disabled(empty):
    """No threshold setting should be able to admit a record with no content.

    Every gate in filter_text is a tunable threshold, and the CLI exposes them
    all. With --min_chars 0 --min_unique 0 the length and diversity gates both
    pass, and the filter used to hand back "" -- which stream_data then yields
    as a document, and filter_data's own main() printed as {"text": ""}.
    """
    assert filter_text(empty, "auto", **_all_thresholds_off()) is None


def test_empty_text_via_filter_record_is_rejected():
    assert filter_record({"text": "   "}, **_all_thresholds_off()) is None
    assert filter_record({"prompt": "", "completion": ""}, **_all_thresholds_off()) is None


def test_the_hindi_and_english_gates_cannot_rescue_an_empty_record():
    for dataset in ("hindi", "english", "auto", "openthoughts"):
        assert filter_text("", dataset, **_all_thresholds_off()) is None, dataset


# --- normalization ---------------------------------------------------------

def test_normalize_strips_bom_zero_width_and_control_but_keeps_indentation():
    assert normalize_text("\ufeffhello\x07 world") == "hello world"
    # Code indentation is content, not whitespace to be trimmed.
    assert normalize_text("\n\n    def f():\n        return 1\n\n") == \
        "    def f():\n        return 1", "the first line's indentation is content"


def test_normalize_drops_blank_lines_at_the_ends_but_keeps_indentation():
    """The behaviour normalize_text's comment promised and did not deliver.

    Its final str.strip() had no argument, so it removed all leading and
    trailing whitespace -- including the first content line's indentation. Any
    code snippet that opened indented lost it, in a corpus that deliberately
    ingests .py/.cpp/.rs/.java.
    """
    assert normalize_text("\n\n    x = 1\n    y = 2\n\n") == "    x = 1\n    y = 2"
    assert normalize_text("\n\n        deep = 1\n    shallow = 2\n\n") == \
        "        deep = 1\n    shallow = 2"
    # Trailing spaces on the last content line are trimmed with the blank lines
    # only if the line is blank; real trailing spaces are harmless and kept.
    assert normalize_text("\n\n  padded  \n\n") == "  padded  "
    # Whitespace-only lines at the ends still go.
    assert normalize_text("\n\n   \n\n  kept  \n \n\n") == "  kept  "


def test_normalize_folds_crlf_and_applies_nfkc():
    assert normalize_text("a\r\nb\rc") == "a\nb\nc"
    # NFKC: fullwidth digits and a ligature fold to ASCII.
    assert normalize_text("１２３ ﬁ") == "123 fi"


# --- the length and diversity gates ----------------------------------------

def test_length_bounds():
    assert filter_text("short", min_chars=20) is None
    assert filter_text(GOOD, min_chars=20) == GOOD
    assert filter_text(GOOD, min_chars=len(GOOD)) == GOOD
    assert filter_text(GOOD, min_chars=len(GOOD) + 1) is None
    assert filter_text(GOOD, max_chars=len(GOOD) - 1) is None


def test_the_cheap_length_guard_precedes_normalization():
    """A 3M-character input must be refused before NFKC and set() run on it."""
    huge = "a" * 3_000_000
    assert filter_text(huge, max_chars=10**6) is None


def test_length_bounds_are_validated():
    for kw in (dict(min_chars=-1), dict(max_chars=-1), dict(min_chars=10, max_chars=5)):
        with pytest.raises(ValueError, match="0 <= min_chars <= max_chars"):
            filter_text(GOOD, "auto", **kw)


def test_a_repetitive_record_is_rejected_for_lack_of_diversity():
    assert filter_text("a" * 500, min_chars=20) is None
    assert filter_text("ab" * 250, min_chars=20) is None, "two distinct characters is not text"


def test_diversity_is_measured_without_spaces():
    assert filter_text("the cat sat on the mat and the dog sat too", min_chars=20)


# --- content gates ---------------------------------------------------------

def test_a_replacement_character_means_the_decode_failed():
    assert filter_text(GOOD + "�", min_chars=20) is None


def test_a_bare_url_is_rejected():
    assert filter_text("https://example.com/a/b/c?q=1", min_chars=5) is None
    assert filter_text("www.example.com", min_chars=5) is None
    # A document that merely contains a URL is fine.
    assert filter_text("See https://example.com for details about the subject.",
                      min_chars=20) is not None


def test_non_string_input_is_rejected_not_raised():
    for value in (None, 42, 3.5, [], {}, b"bytes", object()):
        assert filter_text(value, min_chars=1) is None


# --- the per-dataset script gates -----------------------------------------

def test_hindi_gate():
    assert filter_text(HINDI, "hindi") == HINDI
    assert filter_text(GOOD, "hindi") is None, "pure English must not pass the hindi gate"
    assert filter_text(HINDI, "english") is None, "and not the english one either"


def test_english_gate():
    assert filter_text(GOOD, "english") == GOOD
    assert filter_text(HINDI, "english") is None


def test_auto_and_openthoughts_apply_no_script_gate():
    for dataset in ("auto", "openthoughts"):
        assert filter_text(HINDI, dataset) == HINDI
        assert filter_text(GOOD, dataset) == GOOD


def test_a_short_script_snippet_is_rejected_even_when_pure():
    """Purity alone is not enough; the absolute count matters too."""
    assert filter_text("नमस्ते", "hindi", min_chars=1) is None, "too few Devanagari chars"
    assert filter_text("hi", "english", min_chars=1) is None, "too few Latin chars"


def test_the_script_gates_are_ratios_not_detectors():
    """Mostly-Hindi text with English in it passes the hindi gate, and vice versa.

    The gate is a fraction of alphabetic characters, not a detector, so it only
    rejects text that is predominantly the *other* script. Worth pinning because
    "mixed text must fail both" is a natural assumption and it is wrong -- which
    is the documented reason --dataset auto exists.
    """
    hindi_heavy = HINDI + " and also a few English words here"
    assert filter_text(hindi_heavy, "hindi") == hindi_heavy
    assert filter_text(hindi_heavy, "auto") == hindi_heavy

    latin_heavy = GOOD + " and here is some Hindi भारत देश"
    assert filter_text(latin_heavy, "english") == latin_heavy
    # But the minority script alone does not carry it past the other gate.
    assert filter_text(latin_heavy, "hindi") is None


# --- record shape ----------------------------------------------------------

def test_filter_record_combines_prompt_and_completion():
    record = {"Prompt": "Explain this clearly.", "Completion": "This is the explanation."}
    assert filter_record(record, min_chars=1) == "Explain this clearly.\nThis is the explanation."


def test_filter_record_accepts_prompt_without_completion():
    assert filter_record({"prompt": "A sufficiently varied prompt."}, min_chars=1) == \
        "A sufficiently varied prompt."


def test_filter_record_falls_back_through_the_text_keys():
    for key in ("text", "content", "document", "body", "code", "TEXT", "Content"):
        assert filter_record({key: GOOD}, min_chars=20) == GOOD, key


def test_filter_record_rejects_a_non_dict():
    for value in (None, "a string", 42, ["a", "list"]):
        assert filter_record(value, min_chars=1) is None


def test_filter_record_prefers_a_prompt_completion_pair_over_a_text_field():
    record = {"text": GOOD, "prompt": "What is the capital of France?",
              "completion": "It is Paris, a city of many museums."}
    assert filter_record(record, min_chars=1) == \
        "What is the capital of France?\nIt is Paris, a city of many museums."


@pytest.mark.parametrize("value,expected", [
    ("plain", "plain"),
    (["a", "  ", "b"], "a\nb"),
    ([{"value": "v"}, {"text": "t"}, {"content": "c"}], "v\nt\nc"),
    ([{"value": "v"}, "second"], "v\nsecond"),
    ([{"nothing": "useful"}], None),
    ([], None),
    (42, None),
    (None, None),
])
def test_coerce_str_handles_chat_shapes(value, expected):
    assert _coerce_str(value) == expected


# --- the CLI ---------------------------------------------------------------

def _run_main(monkeypatch, capsys, lines, argv):
    monkeypatch.setattr(sys, "argv", ["filter_data.py", *argv])
    monkeypatch.setattr(sys, "stdin", io.StringIO(lines))
    main()
    out = capsys.readouterr()
    return [json.loads(ln) for ln in out.out.splitlines() if ln.strip()], out.err


def test_cli_filters_jsonl_from_stdin(monkeypatch, capsys):
    lines = "\n".join([
        json.dumps({"text": GOOD}),
        json.dumps({"text": "too short"}),
        json.dumps({"text": "a" * 500}),
        json.dumps({"text": HINDI}),
    ])
    kept, err = _run_main(monkeypatch, capsys, lines, ["--dataset", "english"])
    assert [k["text"] for k in kept] == [GOOD]
    assert "kept 1 records" in err


def test_cli_skips_malformed_lines_without_stopping(monkeypatch, capsys):
    lines = "\n".join([
        "{not json",
        json.dumps({"text": GOOD}),
        "",
        json.dumps(["not", "a", "dict"]),
        json.dumps({"nothing": "useful"}),
    ])
    kept, _err = _run_main(monkeypatch, capsys, lines, [])
    assert [k["text"] for k in kept] == [GOOD], "one bad line must not abort the stream"


def test_cli_never_emits_an_empty_record(monkeypatch, capsys):
    """The end-to-end version of the empty-record bug, through the real CLI."""
    lines = "\n".join([
        json.dumps({"text": ""}),
        json.dumps({"text": "   "}),
        json.dumps({"prompt": "", "completion": ""}),
        json.dumps({"text": GOOD}),
    ])
    kept, _err = _run_main(monkeypatch, capsys, lines,
                           ["--min_chars", "0", "--min_unique", "0",
                            "--min_dev", "0", "--min_latin", "0",
                            "--dev_ratio", "0", "--latin_ratio", "0"])
    assert [k["text"] for k in kept] == [GOOD], kept
    assert all(k["text"].strip() for k in kept), "an empty record got through"


def test_cli_reports_a_zero_count(monkeypatch, capsys):
    kept, err = _run_main(monkeypatch, capsys, json.dumps({"text": "x"}), [])
    assert kept == [] and "kept 0 records" in err
