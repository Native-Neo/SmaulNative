import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from filter_data import filter_record


def test_filter_record_combines_prompt_and_completion():
    record = {"Prompt": "Explain this clearly.", "Completion": "This is the explanation."}
    assert filter_record(record, min_chars=1) == "Explain this clearly.\nThis is the explanation."


def test_filter_record_accepts_prompt_without_completion():
    assert filter_record({"prompt": "A sufficiently varied prompt."}, min_chars=1) == "A sufficiently varied prompt."
