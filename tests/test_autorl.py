import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from autorl import AutoRL, PreferenceModel


def test_preference_model_changes_with_prompt():
    model = PreferenceModel(32, 8)
    tokens = torch.tensor([[2, 3, 1], [2, 4, 1]])
    mask = torch.ones_like(tokens, dtype=torch.bool)
    scores = model(tokens, mask)
    assert scores.shape == (2,)


def test_preference_pair_keeps_prompt_separator():
    obj = AutoRL.__new__(AutoRL)
    obj.eos_id = 1
    obj.device = torch.device("cpu")
    obj._encode = lambda text: [2] if text == "prompt" else [3]
    tokens, mask = obj._batch_pairs("prompt", ["response"])
    assert tokens.tolist() == [[2, 1, 3]]
    assert mask.tolist() == [[True, True, True]]


def test_preference_training_counter_starts_at_existing_records(tmp_path):
    path = tmp_path / "preferences.jsonl"
    path.write_text('{"prompt":"q","responses":["a","b"],"chosen":0}\n')
    obj = AutoRL.__new__(AutoRL)
    obj.preference_path = path
    assert obj.preference_count() == 1
