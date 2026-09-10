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


def test_generate_disables_dropout_and_restores_training_state():
    class Tokenizer:
        def encode(self, text):
            return type("Encoded", (), {"ids": [2]})()

        def decode(self, ids):
            return "x"

    class Model:
        training = True

        def eval(self):
            self.training = False
            return self

        def train(self, mode=True):
            self.training = mode
            return self

        def __call__(self, ids, state=None, use_cache=False, return_logits=True):
            return torch.tensor([[[10.0, 0.0, 0.0]]]), None, state

    obj = AutoRL.__new__(AutoRL)
    obj.model = Model()
    obj.tokenizer = Tokenizer()
    obj.device = torch.device("cpu")
    obj.eos_id = 99
    obj.bos_id = 2
    obj._encode = lambda text: [2]
    obj._decode = lambda ids: "x"
    text, tokens, logprobs = obj.generate("prompt", 1, 1.0, 0, 1.0)
    assert text == "x"
    assert tokens == [0]
    assert len(logprobs) == 1
    assert obj.model.training
