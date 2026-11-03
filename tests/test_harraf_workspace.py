import json
import tempfile
import unittest
import sys
import types
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

import torch

from runtime.farsitree.harraf2 import build_model
from weightspace.loader import load_harraf_folder, export_edited_folder
from weightspace.editor import EditHistory, edit_tensor, preview_edit, validate_proposal
from weightspace.analysis import tensor_role, response_metrics, activation_probe, stats
from weightspace.loader import generate_text, is_huggingface_folder, load_huggingface_folder


class _Encoded:
    def __init__(self, ids):
        self.ids = ids


class _FakeTokenizer:
    def get_vocab_size(self, with_added_tokens=True):
        return 48

    def encode(self, text):
        return _Encoded([1, 2, 3])

    def token_to_id(self, token):
        return None

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"واژه{n}" for n in ids)


class HarrafFolderTests(unittest.TestCase):
    def make_folder(self, root: Path):
        # A small Harraf2 network with the same export contract as the real model.
        cfg = {
            "version": 2,
            "vocab_size": 48,
            "n_layer": 2,
            "n_head": 4,
            "n_kv_head": 2,
            "d_model": 32,
            "mlp_hidden": 64,
            "max_seq_len": 32,
            "tree": False,
            "parse_layer": 1,
            "tree_tau": 4.0,
            "mem_layers": [1],
            "mem_keys": 4,
            "mem_topk": 2,
            "dropout": 0.0,
            "version": 2,
        }
        model = build_model(cfg)
        torch.save({"model": model.state_dict(), "config": cfg}, root / "model.pt")
        (root / "info.json").write_text(json.dumps({"name": "حراف تست", "arch": "harraf"}, ensure_ascii=False), encoding="utf-8")
        (root / "state.json").write_text(json.dumps({"step": 12, "tokens": 1000, "history": []}), encoding="utf-8")
        (root / "morph.json").write_text(json.dumps(["کتاب", "رفت", "خانه"], ensure_ascii=False), encoding="utf-8")
        # A placeholder tokenizer: the loader should report its invalid format but still inspect the model.
        (root / "tokenizer.json").write_text("{}", encoding="utf-8")
        (root / "extra.txt").write_text("این فایل باید در موجودی پوشه دیده شود.", encoding="utf-8")
        return cfg

    def test_detects_sharded_huggingface_folder(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "config.json").write_text(json.dumps({"model_type": "llama", "architectures": ["LlamaForCausalLM"]}), encoding="utf-8")
            (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"model.embed_tokens.weight": "model-00001-of-00002.safetensors"}}), encoding="utf-8")
            (root / "model-00001-of-00002.safetensors").write_bytes(b"placeholder")
            self.assertTrue(is_huggingface_folder(root))

    def test_huggingface_folder_generation_and_export_adapter(self):
        class FakeConfig:
            is_encoder_decoder = False
            model_type = "tiny-test"
            architectures = ["TinyForCausalLM"]
            max_position_embeddings = 32
            vocab_size = 10
            def to_dict(self):
                return {"model_type": self.model_type, "architectures": self.architectures,
                        "max_position_embeddings": self.max_position_embeddings, "vocab_size": self.vocab_size}

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = FakeConfig()
                self.embed_tokens = torch.nn.Embedding(10, 4)
                self.lm_head = torch.nn.Linear(4, 10)
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()
            def generate(self, input_ids, **kwargs):
                out = torch.cat([input_ids, torch.tensor([[3]], dtype=torch.long)], dim=1)
                return SimpleNamespace(sequences=out, scores=(torch.zeros((1, 10)),))
            def forward(self, input_ids, **kwargs):
                hidden = self.embed_tokens(input_ids)
                return SimpleNamespace(hidden_states=(hidden, hidden), logits=self.lm_head(hidden))
            def save_pretrained(self, folder, **kwargs):
                Path(folder, "model.safetensors").write_bytes(b"fake safe weights")
                Path(folder, "config.json").write_text(json.dumps(self.config.to_dict()), encoding="utf-8")

        class FakeAutoConfig:
            @staticmethod
            def from_pretrained(*args, **kwargs): return FakeConfig()
        class FakeTokenizer:
            vocab_size = 10
            pad_token_id = 0
            eos_token_id = 2
            @classmethod
            def from_pretrained(cls, *args, **kwargs): return cls()
            def __call__(self, text, **kwargs): return {"input_ids": torch.tensor([[1, 2]])}
            def decode(self, ids, **kwargs): return "پاسخ آزمایشی"
            def save_pretrained(self, folder): Path(folder, "tokenizer.json").write_text("{}", encoding="utf-8")
        class UnusedAuto:
            @classmethod
            def from_pretrained(cls, *args, **kwargs): raise RuntimeError("unused auto class")

        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoConfig = FakeAutoConfig
        fake_transformers.AutoModel = UnusedAuto
        fake_transformers.AutoModelForCausalLM = FakeModel
        fake_transformers.AutoModelForMaskedLM = UnusedAuto
        fake_transformers.AutoModelForSeq2SeqLM = UnusedAuto
        fake_transformers.AutoModelForSequenceClassification = UnusedAuto
        fake_transformers.AutoTokenizer = FakeTokenizer

        with tempfile.TemporaryDirectory() as td, patch.dict(sys.modules, {"transformers": fake_transformers}):
            root = Path(td) / "hf-model"
            root.mkdir()
            (root / "config.json").write_text("{}", encoding="utf-8")
            (root / "model.safetensors").write_bytes(b"placeholder")
            ws = load_huggingface_folder(root, hf_model_id="owner/tiny-model")
            self.assertEqual(ws.model_kind, "Hugging Face")
            self.assertEqual(ws.generation_task, "causal_lm")
            result = generate_text(ws, "سلام", max_new_tokens=2)
            self.assertEqual(result["text"], "پاسخ آزمایشی")
            out = Path(td) / "exported"
            export_edited_folder(ws, out)
            self.assertTrue((out / "model.safetensors").exists())
            self.assertTrue((out / "tokenizer.json").exists())
            self.assertTrue((out / "weightspace_changes.json").exists())

    def test_loads_config_from_model_pt_without_config_json(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = self.make_folder(root)
            ws = load_harraf_folder(root)
            self.assertEqual(ws.config["d_model"], cfg["d_model"])
            self.assertEqual(ws.config["n_layer"], 2)
            self.assertIsNotNone(ws.morph)
            self.assertTrue(any(r["name"] == "wte.weight" for r in ws.tensor_rows))
            self.assertTrue(any(r["path"] == "extra.txt" for r in ws.inventory))
            self.assertFalse((root / "config.json").exists())
            self.assertIsNotNone(ws.tokenizer_error)

    def test_export_preserves_model_assets(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "src"; root.mkdir()
            self.make_folder(root)
            ws = load_harraf_folder(root)
            out = Path(td) / "nested" / "exports" / "edited"
            export_edited_folder(ws, out)
            for f in ("model.pt", "info.json", "morph.json", "state.json", "tokenizer.json", "extra.txt", "weightspace_changes.json"):
                self.assertTrue((out / f).exists(), f)
            reloaded = torch.load(out / "model.pt", map_location="cpu", weights_only=True)
            self.assertIn("config", reloaded)
            self.assertIn("model", reloaded)

    def test_edit_history_undo_redo(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); self.make_folder(root)
            ws = load_harraf_folder(root)
            params = dict(ws.model.named_parameters(remove_duplicate=False))
            before = params["ln_f.weight"].detach().clone()
            history = EditHistory()
            history.commit(ws.model, [{"tensor": "ln_f.weight", "type": "scale", "factor": 0.995}], True)
            after = params["ln_f.weight"].detach().clone()
            self.assertFalse(torch.equal(before, after))
            history.undo(ws.model)
            self.assertTrue(torch.equal(before, params["ln_f.weight"]))
            history.redo(ws.model)
            self.assertTrue(torch.equal(after, params["ln_f.weight"]))

    def test_preview_and_safe_limits(self):
        x = torch.tensor([1., 2., 3., 4.])
        p = preview_edit(x, {"type": "scale_slice", "start": 1, "end": 3, "factor": 0.995})
        self.assertGreater(p["delta_norm"], 0)
        with self.assertRaises(ValueError):
            edit_tensor(x, {"type": "scale", "factor": 1.2}, safe_mode=True)
        # Safe Mode must not be bypassed by selecting an extreme clamp range.
        with self.assertRaises(ValueError):
            edit_tensor(x, {"type": "clamp", "min_value": 1.5, "max_value": 2.0}, safe_mode=True)

    def test_architecture_role_labels(self):
        self.assertIn("Query", tensor_role("blocks.3.attn.q.weight")[0])
        self.assertIn("حافظه", tensor_role("blocks.4.mem.values.weight")[0])

    def test_real_forward_activation_probe_and_text_generation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = self.make_folder(root)
            # Turn on the real latent parser path to exercise its hook as well.
            cfg["tree"] = True
            model = build_model(cfg)
            torch.save({"model": model.state_dict(), "config": cfg}, root / "model.pt")
            ws = load_harraf_folder(root)
            ws.tokenizer = _FakeTokenizer()
            ws.tokenizer_error = None
            ws.info["chat_tuned"] = True
            probe = activation_probe(ws, "سلام")
            self.assertEqual(len(probe["layers"]), cfg["n_layer"])
            self.assertIn("mean", probe["parser"])
            answer = generate_text(ws, "سلام", max_new_tokens=4, temperature=0.0)
            self.assertEqual(answer["token_count"], 4)
            self.assertTrue(answer["text"])

    def test_repetition_metrics(self):
        m = response_metrics("الف ب پ الف ب پ الف ب پ")
        self.assertGreater(m["repeated_trigram_ratio"], 0.4)

    def test_proposal_ranking_uses_risk_then_change_size(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); self.make_folder(root)
            ws = load_harraf_folder(root)
            proposal = {
                "confidence": 0.7,
                "suggested_edits": [
                    {"tensor": "ln_f.weight", "type": "scale", "factor": 1.01, "risk": "medium", "target_relevance": 0.9},
                    {"tensor": "blocks.0.ln1.weight", "type": "scale", "factor": 1.015, "risk": "low", "target_relevance": 0.6},
                    {"tensor": "blocks.1.ln2.weight", "type": "scale", "factor": 1.002, "risk": "low", "target_relevance": 0.8},
                ],
            }
            checked = validate_proposal(proposal, ws.model, safe_mode=True)
            self.assertEqual(checked["suggested_edits"][0]["tensor"], "blocks.1.ln2.weight")
            self.assertEqual(checked["suggested_edits"][1]["tensor"], "blocks.0.ln1.weight")
            self.assertEqual(checked["suggested_edits"][2]["tensor"], "ln_f.weight")

    def test_stats_finds_nonfinite_outside_histogram_sample(self):
        x = torch.ones(50000)
        x[-1] = float("nan")
        result = stats(x, hist_samples=100)
        self.assertEqual(result["nonfinite"], 1)
        self.assertTrue(result["statistics_sampled"])


if __name__ == "__main__":
    unittest.main()
