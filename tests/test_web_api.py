import json
import tempfile
import unittest
import importlib.util
from pathlib import Path
from unittest.mock import patch

import torch
from fastapi.testclient import TestClient

from runtime.farsitree.harraf2 import build_model
from webapp.app import app, SESSIONS


class WebDashboardApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def make_model_folder(self, root: Path):
        cfg = {
            "version": 2, "vocab_size": 48, "n_layer": 2, "n_head": 4,
            "n_kv_head": 2, "d_model": 32, "mlp_hidden": 64,
            "max_seq_len": 32, "tree": False, "parse_layer": 1,
            "tree_tau": 4.0, "mem_layers": [1], "mem_keys": 4,
            "mem_topk": 2, "dropout": 0.0,
        }
        model = build_model(cfg)
        torch.save({"model": model.state_dict(), "config": cfg}, root / "model.pt")
        (root / "info.json").write_text(json.dumps({"name": "حراف وب تست", "chat_tuned": False}), encoding="utf-8")
        (root / "state.json").write_text(json.dumps({"step": 10, "tokens": 1000}), encoding="utf-8")
        (root / "morph.json").write_text(json.dumps(["سلام", "کتاب"]), encoding="utf-8")
        (root / "tokenizer.json").write_text("{}", encoding="utf-8")

    def test_web_shell_and_health(self):
        page = self.client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn("Vazirmatn", page.text)
        self.assertIn("آزمایشگاه مدل‌های وزن‌باز", page.text)
        self.assertEqual(self.client.get("/api/health").json()["ok"], True)
        self.assertEqual(self.client.get("/static/app.js").status_code, 200)
        self.assertEqual(self.client.get("/static/app.css").status_code, 200)

    def test_local_load_explorer_edit_undo_export(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            self.make_model_folder(folder)
            loaded = self.client.post("/api/load-local-folder", json={"path": str(folder)})
            self.assertEqual(loaded.status_code, 200, loaded.text)
            data = loaded.json()
            sid = data["session_id"]
            headers = {"X-Session-ID": sid}
            summary = data["summary"]
            self.assertEqual(summary["layers"], 2)
            self.assertGreater(summary["tensor_count"], 0)
            tensors = self.client.get("/api/tensors?q=ln_f", headers=headers)
            self.assertEqual(tensors.status_code, 200)
            self.assertTrue(any(x["name"] == "ln_f.weight" for x in tensors.json()["items"]))
            detail = self.client.get("/api/tensor/ln_f.weight", headers=headers)
            self.assertEqual(detail.status_code, 200, detail.text)
            preview = self.client.post("/api/preview", headers=headers, json={"edit": {"tensor": "ln_f.weight", "type": "scale", "factor": 0.995}, "safe_mode": True})
            self.assertEqual(preview.status_code, 200, preview.text)
            macro = self.client.post("/api/validate-macro", headers=headers, json={"edits": [
                {"tensor": "ln_f.weight", "type": "scale", "factor": 0.995},
                {"tensor": "blocks.0.ln1.weight", "type": "scale", "factor": 1.005},
            ], "safe_mode": True})
            self.assertEqual(macro.status_code, 200, macro.text)
            self.assertEqual(macro.json()["count"], 2)
            applied = self.client.post("/api/apply", headers=headers, json={"edits": [{"tensor": "ln_f.weight", "type": "scale", "factor": 0.995}], "safe_mode": True})
            self.assertEqual(applied.status_code, 200, applied.text)
            undo = self.client.post("/api/undo", headers=headers)
            self.assertEqual(undo.status_code, 200, undo.text)
            redo = self.client.post("/api/redo", headers=headers)
            self.assertEqual(redo.status_code, 200, redo.text)
            diff = self.client.get("/api/weight-diff", headers=headers)
            self.assertEqual(diff.status_code, 200, diff.text)
            self.assertEqual(diff.json()["changed_tensors"], 1)
            export = self.client.post("/api/export", headers=headers)
            self.assertEqual(export.status_code, 200, export.text[:500])
            self.assertEqual(export.headers.get("content-type"), "application/zip")
            SESSIONS.pop(sid, None)

    def test_upload_folder_preserves_relative_paths(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.make_model_folder(root)
            files = []
            for p in root.iterdir():
                if p.is_file():
                    files.append(("files", (f"selected_model/{p.name}", p.read_bytes(), "application/octet-stream")))
            rels = [item[1][0] for item in files]
            response = self.client.post("/api/load-folder", data={"relative_paths": json.dumps(rels)}, files=files)
            self.assertEqual(response.status_code, 200, response.text)
            sid = response.json()["session_id"]
            inv = response.json()["summary"]["files"]
            self.assertTrue(any(x["name"] == "model.pt" for x in inv))
            SESSIONS.pop(sid, None)


    def test_full_model_plan_contains_auto_profile_and_applies_as_one_batch(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            self.make_model_folder(folder)
            loaded = self.client.post("/api/load-local-folder", json={"path": str(folder)})
            self.assertEqual(loaded.status_code, 200, loaded.text)
            sid = loaded.json()["session_id"]
            headers = {"X-Session-ID": sid}
            profile = self.client.get("/api/profile", headers=headers)
            self.assertEqual(profile.status_code, 200, profile.text)
            self.assertIn("خودرگرسیو", profile.json()["profile"]["task"])
            self.assertGreater(profile.json()["profile"]["tensor_count"], 0)
            self.client.post("/api/settings", headers=headers, json={
                "base_url": "http://localhost:11434/v1", "api_key": "ollama",
                "model": "fake-analyzer", "safe_mode": True,
            })
            fake_plan = {
                "model_identity": {"name": "حراف تست", "task": "مدل زبانی برای تولید متن"},
                "executive_summary": "گزارش تستی برای بررسی مسیر برنامه جامع.",
                "strengths": [{"title": "معماری مشخص", "evidence": "config داخلی"}],
                "weaknesses": [{"title": "نیازمند آزمون", "certainty": "hypothesis"}],
                "optimization_plan": [{"priority": 1, "title": "آزمون مقیاس نرمال‌ساز", "risk": "low"}],
                "suggested_edits": [{"tensor": "ln_f.weight", "type": "scale", "factor": 0.995,
                                     "reason": "آزمون کوچک و برگشت‌پذیر", "risk": "low", "target_relevance": 0.8}],
                "confidence": 0.7, "risk": "low", "reversible": True, "plan_scope": "whole_model",
                "limitations": "تست ساختاری بهبود را تضمین نمی‌کند."
            }
            with patch("webapp.app.request_full_model_plan", return_value=fake_plan) as mocked:
                result = self.client.post("/api/proposal", headers=headers, json={
                    "symptom": "بررسی کل مدل", "mode": "ai", "full_plan": True,
                    "prompts": ["سلام؛ خودت را معرفی کن."],
                })
                self.assertEqual(result.status_code, 200, result.text)
                self.assertEqual(result.json()["proposal"]["plan_scope"], "whole_model")
                self.assertEqual(len(result.json()["proposal"]["suggested_edits"]), 1)
                sent_dossier = mocked.call_args.args[3]
                self.assertIn("all_model_tensors", sent_dossier)
                self.assertIn("all_trainable_tensors", sent_dossier)
                self.assertIn("model_metadata", sent_dossier)
                self.assertIn("layer_summary", sent_dossier["whole_model"])
                self.assertGreaterEqual(len(sent_dossier["all_model_tensors"]), len(sent_dossier["all_trainable_tensors"]))
                self.assertGreater(len(sent_dossier["all_trainable_tensors"]), 0)
            applied = self.client.post("/api/apply", headers=headers, json={
                "edits": fake_plan["suggested_edits"], "safe_mode": True,
                "label": "اعمال یک‌جای برنامه جامع",
            })
            self.assertEqual(applied.status_code, 200, applied.text)
            history = self.client.get("/api/history", headers=headers).json()
            self.assertEqual(history["items"][-1]["label"], "اعمال یک‌جای برنامه جامع")
            self.assertEqual(len(history["items"][-1]["edits"]), 1)
            SESSIONS.pop(sid, None)


    @unittest.skipUnless(importlib.util.find_spec("transformers"), "Transformers is installed by the project requirements")
    def test_nested_huggingface_bin_checkpoint_is_detected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model_folder = root / "nested" / "model"
            model_folder.mkdir(parents=True)
            state = {
                "embeddings.word_embeddings.weight": torch.randn(20, 8),
                "encoder.layer.0.attention.self.query.weight": torch.randn(8, 8),
            }
            torch.save(state, model_folder / "pytorch_model.bin")
            (model_folder / "config.json").write_text(json.dumps({
                "model_type": "bert", "architectures": ["BertModel"],
                "hidden_size": 8, "num_hidden_layers": 1,
            }), encoding="utf-8")
            loaded = self.client.post("/api/load-local-folder", json={"path": str(root)})
            self.assertEqual(loaded.status_code, 200, loaded.text)
            sid = loaded.json()["session_id"]
            headers = {"X-Session-ID": sid}
            self.assertEqual(loaded.json()["summary"]["model_kind"], "Hugging Face")
            profile = self.client.get("/api/profile", headers=headers).json()["profile"]
            self.assertIn("رمزگذار", profile["task"])
            exported = self.client.post("/api/export", headers=headers)
            self.assertEqual(exported.status_code, 200, exported.text[:300])
            SESSIONS.pop(sid, None)

    def test_generic_state_dict_auto_profile_and_safe_edit(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            state = {
                "model.embed_tokens.weight": torch.randn(24, 8),
                "model.layers.0.self_attn.q_proj.weight": torch.randn(8, 8),
                "model.layers.0.mlp.gate_proj.weight": torch.randn(16, 8),
                "lm_head.weight": torch.randn(24, 8),
            }
            torch.save(state, folder / "model.pt")
            (folder / "config.json").write_text(json.dumps({
                "model_type": "llama", "architectures": ["LlamaForCausalLM"],
                "hidden_size": 8, "num_hidden_layers": 1, "vocab_size": 24,
            }), encoding="utf-8")
            loaded = self.client.post("/api/load-local-folder", json={"path": str(folder)})
            self.assertEqual(loaded.status_code, 200, loaded.text)
            sid = loaded.json()["session_id"]
            headers = {"X-Session-ID": sid}
            summary = loaded.json()["summary"]
            self.assertFalse(summary["model_executable"])
            self.assertEqual(summary["model_kind"], "مدل عمومی (حالت بررسی/ویرایش)")
            profile = self.client.get("/api/profile", headers=headers).json()["profile"]
            self.assertIn("خودرگرسیو", profile["task"])
            rows = self.client.get("/api/tensors?q=q_proj", headers=headers).json()["items"]
            self.assertTrue(rows)
            self.assertIn("Query", rows[0]["role"])
            applied = self.client.post("/api/apply", headers=headers, json={
                "edits": [{"tensor": "model.layers.0.self_attn.q_proj.weight", "type": "scale", "factor": 0.995}],
                "safe_mode": True,
                "label": "generic test edit",
            })
            self.assertEqual(applied.status_code, 200, applied.text)
            diff = self.client.get("/api/weight-diff", headers=headers)
            self.assertEqual(diff.status_code, 200, diff.text)
            self.assertEqual(diff.json()["changed_tensors"], 1)
            self.assertGreater(diff.json()["changed_elements"], 0)
            self.assertFalse(diff.json()["model_executable"])
            generated = self.client.post("/api/generate", headers=headers, json={"prompt": "Hello"})
            self.assertEqual(generated.status_code, 400)
            self.assertIn("adapter", generated.json()["detail"])
            SESSIONS.pop(sid, None)


if __name__ == "__main__":
    unittest.main()
