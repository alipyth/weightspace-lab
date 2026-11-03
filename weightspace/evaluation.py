from __future__ import annotations

import gc
from typing import Any

import torch

from runtime.farsitree.harraf2 import build_model
from .analysis import response_metrics
from .loader import generate_text, load_huggingface_folder


def compare_checkpoint(workspace, prompts: list[str], max_new_tokens: int = 80,
                       temperature: float = 0.0, top_p: float = 0.9) -> dict[str, Any]:
    prompts = [p.strip() for p in prompts if p.strip()]
    if not prompts:
        raise ValueError("حداقل یک پرامپت آزمایشی وارد کن.")
    if workspace.tokenizer is None:
        raise RuntimeError(workspace.tokenizer_error or "توکنایزر آماده نیست.")

    # Re-load the original, untouched weights for a fair before/after comparison.
    # HF folders remain unchanged on disk; Harraf stores its baseline state in model.pt.
    if workspace.model_kind == "Hugging Face":
        baseline_workspace = load_huggingface_folder(workspace.source_dir, workspace.hf_model_id)
        baseline = baseline_workspace.model
        baseline.cpu().eval()
        baseline_generate = lambda prompt: generate_text(baseline_workspace, prompt, max_new_tokens, temperature, top_p)
    else:
        baseline = build_model(dict(workspace.config))
        baseline.load_state_dict(workspace.original_checkpoint["model"], strict=True)
        baseline.cpu().eval()
        baseline_generate = lambda prompt: generate_text(workspace, prompt, max_new_tokens, temperature, top_p, model=baseline)

    rows = []
    try:
        for index, prompt in enumerate(prompts, 1):
            before = baseline_generate(prompt)
            after = generate_text(workspace, prompt, max_new_tokens, temperature, top_p, model=workspace.model)
            bm, am = response_metrics(before["text"]), response_metrics(after["text"])
            rows.append({
                "index": index,
                "prompt": prompt,
                "before": before,
                "after": after,
                "before_metrics": bm,
                "after_metrics": am,
                "delta": {
                    "repetition": float(am["repeated_trigram_ratio"] - bm["repeated_trigram_ratio"]),
                    "diversity": float(am["trigram_diversity"] - bm["trigram_diversity"]),
                    "words": int(am["words"] - bm["words"]),
                    "entropy": float(after["mean_entropy"] - before["mean_entropy"]),
                },
            })
    finally:
        del baseline
        gc.collect()
    summary = {
        "prompts": len(rows),
        "mean_repetition_before": sum(x["before_metrics"]["repeated_trigram_ratio"] for x in rows) / len(rows),
        "mean_repetition_after": sum(x["after_metrics"]["repeated_trigram_ratio"] for x in rows) / len(rows),
        "mean_diversity_before": sum(x["before_metrics"]["trigram_diversity"] for x in rows) / len(rows),
        "mean_diversity_after": sum(x["after_metrics"]["trigram_diversity"] for x in rows) / len(rows),
        "mean_words_before": sum(x["before_metrics"]["words"] for x in rows) / len(rows),
        "mean_words_after": sum(x["after_metrics"]["words"] for x in rows) / len(rows),
        "mean_entropy_before": sum(x["before"]["mean_entropy"] for x in rows) / len(rows),
        "mean_entropy_after": sum(x["after"]["mean_entropy"] for x in rows) / len(rows),
        "note": "این معیارها شاخص‌های سطحی متن هستند و کیفیت معنایی یا صحت پاسخ را تضمین نمی‌کنند.",
    }
    return {"summary": summary, "rows": rows}
