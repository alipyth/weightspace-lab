from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch

OPS = {"مقیاس کل": "scale", "مقیاس بخشی": "scale_slice", "نویز کنترل‌شده": "noise", "محدودسازی مقادیر": "clamp"}


@dataclass
class Snapshot:
    label: str
    changes: dict[str, tuple[torch.Tensor, torch.Tensor]]
    meta: dict[str, Any]


class EditHistory:
    def __init__(self, limit: int = 30):
        self.limit = limit
        self.undo_stack: list[Snapshot] = []
        self.redo_stack: list[Snapshot] = []

    @staticmethod
    def parameters(model):
        try:
            return dict(model.named_parameters(remove_duplicate=False))
        except TypeError:
            return dict(model.named_parameters())

    def _apply(self, model, snapshot: Snapshot, side: int):
        params = self.parameters(model)
        for name, pair in snapshot.changes.items():
            if name not in params:
                raise KeyError(f"پارامتر {name} در مدل وجود ندارد.")
            params[name].data.copy_(pair[side].to(device=params[name].device, dtype=params[name].dtype))

    def commit(self, model, edits: list[dict[str, Any]], safe_mode: bool = True, label: str = "ویرایش دستی"):
        params = self.parameters(model)
        prepared: dict[str, torch.Tensor] = {}
        snapshots_before: dict[str, torch.Tensor] = {}
        snapshots_after: dict[str, torch.Tensor] = {}
        for edit in edits:
            name = str(edit.get("tensor", ""))
            if name not in params:
                raise KeyError(f"پارامتر ناشناخته است یا قابل ویرایش نیست: {name}")
            if name not in prepared:
                snapshots_before[name] = params[name].detach().cpu().clone()
                prepared[name] = params[name].detach().cpu().clone()
            prepared[name] = edit_tensor(prepared[name], edit, safe_mode=safe_mode)
        # Commit only after every operation has passed validation.
        with torch.no_grad():
            for name, value in prepared.items():
                params[name].copy_(value.to(device=params[name].device, dtype=params[name].dtype))
                snapshots_after[name] = value.detach().cpu().clone()
        snap = Snapshot(label, {k: (snapshots_before[k], snapshots_after[k]) for k in prepared}, {"edits": edits})
        self.undo_stack.append(snap)
        self.undo_stack = self.undo_stack[-self.limit:]
        self.redo_stack.clear()
        return snap

    def undo(self, model):
        if not self.undo_stack:
            return None
        snap = self.undo_stack.pop()
        self._apply(model, snap, 0)
        self.redo_stack.append(snap)
        return snap

    def redo(self, model):
        if not self.redo_stack:
            return None
        snap = self.redo_stack.pop()
        self._apply(model, snap, 1)
        self.undo_stack.append(snap)
        return snap


def edit_tensor(tensor: torch.Tensor, edit: Mapping[str, Any], safe_mode: bool = True) -> torch.Tensor:
    op = str(edit.get("type", "")).lower()
    if not tensor.is_floating_point():
        raise ValueError("فقط Tensorهای اعشاری قابل ویرایش هستند.")
    result = tensor.detach().cpu().contiguous().clone()
    flat = result.view(-1)
    n = flat.numel()
    start = int(edit.get("start", 0))
    end = int(edit.get("end", n))
    if start < 0 or end > n or start >= end:
        raise ValueError(f"بازه [{start}, {end}) معتبر نیست؛ تعداد عناصر {n} است.")
    work = flat[start:end].float()
    dtype = result.dtype
    if op in {"scale", "scale_slice"}:
        factor = float(edit.get("factor", 1.0))
        low, high = ((0.98, 1.02) if safe_mode else (0.5, 1.5))
        if not math.isfinite(factor) or not low <= factor <= high:
            raise ValueError(f"ضریب باید بین {low} و {high} باشد.")
        changed = work * factor
    elif op == "noise":
        ratio = float(edit.get("std_ratio", 0.001))
        max_ratio = 0.005 if safe_mode else 0.05
        if not math.isfinite(ratio) or not 0 <= ratio <= max_ratio:
            raise ValueError(f"نسبت نویز باید بین صفر و {max_ratio} باشد.")
        g = torch.Generator(device="cpu").manual_seed(int(edit.get("seed", 1234)))
        noise = torch.randn(work.shape, generator=g) * (float(work.std(unbiased=False)) * ratio)
        changed = work + noise
    elif op == "clamp":
        low = float(edit.get("min_value", -1.0))
        high = float(edit.get("max_value", 1.0))
        if not math.isfinite(low) or not math.isfinite(high) or low >= high:
            raise ValueError("حد پایین باید از حد بالا کمتر باشد.")
        changed = work.clamp(low, high)
        if safe_mode:
            # Clamping can silently erase large outliers. In Safe Mode only accept
            # a clamp whose largest per-value movement is <= 2% of tensor std.
            allowed_delta = max(float(work.std(unbiased=False)) * 0.02, 1e-8)
            actual_delta = float((changed - work).abs().max()) if work.numel() else 0.0
            if actual_delta > allowed_delta:
                raise ValueError("این Clamp در حالت ایمن بیش از حد بزرگ است؛ حالت ایمن را فقط با آگاهی از ریسک خاموش کن یا حدود ملایم‌تری انتخاب کن.")
    else:
        raise ValueError(f"نوع ویرایش پشتیبانی نمی‌شود: {op}")
    if not torch.isfinite(changed).all():
        raise ValueError("نتیجه ویرایش شامل NaN یا Inf شد؛ تغییر اعمال نشد.")
    flat[start:end] = changed.to(dtype=dtype)
    return result


def validate_proposal(proposal: dict[str, Any], model, safe_mode: bool = True) -> dict[str, Any]:
    if not isinstance(proposal, dict):
        raise ValueError("ریشه پیشنهاد باید یک شیء JSON باشد.")
    edits = proposal.get("suggested_edits", [])
    if not isinstance(edits, list):
        raise ValueError("suggested_edits باید آرایه باشد.")
    params = EditHistory.parameters(model)
    clean = []
    for i, edit in enumerate(edits):
        if not isinstance(edit, dict):
            raise ValueError(f"ویرایش شماره {i + 1} ساختار درستی ندارد.")
        name = edit.get("tensor")
        if name not in params:
            raise ValueError(f"پارامتر ناشناخته یا غیرقابل ویرایش: {name}")
        spec = dict(edit)
        # Normalize risk and relevance so the UI can rank candidates consistently.
        risk = str(spec.get("risk", "high" if spec.get("type") == "clamp" else "medium")).lower()
        spec["risk"] = risk if risk in {"low", "medium", "high"} else "high"
        try:
            relevance = float(spec.get("target_relevance", 0.5))
        except (TypeError, ValueError):
            relevance = 0.5
        spec["target_relevance"] = max(0.0, min(1.0, relevance)) if math.isfinite(relevance) else 0.5
        # Dry run against a detached copy; nothing changes in the live model.
        edit_tensor(params[name].detach(), spec, safe_mode=safe_mode)
        clean.append(spec)
    risk_rank = {"low": 0, "medium": 1, "high": 2}
    def perturbation_size(spec):
        kind = str(spec.get("type", ""))
        if kind in {"scale", "scale_slice"}:
            return abs(float(spec.get("factor", 1.0)) - 1.0)
        if kind == "noise":
            return abs(float(spec.get("std_ratio", 0.0)))
        return float("inf") if kind == "clamp" else 0.0
    clean.sort(key=lambda spec: (risk_rank.get(spec.get("risk", "high"), 2),
                                 perturbation_size(spec), -spec.get("target_relevance", 0.5)))
    confidence = float(proposal.get("confidence", 0.0))
    if not math.isfinite(confidence):
        confidence = 0.0
    proposal = dict(proposal)
    proposal["confidence"] = max(0.0, min(1.0, confidence))
    proposal["suggested_edits"] = clean
    return proposal


def preview_edit(tensor: torch.Tensor, edit: dict[str, Any], safe_mode: bool = True) -> dict[str, Any]:
    before = tensor.detach().cpu().float().reshape(-1)
    after = edit_tensor(tensor, edit, safe_mode).detach().cpu().float().reshape(-1)
    diff = after - before
    def s(x):
        return {"mean": float(x.mean()), "std": float(x.std(unbiased=False)), "min": float(x.min()),
                "max": float(x.max()), "norm": float(torch.linalg.vector_norm(x))}
    return {"before": s(before), "after": s(after), "changed_fraction": float((diff != 0).float().mean()),
            "delta_norm": float(torch.linalg.vector_norm(diff)), "before_sample": before[::max(1, len(before)//20000)].tolist()[:20000],
            "after_sample": after[::max(1, len(after)//20000)].tolist()[:20000]}


def load_macro(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    edits = obj.get("edits") if isinstance(obj, dict) else obj
    if not isinstance(edits, list):
        raise ValueError("ماکرو باید آرایه‌ای از ویرایش‌ها یا شیء دارای edits باشد.")
    return edits
