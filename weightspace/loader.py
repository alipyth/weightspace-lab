from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from runtime.farsitree.harraf2 import build_model
from runtime.farsitree.morph import Segmenter


class TensorStore(torch.nn.Module):
    """A safe inspection/edit container for tensor-only checkpoints without a known forward adapter."""
    def __init__(self, state: dict[str, torch.Tensor]):
        super().__init__()
        for raw_name, value in state.items():
            if not isinstance(raw_name, str) or not isinstance(value, torch.Tensor):
                continue
            parts = raw_name.split(".")
            if not parts or any(not part or part in {".."} for part in parts):
                raise ValueError(f"نام Tensor قابل ثبت نیست: {raw_name!r}")
            parent = self
            for part in parts[:-1]:
                if part in parent._parameters or part in parent._buffers:
                    raise ValueError(f"تداخل نام در ساختار Tensorها: {raw_name}")
                if part not in parent._modules:
                    parent.add_module(part, torch.nn.Module())
                parent = parent._modules[part]
            leaf = parts[-1]
            if leaf in parent._modules or leaf in parent._parameters or leaf in parent._buffers:
                raise ValueError(f"نام تکراری یا متداخل Tensor: {raw_name}")
            clone = value.detach().cpu().contiguous().clone()
            if clone.is_floating_point() or clone.is_complex():
                parent.register_parameter(leaf, torch.nn.Parameter(clone, requires_grad=True))
            else:
                parent.register_buffer(leaf, clone)


@dataclass
class LoadedWorkspace:
    source_dir: Path
    model_path: Path
    model: torch.nn.Module
    config: dict[str, Any]
    info: dict[str, Any]
    training_state: dict[str, Any]
    morph: Segmenter | None
    tokenizer: Any | None
    tokenizer_error: str | None
    model_kind: str = "حراف ۲"
    original_checkpoint: dict[str, Any] = field(default_factory=dict)
    original_dtypes: dict[str, torch.dtype] = field(default_factory=dict)
    inventory: list[dict[str, Any]] = field(default_factory=list)
    tensor_rows: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    hf_tokenizer: Any | None = None
    generation_task: str | None = None
    hf_model_id: str | None = None


def _json_read(path: Path, max_bytes: int = 32 * 1024 * 1024) -> dict | list | None:
    if not path.exists() or not path.is_file():
        return None
    if path.stat().st_size > max_bytes:
        raise ValueError(f"فایل JSON از حد مجاز بزرگ‌تر است: {path.name}")
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def _safe_torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError("نسخه PyTorch شما بارگذاری امن weights_only را ندارد؛ PyTorch را به‌روز کنید.") from exc


def _tensor_rows(model: torch.nn.Module) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, tensor in model.state_dict().items():
        if not isinstance(tensor, torch.Tensor):
            continue
        p = tensor.detach().cpu()
        rows.append({
            "name": name,
            "shape": tuple(p.shape),
            "elements": int(p.numel()),
            "dtype": str(p.dtype).replace("torch.", ""),
            "trainable": name in dict(model.named_parameters(remove_duplicate=False)),
        })
    return rows


def _inventory(folder: Path) -> list[dict[str, Any]]:
    rows = []
    for p in sorted(folder.rglob("*")):
        if not p.is_file():
            continue
        rel = str(p.relative_to(folder))
        size = p.stat().st_size
        suffix = p.suffix.lower()
        kind = "فایل داده / باینری"
        detail = "برای جلوگیری از اجرای ناخواسته، فقط شناسایی و اندازه‌گیری شد."
        if suffix == ".json":
            kind = "فایل JSON"
            try:
                obj = _json_read(p)
                if isinstance(obj, dict):
                    detail = "کلیدها: " + "، ".join(list(obj.keys())[:18])
                elif isinstance(obj, list):
                    detail = f"آرایه با {len(obj):,} عضو"
                else:
                    detail = "ساختار JSON معتبر"
            except Exception as exc:
                detail = f"خواندن JSON ناموفق: {exc}"
        elif suffix in {".pt", ".pth", ".bin", ".ckpt", ".safetensors"}:
            kind = "وزن / Checkpoint مدل"
            detail = "به‌عنوان وزن مدل بررسی می‌شود؛ pickle ناامن اجرا نمی‌شود."
        elif suffix in {".txt", ".md", ".csv", ".jsonl"}:
            kind = "متن / داده"
            detail = "فایل متنی شناسایی شد؛ برای تحلیل وزن‌ها به‌صورت خودکار وارد مدل نمی‌شود."
        rows.append({"path": rel, "name": p.name, "size": size, "kind": kind, "detail": detail})
    return rows


def _load_tokenizer(path: Path):
    if not path.exists():
        return None, "فایل tokenizer.json پیدا نشد."
    try:
        from tokenizers import Tokenizer
        return Tokenizer.from_file(str(path)), None
    except Exception as exc:
        return None, f"بارگذاری tokenizer.json ناموفق بود: {exc}"


def _load_generic_folder(root: Path, model_path: Path, ckpt: dict, state_dict: dict,
                         config: dict, info: dict, training_state: dict) -> LoadedWorkspace:
    """Load any tensor-only checkpoint into an inspection/editable tensor tree.

    Generation is disabled unless a real architecture adapter is available.
    The model purpose is inferred from explicit metadata/config first, then common
    architecture names and parameter-key patterns.
    """
    tensor_state = {str(k): v for k, v in state_dict.items() if isinstance(k, str) and isinstance(v, torch.Tensor)}
    if not tensor_state:
        raise ValueError("هیچ Tensor معتبری در checkpoint پیدا نشد.")
    model = TensorStore(tensor_state).cpu().eval()
    if not isinstance(ckpt.get("model"), dict):
        normalized = dict(ckpt) if isinstance(ckpt, dict) else {}
        if isinstance(normalized.get("state_dict"), dict):
            original_format = "state_dict"
        elif normalized and all(isinstance(v, torch.Tensor) for v in normalized.values()):
            original_format = "plain"
            normalized = {}
        else:
            original_format = "plain"
        normalized.pop("state_dict", None)
        normalized["model"] = tensor_state
        normalized["config"] = config
        normalized["_weightspace_original_format"] = original_format
        ckpt = normalized
    else:
        ckpt = dict(ckpt)
        ckpt["_weightspace_original_format"] = "model"
    morph = None
    tokenizer, tokenizer_error = _load_tokenizer(root / "tokenizer.json")
    notes = ["این checkpoint با معماری اجرایی شناخته‌شده حراف ۲ تطبیق داده نشد؛ حالت عمومی بررسی و ویرایش فعال است.",
             "تولید متن، بررسی فعال‌سازی و آزمون رفتاری تا زمان افزودن adapter واقعی این معماری غیرفعال‌اند."]
    if tokenizer_error and (root / "tokenizer.json").exists():
        notes.append(tokenizer_error)
    elif not (root / "tokenizer.json").exists():
        notes.append("فایل tokenizer.json پیدا نشد.")
    inventory = _inventory(root)
    original_dtypes = {k: v.dtype for k, v in tensor_state.items()}
    return LoadedWorkspace(
        source_dir=root, model_path=model_path, model=model,
        config=dict(config or {}), info=dict(info or {}), training_state=dict(training_state or {}),
        morph=morph, tokenizer=tokenizer, tokenizer_error=tokenizer_error,
        model_kind="مدل عمومی (حالت بررسی/ویرایش)", original_checkpoint=ckpt,
        original_dtypes=original_dtypes, inventory=inventory,
        tensor_rows=_tensor_rows(model), notes=notes,
    )


def _hf_weight_files(root: Path) -> list[Path]:
    patterns = (
        "model.safetensors", "pytorch_model.bin", "model.bin",
        "model-*.safetensors", "pytorch_model-*.bin", "model-*.bin",
    )
    found: list[Path] = []
    for pattern in patterns:
        found.extend(root.glob(pattern))
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        if (root / index_name).is_file():
            found.append(root / index_name)
    return sorted(set(p for p in found if p.is_file()))


def is_huggingface_folder(folder: str | Path) -> bool:
    root = Path(folder)
    return (root / "config.json").is_file() and bool(_hf_weight_files(root))


def _load_hf_components(folder: Path, hf_model_id: str | None = None):
    try:
        from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForMaskedLM
        from transformers import AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("برای بارگذاری مدل Hugging Face، بسته transformers نصب نیست. وابستگی‌های پروژه را دوباره نصب کن.") from exc

    cfg = AutoConfig.from_pretrained(str(folder), local_files_only=True, trust_remote_code=False)
    is_seq2seq = bool(getattr(cfg, "is_encoder_decoder", False))
    attempts = ([ ("seq2seq", AutoModelForSeq2SeqLM) ] if is_seq2seq else [
        ("causal_lm", AutoModelForCausalLM),
        ("masked_lm", AutoModelForMaskedLM),
        ("sequence_classification", AutoModelForSequenceClassification),
    ])
    attempts.append(("base_model", AutoModel))
    model = None
    generation_task = None
    errors = []
    for task, model_class in attempts:
        try:
            model = model_class.from_pretrained(
                str(folder), config=cfg, local_files_only=True, trust_remote_code=False,
                low_cpu_mem_usage=True, torch_dtype="auto",
            )
            if task in {"causal_lm", "seq2seq"}:
                generation_task = task
            break
        except Exception as exc:
            errors.append(f"{task}: {str(exc)[:220]}")
    if model is None:
        raise RuntimeError("بارگذاری معماری Hugging Face ناموفق بود. " + " | ".join(errors[-3:]))

    tokenizer = None
    tokenizer_error = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(folder), local_files_only=True, trust_remote_code=False, use_fast=True)
    except Exception as exc:
        tokenizer_error = f"توکنایزر بارگذاری نشد: {exc}"

    model.cpu().eval()
    config = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(getattr(cfg, "__dict__", {}))
    info = {"name": hf_model_id or folder.name, "source": "Hugging Face", "model_id": hf_model_id}
    return model, cfg, config, tokenizer, tokenizer_error, generation_task, info


def load_huggingface_folder(folder: str | Path, hf_model_id: str | None = None) -> LoadedWorkspace:
    root = Path(folder).expanduser().resolve()
    if not is_huggingface_folder(root):
        raise ValueError("پوشه Hugging Face باید config.json و حداقل یک فایل وزن یا index وزن داشته باشد.")
    model, cfg_obj, config, tokenizer, tokenizer_error, generation_task, info = _load_hf_components(root, hf_model_id)
    try:
        vocab = getattr(tokenizer, "vocab_size", None) if tokenizer is not None else None
        cfg_vocab = getattr(cfg_obj, "vocab_size", None)
        if vocab and cfg_vocab and int(vocab) != int(cfg_vocab):
            tokenizer_error = f"اندازه واژگان توکنایزر ({vocab}) با vocab_size پیکربندی ({cfg_vocab}) متفاوت است."
    except Exception:
        pass
    notes = [
        "مدل از قالب استاندارد Hugging Face و با trust_remote_code=False بارگذاری شده است.",
        "وزن‌های چندبخشی و فایل index در صورت وجود توسط Transformers مدیریت می‌شوند.",
    ]
    if generation_task is None:
        notes.append("این مدل برای تولید متن خودرگرسیو یا متن‌به‌متن شناسایی نشد؛ بررسی و ویرایش وزن‌ها فعال است.")
    if tokenizer_error:
        notes.append(tokenizer_error)
    inventory = _inventory(root)
    original_dtypes = {k: v.dtype for k, v in model.state_dict().items() if isinstance(v, torch.Tensor)}
    primary = next((p for p in _hf_weight_files(root) if not p.name.endswith(".index.json")), root / "config.json")
    return LoadedWorkspace(
        source_dir=root, model_path=primary, model=model, config=config, info=info,
        training_state={}, morph=None, tokenizer=tokenizer, tokenizer_error=tokenizer_error,
        model_kind="Hugging Face", original_checkpoint={}, original_dtypes=original_dtypes,
        inventory=inventory, tensor_rows=_tensor_rows(model), notes=notes,
        hf_tokenizer=tokenizer, generation_task=generation_task, hf_model_id=hf_model_id,
    )


def load_harraf_folder(folder: str | Path) -> LoadedWorkspace:
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    if is_huggingface_folder(root):
        return load_huggingface_folder(root)
    checkpoint_names = ("model.pt", "model.pth", "pytorch_model.pt", "checkpoint.pt", "pytorch_model.bin", "model.bin", "model.ckpt")
    safe_names = ("model.safetensors", "pytorch_model.safetensors")
    direct_candidates = [root / n for n in (*checkpoint_names, *safe_names) if (root / n).is_file()]
    if not direct_candidates:
        nested_candidates = [p for p in root.rglob("*") if p.is_file() and p.name in {*checkpoint_names, *safe_names}]
        if len(nested_candidates) == 1:
            root = nested_candidates[0].parent
            direct_candidates = nested_candidates
        elif len(nested_candidates) > 1:
            preferred = [p for p in nested_candidates if p.name == "model.pt"]
            if len(preferred) == 1:
                root = preferred[0].parent
                direct_candidates = preferred
            else:
                raise ValueError("چند checkpoint در پوشه پیدا شد؛ مسیر پوشه دقیق یک مدل را انتخاب کن.")
    model_path = next((p for p in direct_candidates if p.name in checkpoint_names), None)
    if model_path is None:
        model_path = next((p for p in direct_candidates if p.suffix.lower() in {".pt", ".pth", ".bin", ".ckpt"}), None)
    raw_safetensors = None
    if model_path is None:
        safe_path = next((p for p in direct_candidates if p.suffix.lower() == ".safetensors"), None)
        if safe_path is not None:
            try:
                from safetensors.torch import load_file
            except ImportError as exc:
                raise RuntimeError("برای خواندن فایل safetensors، بسته safetensors را نصب کن.") from exc
            raw_safetensors = load_file(str(safe_path), device="cpu")
            model_path = safe_path
        else:
            raise FileNotFoundError("فایل model.pt، model.pth، pytorch_model.bin یا model.safetensors در این پوشه پیدا نشد.")

    # Harraf checkpoints carry both model and config inside model.pt. Other
    # tensor-only checkpoints enter generic inspection/edit mode below.
    if raw_safetensors is not None:
        ckpt = {"model": raw_safetensors, "config": _json_read(root / "config.json") or {}}
    else:
        ckpt = _safe_torch_load(model_path)
    embedded_config = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    external_config = _json_read(root / "config.json")
    config = dict(embedded_config if isinstance(embedded_config, dict) else {})
    if not config and isinstance(external_config, dict):
        config = dict(external_config)
    state_dict = None
    if isinstance(ckpt, dict):
        if isinstance(ckpt.get("model"), dict):
            state_dict = ckpt["model"]
        elif isinstance(ckpt.get("state_dict"), dict):
            state_dict = ckpt["state_dict"]
        elif ckpt and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            state_dict = ckpt
    if not isinstance(state_dict, dict) or not any(isinstance(v, torch.Tensor) for v in state_dict.values()):
        raise ValueError("ساختار checkpoint قابل شناسایی نیست؛ باید state_dict یا کلید model حاوی Tensor داشته باشد.")

    info_obj = _json_read(root / "info.json")
    state_obj = _json_read(root / "state.json")
    info = info_obj if isinstance(info_obj, dict) else {}
    training_state = state_obj if isinstance(state_obj, dict) else {}
    is_harraf2 = config.get("version") == 2 and isinstance(ckpt, dict) and isinstance(ckpt.get("config"), dict)
    if not is_harraf2:
        return _load_generic_folder(root, model_path, ckpt, state_dict, config, info, training_state)

    model = build_model(config)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing or unexpected:
        # Do not continue with a partially initialized network.
        raise RuntimeError(
            "وزن‌ها دقیقاً با معماری داخل config سازگار نیستند. "
            f"کلیدهای مفقود: {missing[:12]} | کلیدهای اضافی: {unexpected[:12]}"
        )
    model.cpu().eval()

    morph = None
    morph_path = root / "morph.json"
    notes: list[str] = []
    if morph_path.exists():
        try:
            morph = Segmenter.load(str(morph_path))
        except Exception as exc:
            notes.append(f"فایل morph.json وجود دارد اما بارگذاری نشد: {exc}")
    else:
        notes.append("morph.json پیدا نشد؛ تولید متن بدون جداسازی صرفی انجام می‌شود.")

    tokenizer, tokenizer_error = _load_tokenizer(root / "tokenizer.json")
    if tokenizer_error:
        notes.append(tokenizer_error)
    elif tokenizer is not None:
        actual_vocab = tokenizer.get_vocab_size(with_added_tokens=True)
        expected_vocab = int(config.get("vocab_size", -1))
        if actual_vocab != expected_vocab:
            notes.append(f"تعداد واژگان توکنایزر ({actual_vocab:,}) با vocab_size مدل ({expected_vocab:,}) یکسان نیست؛ تولید متن غیرفعال شد.")
            tokenizer_error = "عدم تطابق اندازه واژگان توکنایزر و مدل."

    inventory = _inventory(root)
    original_dtypes = {
        k: v.dtype for k, v in ckpt["model"].items() if isinstance(v, torch.Tensor)
    }
    return LoadedWorkspace(
        source_dir=root,
        model_path=model_path,
        model=model,
        config=config,
        info=info,
        training_state=training_state,
        morph=morph,
        tokenizer=tokenizer,
        tokenizer_error=tokenizer_error,
        original_checkpoint=ckpt,
        original_dtypes=original_dtypes,
        inventory=inventory,
        tensor_rows=_tensor_rows(model),
        notes=notes,
    )


def generate_text(workspace: LoadedWorkspace, prompt: str, max_new_tokens: int = 80,
                  temperature: float = 0.0, top_p: float = 0.9,
                  model: torch.nn.Module | None = None) -> dict[str, Any]:
    if workspace.generation_task in {"causal_lm", "seq2seq"} and workspace.hf_tokenizer is not None:
        if not prompt.strip():
            raise ValueError("پرامپت خالی است.")
        tokenizer = workspace.hf_tokenizer
        active_model = model or workspace.model
        active_model.eval()
        max_ctx = int(getattr(active_model.config, "max_position_embeddings", 2048) or 2048)
        encoded = tokenizer(prompt.strip(), return_tensors="pt", truncation=True, max_length=min(8192, max_ctx))
        encoded = {k: v.to("cpu") for k, v in encoded.items() if isinstance(v, torch.Tensor)}
        input_len = int(encoded.get("input_ids", torch.empty((1, 0))).shape[-1])
        kwargs = {"max_new_tokens": max(1, min(int(max_new_tokens), 512)), "do_sample": float(temperature) > 0}
        if float(temperature) > 0:
            kwargs["temperature"] = max(0.05, float(temperature))
            kwargs["top_p"] = max(0.05, min(1.0, float(top_p)))
        if getattr(tokenizer, "pad_token_id", None) is not None:
            kwargs["pad_token_id"] = tokenizer.pad_token_id
        elif getattr(tokenizer, "eos_token_id", None) is not None:
            kwargs["pad_token_id"] = tokenizer.eos_token_id
        with torch.inference_mode():
            generated = active_model.generate(**encoded, return_dict_in_generate=True, output_scores=True, **kwargs)
        sequences = generated.sequences
        token_ids = sequences[0, input_len:] if workspace.generation_task == "causal_lm" else sequences[0]
        output = tokenizer.decode(token_ids.tolist(), skip_special_tokens=True).strip()
        entropies = []
        for score in getattr(generated, "scores", []) or []:
            probs = torch.softmax(score[0].float(), dim=-1)
            entropies.append(float((-(probs * torch.log(probs.clamp_min(1e-12))).sum()).item()))
        return {"text": output, "token_count": int(token_ids.numel()),
                "mean_entropy": sum(entropies) / max(1, len(entropies)), "prompt_tokens": input_len}

    if workspace.model_kind != "حراف ۲":
        raise RuntimeError("برای این معماری هنوز adapter اجرای متن در برنامه تعریف نشده است؛ بررسی و ویرایش وزن‌ها فعال است.")
    if workspace.tokenizer is None:
        raise RuntimeError(workspace.tokenizer_error or "توکنایزر بارگذاری نشده است.")
    if not prompt.strip():
        raise ValueError("پرامپت خالی است.")
    tok = workspace.tokenizer
    # The supplied Harraf chat-training code teaches the prefix form
    # «پرسش: ... پاسخ:». Apply it automatically only for chat-tuned exports.
    chat_prompt = prompt.strip()
    if bool(workspace.info.get("chat_tuned", False)):
        if not chat_prompt.startswith("پرسش:") and " پرسش: " not in chat_prompt:
            chat_prompt = "پرسش: " + chat_prompt
        if not chat_prompt.rstrip().endswith("پاسخ:"):
            chat_prompt = chat_prompt.rstrip() + " پاسخ:"
    processed = workspace.morph(chat_prompt) if workspace.morph else chat_prompt
    ids = tok.encode(processed).ids
    if not ids:
        raise ValueError("پرامپت پس از توکن‌سازی به توکن تبدیل نشد.")
    cfg = workspace.config
    max_context = int(cfg.get("max_seq_len", 1024))
    eos_ids = set()
    for token in ("</s>", "<|endoftext|>", "<|im_end|>", "<|end|>", "<eos>"):
        token_id = tok.token_to_id(token)
        if token_id is not None:
            eos_ids.add(token_id)

    active_model = model or workspace.model
    active_model.eval()
    generated: list[int] = []
    entropies: list[float] = []
    device = next(active_model.parameters()).device
    # Reuse the same random stream for every call so before/after sampling is comparable.
    generator = torch.Generator(device=device).manual_seed(20261103)
    seq = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        for _ in range(max(1, min(int(max_new_tokens), 512))):
            context = seq[:, -max_context:]
            logits = active_model(context)["logits"][:, -1, :].float()
            if not torch.isfinite(logits).all():
                raise RuntimeError("خروجی مدل شامل مقدار غیرمتناهی است؛ تولید متوقف شد.")
            probs_all = torch.softmax(logits, dim=-1)
            entropy = -(probs_all * torch.log(probs_all.clamp_min(1e-12))).sum(-1).item()
            entropies.append(float(entropy))
            if temperature <= 0:
                next_id = int(torch.argmax(logits, dim=-1).item())
            else:
                scaled = logits / max(0.05, float(temperature))
                probs = torch.softmax(scaled, dim=-1)
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumulative = torch.cumsum(sorted_probs, dim=-1)
                remove = cumulative - sorted_probs > max(0.05, min(1.0, float(top_p)))
                sorted_probs[remove] = 0
                filtered = torch.zeros_like(probs).scatter(-1, sorted_idx, sorted_probs)
                filtered = filtered / filtered.sum(-1, keepdim=True).clamp_min(1e-12)
                next_id = int(torch.multinomial(filtered, 1, generator=generator).item())
            if next_id in eos_ids:
                break
            generated.append(next_id)
            seq = torch.cat([seq, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
    output = tok.decode(generated, skip_special_tokens=True)
    if workspace.morph:
        output = workspace.morph.join(output)
    return {"text": output.strip(), "token_count": len(generated),
            "mean_entropy": sum(entropies) / max(1, len(entropies)), "prompt_tokens": len(ids)}


def export_edited_folder(workspace: LoadedWorkspace, destination: str | Path) -> Path:
    dest = Path(destination).expanduser().resolve()
    if dest.exists():
        raise FileExistsError(f"مسیر خروجی از قبل وجود دارد: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(workspace.source_dir, dest)
    if workspace.model_kind == "Hugging Face":
        weight_patterns = ("model.safetensors", "pytorch_model.bin", "model.bin", "model-*.safetensors", "pytorch_model-*.bin", "model-*.bin", "model.safetensors.index.json", "pytorch_model.bin.index.json")
        for pattern in weight_patterns:
            for file in dest.glob(pattern):
                if file.is_file():
                    file.unlink()
        workspace.model.save_pretrained(str(dest), safe_serialization=True)
        if workspace.hf_tokenizer is not None:
            workspace.hf_tokenizer.save_pretrained(str(dest))
        manifest = {
            "tool": "WeightSpace Lab",
            "created_unix": time.time(),
            "source_folder": str(workspace.source_dir),
            "model_id": workspace.hf_model_id,
            "changes": getattr(workspace, "change_log", []),
            "note_fa": "این خروجی یک آزمایش ویرایش وزن است؛ بهتر بودن مدل باید با بنچمارک مستقل بررسی شود.",
        }
        (dest / "weightspace_changes.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return dest
    export_state = {}
    for name, tensor in workspace.model.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        original_dtype = workspace.original_dtypes.get(name)
        if original_dtype is not None and value.is_floating_point():
            value = value.to(original_dtype)
        export_state[name] = value
    new_ckpt = dict(workspace.original_checkpoint)
    original_format = new_ckpt.pop("_weightspace_original_format", "model")
    if workspace.model_kind != "حراف ۲" and original_format == "plain":
        new_ckpt = export_state
    elif workspace.model_kind != "حراف ۲" and original_format == "state_dict":
        new_ckpt.pop("model", None)
        new_ckpt["state_dict"] = export_state
    else:
        new_ckpt["model"] = export_state
    if workspace.model_path.suffix.lower() == ".safetensors":
        from safetensors.torch import save_file
        output_checkpoint = dest / workspace.model_path.name
        if output_checkpoint.exists():
            output_checkpoint.unlink()
        save_file({k: v.detach().cpu().contiguous() for k, v in export_state.items()}, str(output_checkpoint))
    else:
        output_checkpoint = dest / workspace.model_path.name
        torch.save(new_ckpt, output_checkpoint)
    manifest = {
        "tool": "WeightSpace Lab Persian Edition",
        "created_unix": time.time(),
        "source_folder": str(workspace.source_dir),
        "changes": getattr(workspace, "change_log", []),
        "note_fa": "این خروجی یک آزمایش ویرایش وزن است؛ بهتر بودن مدل باید با بنچمارک مستقل بررسی شود.",
    }
    (dest / "weightspace_changes.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return dest
