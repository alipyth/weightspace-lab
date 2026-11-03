from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import torch
try:
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
except Exception:
    pass
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from weightspace.analysis import (
    activation_probe, analyze_model, guided_tensor_candidates, offline_hypotheses,
    response_metrics, stats, tensor_role, build_model_profile, build_model_dossier, activation_probe_huggingface,
)
from weightspace.api import request_proposal, request_full_model_plan
from weightspace.editor import EditHistory, preview_edit, validate_proposal
from weightspace.evaluation import compare_checkpoint
from weightspace.loader import (LoadedWorkspace, export_edited_folder, generate_text, load_harraf_folder,
                                 is_huggingface_folder, load_huggingface_folder)

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
WORK_ROOT = ROOT / "workspace" / "web_sessions"
EXPORT_ROOT = ROOT / "workspace" / "exports"
WORK_ROOT.mkdir(parents=True, exist_ok=True)
EXPORT_ROOT.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="WeightSpace Lab | Open-Weight Model Workbench", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=str(BASE / "static")), name="static")


@dataclass
class SessionState:
    workspace: LoadedWorkspace
    work_dir: Path
    history: EditHistory = field(default_factory=lambda: EditHistory(limit=40))
    findings: list[dict[str, Any]] = field(default_factory=list)
    proposal: dict[str, Any] | None = None
    settings: dict[str, str] = field(default_factory=dict)
    experiments: list[dict[str, Any]] = field(default_factory=list)
    change_log: list[dict[str, Any]] = field(default_factory=list)
    safe_mode: bool = True
    lock: threading.RLock = field(default_factory=threading.RLock)


SESSIONS: dict[str, SessionState] = {}
SESSIONS_LOCK = threading.RLock()


def _session_id(header: str | None) -> str:
    if not header:
        raise HTTPException(400, "شناسه نشست ارسال نشده است.")
    return header


def _state(sid: str) -> SessionState:
    state = SESSIONS.get(sid)
    if state is None:
        raise HTTPException(404, "هنوز مدلی بارگذاری نشده است. ابتدا پوشه مدل را انتخاب کن.")
    return state


def _clean_relpath(value: str) -> Path:
    # Browser directory uploads use forward-slash relative paths.
    p = PurePosixPath(value.replace("\\", "/"))
    if p.is_absolute() or any(part in {"..", ""} for part in p.parts):
        raise ValueError("مسیر فایل ارسالی معتبر نیست.")
    if p.parts and ":" in p.parts[0]:
        raise ValueError("مسیر مطلق مجاز نیست.")
    return Path(*p.parts)


def _find_model_files(folder: Path) -> list[Path]:
    names = {"model.pt", "model.pth", "pytorch_model.pt", "checkpoint.pt", "pytorch_model.bin", "model.bin", "model.ckpt", "model.safetensors", "pytorch_model.safetensors"}
    return [p for p in folder.rglob("*") if p.is_file() and (p.name in names or p.name.endswith((".safetensors", ".bin")))]


def _find_hf_folders(folder: Path) -> list[Path]:
    roots = [folder, *(p for p in folder.rglob("*") if p.is_dir())]
    return [p for p in roots if is_huggingface_folder(p)]


def _supports_generation(ws: LoadedWorkspace) -> bool:
    return ws.model_kind == "حراف ۲" or (ws.generation_task in {"causal_lm", "seq2seq"} and ws.hf_tokenizer is not None)


def _install_workspace(folder: Path) -> dict[str, Any]:
    sid = uuid.uuid4().hex
    try:
        ws = load_harraf_folder(folder)
    except Exception:
        raise
    state = SessionState(workspace=ws, work_dir=folder)
    # Start with exact architecture-grounded diagnostics after successful load.
    with state.lock:
        state.findings = []
    with SESSIONS_LOCK:
        SESSIONS[sid] = state
    return {"session_id": sid, "summary": _summary(state)}


def _summary(state: SessionState) -> dict[str, Any]:
    ws = state.workspace
    cfg = ws.config
    unique_params = list(ws.model.parameters())
    total = sum(int(p.numel()) for p in unique_params)
    trainable = sum(int(p.numel()) for p in unique_params if p.requires_grad)
    return {
        "name": ws.info.get("name") or ws.source_dir.name,
        "model_profile": build_model_profile(ws),
        "model_kind": ws.model_kind,
        "model_executable": _supports_generation(ws),
        "activation_supported": ws.model_kind == "حراف ۲" or (ws.model_kind == "Hugging Face" and ws.generation_task in {"causal_lm", "seq2seq"} and ws.hf_tokenizer is not None),
        "source": str(ws.source_dir),
        "files": ws.inventory,
        "file_count": len(ws.inventory),
        "tensor_count": len(ws.tensor_rows),
        "parameter_count": total,
        "trainable_count": trainable,
        "d_model": cfg.get("d_model", cfg.get("hidden_size")),
        "layers": cfg.get("n_layer", cfg.get("num_hidden_layers", cfg.get("num_layers"))),
        "heads": cfg.get("n_head", cfg.get("num_attention_heads")),
        "kv_heads": cfg.get("n_kv_head", cfg.get("num_key_value_heads")),
        "vocab_size": cfg.get("vocab_size"),
        "context": cfg.get("max_seq_len", cfg.get("max_position_embeddings")),
        "chat_tuned": bool(ws.info.get("chat_tuned", False)),
        "tokenizer_ready": ws.tokenizer is not None,
        "tokenizer_error": ws.tokenizer_error,
        "notes": ws.notes,
        "findings_count": len(state.findings),
        "undo_count": len(state.history.undo_stack),
        "redo_count": len(state.history.redo_stack),
        "change_count": len(state.change_log),
        "safe_mode": state.safe_mode,
        "experiment_count": len(state.experiments),
        "training_state": {
            k: ws.training_state.get(k) for k in ("step", "tokens", "epoch", "loss") if k in ws.training_state
        },
    }


def _tensor_params(ws: LoadedWorkspace):
    return EditHistory.parameters(ws.model)


def _append_change(state: SessionState, label: str, edits: list[dict[str, Any]]) -> None:
    state.change_log.append({
        "index": len(state.change_log) + 1,
        "label": label,
        "edits": edits,
        "tensors": list(dict.fromkeys(str(e.get("tensor", "")) for e in edits)),
    })
    state.workspace.change_log = list(state.change_log)


def _zip_directory(source: Path, zip_path: Path) -> Path:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for file in source.rglob("*"):
            if file.is_file():
                zf.write(file, file.relative_to(source.parent))
    return zip_path


@app.get("/", response_class=HTMLResponse)
def index():
    return (BASE / "templates" / "index.html").read_text(encoding="utf-8")


@app.get("/api/health")
def health():
    return {"ok": True, "app": "WeightSpace Lab", "version": "4.0-web", "torch": torch.__version__}


@app.post("/api/load-folder")
async def load_folder(
    files: list[UploadFile] = File(...),
    relative_paths: str = Form(...),
):
    try:
        paths = json.loads(relative_paths)
        if not isinstance(paths, list) or len(paths) != len(files):
            raise ValueError("فهرست مسیرها با فایل‌های ارسالی هماهنگ نیست.")
        folder = Path(tempfile.mkdtemp(prefix="upload_", dir=WORK_ROOT))
        total_bytes = 0
        max_upload_bytes = 64 * 1024**3
        for upload, rel in zip(files, paths):
            relpath = _clean_relpath(str(rel))
            dest = folder / relpath
            dest.parent.mkdir(parents=True, exist_ok=True)
            # Stream large checkpoint shards to disk instead of buffering a whole
            # model file in RAM during browser-folder uploads.
            with dest.open("wb") as output:
                while True:
                    chunk = await upload.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    total_bytes += len(chunk)
                    if total_bytes > max_upload_bytes:
                        raise ValueError("حجم پوشه از سقف ۶۴ گیگابایت بیشتر است.")
                    output.write(chunk)
        hf_folders = _find_hf_folders(folder)
        if len(hf_folders) == 1:
            return _install_workspace(hf_folders[0])
        if len(hf_folders) > 1:
            raise ValueError("چند پوشه Hugging Face در انتخاب وجود دارد؛ فقط پوشه یک مدل را انتخاب کن.")
        candidates = _find_model_files(folder)
        if not candidates:
            raise FileNotFoundError("پوشه مدل پیدا نشد. یک پوشه حاوی config.json و وزن‌های Hugging Face یا checkpoint با قالب pt/pth/bin/safetensors انتخاب کن.")
        if len(candidates) > 1:
            preferred = [p for p in candidates if p.name == "model.pt"]
            if len(preferred) == 1:
                candidates = preferred
            else:
                raise ValueError("بیش از یک checkpoint پیدا شد؛ پوشه دقیق یک مدل را انتخاب کن.")
        # Browsers include the selected folder name in webkitRelativePath; use the
        # directory containing model.pt as the model root, not the upload temp root.
        return _install_workspace(candidates[0].parent)
    except Exception as exc:
        if "folder" in locals() and folder.exists():
            shutil.rmtree(folder, ignore_errors=True)
        raise HTTPException(400, f"بارگذاری پوشه ناموفق بود: {exc}") from exc


@app.post("/api/load-local-folder")
def load_local_folder(payload: dict[str, Any]):
    path = str(payload.get("path", "")).strip()
    if not path:
        raise HTTPException(400, "مسیر پوشه را وارد کن.")
    folder = Path(path).expanduser()
    try:
        if not folder.is_dir():
            raise NotADirectoryError("پوشه در مسیر واردشده پیدا نشد.")
        resolved = folder.resolve()
        hf_folders = _find_hf_folders(resolved)
        if len(hf_folders) == 1:
            resolved = hf_folders[0]
        elif len(hf_folders) > 1:
            raise ValueError("چند پوشه مدل Hugging Face پیدا شد؛ مسیر دقیق یک مدل را انتخاب کن.")
        elif not _find_model_files(resolved):
            raise FileNotFoundError("در پوشه انتخاب‌شده فایل مدل پشتیبانی‌شده پیدا نشد.")
        return _install_workspace(resolved)
    except Exception as exc:
        raise HTTPException(400, f"بارگذاری مدل ناموفق بود: {exc}") from exc


@app.post("/api/load-hf")
def load_huggingface_hub(payload: dict[str, Any]):
    model_id = str(payload.get("model_id", "")).strip()
    token = str(payload.get("token", "")).strip() or None
    revision = str(payload.get("revision", "")).strip() or None
    if not model_id or "/" not in model_id or len(model_id) > 200:
        raise HTTPException(400, "شناسه مدل باید به شکل owner/model باشد؛ مانند Qwen/Qwen2.5-0.5B-Instruct.")
    try:
        from huggingface_hub import snapshot_download
        target = WORK_ROOT / ("hf_" + uuid.uuid4().hex)
        target.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=model_id, repo_type="model", revision=revision,
            token=token, local_dir=str(target),
            ignore_patterns=["*.msgpack", "flax_model*", "tf_model.h5", "*.ot"],
        )
        ws = load_huggingface_folder(target, hf_model_id=model_id)
        sid = uuid.uuid4().hex
        state = SessionState(workspace=ws, work_dir=target)
        with SESSIONS_LOCK:
            SESSIONS[sid] = state
        return {"session_id": sid, "summary": _summary(state)}
    except Exception as exc:
        if "target" in locals():
            shutil.rmtree(target, ignore_errors=True)
        raise HTTPException(400, f"دریافت یا بارگذاری مدل از Hugging Face ناموفق بود: {exc}") from exc


@app.post("/api/load-zip")
async def load_zip(file: UploadFile = File(...)):
    folder = Path(tempfile.mkdtemp(prefix="zip_", dir=WORK_ROOT))
    try:
        zip_path = folder / "upload.zip"
        total_zip_bytes = 0
        with zip_path.open("wb") as target:
            while True:
                chunk = await file.read(8 * 1024 * 1024)
                if not chunk:
                    break
                total_zip_bytes += len(chunk)
                if total_zip_bytes > 64 * 1024**3:
                    raise ValueError("حجم فایل ZIP از ۶۴ گیگابایت بیشتر است.")
                target.write(chunk)
        extract = folder / "model"
        extract.mkdir()
        with zipfile.ZipFile(zip_path) as zf:
            extracted_bytes = 0
            for info in zf.infolist():
                rel = _clean_relpath(info.filename)
                if info.file_size < 0 or extracted_bytes + info.file_size > 64 * 1024**3:
                    raise ValueError("حجم استخراج‌شده ZIP از سقف ۶۴ گیگابایت بیشتر است.")
                if not info.is_dir():
                    extracted_bytes += info.file_size
                    dest = extract / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(info) as src, dest.open("wb") as dst:
                        shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
        hf_folders = _find_hf_folders(extract)
        if len(hf_folders) == 1:
            return _install_workspace(hf_folders[0])
        if len(hf_folders) > 1:
            raise ValueError("داخل ZIP چند پوشه Hugging Face پیدا شد.")
        candidates = _find_model_files(extract)
        if not candidates:
            raise FileNotFoundError("داخل ZIP فایل وزن یا پوشه Hugging Face پیدا نشد.")
        preferred = [p for p in candidates if p.name == "model.pt"]
        if len(candidates) > 1 and len(preferred) != 1:
            raise ValueError("داخل ZIP چند checkpoint پیدا شد؛ فقط یک مدل را بسته‌بندی کن.")
        return _install_workspace((preferred or candidates)[0].parent)
    except Exception as exc:
        shutil.rmtree(folder, ignore_errors=True)
        raise HTTPException(400, f"بازکردن ZIP ناموفق بود: {exc}") from exc


@app.get("/api/summary")
def get_summary(x_session_id: str | None = Header(default=None)):
    state = _state(_session_id(x_session_id))
    return _summary(state)


@app.get("/api/tensors")
def tensors(q: str = "", role: str = "", limit: int = 500, x_session_id: str | None = Header(default=None)):
    state = _state(_session_id(x_session_id))
    rows = []
    ql = q.strip().lower()
    for row in state.workspace.tensor_rows:
        name = row["name"]
        title, desc, layer = tensor_role(name)
        if ql and ql not in name.lower() and ql not in title.lower() and ql not in desc.lower():
            continue
        if role and role not in title and role not in name:
            continue
        rows.append({**row, "shape": list(row["shape"]), "role": title, "description": desc, "layer": layer})
    return {"total": len(rows), "items": rows[:max(1, min(limit, 2000))]}


@app.get("/api/tensor/{name:path}")
def tensor_detail(name: str, x_session_id: str | None = Header(default=None)):
    state = _state(_session_id(x_session_id))
    params = _tensor_params(state.workspace)
    if name not in params:
        raise HTTPException(404, "Tensor پیدا نشد یا قابل ویرایش نیست.")
    tensor = params[name].detach().cpu()
    role, desc, layer = tensor_role(name)
    result = stats(tensor, hist_samples=12000)
    result.pop("sample", None)
    sample = tensor.reshape(-1)
    step = max(1, sample.numel() // 1500)
    sample = sample[::step][:1500].float()
    sample = sample[torch.isfinite(sample)]
    return {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype).replace("torch.", ""),
            "role": role, "description": desc, "layer": layer, "stats": result, "sample": sample.tolist(),
            "trainable": name in params}


@app.post("/api/analyze")
def run_analysis(x_session_id: str | None = Header(default=None)):
    state = _state(_session_id(x_session_id))
    with state.lock:
        state.findings = analyze_model(state.workspace)
    return {"findings": state.findings, "count": len(state.findings), "summary": _summary(state)}


@app.post("/api/activation")
def activation(payload: dict[str, Any], x_session_id: str | None = Header(default=None)):
    state = _state(_session_id(x_session_id))
    prompt = str(payload.get("prompt", "")).strip()
    with state.lock:
        try:
            if state.workspace.model_kind == "Hugging Face":
                result = activation_probe_huggingface(state.workspace, prompt)
            else:
                result = activation_probe(state.workspace, prompt)
        except Exception as exc:
            raise HTTPException(400, str(exc)) from exc
    return result


@app.post("/api/guided")
def guided(payload: dict[str, Any], x_session_id: str | None = Header(default=None)):
    state = _state(_session_id(x_session_id))
    symptom = str(payload.get("symptom", "تکرار"))
    return {"symptom": symptom, "candidates": guided_tensor_candidates(state.workspace, symptom)}


@app.post("/api/settings")
def save_settings(payload: dict[str, Any], x_session_id: str | None = Header(default=None)):
    state = _state(_session_id(x_session_id))
    with state.lock:
        state.settings = {
            "base_url": str(payload.get("base_url", "")).strip(),
            "api_key": str(payload.get("api_key", "")).strip(),
            "model": str(payload.get("model", "")).strip(),
        }
        state.safe_mode = bool(payload.get("safe_mode", True))
    return {"ok": True, "safe_mode": state.safe_mode, "message": "تنظیمات در حافظه همین نشست ذخیره شد."}


@app.post("/api/proposal")
def create_proposal(payload: dict[str, Any], x_session_id: str | None = Header(default=None)):
    state = _state(_session_id(x_session_id))
    symptom = str(payload.get("symptom", "بررسی جامع کل مدل"))
    mode = str(payload.get("mode", "ai"))
    full_plan = bool(payload.get("full_plan", False))
    with state.lock:
        # Rebuild diagnostics after each edit so the LLM sees the live model state.
        state.findings = analyze_model(state.workspace)
        if full_plan:
            settings = state.settings
            if mode != "ai":
                raise HTTPException(400, "برای ساخت برنامه جامع، ابتدا حالت تحلیل‌گر هوش مصنوعی را انتخاب کن و اتصال API را ذخیره کن.")
            if not settings.get("model"):
                raise HTTPException(400, "ابتدا آدرس API و نام مدل تحلیل‌گر را تنظیم و ذخیره کن.")
            prompts = payload.get("prompts") or [
                "خودت را معرفی کن و بگو چه کاری انجام می‌دهی.",
                "چرا آسمان در روز آبی دیده می‌شود؟",
                "اگر پاسخ پرسشی را نمی‌دانی، بهترین پاسخ چیست؟",
                "۴۸ منهای ۱۹ چند می‌شود؟ مرحله‌به‌مرحله توضیح بده.",
                "در سه جمله یک متن کوتاه و طبیعی فارسی بنویس.",
            ]
            if isinstance(prompts, str):
                prompts = prompts.splitlines()
            prompts = [str(x).strip() for x in prompts if str(x).strip()][:12]
            behavioral_samples = []
            activation = None
            if _supports_generation(state.workspace):
                if state.workspace.model_kind == "حراف ۲":
                    try:
                        activation = activation_probe(state.workspace, prompts[0] if prompts else "خودت را معرفی کن")
                    except Exception as exc:
                        activation = {"available": False, "reason": f"بررسی فعال‌سازی اجرا نشد: {exc}"}
                for prompt in prompts:
                    try:
                        generated = generate_text(state.workspace, prompt, 48, 0.0, 0.9)
                        behavioral_samples.append({
                            "prompt": prompt, "answer": generated["text"][:900],
                            "metrics": response_metrics(generated["text"]),
                            "token_count": generated["token_count"],
                            "mean_entropy": generated["mean_entropy"],
                        })
                    except Exception as exc:
                        behavioral_samples.append({"prompt": prompt, "error": str(exc)[:300]})
            else:
                behavioral_samples.append({"available": False, "reason": "برای این معماری موتور اجرای متنی در دسترس نیست؛ پرونده ساختاری و آماری مدل ارسال شد."})
            dossier = build_model_dossier(state.workspace, state.findings, activation, behavioral_samples)
            dossier["user_goal"] = symptom
            dossier["proposed_test_prompts"] = prompts
            try:
                proposal = request_full_model_plan(
                    settings.get("base_url", ""), settings.get("api_key", ""), settings["model"],
                    dossier, state.workspace.model, state.safe_mode,
                )
            except Exception as exc:
                raise HTTPException(502, f"ساخت برنامه جامع ناموفق بود: {exc}") from exc
            proposal["symptom"] = symptom
            proposal["analysis_scope"] = "کل مدل؛ همه تنسورهای قابل‌آموزش، ساختار بلوک‌ها، یافته‌ها، فعال‌سازی و نمونه خروجی"
            benchmark_plan = proposal.get("benchmark_plan") if isinstance(proposal.get("benchmark_plan"), dict) else {}
            llm_prompts = benchmark_plan.get("prompts", [])
            if isinstance(llm_prompts, str):
                llm_prompts = llm_prompts.splitlines()
            combined_prompts = list(dict.fromkeys([*prompts, *(str(x).strip() for x in llm_prompts if str(x).strip())]))[:12]
            proposal["test_prompts"] = combined_prompts or prompts
        elif mode == "ai":
            settings = state.settings
            if not settings.get("model"):
                raise HTTPException(400, "ابتدا نشانی API و نام مدل تحلیل‌گر را تنظیم کن.")
            try:
                candidate_items = guided_tensor_candidates(state.workspace, symptom)
                candidate_names = {item["name"] for item in candidate_items}
                params = _tensor_params(state.workspace)
                enriched_rows = []
                for row in state.workspace.tensor_rows:
                    if not row.get("trainable"):
                        continue
                    role, description, layer = tensor_role(row["name"])
                    enriched = {k: (list(v) if isinstance(v, tuple) else v) for k, v in row.items()}
                    enriched["role_fa"] = role
                    enriched["description_fa"] = description
                    enriched["layer_fa"] = layer
                    enriched_rows.append(enriched)
                enriched_rows.sort(key=lambda row: (0 if row["name"] in candidate_names else 1, row["name"]))
                for row in enriched_rows[:120]:
                    sample_stats = stats(params[row["name"]].detach(), hist_samples=256)
                    row["weight_summary"] = {k: (round(v, 7) if isinstance(v, float) else v)
                                              for k, v in sample_stats.items() if k != "sample"}
                    row["weight_sample"] = [round(float(v), 7) for v in sample_stats.get("sample", [])[:8]]
                candidate_findings = [f for f in state.findings if not f.get("tensor") or f.get("tensor") in candidate_names]
                proposal = request_proposal(settings.get("base_url", ""), settings.get("api_key", ""),
                                            settings["model"], symptom, enriched_rows,
                                            candidate_findings, state.workspace.model, state.safe_mode)
                proposal["symptom"] = symptom
            except Exception as exc:
                raise HTTPException(502, f"تحلیل‌گر پاسخ نداد: {exc}") from exc
        else:
            candidates = {x["name"] for x in guided_tensor_candidates(state.workspace, symptom)}
            relevant_findings = [f for f in state.findings if not f.get("tensor") or f.get("tensor") in candidates]
            proposals = offline_hypotheses(state.workspace, relevant_findings)
            proposal = proposals[0] if proposals else {
                "hypothesis": "از یافته‌های ساختاری مرتبط با این مسئله، وزن مشخصی برای ویرایش کم‌ریسک به دست نیامد.",
                "confidence": 0.15, "risk": "low", "reversible": True,
                "limitations": "این نتیجه به معنی نبود مشکل رفتاری نیست. شاخص‌های وزن به‌تنهایی علت تکرار، توهم یا فراموشی را ثابت نمی‌کنند؛ آزمون رفتاری و تحلیل‌گر متصل می‌تواند شواهد بیشتری فراهم کند.",
                "suggested_edits": [],
            }
            proposal["symptom"] = symptom
            proposal["alternatives"] = proposals[1:]
        # The full-plan request already validates each edit independently. The
        # ordinary proposal path uses the existing strict validator.
        if not full_plan:
            state.proposal = validate_proposal(proposal, state.workspace.model, state.safe_mode)
        else:
            state.proposal = proposal
        return {"proposal": state.proposal, "profile": build_model_profile(state.workspace)}


@app.get("/api/profile")
def get_model_profile(x_session_id: str | None = Header(default=None)):
    state = _state(_session_id(x_session_id))
    return {"profile": build_model_profile(state.workspace), "summary": _summary(state)}


@app.post("/api/preview")
def preview(payload: dict[str, Any], x_session_id: str | None = Header(default=None)):
    state = _state(_session_id(x_session_id))
    edit = payload.get("edit") or {}
    name = str(edit.get("tensor", ""))
    params = _tensor_params(state.workspace)
    if name not in params:
        raise HTTPException(400, "ابتدا یک Tensor قابل ویرایش انتخاب کن.")
    safe_mode = bool(payload.get("safe_mode", state.safe_mode))
    with state.lock:
        try:
            result = preview_edit(params[name].detach(), edit, safe_mode=safe_mode)
        except Exception as exc:
            raise HTTPException(400, str(exc)) from exc
    return result


@app.post("/api/validate-macro")
def validate_macro(payload: dict[str, Any], x_session_id: str | None = Header(default=None)):
    state = _state(_session_id(x_session_id))
    edits = payload.get("edits")
    if not isinstance(edits, list) or not edits:
        raise HTTPException(400, "ماکرو باید حداقل یک ویرایش داشته باشد.")
    safe_mode = bool(payload.get("safe_mode", state.safe_mode))
    try:
        checked = validate_proposal({"confidence": 1.0, "suggested_edits": edits}, state.workspace.model, safe_mode)
    except Exception as exc:
        raise HTTPException(400, f"اعتبارسنجی ماکرو ناموفق بود: {exc}") from exc
    return {"ok": True, "edits": checked["suggested_edits"], "count": len(checked["suggested_edits"]),
            "message": "همه عملیات روی نسخه جداگانه آزمایش شدند و معتبر هستند؛ هنوز هیچ وزنی تغییر نکرده است."}


@app.post("/api/apply")
def apply_edits(payload: dict[str, Any], x_session_id: str | None = Header(default=None)):
    state = _state(_session_id(x_session_id))
    edits = payload.get("edits")
    if not isinstance(edits, list) or not edits:
        raise HTTPException(400, "هیچ ویرایشی برای اعمال انتخاب نشده است.")
    safe_mode = bool(payload.get("safe_mode", state.safe_mode))
    label = str(payload.get("label", "ویرایش دستی"))
    with state.lock:
        try:
            state.history.commit(state.workspace.model, edits, safe_mode=safe_mode, label=label)
            _append_change(state, label, edits)
            state.safe_mode = safe_mode
        except Exception as exc:
            raise HTTPException(400, str(exc)) from exc
    return {"ok": True, "summary": _summary(state), "history": state.change_log[-30:]}


@app.post("/api/undo")
def undo(x_session_id: str | None = Header(default=None)):
    state = _state(_session_id(x_session_id))
    with state.lock:
        snap = state.history.undo(state.workspace.model)
    if snap is None:
        raise HTTPException(400, "تغییری برای بازگردانی وجود ندارد.")
    return {"ok": True, "label": snap.label, "summary": _summary(state)}


@app.post("/api/redo")
def redo(x_session_id: str | None = Header(default=None)):
    state = _state(_session_id(x_session_id))
    with state.lock:
        snap = state.history.redo(state.workspace.model)
    if snap is None:
        raise HTTPException(400, "تغییری برای انجام دوباره وجود ندارد.")
    return {"ok": True, "label": snap.label, "summary": _summary(state)}


@app.post("/api/reset")
def reset_model(x_session_id: str | None = Header(default=None)):
    state = _state(_session_id(x_session_id))
    with state.lock, torch.no_grad():
        state.workspace.model.load_state_dict(state.workspace.original_checkpoint["model"], strict=True)
        state.history.undo_stack.clear(); state.history.redo_stack.clear()
        state.change_log.clear(); state.workspace.change_log = []
    return {"ok": True, "summary": _summary(state)}


@app.post("/api/generate")
def generate(payload: dict[str, Any], x_session_id: str | None = Header(default=None)):
    state = _state(_session_id(x_session_id))
    with state.lock:
        try:
            result = generate_text(state.workspace, str(payload.get("prompt", "")),
                                   int(payload.get("max_new_tokens", 80)),
                                   float(payload.get("temperature", 0.0)),
                                   float(payload.get("top_p", 0.9)))
        except Exception as exc:
            raise HTTPException(400, str(exc)) from exc
    result["metrics"] = response_metrics(result["text"])
    return result


@app.get("/api/weight-diff")
def weight_diff(x_session_id: str | None = Header(default=None)):
    """Report exact changed-element counts and sampled before/after distributions."""
    state = _state(_session_id(x_session_id))
    if state.workspace.model_kind == "Hugging Face":
        try:
            baseline_workspace = load_huggingface_folder(state.workspace.source_dir, state.workspace.hf_model_id)
            original = baseline_workspace.model.state_dict()
        except Exception as exc:
            raise HTTPException(400, f"بارگذاری نسخه اصلی برای مقایسه ممکن نشد: {exc}") from exc
    else:
        original = state.workspace.original_checkpoint.get("model", {})
    if not isinstance(original, dict):
        raise HTTPException(400, "وزن‌های نسخه اصلی برای مقایسه در دسترس نیستند.")
    current = state.workspace.model.state_dict()
    rows = []
    for name, after in current.items():
        before = original.get(name)
        if not isinstance(before, torch.Tensor) or not isinstance(after, torch.Tensor):
            continue
        if tuple(before.shape) != tuple(after.shape):
            rows.append({"name": name, "role": tensor_role(name)[0], "shape": list(after.shape),
                         "shape_changed": True, "changed_elements": None, "changed_fraction": None})
            continue
        b = before.detach().cpu().reshape(-1)
        a = after.detach().cpu().reshape(-1)
        changed = 0
        max_delta = 0.0
        delta_sq_sum = 0.0
        finite_delta_count = 0
        for start in range(0, b.numel(), 1_000_000):
            bc = b[start:start + 1_000_000]
            ac = a[start:start + 1_000_000]
            if bc.is_floating_point() or ac.is_floating_point():
                same = (bc == ac) | (torch.isnan(bc) & torch.isnan(ac))
            else:
                same = bc == ac
            changed += int((~same).sum().item())
            d = ac.to(torch.float64) - bc.to(torch.float64)
            fd = d[torch.isfinite(d)]
            if fd.numel():
                max_delta = max(max_delta, float(fd.abs().max().item()))
                delta_sq_sum += float((fd * fd).sum().item())
                finite_delta_count += int(fd.numel())
        if changed == 0:
            continue
        before_stats = stats(before, hist_samples=4096)
        after_stats = stats(after, hist_samples=4096)
        role_name, role_description, layer_label = tensor_role(name)
        before_rms = (float(before_stats["mean"]) ** 2 + float(before_stats["std"]) ** 2) ** 0.5
        after_rms = (float(after_stats["mean"]) ** 2 + float(after_stats["std"]) ** 2) ** 0.5
        rows.append({
            "name": name, "role": role_name, "description": role_description, "layer": layer_label,
            "shape": list(after.shape), "elements": int(after.numel()),
            "changed_elements": changed, "changed_fraction": changed / max(1, after.numel()),
            "max_abs_delta": max_delta,
            "rms_delta": (delta_sq_sum / finite_delta_count) ** 0.5 if finite_delta_count else 0.0,
            "before": {k: before_stats.get(k) for k in ("mean", "std", "zero_ratio", "nonfinite")} | {"rms": before_rms},
            "after": {k: after_stats.get(k) for k in ("mean", "std", "zero_ratio", "nonfinite")} | {"rms": after_rms},
        })
    if state.workspace.model_kind == "Hugging Face":
        del baseline_workspace
    rows.sort(key=lambda x: (x.get("changed_elements") or 0), reverse=True)
    def group_rows(key: str, fallback: str):
        groups: dict[str, dict[str, Any]] = {}
        for row in rows:
            label = str(row.get(key) or fallback)
            group = groups.setdefault(label, {"name": label, "changed_tensors": 0, "changed_elements": 0, "max_abs_delta": 0.0})
            group["changed_tensors"] += 1
            group["changed_elements"] += int(row.get("changed_elements") or 0)
            group["max_abs_delta"] = max(group["max_abs_delta"], float(row.get("max_abs_delta") or 0.0))
        return sorted(groups.values(), key=lambda x: x["changed_elements"], reverse=True)
    return {"items": rows, "changed_tensors": len(rows),
            "changed_elements": sum(int(x.get("changed_elements") or 0) for x in rows),
            "role_summary": group_rows("role", "نقش نامشخص"),
            "layer_summary": group_rows("layer", "سراسری / بدون شماره لایه"),
            "model_executable": _supports_generation(state.workspace),
            "performance_note": ("برای این معماری امکان مقایسه رفتاری قبل/بعد وجود دارد." if _supports_generation(state.workspace)
                                 else "این checkpoint در حالت بررسی/ویرایش است؛ برای سنجش عملکرد به adapter اجرای معماری نیاز است.")}


@app.post("/api/evaluate")
def evaluate(payload: dict[str, Any], x_session_id: str | None = Header(default=None)):
    state = _state(_session_id(x_session_id))
    prompts = payload.get("prompts", [])
    if isinstance(prompts, str):
        prompts = prompts.splitlines()
    with state.lock:
        try:
            result = compare_checkpoint(state.workspace, prompts,
                                        int(payload.get("max_new_tokens", 80)),
                                        float(payload.get("temperature", 0.0)),
                                        float(payload.get("top_p", 0.9)))
            experiment = {"id": len(state.experiments) + 1, "name": str(payload.get("name", "آزمایش A/B")),
                          "summary": result["summary"], "rows": result["rows"]}
            state.experiments.append(experiment)
        except Exception as exc:
            raise HTTPException(400, str(exc)) from exc
    return experiment


@app.get("/api/experiments")
def experiments(x_session_id: str | None = Header(default=None)):
    state = _state(_session_id(x_session_id))
    return {"items": state.experiments}


@app.get("/api/history")
def history(x_session_id: str | None = Header(default=None)):
    state = _state(_session_id(x_session_id))
    return {"items": state.change_log, "undo": len(state.history.undo_stack), "redo": len(state.history.redo_stack)}


@app.post("/api/export")
def export_model(x_session_id: str | None = Header(default=None)):
    state = _state(_session_id(x_session_id))
    out_dir = Path(tempfile.mkdtemp(prefix="wsl_export_", dir=EXPORT_ROOT))
    shutil.rmtree(out_dir)
    dest = out_dir / "harraf_edited"
    with state.lock:
        try:
            export_edited_folder(state.workspace, dest)
            zip_path = EXPORT_ROOT / f"WeightSpace_Harraf_Export_{uuid.uuid4().hex[:8]}.zip"
            _zip_directory(dest, zip_path)
        except Exception as exc:
            shutil.rmtree(out_dir, ignore_errors=True)
            raise HTTPException(500, f"خروجی گرفتن ناموفق بود: {exc}") from exc
    shutil.rmtree(out_dir, ignore_errors=True)
    return FileResponse(zip_path, filename=zip_path.name, media_type="application/zip")


@app.exception_handler(Exception)
async def unhandled_exception_handler(_request, exc: Exception):
    # Do not leak stack traces or local paths into the browser.
    return JSONResponse(status_code=500, content={"detail": f"خطای داخلی برنامه: {type(exc).__name__}: {exc}"})
