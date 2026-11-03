from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

import torch


ROLE_HINTS = [
    (r"^wte\.weight$", "Embedding توکن و ماتریس خروجی مشترک", "هر شناسه توکن را به بردار d_model تبدیل می‌کند. در معماری حراف، وزن head نیز با آن مشترک است."),
    (r"^head\.weight$", "سر خروجی زبان", "بردارهای خروجی برای امتیازدهی واژگان؛ در پیاده‌سازی حراف با wte.weight مشترک است."),
    (r"^blocks\.(\d+)\.ln1\.weight$", "RMSNorm ورودی Attention", "مقیاس‌بندی بردار پنهان پیش از توجه در این بلوک."),
    (r"^blocks\.(\d+)\.ln2\.weight$", "RMSNorm ورودی MLP", "نرمال‌سازی ورودی شبکه SwiGLU در این بلوک."),
    (r"^blocks\.(\d+)\.ln3\.weight$", "RMSNorm ورودی حافظه", "در بلوک‌های دارای حافظه، ورودی مسیر حافظه را نرمال می‌کند."),
    (r"^blocks\.(\d+)\.attn\.q\.weight$", "پروجکشن Query", "بردار پرس‌وجو برای هر سر توجه را می‌سازد."),
    (r"^blocks\.(\d+)\.attn\.kv\.weight$", "پروجکشن Key و Value", "کلید و مقدار توجه را برای سرهای KV می‌سازد."),
    (r"^blocks\.(\d+)\.attn\.proj\.weight$", "پروجکشن خروجی Attention", "خروجی سرهای توجه را دوباره به فضای d_model می‌برد."),
    (r"^blocks\.(\d+)\.attn\.tree_slope$", "شیب توجه درختی", "شدت جریمه فاصله ساختاری را برای هر سر توجه کنترل می‌کند."),
    (r"^blocks\.(\d+)\.mlp\.w1\.weight$", "شاخه اول SwiGLU", "یکی از دو شاخه ورودی MLP را می‌سازد."),
    (r"^blocks\.(\d+)\.mlp\.w3\.weight$", "دروازه SwiGLU", "شاخه دروازه‌ای MLP را می‌سازد و با خروجی شاخه دیگر ترکیب می‌شود."),
    (r"^blocks\.(\d+)\.mlp\.w2\.weight$", "پروجکشن خروجی SwiGLU", "نمایش میانی MLP را به d_model بازمی‌گرداند."),
    (r"^blocks\.(\d+)\.mem\.query\.weight$", "پروجکشن پرس‌وجوی حافظه", "بردار پرس‌وجو برای جست‌وجوی حافظه محصولی را می‌سازد."),
    (r"^blocks\.(\d+)\.mem\.keys$", "کلیدهای حافظه محصولی", "کلیدهای دو نیمه فضای برداری برای بازیابی اسلات‌های حافظه."),
    (r"^blocks\.(\d+)\.mem\.values\.weight$", "مقادیر حافظه محصولی", "بردارهای محتوایی اسلات‌های حافظه؛ در هر توکن فقط تعدادی از آن‌ها خوانده می‌شوند."),
    (r"^blocks\.(\d+)\.mem\.hits$", "شمارنده بازدید حافظه", "بافر آماری تعداد خواندن اسلات‌ها؛ پارامتر قابل‌آموزش نیست."),
    (r"^parser\.conv\.(weight|bias)$", "فیلتر علّی Parser", "ویژگی‌های محلی دنباله را برای تخمین فاصله نحوی استخراج می‌کند."),
    (r"^parser\.norm\.(weight|bias)$", "نرمال‌سازی Parser", "ورودی Parser را نرمال می‌کند."),
    (r"^parser\.mlp\.0\.(weight|bias)$", "لایه میانی Parser", "نمایش ویژگی‌های Parser را تبدیل می‌کند."),
    (r"^parser\.mlp\.2\.(weight|bias)$", "خروجی Parser", "برای هر موقعیت یک فاصله نحوی پیش‌بینی می‌کند."),
    (r"^ln_f\.weight$", "RMSNorm نهایی", "نمایش آخرین بلوک را پیش از سر خروجی نرمال می‌کند."),
]


def tensor_role(name: str) -> tuple[str, str, str | None]:
    for pattern, title, desc in ROLE_HINTS:
        m = re.match(pattern, name)
        if m:
            layer = int(m.group(1)) if m.lastindex and m.group(1).isdigit() else None
            return title, desc, (f"بلوک {layer}" if layer is not None else None)
    lower = name.lower()
    layer_match = re.search(r"(?:blocks|layers|h|layer)\.(\d+)", lower)
    layer_label = f"لایه {int(layer_match.group(1))}" if layer_match else None
    generic_rules = [
        (("embed_tokens", "word_embeddings", "wte", "token_embedding", "tok_embeddings"), "Embedding توکن", "شناسه هر توکن را به بردار ورودی مدل تبدیل می‌کند."),
        (("q_proj", "query", "query_proj"), "پروجکشن Query در Attention", "بخش پرس‌وجوی سازوکار توجه را می‌سازد."),
        (("k_proj", "key", "key_proj"), "پروجکشن Key در Attention", "کلیدهایی را می‌سازد که برای سنجش ارتباط توکن‌ها استفاده می‌شوند."),
        (("v_proj", "value", "value_proj"), "پروجکشن Value در Attention", "محتوایی را می‌سازد که توجه از توکن‌های مرتبط جمع‌آوری می‌کند."),
        (("o_proj", "out_proj", "attn.proj", "attention.dense"), "پروجکشن خروجی Attention", "ترکیب سرهای توجه را به فضای پنهان برمی‌گرداند."),
        (("gate_proj", "up_proj", "w1", "fc1", "wi_0"), "لایه ورودی MLP", "ویژگی‌ها را برای پردازش غیرخطی گسترش می‌دهد."),
        (("down_proj", "w2", "fc2", "wo"), "لایه خروجی MLP", "ویژگی‌های میانی را به اندازه فضای پنهان بازمی‌گرداند."),
        (("layernorm", "layer_norm", "rmsnorm", "input_layernorm", "post_attention_layernorm", "ln_f", ".norm"), "وزن نرمال‌سازی", "مقیاس فعال‌سازی‌ها را در بخشی از شبکه تنظیم می‌کند."),
        (("lm_head", "output_projection", "vocab_proj"), "سر خروجی واژگان", "برای توکن‌های خروجی امتیاز تولید می‌کند."),
        (("position_embeddings", "pos_embed", "position_embedding"), "Embedding موقعیت", "اطلاعات موقعیت توکن‌ها را وارد نمایش مدل می‌کند."),
    ]
    for terms, title, desc in generic_rules:
        if any(term in lower for term in terms):
            return title, desc, layer_label
    return "Tensor عمومی", "نقش دقیق این وزن از روی نام به‌تنهایی مشخص نیست؛ شکل Tensor، معماری ثبت‌شده و مستندات مدل باید بررسی شوند.", layer_label


def stats(tensor: torch.Tensor, hist_samples: int = 20000) -> dict[str, Any]:
    x = tensor.detach().cpu().reshape(-1)
    n = x.numel()
    if n == 0:
        return {"elements": 0, "mean": 0., "std": 0., "min": 0., "max": 0., "norm": 0., "zero_ratio": 0., "nonfinite": 0, "sample": []}
    step = max(1, n // hist_samples)
    sample = x[::step][:hist_samples].float()
    finite_mask = torch.isfinite(sample)
    finite = sample[finite_mask]
    if finite.numel() == 0:
        mean = std = min_v = max_v = norm = 0.0
    else:
        mean = float(finite.mean())
        std = float(finite.std(unbiased=False))
        min_v = float(finite.min())
        max_v = float(finite.max())
        norm = float(torch.linalg.vector_norm(finite))

    # Scan the full tensor in bounded chunks for exact corruption/zero counts.
    # Distribution statistics and the histogram remain a reproducible sample to
    # keep exploration responsive for very large embedding/memory tables.
    nonfinite = 0
    zero_count = 0
    for chunk in x.split(4_000_000):
        nonfinite += int((~torch.isfinite(chunk)).sum())
        zero_count += int((chunk == 0).sum())
    zero_ratio = zero_count / n
    return {"elements": int(n), "mean": mean, "std": std, "min": min_v, "max": max_v,
            "norm": norm, "zero_ratio": zero_ratio, "nonfinite": nonfinite,
            "statistics_sampled": True, "sample": finite[:hist_samples].tolist()}


def analyze_model(workspace) -> list[dict[str, Any]]:
    """Conservative structural diagnostics. These are indicators, not causal conclusions."""
    model = workspace.model
    state = model.state_dict()
    findings: list[dict[str, Any]] = []
    by_layer: dict[int, list[tuple[str, float]]] = {}

    for name, tensor in state.items():
        if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
            continue
        s = stats(tensor, hist_samples=3000)
        title, desc, layer_label = tensor_role(name)
        if s["nonfinite"]:
            findings.append({"severity": "بحرانی", "title": f"مقدار غیرمتناهی در {name}",
                             "detail": f"در کل Tensor، {s['nonfinite']} مقدار NaN یا Inf پیدا شد.",
                             "recommendation": "ویرایش را متوقف کن؛ فایل خروجی نگیر و نسخه اصلی را حفظ کن.", "tensor": name})
        if name.endswith(("ln1.weight", "ln2.weight", "ln3.weight", "ln_f.weight")):
            ratio = max(abs(s["min"]), abs(s["max"])) / max(1e-8, abs(s["mean"]) + s["std"])
            if s["std"] > 0.25 or s["min"] < 0.05 or s["max"] > 3.0 or ratio > 8:
                findings.append({"severity": "نیازمند بررسی", "title": f"پراکندگی غیرمعمول وزن نرمال‌سازی: {name}",
                                 "detail": f"میانگین {s['mean']:.4g}، انحراف معیار {s['std']:.4g}، بازه [{s['min']:.4g}, {s['max']:.4g}]. این حدود فقط هشدار آماری‌اند.",
                                 "recommendation": "ابتدا خروجی مدل را روی چند پرامپت ثابت بررسی کن؛ کاهش مقیاس فقط یک آزمایش A/B برگشت‌پذیر است، نه اصلاح قطعی.", "tensor": name})
        if name.startswith("blocks.") and name.endswith(("attn.q.weight", "attn.kv.weight", "attn.proj.weight")):
            m = re.match(r"blocks\.(\d+)\.", name)
            if m:
                by_layer.setdefault(int(m.group(1)), []).append((name, s["norm"]))
        if name == "wte.weight" and tensor.ndim == 2:
            row_norms = torch.linalg.vector_norm(tensor.detach().float().cpu(), dim=1)
            med = float(row_norms.median())
            p99 = float(torch.quantile(row_norms, 0.99))
            max_v = float(row_norms.max())
            if p99 > max(med * 2.5, 1e-6) or max_v > max(p99 * 2.5, 1e-6):
                findings.append({"severity": "اطلاعاتی", "title": "ناهمگنی اندازه بردارهای embedding",
                                 "detail": f"میانه نرم ردیف‌ها {med:.4g}، صدک ۹۹ برابر {p99:.4g} و بیشینه {max_v:.4g} است. این معیار فراوانی توکن‌ها را نشان نمی‌دهد.",
                                 "recommendation": "برای بررسی غلبه embedding، توزیع نُرم را با فراوانی واقعی توکن‌ها و آزمون پرامپت مقایسه کن؛ از کوچک‌کردن کل embedding بدون آزمون پرهیز کن.", "tensor": name})
        if ".mem.values.weight" in name:
            row_norms = torch.linalg.vector_norm(tensor.detach().float().cpu(), dim=1)
            med = float(row_norms.median())
            p99 = float(torch.quantile(row_norms, 0.99))
            if p99 > max(med * 3.0, 1e-6):
                findings.append({"severity": "اطلاعاتی", "title": f"ناهمگنی در اسلات‌های حافظه: {name}",
                                 "detail": f"نرم میانه {med:.4g} و صدک ۹۹ {p99:.4g} است.",
                                 "recommendation": "این اختلاف می‌تواند طبیعی باشد؛ با شمارنده hits و تست رفتاری بررسی کن.", "tensor": name})

    all_norms = [(layer, name, norm) for layer, pairs in by_layer.items() for name, norm in pairs]
    if all_norms:
        vals = torch.tensor([x[2] for x in all_norms], dtype=torch.float64)
        med = float(vals.median())
        for layer, name, norm in all_norms:
            if med > 0 and (norm > med * 2.5 or norm < med / 2.5):
                findings.append({"severity": "نیازمند بررسی", "title": f"نرم پروجکشن نامتوازن در بلوک {layer}",
                                 "detail": f"نرم {name} برابر {norm:.4g} است؛ میانه همه پروجکشن‌های توجه {med:.4g} است.",
                                 "recommendation": "لایه را با خروجی آزمون و سایر پروجکشن‌های همان بلوک بررسی کن. مقایسه نرم به‌تنهایی نشانه خرابی نیست.", "tensor": name})

    if "wte.weight" in state and "head.weight" in state:
        tied = torch.equal(state["wte.weight"].detach().cpu(), state["head.weight"].detach().cpu())
        if not tied:
            findings.append({"severity": "نیازمند بررسی", "title": "وزن ورودی و خروجی مشترک نیستند",
                             "detail": "در معماری حراف ۲ انتظار می‌رود head.weight و wte.weight به یک پارامتر مشترک اشاره کنند.",
                             "recommendation": "در خروجی استاندارد حراف ۲، اتصال وزن مشترک را حفظ کن؛ این نسخه در زمان ساخت مدل آن را از معماری بازیابی می‌کند.", "tensor": "wte.weight"})

    cfg = workspace.config
    mem_layers = list(cfg.get("mem_layers", []))
    if mem_layers:
        missing = [i for i in mem_layers if f"blocks.{i}.mem.values.weight" not in state]
        if missing:
            findings.append({"severity": "بحرانی", "title": "اسلات‌های حافظه مورد انتظار پیدا نشدند",
                             "detail": f"برای بلوک‌های {missing} وزن values پیدا نشد.", "recommendation": "پیش از ویرایش، سازگاری checkpoint و config را بررسی کن.", "tensor": ""})

    if not findings:
        findings.append({"severity": "اطلاعاتی", "title": "هشدار ساختاری جدی پیدا نشد",
                         "detail": "این فقط یعنی آزمون‌های آماری ساده نشانه واضحی پیدا نکردند؛ سلامت رفتاری مدل تأیید نشده است.",
                         "recommendation": "چند پرامپت ثابت اجرا کن و تکرار، تنوع، طول پاسخ و صحت پاسخ را بسنج.", "tensor": ""})
    return findings



def guided_tensor_candidates(workspace, symptom: str) -> list[dict[str, Any]]:
    """Return architecture-grounded candidate tensors for a selected symptom."""
    key = symptom.lower()
    families = {
        "تکرار": ("attn", "mlp", "parser", "tree_slope", "mem", "ln_f", "wte"),
        "کوتاه": ("head", "wte", "attn", "mem", "ln_f"),
        "تنوع": ("wte", "head", "attn", "mlp", "parser"),
        "توهم": ("attn", "mlp", "parser", "mem", "ln_f", "wte"),
        "فراموشی": ("mem", "attn", "mlp", "wte", "ln_f"),
        "نرمال": ("ln1", "ln2", "ln3", "ln_f"),
        "embedding": ("wte", "head"),
        "حافظه": ("mem",),
        "parser": ("parser", "tree_slope"),
        "فرضیه": tuple(),
    }
    if "تکرار" in key or "حلقه" in key:
        terms = families["تکرار"]
    elif "کوتاه" in key:
        terms = families["کوتاه"]
    elif "تنوع" in key:
        terms = families["تنوع"]
    elif "توهم" in key or "hallucination" in key:
        terms = families["توهم"]
    elif "فراموش" in key or "بازیابی" in key:
        terms = families["فراموشی"]
    elif "نرمال" in key:
        terms = families["نرمال"]
    elif "embedding" in key or "غلبه" in key:
        terms = families["embedding"]
    elif "اعتماد" in key or "لایه اولیه" in key or "لایه‌های اولیه" in key:
        terms = ("attn", "mlp", "ln1", "ln2", "wte", "ln_f")
    elif "حافظه" in key:
        terms = families["حافظه"]
    elif "parser" in key or "درخت" in key:
        terms = families["parser"]
    else:
        terms = tuple()
    result = []
    for row in workspace.tensor_rows:
        name = row["name"]
        if not row.get("trainable"):
            continue
        if not terms or any(t in name.lower() for t in terms):
            role, desc, layer = tensor_role(name)
            result.append({"name": name, "role": role, "description": desc, "layer": layer,
                           "shape": list(row["shape"]), "elements": row["elements"]})
    # Keep the view navigable on large models; names remain searchable in the weight explorer.
    return result[:180]


def response_metrics(text: str) -> dict[str, float | int]:
    words = re.findall(r"[\w\u200c]+", text.lower(), flags=re.UNICODE)
    grams = [tuple(words[i:i + 3]) for i in range(max(0, len(words) - 2))]
    total = len(grams)
    unique = len(set(grams))
    repeated = 0.0 if total == 0 else 1.0 - unique / total
    return {
        "words": len(words),
        "characters": len(text),
        "trigram_count": total,
        "trigram_diversity": 0.0 if total == 0 else unique / total,
        "repeated_trigram_ratio": repeated,
    }


def offline_hypotheses(workspace, findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    proposals = []
    for f in findings:
        name = f.get("tensor")
        if not name or name not in workspace.model.state_dict():
            continue
        # Candidate tests are intentionally small and explicitly not asserted as repairs.
        if name.endswith(("ln1.weight", "ln2.weight", "ln3.weight", "ln_f.weight")):
            proposals.append({
                "hypothesis": f"مقیاس {name} ممکن است روی رفتار این مسیر اثر قابل‌اندازه‌گیری داشته باشد؛ این صرفاً یک فرضیه آزمایشی است.",
                "confidence": 0.35, "risk": "medium", "reversible": True,
                "limitations": "نرم نرمال‌سازی به‌تنهایی ضعف رفتاری را ثابت نمی‌کند. خروجی باید با پرامپت ثابت سنجیده شود.",
                "suggested_edits": [{"tensor": name, "type": "scale", "factor": 0.995,
                                     "reason": "آزمایش A/B با تغییر ۰٫۵ درصدی و برگشت‌پذیر؛ نه اصلاح تضمینی.", "risk": "medium", "target_relevance": 0.45}],
            })
        elif name.startswith("blocks.") and ".attn." in name and name.endswith(".weight"):
            proposals.append({
                "hypothesis": f"بررسی تغییر بسیار کوچک در {name} می‌تواند نشان دهد این پروجکشن به رفتار آزمون‌شده حساس است یا نه.",
                "confidence": 0.25, "risk": "medium", "reversible": True,
                "limitations": "نرم بزرگ یا کوچک به‌تنهایی نشانه مشکل نیست؛ نتیجه ممکن است بدتر شود.",
                "suggested_edits": [{"tensor": name, "type": "scale", "factor": 0.999,
                                     "reason": "آزمون حساسیت با تغییر ۰٫۱ درصدی.", "risk": "medium", "target_relevance": 0.35}],
            })
    return proposals[:8]


def activation_probe(workspace, prompt: str) -> dict[str, Any]:
    """Run one real forward pass and report per-block output activation statistics."""
    if workspace.tokenizer is None:
        raise RuntimeError(workspace.tokenizer_error or "توکنایزر بارگذاری نشده است.")
    if not prompt.strip():
        raise ValueError("پرامپت بررسی فعال‌سازی خالی است.")
    text = workspace.morph(prompt) if workspace.morph else prompt
    ids = workspace.tokenizer.encode(text).ids
    if not ids:
        raise ValueError("پرامپت به توکن تبدیل نشد.")
    limit = int(workspace.config.get("max_seq_len", 1024))
    input_ids = torch.tensor([ids[-limit:]], dtype=torch.long)
    layers: list[dict[str, Any]] = []
    handles = []

    def hook_factory(i):
        def hook(_module, _inputs, output):
            x = output[0] if isinstance(output, (tuple, list)) else output
            if isinstance(x, torch.Tensor):
                z = x.detach().float().cpu()
                row = {"layer": i, "mean": float(z.mean()), "std": float(z.std(unbiased=False)),
                       "rms": float(torch.sqrt(torch.mean(z * z))), "max_abs": float(z.abs().max()),
                       "shape": list(z.shape)}
                # Diagnostic-only proxy: project the last position of this block through
                # the model's final norm and tied language head. This is not an auxiliary
                # head trained at intermediate layers and must not be treated as calibrated.
                with torch.inference_mode():
                    proxy = workspace.model.head(workspace.model.ln_f(x[:, -1:, :])).float()
                    proxy_probs = torch.softmax(proxy[:, -1, :], dim=-1)
                    row["proxy_top1_probability"] = float(proxy_probs.max())
                    row["proxy_entropy"] = float((-(proxy_probs * torch.log(proxy_probs.clamp_min(1e-12))).sum(-1)).item())
                layers.append(row)
        return hook

    for i, block in enumerate(workspace.model.blocks):
        handles.append(block.register_forward_hook(hook_factory(i)))
    parser_rows = {}
    if getattr(workspace.model, "parser", None) is not None:
        def parser_hook(_module, _inputs, output):
            z = output.detach().float().cpu()
            parser_rows.update({"mean": float(z.mean()), "std": float(z.std(unbiased=False)),
                                "min": float(z.min()), "max": float(z.max())})
        handles.append(workspace.model.parser.register_forward_hook(parser_hook))

    try:
        workspace.model.eval()
        with torch.inference_mode():
            output = workspace.model(input_ids, return_dist=True)
        logits = output["logits"][:, -1, :].float()
        probs = torch.softmax(logits, dim=-1)
        top = torch.topk(probs, k=min(2, probs.size(-1)), dim=-1).values[0].tolist()
        entropy = float((-(probs * torch.log(probs.clamp_min(1e-12))).sum(-1)).item())
        top1 = float(top[0])
        margin = float(top[0] - top[1]) if len(top) > 1 else top1
        if "dist" in output and output["dist"] is not None:
            d = output["dist"].detach().float().cpu()
            parser_rows.update({"mean": float(d.mean()), "std": float(d.std(unbiased=False)),
                                "min": float(d.min()), "max": float(d.max())})
    finally:
        for h in handles:
            h.remove()

    layers.sort(key=lambda x: x["layer"])
    rms_values = [x["rms"] for x in layers]
    median_rms = float(torch.tensor(rms_values).median()) if rms_values else 0.0
    for row in layers:
        row["rms_vs_median"] = row["rms"] / max(1e-8, median_rms)
    return {"prompt_tokens": len(ids[-limit:]), "layers": layers, "parser": parser_rows,
            "last_token_top1_probability": top1, "last_token_top1_margin": margin,
            "last_token_entropy": entropy,
            "note": "احتمال‌های میانی با عبور از نرمال‌ساز نهایی و سر خروجی، فقط شاخص تقریبی‌اند و سر مستقلِ آموزش‌دیده نیستند. این بررسی یک پرامپت است؛ برای تشخیص قطعی چند ورودی و مجموعه ارزیابی لازم است."}


def activation_probe_huggingface(workspace, prompt: str) -> dict[str, Any]:
    """Collect hidden-state summaries from a standard Transformers model."""
    tokenizer = workspace.hf_tokenizer
    if tokenizer is None:
        raise RuntimeError(workspace.tokenizer_error or "توکنایزر Hugging Face بارگذاری نشده است.")
    if not prompt.strip():
        raise ValueError("پرامپت خالی است.")
    encoded = tokenizer(prompt.strip(), return_tensors="pt", truncation=True, max_length=2048)
    encoded = {k: v.to("cpu") for k, v in encoded.items() if isinstance(v, torch.Tensor)}
    with torch.inference_mode():
        if bool(getattr(workspace.model.config, "is_encoder_decoder", False)) and hasattr(workspace.model, "get_encoder"):
            output = workspace.model.get_encoder()(**encoded, output_hidden_states=True, return_dict=True)
        else:
            output = workspace.model(**encoded, output_hidden_states=True, return_dict=True)
    hidden_states = getattr(output, "hidden_states", None)
    if not hidden_states:
        raise RuntimeError("مدل hidden_states را برنگرداند؛ این معماری از بررسی فعال‌سازی عمومی پشتیبانی نمی‌کند.")
    rows = []
    for index, hidden in enumerate(hidden_states):
        x = hidden.detach().float().cpu()
        rows.append({
            "layer": max(0, index - 1),
            "mean": float(x.mean()),
            "std": float(x.std(unbiased=False)),
            "rms": float(torch.sqrt(torch.mean(x.square()))),
            "max_abs": float(x.abs().max()),
            "proxy_entropy": 0.0,
        })
    logits = getattr(output, "logits", None)
    last_entropy = None
    if logits is not None:
        probs = torch.softmax(logits[:, -1, :].float(), dim=-1)
        last_entropy = float((-(probs * torch.log(probs.clamp_min(1e-12))).sum(-1)).item())
    return {
        "prompt_tokens": int(encoded.get("input_ids", torch.empty((1, 0))).shape[-1]),
        "layers": rows,
        "note": "آمار از hidden_states واقعی مدل Transformers به دست آمده است. ستون آنتروپی پروکسی برای این مسیر عمومی محاسبه نمی‌شود؛ در صورت وجود logits، آنتروپی توزیع آخرین توکن جداگانه گزارش می‌شود.",
        "last_token_entropy": last_entropy,
    }


def _tensor_family(name: str) -> tuple[str, str]:
    n = name.lower()
    if n.startswith("wte.") or n.startswith("head."):
        return "embedding_output", "ورودی واژگان و خروجی پیش‌بینی توکن"
    if any(tag in n for tag in (".attn.", "self_attn", "attention", "q_proj", "k_proj", "v_proj", "o_proj", "query", "key", "value")) or "tree_slope" in n:
        return "attention", "پردازش وابستگی‌ها و ترکیب اطلاعات توکن‌ها"
    if any(tag in n for tag in (".mlp.", "gate_proj", "up_proj", "down_proj", "feed_forward", "intermediate", "fc1", "fc2")):
        return "mlp", "تبدیل غیرخطی ویژگی‌ها و پردازش درونی بلوک"
    if ".mem." in n:
        return "memory", "بازیابی و ترکیب اطلاعات مسیر حافظه"
    if any(tag in n for tag in ("ln", "norm", "layernorm", "layer_norm", "rmsnorm")):
        return "normalization", "تنظیم مقیاس فعال‌سازی‌ها در مسیر شبکه"
    if n.startswith("parser."):
        return "parser", "استخراج نشانه‌های نحوی برای توجه ساختاری"
    return "other", "پارامتر یا بافر با نقش تخصصی‌تر"


def build_model_profile(workspace) -> dict[str, Any]:
    """Describe what the loaded, supported architecture is built to do.

    The description is evidence-based: config and implementation define the task;
    chat tuning and training metadata refine the interpretation but do not prove quality.
    """
    cfg = workspace.config
    info = workspace.info if isinstance(workspace.info, dict) else {}
    task_hint = str(info.get("task") or info.get("pipeline_tag") or cfg.get("task") or cfg.get("problem_type") or "").strip()
    chat_tuned = bool(info.get("chat_tuned", False))
    if workspace.model_kind == "حراف ۲":
        arch = "حراف ۲ (معماری اختصاصی Decoder با Attention، MLP، حافظه و Parser)"
        task = task_hint or "مدل زبانی خودرگرسیو برای پیش‌بینی توکن بعدی و تولید متن"
        if chat_tuned:
            purpose = "دستیار متنی گفتگو‌محور فارسی؛ برای پاسخ‌دادن به ورودی کاربر با قالب گفتگو تنظیم شده است."
            task += "؛ نسخه گفتگو‌محور (Chat-tuned)"
        else:
            purpose = "مدل زبانی پایه برای ادامه متن و تولید توکن بعدی؛ گفتگو‌محور بودن آن از اطلاعات موجود تأیید نشده است."
        if task_hint:
            purpose = str(info.get("purpose") or task_hint)
        confidence = 0.95
        confidence_note = "اطمینان بالا به نوع معماری و هدف تولید متن؛ نه به کیفیت یا توانایی واقعی مدل."
    else:
        arch_names = cfg.get("architectures") or cfg.get("architecture") or info.get("architecture") or []
        if isinstance(arch_names, str):
            arch_names = [arch_names]
        model_type = str(cfg.get("model_type") or info.get("model_type") or "").lower()
        arch = ", ".join(str(x) for x in arch_names) or (model_type or "معماری ثبت‌نشده")
        key_names = [str(r.get("name", "")) for r in workspace.tensor_rows[:80]]
        identity = (model_type + " " + " ".join(str(x) for x in arch_names) + " " + " ".join(key_names)).lower()
        patterns = [
            (("llama", "mistral", "qwen", "gpt", "gemma", "falcon", "bloom", "phi", "causallm"), "مدل زبانی خودرگرسیو / تولید متن", "پیش‌بینی توکن بعدی و تولید دنباله متن"),
            (("bert", "roberta", "deberta", "maskedlm"), "مدل زبانی رمزگذار / درک متن", "نمایش و تحلیل متن؛ نه لزوماً تولید آزاد متن"),
            (("t5", "bart", "seq2seq", "conditionalgeneration"), "مدل تبدیل متن به متن", "تبدیل ورودی به خروجی مانند خلاصه‌سازی یا ترجمه"),
            (("whisper", "wav2vec", "speech", "audio"), "مدل صوت و گفتار", "پردازش صوت یا تبدیل گفتار به متن، بسته به سر مدل"),
            (("vit", "swin", "imageclassification", "vision"), "مدل بینایی ماشین", "استخراج ویژگی یا طبقه‌بندی تصویر"),
            (("clip", "multimodal"), "مدل چندوجهی / embedding", "هم‌ترازسازی یا نمایش مشترک متن و تصویر"),
        ]
        inferred = next((x for terms, *x in patterns if any(t in identity for t in terms)), None)
        if task_hint:
            task = task_hint
            purpose = str(info.get("purpose") or task_hint)
            confidence = 0.95
            confidence_note = "وظیفه از metadata/config صریح مدل خوانده شده؛ کیفیت آن هنوز ارزیابی نشده است."
        elif inferred:
            task, purpose = inferred[:2]
            confidence = 0.72
            confidence_note = "وظیفه از نام معماری و الگوی پارامترها استنباط شده؛ نوع دقیق head و کیفیت باید بررسی شود."
        else:
            task = "وظیفه مدل از metadata فعلی قابل تشخیص قطعی نیست."
            purpose = "نام معماری و نام وزن‌ها ثبت شده‌اند، اما برای تشخیص کاربرد دقیق به config، model card یا metadata روشن‌تری نیاز است."
            confidence = 0.2
            confidence_note = "تشخیص خودکار کم‌اطمینان است؛ از حدس زدن کاربرد مدل خودداری شده است."
    n_layer = int(cfg.get("n_layer", cfg.get("num_hidden_layers", cfg.get("num_layers", 0))) or 0)
    tensor_count = len(workspace.tensor_rows)
    trainable_count = sum(int(r.get("elements", 0)) for r in workspace.tensor_rows if r.get("trainable"))
    memory_layers = list(cfg.get("mem_layers", []))
    if workspace.model_kind == "حراف ۲":
        evidence = [
            "ساختار config ذخیره‌شده داخل model.pt با نسخه معماری ۲ سازگار است.",
            "خروجی forward شبکه شامل logits برای پیش‌بینی توکن بعدی است.",
            f"توکنایزر {'بارگذاری شده' if workspace.tokenizer is not None else 'در دسترس نیست'}.",
            f"وضعیت chat_tuned در info.json: {'فعال' if chat_tuned else 'غیرفعال یا نامشخص'}.",
        ]
    else:
        evidence = [
            f"نوع مدل از config/metadata و نام معماری‌ها بررسی شد: {arch}.",
            f"الگوی نام {len(workspace.tensor_rows)} Tensor برای کمک به تشخیص ساختار بررسی شد.",
            f"توکنایزر {'بارگذاری شده' if workspace.tokenizer is not None else 'در دسترس نیست'}.",
            ("مدل Hugging Face با موتور Transformers بارگذاری شده و اجرای forward/generate آن در دسترس است." if getattr(workspace, "generation_task", None) in {"causal_lm", "seq2seq"} else "این checkpoint در حالت عمومی بررسی/ویرایش است و موتور اجرای متنی آن در دسترس نیست."),
        ]
    training = workspace.training_state or {}
    training_text = {
        "step": training.get("step"),
        "tokens": training.get("tokens"),
        "loss": training.get("loss"),
        "epoch": training.get("epoch"),
    }
    return {
        "name": info.get("name") or workspace.source_dir.name,
        "family": "Harraf-2" if workspace.model_kind == "حراف ۲" else (str(cfg.get("model_type") or ("Hugging Face" if workspace.model_kind == "Hugging Face" else "مدل عمومی"))),
        "architecture": arch,
        "task": task,
        "purpose": purpose,
        "confidence": confidence,
        "confidence_note": confidence_note,
        "chat_tuned": chat_tuned,
        "config": {k: cfg.get(k) for k in ("version", "model_type", "architectures", "d_model", "hidden_size", "n_layer", "num_hidden_layers", "num_layers", "n_head", "num_attention_heads", "n_kv_head", "num_key_value_heads", "mlp_hidden", "intermediate_size", "vocab_size", "max_seq_len", "max_position_embeddings", "tree", "parse_layer", "mem_layers", "mem_keys", "mem_topk", "problem_type", "task") if k in cfg},
        "parameter_count": trainable_count,
        "tensor_count": tensor_count,
        "layer_count": n_layer,
        "memory_layers": memory_layers,
        "tokenizer_ready": workspace.tokenizer is not None,
        "tokenizer_vocab": (workspace.tokenizer.get_vocab_size(with_added_tokens=True) if workspace.tokenizer is not None and hasattr(workspace.tokenizer, "get_vocab_size") else (getattr(workspace.tokenizer, "vocab_size", None) if workspace.tokenizer is not None else None)),
        "training_metadata": training_text,
        "evidence": evidence,
        "limits": [
            "نوع وظیفه از معماری و metadata استنباط شده؛ معیار کیفیت یا دقت از config قابل استخراج نیست.",
            "برای تشخیص توانایی‌هایی مثل دانش، استدلال، صحت factual یا پیروی از دستور، مجموعه آزمون دارای پاسخ مرجع لازم است.",
            "برای checkpointهای عمومی، وظیفه از config/metadata و نام لایه‌ها به‌صورت احتمالی استنباط می‌شود؛ تشخیص قطعی به model card یا معماری اجرایی نیاز دارد.",
            "مدل‌های Hugging Face با معماری‌های استاندارد Transformers در صورت شناسایی head متنی، تولید و A/B دارند؛ checkpointهای خام بدون adapter فقط بررسی/ویرایش می‌شوند.",
        ],
    }


def build_model_dossier(workspace, findings: list[dict[str, Any]] | None = None,
                        activation: dict[str, Any] | None = None,
                        behavioral_samples: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Create compact but broad context for a model-analysis LLM.

    Never serializes the checkpoint or large raw matrices. It includes every model
    tensor (trainable parameters and buffers), with role, dimensions and compact stats.
    """
    params = dict(workspace.model.named_parameters(remove_duplicate=False))
    buffers = dict(workspace.model.named_buffers())
    tensor_values = {**buffers, **params}
    tensors: list[dict[str, Any]] = []
    layers: dict[str, dict[str, Any]] = {}
    families: dict[str, dict[str, Any]] = {}
    for row in workspace.tensor_rows:
        name = row["name"]
        if name not in tensor_values:
            continue
        role, description, layer = tensor_role(name)
        family, family_desc = _tensor_family(name)
        tensor = tensor_values[name].detach()
        st = stats(tensor, hist_samples=48)
        entry = {
            "name": name,
            "shape": list(row["shape"]),
            "dtype": row["dtype"],
            "elements": int(row["elements"]),
            "trainable": name in params,
            "role": role,
            "what_it_does": description,
            "where_it_affects": family_desc,
            "family": family,
            "layer": layer,
            "stats": {k: round(v, 7) if isinstance(v, float) else v
                      for k, v in st.items() if k not in {"sample", "statistics_sampled"}},
            "sample_values": [round(float(x), 6) for x in st.get("sample", [])[:6]],
        }
        tensors.append(entry)
        group = families.setdefault(family, {"description": family_desc, "tensor_count": 0, "parameters": 0, "names": []})
        group["tensor_count"] += 1
        group["parameters"] += int(row["elements"])
        group["names"].append(name)
        layer_key = layer or "سراسری"
        lg = layers.setdefault(layer_key, {"tensor_count": 0, "parameters": 0, "families": {}, "tensors": []})
        lg["tensor_count"] += 1
        lg["parameters"] += int(row["elements"])
        lg["families"][family] = lg["families"].get(family, 0) + 1
        lg["tensors"].append({"name": name, "shape": list(row["shape"]), "family": family,
                              "rms": round((sum(float(v) * float(v) for v in st.get("sample", [])) / max(1, len(st.get("sample", [])))) ** 0.5, 7),
                              "std": round(float(st.get("std", 0.0)), 7), "max_abs": round(max(abs(float(st.get("min", 0.0))), abs(float(st.get("max", 0.0)))), 7),
                              "nonfinite": int(st.get("nonfinite", 0))})
    return {
        "model_profile": build_model_profile(workspace),
        "whole_model": {
            "tensor_count": len(tensors),
            "parameter_count": sum(x["elements"] for x in tensors),
            "family_summary": families,
            "layer_summary": layers,
        },
        "all_model_tensors": tensors,
        "all_trainable_tensors": [item for item in tensors if item["trainable"]],
        "model_metadata": {
            "config": workspace.config,
            "info": workspace.info,
            "training_state": workspace.training_state,
            "file_inventory": [{k: row.get(k) for k in ("path", "size", "kind")} for row in workspace.inventory],
        },
        "structural_findings": (findings or [])[:100],
        "activation_probe": activation or {"available": False, "reason": "هنوز بررسی فعال‌سازی اجرا نشده یا توکنایزر در دسترس نیست."},
        "behavioral_samples": behavioral_samples or [],
        "change_history": list(getattr(workspace, "change_log", []))[-30:],
        "interpretation_rules": [
            "آمار وزن، علت رفتار را اثبات نمی‌کند؛ رابطه علّی باید با آزمون کنترل‌شده سنجیده شود.",
            "نمونه وزن‌ها فقط چند مقدار نماینده‌اند و کل ماتریس به LLM ارسال نشده است.",
            "بهبود تواناییِ آموزش‌ندیده یا افزودن دانش جدید صرفاً با یک تغییر مقیاس تضمین نمی‌شود.",
            "هر تغییر پیشنهادی باید برگشت‌پذیر و همراه با آزمون قبل/بعد باشد.",
        ],
    }
