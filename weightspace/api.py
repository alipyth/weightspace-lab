from __future__ import annotations

import json
import re
from typing import Any

from .editor import validate_proposal

SYSTEM_PROMPT_FA = """تو تحلیل‌گر تخصصی وزن‌های شبکه عصبی حراف ۲ هستی. فقط بر اساس شواهدی که داده شده فرضیه بساز؛ از نام Tensor به‌تنهایی نتیجه علّی نگیر. پاسخ فقط یک JSON معتبر و بدون Markdown باشد با کلیدهای:
{
 "hypothesis":"فرضیه فارسی و قابل‌آزمون",
 "confidence":0.0,
 "risk":"low|medium|high",
 "reversible":true,
 "limitations":"محدودیت تحلیل به فارسی",
 "suggested_edits":[{"tensor":"نام دقیق پارامتر","type":"scale|scale_slice|noise|clamp","factor":0.995,"reason":"دلیل فارسی","risk":"medium","target_relevance":0.5}]
}
برای scale و scale_slice در حالت ایمن فقط factor بین 0.98 و 1.02. برای noise فقط std_ratio حداکثر 0.005. برای clamp حداقل و حداکثر finite و مرتب بده. نام پارامتر باید دقیقاً از فهرست داده‌شده باشد. خلاصه عددی و نمونه کوچک هر وزن را در صورت وجود بررسی کن؛ این نمونه‌ها کل وزن نیستند. حداکثر ۳ ویرایش پیشنهاد بده. اگر شواهد کافی نیست، suggested_edits را خالی کن. هر ویرایش فقط آزمایش برگشت‌پذیر است، نه تضمین بهبود. هیچ‌وقت ادعا نکن فقط از روی آمار وزن‌ها مشکل رفتاری را قطعی تشخیص داده‌ای."""


def parse_json(text: str) -> dict[str, Any]:
    raw = text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        obj = json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            raise ValueError("تحلیل‌گر JSON معتبر برنگرداند.")
        obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError("خروجی تحلیل‌گر باید شیء JSON باشد.")
    return obj


def request_proposal(base_url: str, api_key: str, model_name: str, symptom: str,
                     tensor_rows: list[dict], findings: list[dict], model,
                     safe_mode: bool = True) -> dict[str, Any]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("کتابخانه openai نصب نیست.") from exc
    if not model_name.strip():
        raise ValueError("نام مدل تحلیل‌گر را وارد کن.")
    client = OpenAI(base_url=base_url.strip() or None, api_key=api_key.strip() or "ollama")
    compact_tensors = tensor_rows[:300]
    user_data = {
        "مشکل انتخاب‌شده": symptom,
        "حالت ایمن": safe_mode,
        "تنسرها": compact_tensors,
        "یافته‌های آماری": findings[:60],
    }
    messages = [{"role": "system", "content": SYSTEM_PROMPT_FA},
                {"role": "user", "content": json.dumps(user_data, ensure_ascii=False)}]
    try:
        response = client.chat.completions.create(model=model_name.strip(), messages=messages,
                                                  temperature=0.15, response_format={"type": "json_object"})
    except Exception:
        # Some Ollama-compatible servers do not implement response_format. The prompt
        # still demands strict JSON; we parse and validate it below.
        response = client.chat.completions.create(model=model_name.strip(), messages=messages, temperature=0.15)
    content = response.choices[0].message.content or "{}"
    proposal = parse_json(content)
    return validate_proposal(proposal, model, safe_mode=safe_mode)

FULL_PLAN_SYSTEM_PROMPT_FA = r'''تو یک پژوهشگر ارشد عیب‌یابی و بهبود مدل‌های هوش مصنوعی هستی. باید کل پرونده‌ای را که کاربر فرستاده بررسی کنی: معماری، هدف مدل، همه تنسورها (پارامترهای قابل‌آموزش و بافرها)، شکل تنسورها، نقش هرکدام، آمار وزن‌ها، بلوک‌ها، یافته‌های سلامت، فعال‌سازی واقعی و نمونه خروجی‌های رفتاری در صورت پشتیبانی موتور اجرا.

وظیفه تو ارائه یک «برنامه جامع اما محتاطانه برای بهبود» است؛ نه فهرست حدسی از دستکاری وزن‌ها. ابتدا از روی شواهد بگو مدل چه کاری انجام می‌دهد و چه قسمت‌هایی از معماری این وظیفه را اجرا می‌کنند. سپس نقاط قوت، ضعف‌های قابل مشاهده، ابهام‌ها و فرصت‌های بهبود را به زبان فارسی و قابل فهم توضیح بده.

قواعد مهم:
۱) تمام نام‌گذاری‌ها، توضیحات و استدلال‌ها فارسی باشد؛ فقط نام دقیق Tensor و نام عملیات انگلیسی باقی بماند.
۲) هرگز ضعف رفتاری را فقط از روی شکل یا آمار وزن قطعی اعلام نکن. بین «مشاهده»، «فرضیه» و «نیاز به آزمون» تمایز بگذار.
۳) حداکثر ۱۲ ویرایش وزن پیشنهاد بده و فقط وقتی پیشنهاد بده که برای آن شواهد و دلیل مشخص داری. اگر پرونده اعلام کرده موتور اجرای واقعی در دسترس نیست، این محدودیت را در گزارش و benchmark_plan رعایت کن و نتیجه عملکردی نساز. اگر تغییر وزن دلیل کافی ندارد، آن را در کارهای غیر وزنی/آزمون‌های پیشنهادی قرار بده و suggested_edits را خالی بگذار.
۴) پیشنهادها باید با نام دقیق Tensor موجود در dossier باشند. نوع مجاز: scale، scale_slice، noise، clamp. در حالت ایمن: scale و scale_slice فقط factor بین ۰٫۹۸ تا ۱٫۰۲؛ noise با std_ratio حداکثر ۰٫۰۰۵؛ clamp فقط اگر در محدوده ایمن تغییر بسیار کوچکی ایجاد کند. پارامترهای غیرقابل‌آموزش مثل شمارنده‌ها را پیشنهاد نده.
۵) از تغییر هم‌زمان تعداد زیادی وزن نامرتبط، تغییرهای تهاجمی روی embedding، حذف یا صفرکردن head، تغییر مشترک wte/head به شکلی که وزن مشترک را بشکند، و دستکاری تصادفی حافظه/پارسر خودداری کن.
۶) برای هر تغییر بگو دقیقاً چرا آن Tensor انتخاب شده، کدام مسیر رفتاری را هدف می‌گیرد، اثر مورد انتظار چیست، چه اثر جانبی ممکن است داشته باشد و چگونه باید سنجیده شود.
۷) اگر داده کافی برای تغییر نیست، صریح بگو «فعلاً ویرایش وزن پیشنهاد نمی‌شود». لازم نیست برای راضی کردن کاربر حتماً edit بسازی.
۸) پیشنهادها مرتب شوند: اولویت بالا، ریسک کم، مقدار تغییر کوچک و ارتباط شواهدی بیشتر جلوتر.
۹) هر پیشنهاد فقط فرضیه آزمایشی است؛ بهبود کیفیت، دانش، استدلال یا صحت را تضمین نکن.
۱۰) فقط یک شیء JSON معتبر و بدون Markdown برگردان. هیچ متن اضافه‌ای ننویس.

ساختار خروجی:
{
 "model_identity":{"name":"...","architecture":"...","task":"...","purpose":"...","evidence":["..."],"confidence":0.0},
 "executive_summary":"جمع‌بندی روشن برای کاربر تازه‌کار",
 "strengths":[{"title":"...","evidence":"...","meaning":"..."}],
 "weaknesses":[{"title":"...","evidence":"...","certainty":"observed|hypothesis|unknown","meaning":"...","what_to_check":"..."}],
 "optimization_plan":[{"priority":1,"title":"...","goal":"...","why":"...","method":"...","validation":"...","risk":"low|medium|high","weight_edit_required":true}],
 "suggested_edits":[{"tensor":"نام دقیق","type":"scale","factor":0.995,"reason":"دلیل روشن فارسی","target_relevance":0.8,"risk":"low","expected_effect":"...","possible_side_effect":"...","validation":"..."}],
 "non_weight_actions":[{"title":"...","why":"...","steps":["..."]}],
 "benchmark_plan":{"prompts":["..."],"metrics":["..."],"missing_evidence":"..."},
 "not_recommended":[{"action":"...","reason":"..."}],
 "confidence":0.0,
 "risk":"low|medium|high",
 "reversible":true,
 "limitations":"محدودیت‌ها و مرز قطعیت به فارسی"
}
'''


def request_full_model_plan(base_url: str, api_key: str, model_name: str,
                            dossier: dict[str, Any], model, safe_mode: bool = True) -> dict[str, Any]:
    """Ask an LLM for a model-wide, evidence-grounded and executable edit plan."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("کتابخانه openai نصب نیست.") from exc
    if not model_name.strip():
        raise ValueError("نام مدل تحلیل‌گر را وارد کن.")
    client = OpenAI(base_url=base_url.strip() or None, api_key=api_key.strip() or "ollama", timeout=300.0)
    payload = {"حالت ایمن": safe_mode, "پرونده جامع مدل": dossier}
    content = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    messages = [
        {"role": "system", "content": FULL_PLAN_SYSTEM_PROMPT_FA},
        {"role": "user", "content": content},
    ]
    try:
        response = client.chat.completions.create(
            model=model_name.strip(), messages=messages, temperature=0.1,
            response_format={"type": "json_object"},
        )
    except Exception:
        response = client.chat.completions.create(
            model=model_name.strip(), messages=messages, temperature=0.1,
        )
    raw = parse_json(response.choices[0].message.content or "{}")
    # Validate each proposed change separately. A single bad tensor/operation
    # must not discard an otherwise useful report; rejected items are disclosed.
    raw_edits = raw.get("suggested_edits", [])
    if not isinstance(raw_edits, list):
        raw_edits = []
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in raw_edits[:12]:
        try:
            checked = validate_proposal({"confidence": 1.0, "suggested_edits": [item]},
                                        model, safe_mode=safe_mode)
        except Exception as exc:
            rejected.append({"tensor": str(item.get("tensor", "نامشخص")) if isinstance(item, dict) else "نامشخص",
                             "reason": str(exc)})
            continue
        clean = checked["suggested_edits"][0]
        identity = (str(clean.get("tensor")), str(clean.get("type")))
        if identity in seen:
            continue
        seen.add(identity)
        accepted.append(clean)
    # Apply the same deterministic global ranking used by the manual proposal path.
    if accepted:
        accepted = validate_proposal({"confidence": raw.get("confidence", 0.0), "suggested_edits": accepted},
                                     model, safe_mode=safe_mode)["suggested_edits"]
    raw["suggested_edits"] = accepted
    raw["rejected_edits"] = rejected
    raw["plan_scope"] = "whole_model"
    raw["safe_mode"] = safe_mode
    return raw
