"""G-1 support code: smoke suite, config validation, records, summary, report.

Everything here is pure standard library plus PyYAML, so the whole module -- and
therefore the whole test suite -- imports inside the lightweight local `.venv`
(`requirements/dev.txt`). The model itself is never touched from this module;
loading and inference live in `scripts/g_minus1_restore_smoke.py`, which imports
torch and transformers lazily and only runs on Colab.

Gate G-1 asks one narrow question: is the pinned off-the-shelf Vietnamese
diacritic restorer a *trustworthy candidate* for the `RESTORE` system of the
UNMARK study? It does not ask whether the restorer is accurate, and it does not
compare anything to UNMARK.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from unmark.orthography import base_signature, first_divergence, nfc, rewrite_signature, word_diff

SCRIPT_VERSION = "g_minus1_restore_smoke/1.1"

DEFAULT_CONFIG = "configs/restore/nrl_vit5_base.yaml"
DEFAULT_OUTPUT_ROOT = "results/g_minus1"

STATUS_PASS = "ENGINEERING_SMOKE_PASS"
STATUS_FAIL = "ENGINEERING_SMOKE_FAIL"

# Shown when someone runs the real smoke test without the experiment stack.
EXPERIMENT_DEPS_MESSAGE = """\
G-1 model inference requires the experiment environment.
Run this on Google Colab using requirements/experiment.txt.

The local .venv is deliberately lightweight: it carries no torch, transformers,
sentencepiece or safetensors, and no Hugging Face checkpoint is ever downloaded
onto the local machine. Nothing will be installed automatically.

On a Colab GPU runtime, inside the cloned repository:

    export HF_HOME="$PWD/.hf-cache"
    python -m venv .venv-colab --system-site-packages
    .venv-colab/bin/python -m pip install --upgrade pip
    .venv-colab/bin/python -m pip install -r requirements/experiment.txt
    .venv-colab/bin/python scripts/g_minus1_restore_smoke.py

Locally you can still run everything that does not need the model:

    pytest -q
    python scripts/g_minus1_restore_smoke.py --list-cases
"""


# ---------------------------------------------------------------------------
# Built-in smoke suite
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SmokeCase:
    id: str
    category: str
    text: str


# Categories whose correct restoration cannot be decided even by a human without
# further context: several readings are defensible. Nothing anywhere in this
# harness carries restoration ground truth -- the proposal forbids inventing it --
# so this marks the cases a human reviewer cannot adjudicate either, rather than
# the cases where a label is missing.
NOT_HUMANLY_ADJUDICABLE_CATEGORIES = ("ambiguity_short", "ambiguity_context")

SMOKE_CASES: tuple[SmokeCase, ...] = (
    # --- fully stripped Vietnamese -------------------------------------------
    SmokeCase("fs_01", "full_strip", "toi dang nghien cuu xu ly ngon ngu tu nhien"),
    SmokeCase("fs_02", "full_strip", "toi rat thich bo phim nay"),
    SmokeCase("fs_03", "full_strip", "hom nay thoi tiet o thanh pho ho chi minh rat dep"),
    SmokeCase("fs_04", "full_strip", "chung toi se hop vao chieu mai de thao luan ve ke hoach nam sau"),
    SmokeCase("fs_05", "full_strip", "quan an nay co mon pho bo rat ngon va gia ca hop ly"),
    SmokeCase("fs_06", "full_strip", "sinh vien can dang ky mon hoc truoc ngay mai"),
    # --- partially stripped: accented and unaccented words in one sentence ---
    SmokeCase("ps_01", "partial_strip", "tôi đang nghien cứu xử lý ngon ngu tự nhiên"),
    SmokeCase("ps_02", "partial_strip", "hôm nay toi muốn đi học ở truong đại học"),
    SmokeCase("ps_03", "partial_strip", "cô ây noi rằng buoi họp đã bị hoãn"),
    SmokeCase("ps_04", "partial_strip", "chúng toi dang cho ket qua tu hội đồng"),
    SmokeCase("ps_05", "partial_strip", "Ha Noi hôm nay trời rất đep"),
    # --- already correct Vietnamese ------------------------------------------
    SmokeCase("ac_01", "already_clean", "Tôi đang nghiên cứu xử lý ngôn ngữ tự nhiên."),
    SmokeCase("ac_02", "already_clean", "Hôm nay thời tiết ở Thành phố Hồ Chí Minh rất đẹp."),
    SmokeCase("ac_03", "already_clean", "Bạn có thể bán cho tôi một cái bàn không?"),
    SmokeCase(
        "ac_04",
        "already_clean",
        "Trường Đại học Khoa học Tự nhiên trực thuộc Đại học Quốc gia Thành phố Hồ Chí Minh.",
    ),
    SmokeCase("ac_05", "already_clean", "Đây là một câu tiếng Việt hoàn chỉnh với đầy đủ dấu."),
    # --- short, context-poor, genuinely ambiguous ----------------------------
    # Diagnostic only: several restorations are defensible for each of these.
    SmokeCase("as_01", "ambiguity_short", "ban"),
    SmokeCase("as_02", "ambiguity_short", "nam"),
    SmokeCase("as_03", "ambiguity_short", "ma"),
    SmokeCase("as_04", "ambiguity_short", "hoa"),
    SmokeCase("as_05", "ambiguity_short", "bo"),
    SmokeCase("as_06", "ambiguity_short", "dan"),
    SmokeCase("as_07", "ambiguity_short", "cam"),
    SmokeCase("as_08", "ambiguity_short", "toi"),
    # --- ambiguous base repeated with disambiguating context ------------------
    SmokeCase("ax_01", "ambiguity_context", "ban co the ban cho toi mot cai ban khong"),
    SmokeCase("ax_02", "ambiguity_context", "nam nay co nam muoi sinh vien dang ky"),
    SmokeCase("ax_03", "ambiguity_context", "co ay ten la hoa va rat thich hoa hong"),
    SmokeCase("ax_04", "ambiguity_context", "toi mua mot bo sach ve bo o mien bac"),
    # --- proper names ---------------------------------------------------------
    SmokeCase("pn_01", "proper_names", "Nguyen Viet Anh dang lam viec tai Ha Noi"),
    SmokeCase("pn_02", "proper_names", "Vo Quoc Linh dang hoc tai VNU-HCM"),
    SmokeCase("pn_03", "proper_names", "Truong Dai hoc Khoa hoc Tu nhien nam o Quan 5"),
    SmokeCase("pn_04", "proper_names", "Anh Hung va chi Huong den tu Da Nang"),
    SmokeCase("pn_05", "proper_names", "Le Thi Bich Ngoc sinh nam 1998 tai Hue"),
    # --- mixed Vietnamese / English ------------------------------------------
    SmokeCase("ms_01", "mixed_script", "toi dang hoc machine learning tai VNU-HCM"),
    SmokeCase("ms_02", "mixed_script", "hom nay toi dung Python va PyTorch de train model"),
    SmokeCase("ms_03", "mixed_script", "team cua toi vua deploy mot API moi len server"),
    SmokeCase("ms_04", "mixed_script", "ban co the gui file PDF qua Zalo cho minh khong"),
    # --- punctuation, numbers, percentages, dates ----------------------------
    SmokeCase("pu_01", "punctuation_numbers", "Nam 2026, GDP tang 6,5% so voi nam truoc."),
    SmokeCase("pu_02", "punctuation_numbers", "Ban co chac khong? Toi nghi la khong!"),
    SmokeCase("pu_03", "punctuation_numbers", "Cuoc hop bat dau luc 14:30 ngay 19/08/2026."),
    SmokeCase("pu_04", "punctuation_numbers", "Gia ve la 250.000 dong (da bao gom thue VAT 10%)."),
    # --- URLs and e-mail addresses: ideally untouched -------------------------
    SmokeCase("ue_01", "url_email", "Xem chi tiet tai https://example.edu.vn/tuyen-sinh nhe"),
    SmokeCase("ue_02", "url_email", "Lien he qua email lien.he@example.com de biet them"),
    SmokeCase("ue_03", "url_email", "Tai lieu o duong dan http://example.com/tai-lieu?id=42&lang=vi"),
    SmokeCase("ue_04", "url_email", "Gui mail cho support@example.org truoc 17h hom nay"),
    # --- emoji ----------------------------------------------------------------
    SmokeCase("em_01", "simple_emoji", "hom nay toi rat vui \U0001F604"),
    SmokeCase("em_02", "simple_emoji", "chuc mung sinh nhat ban \U0001F389\U0001F382"),
    SmokeCase("em_03", "simple_emoji", "troi mua to qua ☔ nen toi o nha"),
)

CATEGORY_ORDER: tuple[str, ...] = (
    "full_strip",
    "partial_strip",
    "already_clean",
    "ambiguity_short",
    "ambiguity_context",
    "proper_names",
    "mixed_script",
    "punctuation_numbers",
    "url_email",
    "simple_emoji",
)


def validate_suite(cases: Sequence[SmokeCase] = SMOKE_CASES) -> None:
    """Fail loudly on a malformed suite (duplicate ids, stray category)."""
    ids = [c.id for c in cases]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"duplicate smoke-case ids: {duplicates}")
    unknown = sorted({c.category for c in cases} - set(CATEGORY_ORDER))
    if unknown:
        raise ValueError(f"unknown categories: {unknown}")
    empty = [c.id for c in cases if not c.text.strip()]
    if empty:
        raise ValueError(f"empty smoke-case text: {empty}")


def select_cases(categories: str | None) -> list[SmokeCase]:
    """The built-in suite, optionally filtered to a comma-separated category list."""
    validate_suite()
    if not categories:
        return list(SMOKE_CASES)
    wanted = [c.strip() for c in categories.split(",") if c.strip()]
    unknown = sorted(set(wanted) - set(CATEGORY_ORDER))
    if unknown:
        raise SystemExit(f"unknown categories: {unknown}; known: {list(CATEGORY_ORDER)}")
    return [c for c in SMOKE_CASES if c.category in wanted]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def load_config(path: str | Path) -> dict[str, Any]:
    """Read and validate the locked YAML config."""
    import yaml  # PyYAML is in requirements/base.txt, so it is present everywhere

    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: config must be a YAML mapping")
    return validate_config(cfg)


def validate_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Enforce the G-1 invariants: pinned revision, deterministic greedy decoding."""
    for key in ("model_id", "revision", "generation", "repeats"):
        if key not in cfg:
            raise ValueError(f"config is missing required key: {key!r}")
    if not isinstance(cfg["revision"], str) or len(cfg["revision"]) < 7:
        raise ValueError("config: 'revision' must be a full commit hash (the checkpoint is pinned)")

    gen = cfg["generation"]
    if not isinstance(gen, dict):
        raise ValueError("config: 'generation' must be a mapping")
    if gen.get("do_sample", False):
        raise ValueError("config: G-1 requires deterministic greedy decoding, so do_sample must be false")
    if gen.get("num_beams", 1) != 1:
        raise ValueError("config: G-1 requires greedy decoding, so num_beams must be 1")
    if "max_new_tokens" not in gen and "max_length" not in gen:
        raise ValueError(
            "config: generation must bound the output length; this checkpoint's "
            "generation_config.json has no max_length, so transformers would "
            "silently fall back to max_length=20"
        )
    repeats = cfg["repeats"]
    if not isinstance(repeats, int) or repeats < 2:
        raise ValueError("config: 'repeats' must be an integer >= 2 for the determinism check")
    return cfg


def engineering_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    """Engineering-only thresholds, with defaults if the config omits them."""
    eng = dict(cfg.get("engineering") or {})
    eng.setdefault(
        "core_categories",
        [
            "full_strip",
            "partial_strip",
            "already_clean",
            "ambiguity_context",
            "proper_names",
            "punctuation_numbers",
        ],
    )
    eng.setdefault("advisory_categories", ["ambiguity_short", "mixed_script", "url_email", "simple_emoji"])
    eng.setdefault("min_core_rewrite_preservation_rate", 0.9)
    return eng


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------
def build_record(
    case: SmokeCase,
    outputs: list[str],
    latencies_ms: list[float],
    error: str | None,
    n_input_tokens: int | None = None,
    input_truncated: bool | None = None,
) -> dict[str, Any]:
    """Assemble the per-case result record written to ``cases.jsonl``.

    Pure: no model, no I/O. Everything a reader needs to audit the case is in
    here, including the exact signature mismatches when one occurs.

    Two independent diagnostics are recorded for every case, and they are not
    interchangeable:

    * ``base_preserved`` -- strict. Compares :func:`base_signature`, which keeps
      case and punctuation, so it flags *any* change the restorer made beyond
      the diacritics. This is the honest record of what happened.
    * ``rewrite_preserved`` -- the engineering lexical check. Compares
      :func:`rewrite_signature`, which additionally tolerates letter case and a
      trailing sentence stop, because a restorer capitalising a sentence or
      adding a final period has not rewritten any word. This, not
      ``base_preserved``, drives ``no_catastrophic_lexical_rewriting``.

    A case with ``base_preserved=False`` and ``rewrite_preserved=True`` is the
    normal, expected shape for a lowercase unpunctuated input.
    """
    input_sig = base_signature(case.text)
    input_rewrite_sig = rewrite_signature(case.text)
    record: dict[str, Any] = {
        "id": case.id,
        "category": case.category,
        "input": case.text,
        "outputs": outputs,
        "final_output": outputs[-1] if outputs else None,
        "deterministic": None,
        "input_base_signature": input_sig,
        "output_base_signature": None,
        "base_preserved": None,
        "input_rewrite_signature": input_rewrite_sig,
        "output_rewrite_signature": None,
        "rewrite_preserved": None,
        "clean_exact_preserved": None,
        "error": error,
        "latency_ms": None,
        # --- supporting detail -------------------------------------------
        "latencies_ms": latencies_ms,
        "base_preserved_strict": None,
        "whitespace_only_difference": None,
        "formatting_only_difference": None,
        "base_diff": None,
        "rewrite_diff": None,
        "first_divergence_char_index": None,
        "n_input_tokens": n_input_tokens,
        "input_truncated": input_truncated,
        "humanly_adjudicable": case.category not in NOT_HUMANLY_ADJUDICABLE_CATEGORIES,
    }
    if error is not None or not outputs:
        return record

    record["deterministic"] = all(o == outputs[0] for o in outputs)
    final = outputs[-1]
    output_sig = base_signature(final)
    record["output_base_signature"] = output_sig
    record["base_preserved"] = input_sig == output_sig
    output_rewrite_sig = rewrite_signature(final)
    record["output_rewrite_signature"] = output_rewrite_sig
    record["rewrite_preserved"] = input_rewrite_sig == output_rewrite_sig
    strict_equal = base_signature(case.text, collapse_whitespace=False) == base_signature(
        final, collapse_whitespace=False
    )
    record["base_preserved_strict"] = strict_equal
    record["whitespace_only_difference"] = bool(record["base_preserved"]) and not strict_equal
    # The signature that the strict check rejects but the lexical check accepts:
    # capitalisation and/or a trailing stop, and nothing else.
    record["formatting_only_difference"] = not record["base_preserved"] and record["rewrite_preserved"]
    if not record["base_preserved"]:
        record["base_diff"] = word_diff(input_sig, output_sig)
        record["first_divergence_char_index"] = first_divergence(input_sig, output_sig)
    if not record["rewrite_preserved"]:
        record["rewrite_diff"] = word_diff(input_rewrite_sig, output_rewrite_sig)
    if case.category == "already_clean":
        record["clean_exact_preserved"] = final == nfc(case.text)
    if latencies_ms:
        record["latency_ms"] = statistics.mean(latencies_ms)
    return record


# ---------------------------------------------------------------------------
# Summarisation
# ---------------------------------------------------------------------------
def _rate(numerator: int, denominator: int) -> float | None:
    """Rate, or ``None`` when the denominator is zero (never a misleading 0.0)."""
    return None if denominator == 0 else numerator / denominator


def _category_block(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    errors = [r for r in records if r["error"] is not None]
    ok = [r for r in records if r["error"] is None]
    det = [r for r in ok if r["deterministic"]]
    base = [r for r in ok if r["base_preserved"]]
    rewrite = [r for r in ok if r["rewrite_preserved"]]
    formatting_only = [r for r in ok if r["formatting_only_difference"]]
    clean_applicable = [r for r in ok if r["clean_exact_preserved"] is not None]
    clean_ok = [r for r in clean_applicable if r["clean_exact_preserved"]]
    latencies = [r["latency_ms"] for r in ok if r["latency_ms"] is not None]
    return {
        "n": n,
        "n_success": len(ok),
        "num_errors": len(errors),
        "num_deterministic": len(det),
        "num_base_preserved": len(base),
        "num_rewrite_preserved": len(rewrite),
        "num_formatting_only_difference": len(formatting_only),
        "n_clean_applicable": len(clean_applicable),
        "num_clean_exact_preserved": len(clean_ok),
        "deterministic_rate": _rate(len(det), len(ok)),
        "base_preservation_rate": _rate(len(base), len(ok)),
        "rewrite_preservation_rate": _rate(len(rewrite), len(ok)),
        "clean_exact_preservation_rate": _rate(len(clean_ok), len(clean_applicable)),
        "mean_latency_ms": statistics.mean(latencies) if latencies else None,
    }


def summarize(records: Sequence[dict[str, Any]], model_id: str, revision: str) -> dict[str, Any]:
    """Aggregate per-case records into ``summary.json``.

    Rates are computed over *successful* cases only; a case that raised is
    counted in ``num_errors`` and excluded from every rate, so an error can
    never inflate a preservation rate.

    Both preservation metrics are reported side by side and never merged:
    ``base_preservation_rate`` is the strict one, ``rewrite_preservation_rate``
    is the engineering lexical one that drives the status. Expect the strict
    rate to be the lower of the two whenever inputs are lowercase and
    unpunctuated; the gap is counted in ``num_formatting_only_difference``.
    """
    overall = _category_block(records)
    latencies = [r["latency_ms"] for r in records if r["error"] is None and r["latency_ms"] is not None]
    by_category: dict[str, Any] = {}
    for category in CATEGORY_ORDER:
        block = [r for r in records if r["category"] == category]
        if block:
            by_category[category] = _category_block(block)
    for category in sorted({r["category"] for r in records} - set(CATEGORY_ORDER)):
        by_category[category] = _category_block([r for r in records if r["category"] == category])
    return {
        "num_cases": overall["n"],
        "num_success": overall["n_success"],
        "num_errors": overall["num_errors"],
        "deterministic_rate": overall["deterministic_rate"],
        "base_preservation_rate": overall["base_preservation_rate"],
        "rewrite_preservation_rate": overall["rewrite_preservation_rate"],
        "clean_exact_preservation_rate": overall["clean_exact_preservation_rate"],
        "rates_by_category": by_category,
        "mean_latency_ms": statistics.mean(latencies) if latencies else None,
        "median_latency_ms": statistics.median(latencies) if latencies else None,
        "model_id": model_id,
        "revision": revision,
        "overall_counts": overall,
    }


def engineering_status(
    summary: dict[str, Any],
    eng: dict[str, Any],
    *,
    model_loaded: bool,
) -> tuple[str, list[dict[str, Any]]]:
    """Engineering-only PASS/FAIL, plus the checks that produced it.

    Deliberately blind to restoration *quality*: it never asks whether a
    restored sentence is correct Vietnamese, only whether the checkpoint is
    usable as a frozen component.

    The rewriting check is thresholded on ``rewrite_preservation_rate``, not on
    the strict ``base_preservation_rate``: capitalising a sentence or adding a
    final stop is a formatting change, and failing a gate for that would be
    failing it for the wrong reason. The strict rate is still computed and is
    quoted in the check's detail string so the difference stays visible.
    """
    core = list(eng["core_categories"])
    min_rewrite = float(eng["min_core_rewrite_preservation_rate"])
    by_cat = summary.get("rates_by_category", {})
    core_blocks = [by_cat[c] for c in core if c in by_cat]

    core_errors = sum(b["num_errors"] for b in core_blocks)
    core_ok = sum(b["n_success"] for b in core_blocks)
    core_rewrite_preserved = sum(b["num_rewrite_preserved"] for b in core_blocks)
    core_rewrite_rate = _rate(core_rewrite_preserved, core_ok)
    core_base_preserved = sum(b["num_base_preserved"] for b in core_blocks)
    core_base_rate = _rate(core_base_preserved, core_ok)
    det_rate = summary.get("deterministic_rate")

    checks = [
        {
            "check": "model_loaded",
            "passed": bool(model_loaded),
            "detail": "checkpoint downloaded, loaded and moved to the device",
        },
        {
            "check": "core_inference_completed",
            "passed": model_loaded and core_errors == 0 and core_ok > 0,
            "detail": f"{core_ok} successful / {core_errors} errored across core categories {core}",
        },
        {
            "check": "greedy_decoding_deterministic",
            "passed": det_rate == 1.0,
            "detail": (
                f"{summary['overall_counts']['num_deterministic']}/{summary['overall_counts']['n_success']} "
                f"successful cases produced identical output across all repeats"
            ),
        },
        {
            "check": "no_catastrophic_lexical_rewriting",
            "passed": core_rewrite_rate is not None and core_rewrite_rate >= min_rewrite,
            "detail": (
                f"core rewrite-preservation rate "
                f"{core_rewrite_rate if core_rewrite_rate is None else round(core_rewrite_rate, 4)} "
                f"(threshold {min_rewrite}) over {core_ok} successful core cases; "
                f"strict base-preservation rate over the same cases was "
                f"{core_base_rate if core_base_rate is None else round(core_base_rate, 4)} "
                f"(recorded, not thresholded)"
            ),
        },
    ]
    status = STATUS_PASS if all(c["passed"] for c in checks) else STATUS_FAIL
    return status, checks


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _fmt_rate(rate: float | None) -> str:
    return "n/a" if rate is None else f"{rate * 100:.1f}%"


def _fmt_ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}"


def _fmt_frac(num: int, den: int) -> str:
    return "-" if den == 0 else f"{num}/{den}"


def format_console_table(summary: dict[str, Any]) -> str:
    """Compact end-of-run table printed to stdout."""
    headers = ("Category", "N", "Errors", "Deterministic", "Base preserved", "Lexical kept", "Clean exact")
    rows: list[tuple[str, ...]] = []
    for category, block in summary["rates_by_category"].items():
        rows.append(
            (
                category,
                str(block["n"]),
                str(block["num_errors"]),
                _fmt_frac(block["num_deterministic"], block["n_success"]),
                _fmt_frac(block["num_base_preserved"], block["n_success"]),
                _fmt_frac(block["num_rewrite_preserved"], block["n_success"]),
                _fmt_frac(block["num_clean_exact_preserved"], block["n_clean_applicable"]),
            )
        )
    overall = summary["overall_counts"]
    total = (
        "TOTAL",
        str(overall["n"]),
        str(overall["num_errors"]),
        _fmt_frac(overall["num_deterministic"], overall["n_success"]),
        _fmt_frac(overall["num_base_preserved"], overall["n_success"]),
        _fmt_frac(overall["num_rewrite_preserved"], overall["n_success"]),
        _fmt_frac(overall["num_clean_exact_preserved"], overall["n_clean_applicable"]),
    )
    widths = [max(len(headers[i]), max((len(r[i]) for r in rows + [total]), default=0)) for i in range(len(headers))]

    def line(cells: Sequence[str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells)).rstrip()

    out = [line(headers), line(["-" * w for w in widths])]
    out.extend(line(r) for r in rows)
    out.append(line(["-" * w for w in widths]))
    out.append(line(total))
    out.append("")
    out.append("Base preserved = strict (case and punctuation significant).")
    out.append("Lexical kept   = engineering check: tolerates capitalisation and a final stop.")
    return "\n".join(out)


def _md_cell(text: str | None) -> str:
    if text is None:
        return "_(none)_"
    return text.replace("|", "\\|").replace("\n", " ")


def render_report(
    run_config: dict[str, Any],
    summary: dict[str, Any],
    records: Sequence[dict[str, Any]],
    status: str,
    checks: Sequence[dict[str, Any]],
    eng: dict[str, Any],
) -> str:
    """Human-readable ``report.md``."""
    env = run_config["environment"]
    model = run_config["model"]
    gen = run_config["generation"]
    lines: list[str] = []
    a = lines.append

    a(f"# G-1 RESTORE smoke test - `{model['model_id']}`")
    a("")
    a(f"Run id: `{run_config['run_id']}`  ")
    a(f"Timestamp (UTC): `{run_config['timestamp_utc']}`  ")
    a(f"Script: `{run_config['script_version']}`")
    a("")
    a("> This report answers an **engineering** question only: is this pinned checkpoint")
    a("> loadable, deterministic and non-destructive enough to serve as the frozen")
    a("> `RESTORE` baseline? It deliberately does **not** score restoration accuracy,")
    a("> does not compare anything to UNMARK, and does not decide G-1.")
    a("")

    a("## Environment")
    a("")
    a("| Field | Value |")
    a("|---|---|")
    for key in (
        "python_version",
        "platform",
        "torch_version",
        "transformers_version",
        "device",
        "device_name",
        "cuda_version",
    ):
        if env.get(key) is not None:
            a(f"| `{key}` | {_md_cell(str(env[key]))} |")
    a("")

    a("## Model")
    a("")
    a("| Field | Value |")
    a("|---|---|")
    a(f"| model id | `{model['model_id']}` |")
    a(f"| pinned revision | `{model['revision']}` |")
    for key, value in (model.get("metadata") or {}).items():
        a(f"| {key} | {_md_cell(str(value))} |")
    if run_config.get("expected_dtype") is not None:
        a(f"| expected dtype (diagnostic only) | `{run_config['expected_dtype']}` |")
        a(f"| dtype matches expectation | {run_config.get('dtype_matches_expectation')} |")
    a("")
    a("`expected_dtype` is an expectation recorded for reproducibility, not a runtime")
    a("control: the checkpoint is loaded exactly as stored and nothing is cast. The")
    a("authoritative value is the observed `parameter_dtype` above.")
    a("")

    a("## Generation settings")
    a("")
    a("| Field | Value |")
    a("|---|---|")
    for key, value in gen.items():
        a(f"| `{key}` | `{value}` |")
    a(f"| repeats per case | `{run_config['repeats']}` |")
    a(f"| max input tokens | `{run_config.get('max_input_tokens')}` |")
    a("")
    a("Decoding is greedy and deterministic by construction (`do_sample=false`,")
    a("`num_beams=1`). `max_new_tokens` is used rather than `max_length` because this")
    a("checkpoint's `generation_config.json` carries no length bound, so an unbounded")
    a("call would silently inherit the library default of `max_length=20`; and because")
    a("for an encoder-decoder `max_new_tokens` counts only newly generated tokens,")
    a("independently of the encoder input length.")
    a("")

    a("## Overall summary")
    a("")
    a("| Metric | Value |")
    a("|---|---|")
    a(f"| cases | {summary['num_cases']} |")
    a(f"| successful | {summary['num_success']} |")
    a(f"| errors | {summary['num_errors']} |")
    a(f"| deterministic rate | {_fmt_rate(summary['deterministic_rate'])} |")
    a(f"| strict base-preservation rate | {_fmt_rate(summary['base_preservation_rate'])} |")
    a(f"| engineering lexical-preservation rate | {_fmt_rate(summary['rewrite_preservation_rate'])} |")
    a(f"| formatting-only differences | {summary['overall_counts']['num_formatting_only_difference']} |")
    a(f"| clean-exact preservation rate | {_fmt_rate(summary['clean_exact_preservation_rate'])} |")
    a(f"| mean latency (ms) | {_fmt_ms(summary['mean_latency_ms'])} |")
    a(f"| median latency (ms) | {_fmt_ms(summary['median_latency_ms'])} |")
    a("")

    a("## Summary by category")
    a("")
    a("| Category | Role | N | Errors | Deterministic | Base preserved (strict) | Lexical kept (engineering) | Clean exact | Mean ms |")
    a("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    core = set(eng["core_categories"])
    for category, block in summary["rates_by_category"].items():
        role = "core" if category in core else "advisory"
        a(
            f"| {category} | {role} | {block['n']} | {block['num_errors']} | "
            f"{_fmt_frac(block['num_deterministic'], block['n_success'])} | "
            f"{_fmt_frac(block['num_base_preserved'], block['n_success'])} | "
            f"{_fmt_frac(block['num_rewrite_preserved'], block['n_success'])} | "
            f"{_fmt_frac(block['num_clean_exact_preserved'], block['n_clean_applicable'])} | "
            f"{_fmt_ms(block['mean_latency_ms'])} |"
        )
    a("")
    a("*core* categories drive the engineering status; *advisory* categories")
    a("(short ambiguous strings, mixed script, URLs/e-mail, emoji) are reported")
    a("separately and never flip it, per the G-1 brief.")
    a("")
    a("### Two preservation measurements, deliberately not merged")
    a("")
    a("| | Compares | Sensitive to | Used for |")
    a("|---|---|---|---|")
    a("| **Base preserved** (strict) | `base_signature` | case, punctuation, every word | the honest record of what the restorer changed |")
    a("| **Lexical kept** (engineering) | `rewrite_signature` | every word; *not* case, *not* a trailing `.!?…` | the `no_catastrophic_lexical_rewriting` check |")
    a("")
    a("Both ignore Vietnamese diacritics -- that is the point of the gate. The strict")
    a("measurement will read low whenever the inputs are lowercase and unpunctuated and")
    a("the restorer capitalises them; that is a formatting difference, not a rewrite, and")
    a("failing the gate on it would fail it for the wrong reason. Every such case is")
    a("listed under *Formatting-only differences* below, so the strict result is visible")
    a("rather than absorbed. Word substitutions, insertions and deletions, internal")
    a("punctuation, digits, URLs and e-mail addresses register in **both** measurements.")
    a("")

    flagged = [
        r
        for r in records
        if r["error"] is not None or r["deterministic"] is False or r["rewrite_preserved"] is False
    ]
    a("## Failed / lexically rewritten examples")
    a("")
    a("Listed here when a case errored, was non-deterministic, or failed the *engineering*")
    a("lexical check (`rewrite_preserved`). Cases that changed only capitalisation or a")
    a("final stop are not failures and are listed in the next section instead.")
    a("")
    if not flagged:
        a("None. Every case completed, was deterministic across repeats, and kept its")
        a("lexical content.")
        a("")
    else:
        a(f"{len(flagged)} case(s) errored, were non-deterministic, or rewrote the lexical base.")
        a("")
        for r in flagged:
            reasons = []
            if r["error"] is not None:
                reasons.append("error")
            if r["deterministic"] is False:
                reasons.append("non-deterministic")
            if r["rewrite_preserved"] is False:
                reasons.append("lexical rewrite")
            a(f"### `{r['id']}` ({r['category']}) - {', '.join(reasons)}")
            a("")
            a("```text")
            a(f"input             : {r['input']}")
            a(f"output            : {r['final_output']}")
            if r["error"] is not None:
                a(f"error             : {r['error']}")
            if r["deterministic"] is False:
                for i, o in enumerate(r["outputs"], start=1):
                    a(f"repeat {i}          : {o}")
            if r["rewrite_preserved"] is False:
                a(f"input  (lexical)  : {r['input_rewrite_signature']}")
                a(f"output (lexical)  : {r['output_rewrite_signature']}")
                for change in r["rewrite_diff"] or []:
                    a(f"  {change['op']:<8} input{change['input_words']} -> output{change['output_words']}")
            if r["base_preserved"] is False:
                a(f"input  (strict)   : {r['input_base_signature']}")
                a(f"output (strict)   : {r['output_base_signature']}")
                a(f"first divergence  : char index {r['first_divergence_char_index']}")
                for change in r["base_diff"] or []:
                    a(f"  {change['op']:<8} input{change['input_words']} -> output{change['output_words']}")
            a("```")
            a("")

    formatting = [r for r in records if r["formatting_only_difference"]]
    a("## Formatting-only differences (strict check failed, lexical check passed)")
    a("")
    a("These cases changed the strict base signature but kept every word: the restorer")
    a("capitalised the sentence, or added a sentence-final stop, or both. They are")
    a("recorded rather than hidden, and they do not affect the engineering status.")
    a("")
    if not formatting:
        a("None.")
        a("")
    else:
        a("| id | category | input | output |")
        a("|---|---|---|---|")
        for r in formatting:
            a(f"| `{r['id']}` | {r['category']} | {_md_cell(r['input'])} | {_md_cell(r['final_output'])} |")
        a("")

    clean = [r for r in records if r["category"] == "already_clean"]
    if clean:
        a("## Already-correct Vietnamese (what the restorer does to clean input)")
        a("")
        a("| id | exact | input | output |")
        a("|---|---|---|---|")
        for r in clean:
            mark = "yes" if r["clean_exact_preserved"] else ("n/a" if r["clean_exact_preserved"] is None else "NO")
            a(f"| `{r['id']}` | {mark} | {_md_cell(r['input'])} | {_md_cell(r['final_output'])} |")
        a("")
        a("`clean_exact_preserved` compares the output against `NFC(input)`. The input is")
        a("otherwise passed to the model untouched, exactly as a deployed pipeline would.")
        a("")

    short = [r for r in records if r["category"] == "ambiguity_short"]
    if short:
        a("## Short ambiguous inputs (diagnostic only, no ground truth)")
        a("")
        a("Several restorations are defensible for each of these strings. Nothing here is")
        a("scored; the outputs are recorded so the researcher can see which reading the")
        a("model commits to when context is absent.")
        a("")
        a("| id | input | output | deterministic | base preserved | lexical kept |")
        a("|---|---|---|---|---|---|")
        for r in short:
            a(
                f"| `{r['id']}` | `{_md_cell(r['input'])}` | `{_md_cell(r['final_output'])}` | "
                f"{r['deterministic']} | {r['base_preserved']} | {r['rewrite_preserved']} |"
            )
        a("")

    advisory = [c for c in eng["advisory_categories"] if c != "ambiguity_short"]
    adv_records = [r for r in records if r["category"] in advisory]
    if adv_records:
        a("## Advisory categories (mixed script, URLs/e-mail, emoji)")
        a("")
        a("Reported separately. Per the G-1 brief, failures here do not by themselves")
        a("disqualify the checkpoint; the model card explicitly lists heavy emoji and")
        a("mixed-script text as out-of-distribution.")
        a("")
        a("| id | category | base preserved | lexical kept | input | output |")
        a("|---|---|---|---|---|---|")
        for r in adv_records:
            a(
                f"| `{r['id']}` | {r['category']} | {r['base_preserved']} | {r['rewrite_preserved']} | "
                f"{_md_cell(r['input'])} | {_md_cell(r['final_output'])} |"
            )
        a("")

    ctx = [r for r in records if r["category"] == "ambiguity_context"]
    if ctx:
        a("## Ambiguous bases with context (diagnostic only, no ground truth)")
        a("")
        a("| id | input | output |")
        a("|---|---|---|")
        for r in ctx:
            a(f"| `{r['id']}` | {_md_cell(r['input'])} | {_md_cell(r['final_output'])} |")
        a("")

    a("## G-1 Assessment")
    a("")
    a(f"**Engineering status: `{status}`**")
    a("")
    a("| Check | Result | Detail |")
    a("|---|---|---|")
    for c in checks:
        a(f"| `{c['check']}` | {'PASS' if c['passed'] else 'FAIL'} | {_md_cell(c['detail'])} |")
    a("")
    a("### What this status does and does not mean")
    a("")
    a("`ENGINEERING_SMOKE_PASS` means only: the pinned checkpoint loads, ordinary")
    a("Vietnamese input runs without error, greedy decoding is reproducible, and the")
    a("model is not rewriting the lexical base (measured with `rewrite_signature`, which")
    a("tolerates capitalisation and a final stop). It is a statement about *mechanical")
    a("usability*, not about restoration quality, and it is computed without any")
    a("restoration ground truth.")
    a("")
    a("It explicitly does **not** establish that:")
    a("")
    a("* the restorations are correct (no ground truth was used, by design);")
    a("* the model handles short, context-poor input acceptably for the study;")
    a("* the proper-name behaviour is acceptable (the model card warns it is not);")
    a("* the checkpoint is the right `RESTORE` baseline for H2;")
    a("* the casing and punctuation the model introduces are acceptable for the")
    a("  downstream tasks -- the engineering check deliberately ignores them, so that")
    a("  judgement is yours to make from the formatting-only section.")
    a("")
    a("### Researcher checklist before locking this as RESTORE")
    a("")
    a("1. Read every `full_strip` and `partial_strip` output. Are the restorations")
    a("   linguistically plausible, or merely well-formed?")
    a("2. Read the already-clean section. If the model *edits* correct Vietnamese, the")
    a("   `RESTORE` pipeline must decide whether to run it on the `FULL` condition at")
    a("   all - and that decision has to be made once and frozen.")
    a("3. Read the short-ambiguity outputs. These are the inputs where restoration must")
    a("   commit, and where a wrong commitment is unrecoverable downstream (proposal")
    a("   section 1.4). Judge whether the failure mode looks like the one H2 predicts.")
    a("4. Read the proper-name outputs against the model card's own warning about")
    a("   proper-noun ambiguity.")
    a("5. Decide whether the advisory-category behaviour matters for the chosen tasks.")
    a("6. Record the decision, with this run id, before any G2 measurement is taken.")
    a("")
    a("**The G-1 decision is the researcher's. This script does not make it.**")
    a("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------
def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=False)
        fh.write("\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_artifacts(
    run_dir: Path,
    run_config: dict[str, Any],
    records: Sequence[dict[str, Any]],
    summary: dict[str, Any],
    report: str,
) -> None:
    """Write the four G-1 artifacts to ``run_dir``."""
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", run_config)
    write_jsonl(run_dir / "cases.jsonl", records)
    write_json(run_dir / "summary.json", summary)
    (run_dir / "report.md").write_text(report, encoding="utf-8")


def make_run_id(now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    return now.strftime("%Y%m%dT%H%M%SZ")


def unique_run_dir(output_root: Path, run_id: str) -> Path:
    """A run directory that never overwrites an existing one."""
    run_dir = output_root / run_id
    suffix = 1
    while run_dir.exists():
        run_dir = output_root / f"{run_id}-{suffix}"
        suffix += 1
    return run_dir
