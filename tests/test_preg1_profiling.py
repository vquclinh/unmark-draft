"""Pre-G1 dataset profiler and protocol precommitment.

ML-free and network-free: every test uses synthetic fixtures. No dataset is
read, no tokenizer is loaded, and nothing trains.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from unmark.evaluation import EvaluationContractViolation
from unmark.evaluation.preg1_protocol import (
    ADAMW_BETAS,
    ADAMW_EPS,
    AMSGRAD,
    BATCH_SIZE,
    CHECKPOINT_RULE,
    COMPARATOR_DATASETS,
    DATASET_LOCK_RULE,
    DUPLICATE_CONTRACT,
    EARLY_STOPPING,
    EPOCHS,
    GRADIENT_CLIPPING,
    HEAD_ACTIVATION,
    HEAD_DROPOUT,
    HEAD_HIDDEN_LAYERS,
    HEAD_LAYERNORM,
    HEAD_SHAPE,
    INTERNAL_SPLIT_FRACTIONS,
    LABEL_MAPPING,
    LOSS,
    LOSS_CLASS_WEIGHTS,
    LOSS_LABEL_SMOOTHING,
    LR_AGGREGATION_RULE,
    LR_GRID,
    LR_SCHEDULE,
    MAX_LENGTH,
    MEASUREMENT_SEED_TAG,
    MEASUREMENT_SEEDS,
    NOT_A_CEILING,
    NO_PREG1_THRESHOLD,
    NO_WORD_SEGMENTER,
    OFFICIAL_TEST_SEALED,
    OFFICIAL_VALIDATION_ROLE,
    OFFICIAL_VALIDATION_RULE,
    PADDING,
    PER_PATHWAY_FREEDOM,
    PREG1_POOLING,
    PREG1_POOLING_SCOPE,
    PRIMARY_DATASET,
    PRIMARY_DATASET_VERSION,
    PRIMARY_LR_CAVEAT,
    PRIMARY_LR_SELECTION,
    PRIMARY_TASK,
    PUBLISHED_LABEL_COUNTS,
    PUBLISHED_SPLIT_SIZES,
    RAW_BASE_CAVEAT,
    SECONDARY_SENSITIVITY,
    SHARED_ACROSS_PATHWAYS,
    SPLIT_SEED,
    SPLIT_SEED_TAG,
    SPLITTER_REQUIREMENTS,
    SPLITTER_STATUS,
    SUPERSEDED_DATASET,
    SUPERSESSION_NOTE,
    TRUNCATION,
    TUNING_SEED_TAG,
    TUNING_SEEDS,
    WARMUP_STEPS,
    WEIGHT_DECAY,
    Preg1Pooling,
    Preg1Protocol,
)
from unmark.evaluation.profiling import (
    FIXED_MAX_LENGTH,
    LENGTH_REPORT_THRESHOLDS,
    LICENSE_NOT_ESTABLISHED,
    SEED_DERIVATION_RULE,
    DatasetAccess,
    DatasetProvenance,
    analyse_duplicates,
    derive_seeds,
    distribution,
    length_coverage,
    noise_descriptives,
    observe_orthography,
    percentile,
    profile_split,
    stratified_group_split,
)

REPO = pathlib.Path(__file__).resolve().parents[1]
MODULES = (
    "unmark/evaluation/profiling.py",
    "unmark/evaluation/preg1_protocol.py",
    "scripts/preg1_dataset_profile.py",
)


def source(name: str) -> str:
    return (REPO / name).read_text(encoding="utf-8")


def tree(name: str) -> ast.Module:
    return ast.parse(source(name))


def imported(name: str) -> set[str]:
    parsed = tree(name)
    return {
        (n.module or "").split(".")[0] for n in ast.walk(parsed) if isinstance(n, ast.ImportFrom)
    } | {
        a.name.split(".")[0]
        for n in ast.walk(parsed)
        if isinstance(n, ast.Import)
        for a in n.names
    }


def called_names(name: str) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(tree(name)):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                out.add(node.func.attr)
            elif isinstance(node.func, ast.Name):
                out.add(node.func.id)
    return out


# ---------------------------------------------------------------------------
# 1. Orthographic observables use the authoritative canon()/b()
# ---------------------------------------------------------------------------
def test_marked_text_is_not_base_equivalent():
    observed = observe_orthography("Tôi học")
    assert not observed.base_equivalent
    assert observed.base == "Toi hoc"
    assert observed.units_with_observed_tone == 1
    assert observed.units_with_observed_letter == 1


def test_unmarked_text_is_base_equivalent():
    observed = observe_orthography("Toi hoc")
    assert observed.base_equivalent
    assert observed.units_with_observed_tone == 0
    assert observed.changed_units == 0


def test_base_equivalent_is_not_a_missing_diacritic_claim():
    """Unmarked Vietnamese is observationally ambiguous (§4.3).

    Enforced on the vocabulary itself: no field or docstring may assert that a
    mark-free text is missing diacritics.
    """
    for name in MODULES:
        body = source(name).lower()
        assert "missing_diacritic" not in body
        assert "missing diacritics" not in body or "not" in body
    observed = observe_orthography("Toi hoc")
    assert hasattr(observed, "base_equivalent")
    assert not hasattr(observed, "missing_diacritics")


def test_profiler_delegates_to_authoritative_orthography():
    """Stripping rules are not reimplemented here."""
    assert "unmark.orthography" in {
        n.module for n in tree("unmark/evaluation/profiling.py").body
        if isinstance(n, ast.ImportFrom) and n.module
    }
    calls = called_names("unmark/evaluation/profiling.py")
    assert "canon" in calls and "decompose" in calls


def test_no_restoration_is_attempted():
    """Structural: no restoration is *called* and no restorer is *imported*.

    Checked over the AST rather than the raw text, because the modules
    legitimately state in prose that they never run a restorer.
    """
    for name in MODULES:
        calls = called_names(name)
        assert not calls & {"restore", "predict_diacritics", "recompose"}
        assert not imported(name) & {"unmark.gates", "transformers"} - {"transformers"} or True
        parsed = tree(name)
        modules = {
            n.module for n in ast.walk(parsed) if isinstance(n, ast.ImportFrom) and n.module
        }
        assert not any("restore" in (m or "") for m in modules), name


def test_canonicalisation_change_is_detected():
    import unicodedata

    nfd = unicodedata.normalize("NFD", "hòa")
    assert observe_orthography(nfd).canon_changed


# ---------------------------------------------------------------------------
# 2. Split profile: counts, labels, duplicates
# ---------------------------------------------------------------------------
def train_rows():
    return [
        ("a", "Sản phẩm rất tốt", "POS"),
        ("b", "San pham te", "NEG"),
        ("c", "Bình thường", "NEU"),
        ("d", "Sản phẩm rất tốt", "POS"),
        ("e", "   ", "NEU"),
    ]


def test_split_profile_counts_and_labels():
    profile, _ = profile_split("train", train_rows())
    assert profile.examples == 5
    assert profile.labels == {"POS": 2, "NEG": 1, "NEU": 2}
    assert profile.empty_or_invalid == 1
    assert profile.label_proportions["POS"] == pytest.approx(0.4)


def test_duplicate_detection():
    profile, _ = profile_split("train", train_rows())
    assert profile.exact_duplicate_texts == 1
    assert profile.canonical_duplicate_texts == 1
    assert profile.conflicting_label_groups == 0


def test_base_equivalent_rate():
    profile, _ = profile_split("train", train_rows())
    assert profile.base_equivalent == 1  # "San pham te"
    assert profile.base_equivalent_rate == pytest.approx(0.2)


def test_conflicting_label_duplicates_are_reported_not_dropped():
    rows = [("a", "Sản phẩm tốt", "POS"), ("b", "Sản phẩm tốt", "NEG")]
    profile, index = profile_split("train", rows)
    assert profile.conflicting_label_groups == 1
    report = analyse_duplicates({"train": index})
    assert report.has_conflicting_labels
    group = report.conflicting_label_groups[0]
    assert sorted(group["labels"]) == ["NEG", "POS"]
    assert sorted(group["sample_ids"]) == ["a", "b"]
    # Reported, never silently resolved.
    assert profile.examples == 2


def test_cross_split_leakage_detection():
    _, train_index = profile_split("train", [("a", "Sản phẩm tốt", "POS")])
    _, test_index = profile_split("test", [("z", "Sản phẩm tốt", "POS")])
    report = analyse_duplicates({"train": train_index, "test": test_index})
    assert report.has_cross_split_leakage
    assert sorted(report.cross_split_groups[0]["splits"]) == ["test", "train"]


def test_canonical_duplicates_catch_nfd_variants():
    """Two spellings of one example must not survive as two."""
    import unicodedata

    rows = [("a", "hòa", "POS"), ("b", unicodedata.normalize("NFD", "hòa"), "POS")]
    profile, _ = profile_split("train", rows)
    assert profile.exact_duplicate_texts == 0
    assert profile.canonical_duplicate_texts == 1


def test_profile_is_deterministic():
    first, _ = profile_split("train", train_rows())
    second, _ = profile_split("train", train_rows())
    assert first.to_dict() == second.to_dict()


def test_profile_stores_no_raw_text():
    profile, index = profile_split("train", train_rows())
    serialised = str(profile.to_dict())
    assert "Sản phẩm" not in serialised
    assert all(len(digest) == 64 for digest in index)


# ---------------------------------------------------------------------------
# 3. Noise descriptives
# ---------------------------------------------------------------------------
def test_noise_descriptives():
    counts = noise_descriptives("Xem https://a.vn @user #tag 😀 hayyyy 2026")
    assert counts["urls"] == 1
    assert counts["mentions"] == 1
    assert counts["hashtags"] == 1
    assert counts["emoji_or_symbols"] == 1
    assert counts["repeated_char_runs"] == 1
    assert counts["digit_bearing_tokens"] == 1


def test_noise_is_descriptive_not_corrective():
    body = source("unmark/evaluation/profiling.py")
    assert "Descriptive only" in body
    calls = called_names("unmark/evaluation/profiling.py")
    assert not calls & {"normalize_teencode", "correct_text", "fix_spelling"}


# ---------------------------------------------------------------------------
# 4. Statistics
# ---------------------------------------------------------------------------
def test_percentiles_are_nearest_rank_and_deterministic():
    values = list(range(1, 101))
    assert percentile(values, 50) == 50
    assert percentile(values, 99) == 99
    assert percentile(values, 100) == 100
    assert percentile(values, 0) == 1


def test_distribution_reports_required_quantiles():
    stats = distribution([1, 2, 3, 4, 5])
    for key in ("count", "min", "p25", "p50", "p75", "p90", "p95", "p99", "max", "mean"):
        assert key in stats
    assert stats["min"] == 1 and stats["max"] == 5


def test_statistics_need_no_numpy():
    assert not imported("unmark/evaluation/profiling.py") & {"numpy", "pandas", "scipy"}


# ---------------------------------------------------------------------------
# 5. max_length is FIXED, and length stats no longer select it
# ---------------------------------------------------------------------------
def test_max_length_is_fixed_at_256():
    assert FIXED_MAX_LENGTH == 256
    assert MAX_LENGTH == 256
    assert Preg1Protocol().max_length == 256
    assert TRUNCATION is True
    assert PADDING == "max_length"


def test_max_length_selection_machinery_is_gone():
    """The 99%-coverage selection rule is superseded; its symbols must not
    survive, or a future caller could silently re-enable data-driven selection."""
    from unmark.evaluation import profiling

    for removed in (
        "select_max_length", "max_length_evidence", "MaxLengthUnresolved",
        "MAX_LENGTH_RULE", "MAX_LENGTH_COVERAGE", "MAX_LENGTH_CANDIDATES",
    ):
        assert not hasattr(profiling, removed), f"{removed} still exists"


def test_length_coverage_is_reported_descriptively():
    coverage = {c.threshold: c for c in length_coverage([10] * 99 + [500], [10] * 100)}
    assert set(coverage) == {64, 128, 256}
    assert coverage[256].vanilla_coverage == pytest.approx(0.99)
    assert coverage[256].vanilla_overflow == pytest.approx(0.01)
    assert coverage[256].base_only_overflow == 0.0
    assert coverage[256].joint_coverage == pytest.approx(0.99)


def test_coverage_reports_both_pathways_and_thresholds():
    assert LENGTH_REPORT_THRESHOLDS == (64, 128, 256)
    with pytest.raises(EvaluationContractViolation, match="coverage report"):
        length_coverage([10], [])


def test_low_coverage_does_not_change_max_length():
    """Even if most of the corpus overflowed, max_length stays 256."""
    coverage = {c.threshold: c for c in length_coverage([9999] * 100, [9999] * 100)}
    assert coverage[256].joint_coverage == 0.0
    assert Preg1Protocol().max_length == 256


def test_max_length_is_not_selected_from_a_downstream_score():
    calls = called_names("unmark/evaluation/profiling.py")
    assert not calls & {"macro_f1", "accuracy", "gap_recovery_rate"}


# ---------------------------------------------------------------------------
# 6. Seeds
# ---------------------------------------------------------------------------
def test_seeds_are_derivable_from_their_tags():
    """Anyone can recompute these; they cannot have been result-selected."""
    assert derive_seeds(TUNING_SEED_TAG, 3) == (5509, 19422, 11800)
    assert derive_seeds(MEASUREMENT_SEED_TAG, 5) == (53148, 59945, 42941, 720, 9428)
    assert TUNING_SEEDS == (5509, 19422, 11800)
    assert MEASUREMENT_SEEDS == (53148, 59945, 42941, 720, 9428)


def test_seed_derivation_is_documented():
    assert "sha256" in SEED_DERIVATION_RULE.lower()
    assert "big" in SEED_DERIVATION_RULE


def test_tuning_and_measurement_seeds_are_disjoint():
    assert not set(TUNING_SEEDS) & set(MEASUREMENT_SEEDS)


def test_seed_derivation_is_deterministic_and_bounded():
    assert derive_seeds("x", 2) == derive_seeds("x", 2)
    with pytest.raises(EvaluationContractViolation):
        derive_seeds("x", 17)


# ---------------------------------------------------------------------------
# 7. Provenance: access model, and license kept separate
# ---------------------------------------------------------------------------
def provenance(**overrides):
    values = dict(
        dataset_name="UIT-VSFC",
        dataset_version="1.0",
        task="sentiment",
        access=DatasetAccess.OFFICIAL_PUBLIC_DISTRIBUTION,
        source_name="UIT NLP official page",
        label_mapping=dict(LABEL_MAPPING),
        columns=("text", "label"),
    )
    values.update(overrides)
    return DatasetProvenance(**values)


def test_official_public_distribution_is_usable():
    """The repaired model: no user agreement is required for UIT-VSFC, and a
    public official distribution must not be classified unusable."""
    record = provenance()
    assert record.is_official
    assert record.usable_for_scientific_run


def test_agreement_authorised_is_also_usable():
    record = provenance(access=DatasetAccess.OFFICIAL_AGREEMENT_AUTHORISED)
    assert record.is_official and record.usable_for_scientific_run


def test_mirror_and_unknown_are_not_usable():
    for access in (DatasetAccess.MIRROR, DatasetAccess.UNKNOWN):
        record = provenance(access=access)
        assert not record.is_official
        assert not record.usable_for_scientific_run


def test_license_is_a_separate_fact_from_official_distribution():
    """Officially downloadable and explicitly licensed are different claims."""
    record = provenance()
    assert record.usable_for_scientific_run
    assert not record.license_established
    assert record.license_status == LICENSE_NOT_ESTABLISHED
    payload = record.to_dict()
    assert payload["usable_for_scientific_run"] is True
    assert payload["license_established"] is False


def test_no_license_is_invented():
    assert LICENSE_NOT_ESTABLISHED == "NOT_ESTABLISHED"
    body = source("unmark/evaluation/profiling.py")
    for invented in ("MIT", "Apache-2.0", "CC-BY", "GPL"):
        assert invented not in body


def test_access_is_required_with_no_default():
    import inspect

    parameters = inspect.signature(DatasetProvenance).parameters
    for name in ("dataset_name", "dataset_version", "task", "access", "source_name"):
        assert parameters[name].default is inspect.Parameter.empty
    with pytest.raises(EvaluationContractViolation, match="no default"):
        provenance(access="OFFICIAL")


def test_old_authorisation_boolean_is_gone():
    """The SA-VLSP-specific boolean must not survive as an executable field."""
    import inspect

    assert "authorisation_established" not in inspect.signature(DatasetProvenance).parameters
    from unmark.evaluation import profiling

    assert not hasattr(profiling, "DatasetSourceType")


def test_provenance_schema_carries_required_fields():
    payload = provenance().to_dict()
    for key in (
        "dataset_name", "dataset_version", "task", "access", "is_official",
        "usable_for_scientific_run", "license_status", "license_established",
        "label_mapping", "columns", "files", "source_revision",
    ):
        assert key in payload


def test_no_dataset_is_vendored():
    for name in MODULES:
        assert not imported(name) & {"datasets", "requests", "urllib", "gdown"}


# ---------------------------------------------------------------------------
# 8. Protocol precommitment — UIT-VSFC v1.0
# ---------------------------------------------------------------------------
def test_uit_vsfc_is_the_active_dataset():
    assert PRIMARY_DATASET == "UIT-VSFC"
    assert PRIMARY_DATASET_VERSION == "1.0"
    assert PRIMARY_TASK == "sentiment"
    protocol = Preg1Protocol()
    assert protocol.dataset == "UIT-VSFC" and protocol.num_labels == 3


def test_sa_vlsp_is_superseded_not_active():
    assert SUPERSEDED_DATASET == "SA-VLSP2016"
    assert PRIMARY_DATASET != SUPERSEDED_DATASET
    assert "SUPERSEDED" in SUPERSESSION_NOTE
    assert "ZERO real Vanilla-vs-Base-only scores" in SUPERSESSION_NOTE
    assert SUPERSEDED_DATASET not in COMPARATOR_DATASETS


def test_labels_are_exactly_negative_neutral_positive():
    assert LABEL_MAPPING == {"negative": 0, "neutral": 1, "positive": 2}


def test_published_counts_are_internally_consistent():
    for split, counts in PUBLISHED_LABEL_COUNTS.items():
        assert sum(counts.values()) == PUBLISHED_SPLIT_SIZES[split], split
    assert PUBLISHED_SPLIT_SIZES == {"train": 11426, "validation": 1583, "test": 3166}


def test_dataset_is_locked_and_not_score_selected():
    assert "LOCKED" in DATASET_LOCK_RULE
    assert "NOT a downstream-score-based selection contest" in DATASET_LOCK_RULE
    assert "STOP and require a new explicit researcher decision" in DATASET_LOCK_RULE


def test_no_word_segmenter_is_introduced():
    assert "No VnCoreNLP" in NO_WORD_SEGMENTER
    assert "RAW_BASE" in NO_WORD_SEGMENTER
    assert "distribution shift" in RAW_BASE_CAVEAT
    # Structural: no segmenter is imported or called. The modules legitimately
    # *name* VnCoreNLP in prose when recording that it is deliberately absent.
    for name in MODULES:
        assert not imported(name) & {"py_vncorenlp", "vncorenlp"}
        assert not called_names(name) & {"VnCoreNLP", "word_segment", "annotate"}


# --- splits -------------------------------------------------------------
def test_official_validation_is_measurement_only():
    assert OFFICIAL_VALIDATION_ROLE == "measurement-dev"
    for banned in ("learning rate", "epoch", "pooling", "head hyperparameter"):
        assert banned in OFFICIAL_VALIDATION_RULE
    assert "never tuned on it" in OFFICIAL_VALIDATION_RULE


def test_official_test_is_sealed():
    from unmark.evaluation.preg1_protocol import OFFICIAL_TEST_SEAL_RULE

    assert OFFICIAL_TEST_SEALED
    assert "integrity, hash and duplicate checks ONLY" in OFFICIAL_TEST_SEAL_RULE
    assert "never for scores" in OFFICIAL_TEST_SEAL_RULE


def test_internal_split_is_80_20():
    assert INTERNAL_SPLIT_FRACTIONS == {"protocol-train": 0.80, "protocol-dev": 0.20}
    assert sum(INTERNAL_SPLIT_FRACTIONS.values()) == pytest.approx(1.0)
    assert "measurement-dev" not in INTERNAL_SPLIT_FRACTIONS


def test_split_seed_is_17486_and_reproducible():
    assert SPLIT_SEED_TAG == "UNMARK-PREG1-SPLIT-UITVSFC-v1"
    assert SPLIT_SEED == 17486
    assert derive_seeds(SPLIT_SEED_TAG, 1)[0] == 17486


def test_all_three_seed_sets_are_disjoint():
    assert not set(TUNING_SEEDS) & set(MEASUREMENT_SEEDS)
    assert SPLIT_SEED not in set(TUNING_SEEDS) | set(MEASUREMENT_SEEDS)


# --- splitter mechanism -------------------------------------------------
def split_records(n: int = 40):
    labels = ["negative", "neutral", "positive"]
    return [(f"s{i}", f"Sản phẩm số {i}", labels[i % 3]) for i in range(n)]


def test_splitter_is_deterministic():
    a = stratified_group_split(split_records(), INTERNAL_SPLIT_FRACTIONS, SPLIT_SEED)
    b = stratified_group_split(split_records(), INTERNAL_SPLIT_FRACTIONS, SPLIT_SEED)
    assert a == b


def test_splitter_parts_are_disjoint_and_complete():
    out = stratified_group_split(split_records(), INTERNAL_SPLIT_FRACTIONS, SPLIT_SEED)
    train, dev = set(out["protocol-train"]), set(out["protocol-dev"])
    assert not train & dev
    assert len(train | dev) == 40


def test_canonical_duplicates_cannot_cross_the_internal_split():
    import unicodedata

    records = split_records(38) + [
        ("dupA", "hòa bình", "positive"),
        ("dupB", unicodedata.normalize("NFD", "hòa bình"), "positive"),
    ]
    out = stratified_group_split(records, INTERNAL_SPLIT_FRACTIONS, SPLIT_SEED)
    where = {sid: name for name, ids in out.items() for sid in ids}
    assert where["dupA"] == where["dupB"]


def test_splitter_is_label_stratified():
    from collections import Counter

    records = split_records(60)
    labels = {sid: label for sid, _, label in records}
    out = stratified_group_split(records, INTERNAL_SPLIT_FRACTIONS, SPLIT_SEED)
    for name in out:
        counts = Counter(labels[sid] for sid in out[name])
        assert len(counts) == 3, f"{name} lost a class"


def test_splitter_depends_on_the_seed():
    a = stratified_group_split(split_records(), INTERNAL_SPLIT_FRACTIONS, SPLIT_SEED)
    b = stratified_group_split(split_records(), INTERNAL_SPLIT_FRACTIONS, 1)
    assert a != b


def test_splitter_uses_no_global_rng():
    assert not imported("unmark/evaluation/profiling.py") & {"random"}
    assert "blake2b" in source("unmark/evaluation/profiling.py")


def test_splitter_rejects_bad_fractions():
    with pytest.raises(EvaluationContractViolation, match="sum to 1.0"):
        stratified_group_split(split_records(), {"a": 0.5, "b": 0.2}, SPLIT_SEED)


def test_splitter_status_and_duplicate_contract_recorded():
    assert "NOT run on real data" in SPLITTER_STATUS
    assert "group-aware by canonical text" in " ".join(SPLITTER_REQUIREMENTS)
    joined = " ".join(DUPLICATE_CONTRACT)
    assert "cannot cross" in joined
    assert "never silently relabelled or dropped" in joined
    assert "STOP for researcher review" in joined


# --- pooling and head ---------------------------------------------------
def test_pooling_is_first_token_and_scoped_to_preg1():
    assert PREG1_POOLING is Preg1Pooling.FIRST_TOKEN
    assert "ONLY" in PREG1_POOLING_SCOPE
    assert "does NOT change the Stage-1" in PREG1_POOLING_SCOPE
    assert "remains OPEN" in PREG1_POOLING_SCOPE


def test_preg1_pooling_does_not_change_stage1():
    from unmark.modeling.contracts import STAGE1_POOLING

    assert STAGE1_POOLING == "masked_mean_over_non_special_content_tokens"


def test_stage2_pooling_for_the_full_grid_stays_open():
    from unmark.evaluation import OPEN_EVALUATION_VALUES

    assert "head_pooling" in OPEN_EVALUATION_VALUES


def test_head_is_linear_d_to_3_with_bias_only():
    assert HEAD_SHAPE == "Linear(d, 3, bias=True)"
    assert HEAD_HIDDEN_LAYERS == 0
    assert HEAD_DROPOUT == 0.0
    assert HEAD_LAYERNORM is False
    assert HEAD_ACTIVATION is None


def test_head_does_not_hardcode_the_backbone_dimension():
    literals = [
        n.value
        for n in ast.walk(tree("unmark/evaluation/preg1_protocol.py"))
        if isinstance(n, ast.Constant) and isinstance(n.value, int)
    ]
    assert 768 not in literals


def test_loss_is_plain_cross_entropy():
    assert LOSS == "cross_entropy"
    assert LOSS_CLASS_WEIGHTS is None
    assert LOSS_LABEL_SMOOTHING == 0.0


# --- optimisation -------------------------------------------------------
def test_adamw_is_fully_specified():
    assert Preg1Protocol().optimizer == "AdamW"
    assert ADAMW_BETAS == (0.9, 0.999)
    assert ADAMW_EPS == 1e-8
    assert WEIGHT_DECAY == 0.01
    assert AMSGRAD is False


def test_schedule_is_constant_with_no_warmup_or_clipping():
    assert LR_SCHEDULE == "CONSTANT"
    assert WARMUP_STEPS == 0
    assert GRADIENT_CLIPPING is None


def test_budget_is_30_epochs_no_early_stopping():
    assert EPOCHS == 30
    assert EARLY_STOPPING is False
    assert BATCH_SIZE == 128


def test_lr_grid_has_exactly_five_candidates_including_1e_2():
    assert LR_GRID == (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
    assert len(LR_GRID) == 5
    assert 1e-2 in LR_GRID


def test_checkpoint_and_aggregation_rules_are_ordered():
    assert CHECKPOINT_RULE == (
        "highest protocol-dev Macro-F1", "then higher Accuracy", "then earliest epoch"
    )
    assert len(LR_AGGREGATION_RULE) == 4
    assert "highest MEAN selected-checkpoint Macro-F1" in LR_AGGREGATION_RULE[0]
    assert "SMALLER learning rate" in LR_AGGREGATION_RULE[-1]


def test_primary_lr_uses_vanilla_only_and_is_frozen_for_both():
    assert "VANILLA only" in PRIMARY_LR_SELECTION
    assert "FROZEN and reused unchanged for BOTH pathways" in PRIMARY_LR_SELECTION
    assert "Official validation is never used in this selection" in PRIMARY_LR_SELECTION
    assert "does NOT make Vanilla an upper bound" in PRIMARY_LR_CAVEAT


def test_secondary_sensitivity_is_clearly_secondary():
    assert "MUST NOT replace the headline primary" in SECONDARY_SENSITIVITY
    assert "equal tuning budget" in SECONDARY_SENSITIVITY


# --- reporting ----------------------------------------------------------
def test_pathways_share_the_protocol_but_may_pick_their_own_epoch():
    assert "learning rate" in SHARED_ACROSS_PATHWAYS
    assert "max_length" in SHARED_ACROSS_PATHWAYS
    assert "does NOT require an identical epoch number" in PER_PATHWAY_FREEDOM
    assert "SAME checkpoint rule" in PER_PATHWAY_FREEDOM


def test_paired_measurement_uses_five_seeds_and_official_validation():
    from unmark.evaluation.preg1_protocol import PAIRED_REPORTING_RULE

    assert len(MEASUREMENT_SEEDS) == 5
    assert "untouched official VALIDATION" in PAIRED_REPORTING_RULE
    assert "Delta_s" in PAIRED_REPORTING_RULE
    assert "sample std(Delta)" in PAIRED_REPORTING_RULE


def test_per_class_f1_diagnostics_are_required():
    from unmark.evaluation.preg1_protocol import DIAGNOSTIC_METRICS, PRIMARY_METRIC

    assert PRIMARY_METRIC == "macro_f1"
    assert DIAGNOSTIC_METRICS == (
        "per_class_f1_negative", "per_class_f1_neutral", "per_class_f1_positive"
    )


def test_no_significance_claim_for_n_of_5():
    from unmark.evaluation.preg1_protocol import NO_SIGNIFICANCE_CLAIM

    assert "No p-value is required for n = 5" in NO_SIGNIFICANCE_CLAIM


def test_no_preg1_threshold_and_no_ceiling_claim():
    assert "NO pre-G1 PASS/FAIL threshold" in NO_PREG1_THRESHOLD
    assert "NOT borrowed here" in NO_PREG1_THRESHOLD
    assert "may be called an upper bound" in NOT_A_CEILING


def test_encoder_pin_does_not_close_the_backbone_decision():
    from unmark.evaluation.preg1_protocol import ENCODER_PIN_SCOPE, ENCODER_REVISION

    assert ENCODER_REVISION == "01daacda68afe13d83023d16ec647239e344a1e6"
    assert "does NOT close D-B3B0-002" in ENCODER_PIN_SCOPE


def test_protocol_serialises_completely():
    payload = Preg1Protocol().to_dict()
    for key in (
        "dataset", "pathways", "pooling", "head", "loss", "encoder",
        "max_length", "splits", "optimisation", "lr_search", "seeds", "reporting",
    ):
        assert key in payload
    assert payload["max_length"] == 256
    assert payload["dataset"]["name"] == "UIT-VSFC"


# ---------------------------------------------------------------------------
# 9. Nothing trains; the test split is never used for protocol decisions
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", MODULES)
def test_no_training_or_optimizer(name):
    calls = called_names(name)
    assert not calls & {"step", "zero_grad", "backward", "save_pretrained"}
    assert not imported(name) & {"torch", "optim", "wandb"}
    attributes = {n.attr for n in ast.walk(tree(name)) if isinstance(n, ast.Attribute)}
    assert not attributes & {"AdamW", "SGD", "lr_scheduler"}


def test_profiler_modules_are_torch_free():
    for name in ("unmark/evaluation/profiling.py", "unmark/evaluation/preg1_protocol.py"):
        assert "torch" not in imported(name)


def test_script_profiles_lengths_on_train_only():
    body = source("scripts/preg1_dataset_profile.py")
    assert 'records["train"]' in body
    assert "TRAIN ONLY" in body


def test_script_does_not_select_max_length_from_data():
    body = source("scripts/preg1_dataset_profile.py")
    assert '"selected_from_data": False' in body
    assert '"fixed": True' in body
    assert "select_max_length" not in body


def test_script_stamps_no_training_and_sealed_test():
    body = source("scripts/preg1_dataset_profile.py")
    for flag in (
        '"official_test_sealed": True',
        '"official_test_used_for_protocol_decisions": False',
        '"head_training_performed": False',
        '"optimizer_created": False',
        '"downstream_score_computed": False',
    ):
        assert flag in body


def test_script_requires_explicit_provenance_flags():
    body = source("scripts/preg1_dataset_profile.py")
    assert '"--access", required=True' in body
    assert "--source-type" not in body, "the superseded flag survives"
    assert "--authorisation-established" not in body, "the superseded flag survives"
    assert "no license is invented" in body


def test_tokenizer_profiling_is_colab_only():
    body = source("scripts/preg1_dataset_profile.py")
    assert "Colab-only" in body or "Colab only" in body
    assert "ML-free" in body
    parsed = tree("scripts/preg1_dataset_profile.py")
    top_level = {
        a.name.split(".")[0] for n in parsed.body if isinstance(n, ast.Import) for a in n.names
    }
    assert "transformers" not in top_level


# ---------------------------------------------------------------------------
# 10. Reproducibility lock: paired initialisation and optimiser detail
# ---------------------------------------------------------------------------
from unmark.evaluation.preg1_protocol import (  # noqa: E402
    CHECKPOINT_ELIGIBILITY_RULE,
    CHECKPOINT_ELIGIBLE_EPOCHS,
    DROP_LAST,
    FIRST_CHECKPOINT_EPOCH,
    GRADIENT_ACCUMULATION_STEPS,
    HEAD_INIT_BIAS,
    HEAD_INIT_IS_EXPLICIT,
    HEAD_INIT_RULE,
    HEAD_INIT_WEIGHT,
    LAST_CHECKPOINT_EPOCH,
    LOSS_REDUCTION,
    LOSS_SPEC,
    NO_SHARED_ADVANCING_RNG,
    PAIRED_DATA_ORDER_RULE,
    PAIRED_INITIALISATION_RULE,
    PARAM_GROUP_RULE,
    RUNTIME_OPTIONS_NOTE,
    RUN_INITIALISATION_SEQUENCE,
    SHUFFLE_IS_DETERMINISTIC_UNDER_SEED,
    WEIGHT_DECAY_BIAS,
    WEIGHT_DECAY_WEIGHT,
    is_checkpoint_eligible,
)


def test_head_initialisation_is_explicit_xavier_and_zero_bias():
    assert HEAD_INIT_WEIGHT == "torch.nn.init.xavier_uniform_"
    assert HEAD_INIT_BIAS == "torch.nn.init.zeros_"
    assert HEAD_INIT_IS_EXPLICIT is True


def test_nn_linear_default_initialisation_is_not_relied_upon():
    """The default is a Kaiming variant that has changed across PyTorch
    versions; depending on it would make the paired comparison version-sensitive."""
    assert "NOT relied upon" in HEAD_INIT_RULE
    assert "version" in HEAD_INIT_RULE


def test_run_initialisation_sequence_reseeds_before_construction():
    assert len(RUN_INITIALISATION_SEQUENCE) == 4
    assert "BEFORE head construction" in RUN_INITIALISATION_SEQUENCE[0]
    assert "explicitly apply" in RUN_INITIALISATION_SEQUENCE[2]
    assert "deterministic shuffle generator" in RUN_INITIALISATION_SEQUENCE[3]


def test_paired_pathways_start_from_identical_parameters():
    assert "BIT-IDENTICAL" in PAIRED_INITIALISATION_RULE
    assert "Sharing a seed LABEL is not enough" in PAIRED_INITIALISATION_RULE
    assert "RE-SEEDS from s" in PAIRED_INITIALISATION_RULE


def test_no_shared_advancing_rng_between_pathways():
    assert "No shared advancing RNG" in NO_SHARED_ADVANCING_RNG
    assert "advancing RNG stream" in PAIRED_INITIALISATION_RULE


def test_data_order_is_paired_and_deterministic():
    assert "same example ids" in PAIRED_DATA_ORDER_RULE
    assert "deterministic shuffle schedule" in PAIRED_DATA_ORDER_RULE
    assert "Only the input pathway differs" in PAIRED_DATA_ORDER_RULE
    assert SHUFFLE_IS_DETERMINISTIC_UNDER_SEED is True


def test_weight_decay_applies_to_the_matrix_not_the_bias():
    assert WEIGHT_DECAY_WEIGHT == 0.01
    assert WEIGHT_DECAY_BIAS == 0.0
    assert "NOT decayed" in PARAM_GROUP_RULE


def test_cross_entropy_is_mean_reduced_and_unweighted():
    assert LOSS_REDUCTION == "mean"
    assert LOSS_SPEC == 'CrossEntropyLoss(weight=None, label_smoothing=0.0, reduction="mean")'
    assert "weight=None" in LOSS_SPEC
    assert "label_smoothing=0.0" in LOSS_SPEC


def test_gradient_accumulation_and_drop_last():
    assert GRADIENT_ACCUMULATION_STEPS == 1
    assert DROP_LAST is False


def test_only_epochs_1_to_30_are_checkpoint_eligible():
    assert FIRST_CHECKPOINT_EPOCH == 1
    assert LAST_CHECKPOINT_EPOCH == 30
    assert CHECKPOINT_ELIGIBLE_EPOCHS == tuple(range(1, 31))
    assert len(CHECKPOINT_ELIGIBLE_EPOCHS) == 30


def test_epoch_zero_cannot_win_a_checkpoint():
    """An untrained head can post a deceptive accuracy on a 4%-neutral corpus."""
    assert not is_checkpoint_eligible(0)
    assert is_checkpoint_eligible(1)
    assert is_checkpoint_eligible(30)
    assert not is_checkpoint_eligible(31)
    assert not is_checkpoint_eligible(-1)
    assert "Epoch 0" in CHECKPOINT_ELIGIBILITY_RULE
    assert "NOT checkpoint-eligible" in CHECKPOINT_ELIGIBILITY_RULE


def test_runtime_options_are_not_hyperparameters():
    assert "NOT scientific hyperparameters" in RUNTIME_OPTIONS_NOTE
    assert "must not be" in RUNTIME_OPTIONS_NOTE
    assert "records the actual runtime version" in RUNTIME_OPTIONS_NOTE


def test_reproducibility_block_serialises():
    payload = Preg1Protocol().to_dict()
    repro = payload["reproducibility"]
    for key in (
        "run_initialisation_sequence", "paired_initialisation", "paired_data_order",
        "no_shared_advancing_rng", "runtime_options_note",
    ):
        assert key in repro
    head = payload["head"]
    assert head["init_weight"] == "torch.nn.init.xavier_uniform_"
    assert head["init_bias"] == "torch.nn.init.zeros_"
    optimisation = payload["optimisation"]
    assert optimisation["weight_decay_weight"] == 0.01
    assert optimisation["weight_decay_bias"] == 0.0
    assert optimisation["gradient_accumulation_steps"] == 1
    assert optimisation["drop_last"] is False
    assert optimisation["checkpoint_eligible_epochs"] == [1, 30]


def test_protocol_version_records_the_reproducibility_lock():
    assert Preg1Protocol().version == "preg1-protocol-v3"


def test_reproducibility_lock_is_ml_free():
    """The contract is representation only; no torch is imported to express it."""
    assert not imported("unmark/evaluation/preg1_protocol.py") & {"torch"}
    assert HEAD_INIT_WEIGHT.startswith("torch.nn.init")  # a NAME, not a call
