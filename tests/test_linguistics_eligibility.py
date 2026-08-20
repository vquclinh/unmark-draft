"""Offline tests for B3A Vietnamese syllable eligibility.

The whole suite runs without network. Ordinary unit tests use the small
committed fixture in `tests/fixtures/`; the real pinned inventory is exercised
only by tests that skip when the git-ignored cache is absent.

The fixture was written by hand from well-known Vietnamese syllables rather than
copied from the upstream list, so nothing unlicensed is redistributed here.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import unicodedata
from pathlib import Path

import pytest

from unmark.linguistics import (
    ELIGIBILITY_SCHEMA_VERSION,
    InventoryChecksumMismatch,
    InventoryUnavailable,
    build_inventory,
    classify_candidate,
    clear_inventory_cache,
    is_vietnamese_candidate,
    load_inventory,
    load_manifest,
    make_classifier,
    membership_form,
    try_load_inventory,
)
from unmark.orthography import Eligibility, canon, decompose, recompose

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "configs" / "linguistics" / "vietnamese_syllables.yaml"
FIXTURE = Path(__file__).parent / "fixtures" / "vietnamese_syllables_sample.txt"


@pytest.fixture(scope="module")
def fixture_inventory():
    return build_inventory(FIXTURE.read_text(encoding="utf-8").splitlines())


def _real_inventory_or_skip():
    inventory = try_load_inventory(MANIFEST, REPO_ROOT)
    if inventory is None:
        pytest.skip("pinned inventory not fetched; run scripts/fetch_vietnamese_syllable_inventory.py")
    return inventory


# ---------------------------------------------------------------------------
# 1-3. Provenance is pinned
# ---------------------------------------------------------------------------
def test_manifest_records_full_provenance():
    provenance = load_manifest(MANIFEST)
    assert provenance.source_author == "hieuthi"
    assert provenance.source_name == "all-vietnamese-syllables.txt"
    assert provenance.source_url.startswith("https://gist.github.com/")


def test_exact_revision_is_pinned():
    provenance = load_manifest(MANIFEST)
    assert len(provenance.source_revision) == 40
    assert provenance.source_revision == "135a4d9716e49a981624474156d6f247b9b46f6a"
    assert provenance.source_revision in provenance.raw_url, "raw_url must pin the revision, not 'raw/'"


def test_sha256_is_recorded():
    provenance = load_manifest(MANIFEST)
    assert len(provenance.sha256) == 64
    assert provenance.sha256 == "78eeb840d50455b14bd564da5aed7318d96468b8deaad5986b77bf5c538315d2"
    assert provenance.expected_entry_count == 17974


def test_manifest_contains_no_absolute_local_path():
    text = MANIFEST.read_text(encoding="utf-8")
    assert "/home/" not in text and "/mnt/" not in text and "C:\\" not in text
    assert load_manifest(MANIFEST).cache_relative_path.startswith(".resources-cache/")


def test_eligibility_schema_version_is_recorded_and_distinct():
    from unmark.corruption import CORRUPTION_SCHEMA_VERSION

    assert ELIGIBILITY_SCHEMA_VERSION == "vn-syllables-v1"
    assert load_manifest(MANIFEST).schema_version == ELIGIBILITY_SCHEMA_VERSION
    assert ELIGIBILITY_SCHEMA_VERSION != CORRUPTION_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# 4-8. Fail-closed, repo-local, no global cache
# ---------------------------------------------------------------------------
def test_checksum_mismatch_fails_closed(tmp_path):
    manifest = tmp_path / "m.yaml"
    cache = tmp_path / "cache" / "list.txt"
    cache.parent.mkdir(parents=True)
    cache.write_text("ba\nban\n", encoding="utf-8")
    manifest.write_text(
        "inventory_schema_version: t\nsource_name: t\nsource_author: t\nsource_url: t\n"
        "source_revision: 0123456789012345678901234567890123456789\nraw_url: t\n"
        f"sha256: {'0' * 64}\nretrieved_at: t\ncanonicalization_mode: t\nmembership_form: t\n"
        "license_status: t\nexpected_entry_count: 2\ncache_relative_path: cache/list.txt\n",
        encoding="utf-8",
    )
    clear_inventory_cache()
    with pytest.raises(InventoryChecksumMismatch) as excinfo:
        load_inventory(manifest, tmp_path)
    assert "checksum mismatch" in str(excinfo.value)
    assert "scientific spec change" in str(excinfo.value)


def test_missing_inventory_fails_with_a_clear_message(tmp_path):
    manifest = tmp_path / "m.yaml"
    manifest.write_text(
        "inventory_schema_version: t\nsource_name: t\nsource_author: t\nsource_url: t\n"
        f"source_revision: r\nraw_url: t\nsha256: {'0' * 64}\nretrieved_at: t\n"
        "canonicalization_mode: t\nmembership_form: t\nlicense_status: t\n"
        "expected_entry_count: 0\ncache_relative_path: cache/absent.txt\n",
        encoding="utf-8",
    )
    clear_inventory_cache()
    with pytest.raises(InventoryUnavailable) as excinfo:
        load_inventory(manifest, tmp_path)
    message = str(excinfo.value)
    assert "NOT committed" in message
    assert "no license statement" in message
    assert "fetch_vietnamese_syllable_inventory" in message


def test_entry_count_change_is_rejected(tmp_path):
    manifest = tmp_path / "m.yaml"
    cache = tmp_path / "c.txt"
    cache.write_bytes(b"ba\nban\n")
    digest = hashlib.sha256(cache.read_bytes()).hexdigest()
    manifest.write_text(
        "inventory_schema_version: t\nsource_name: t\nsource_author: t\nsource_url: t\n"
        f"source_revision: r\nraw_url: t\nsha256: {digest}\nretrieved_at: t\n"
        "canonicalization_mode: t\nmembership_form: t\nlicense_status: t\n"
        "expected_entry_count: 99\ncache_relative_path: c.txt\n",
        encoding="utf-8",
    )
    clear_inventory_cache()
    with pytest.raises(InventoryUnavailable, match="entry count"):
        load_inventory(manifest, tmp_path)


def test_cache_location_is_repo_local_and_gitignored():
    provenance = load_manifest(MANIFEST)
    path = REPO_ROOT / provenance.cache_relative_path
    assert REPO_ROOT in path.parents
    result = subprocess.run(
        ["git", "check-ignore", "-q", provenance.cache_relative_path],
        cwd=REPO_ROOT,
        capture_output=True,
    )
    assert result.returncode == 0, "the resource cache must be git-ignored"


def test_no_global_user_cache_is_used():
    source = (REPO_ROOT / "unmark" / "linguistics" / "inventory.py").read_text(encoding="utf-8")
    fetch = (REPO_ROOT / "scripts" / "fetch_vietnamese_syllable_inventory.py").read_text(encoding="utf-8")
    for forbidden in ("~/.cache", "expanduser", "XDG_CACHE", "tempfile.gettempdir", "/tmp"):
        assert forbidden not in source, f"inventory.py references {forbidden}"
        assert forbidden not in fetch, f"fetch script references {forbidden}"


def test_raw_upstream_list_is_not_committed():
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True
    ).stdout.splitlines()
    assert not any("all-vietnamese-syllables" in path for path in tracked)


# ---------------------------------------------------------------------------
# 9-12. Deterministic construction
# ---------------------------------------------------------------------------
def test_inventory_construction_is_deterministic(fixture_inventory):
    again = build_inventory(FIXTURE.read_text(encoding="utf-8").splitlines())
    assert again.forms == fixture_inventory.forms
    assert again.raw_entry_count == fixture_inventory.raw_entry_count


def test_construction_is_order_independent():
    lines = FIXTURE.read_text(encoding="utf-8").splitlines()
    assert build_inventory(lines).forms == build_inventory(list(reversed(lines))).forms


def test_duplicate_stripped_forms_collapse_deterministically(fixture_inventory):
    # ma má mà mả mã mạ -> one form
    assert membership_form("má") == membership_form("mạ") == "ma"
    assert fixture_inventory.collisions_after_stripping > 0
    assert fixture_inventory.unique_stripped_form_count < fixture_inventory.unique_canonical_entry_count


def test_case_policy_is_deterministic_casefold():
    assert membership_form("BAN") == membership_form("Ban") == membership_form("ban") == "ban"
    assert membership_form("ĐƯỜNG") == membership_form("đường") == "duong"


def test_nfc_and_nfd_source_forms_produce_the_same_membership_form():
    for text in ["má", "đường", "Tiếng", "hoà"]:
        assert membership_form(unicodedata.normalize("NFC", text)) == membership_form(
            unicodedata.normalize("NFD", text)
        )


def test_membership_form_is_independent_of_diacritics():
    """The property the whole design rests on: a pure function of the stripped form."""
    for clean, stripped in [("học", "hoc"), ("đường", "duong"), ("nghiên", "nghien"), ("hoà", "hoa")]:
        assert membership_form(clean) == membership_form(stripped) == stripped


# ---------------------------------------------------------------------------
# Eligibility classification
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("syllable", ["ba", "ban", "bàn", "bạn", "má", "mà", "mả", "mã", "mạ", "ma"])
def test_all_six_tones_of_an_inventory_syllable_are_eligible(syllable, fixture_inventory):
    assert classify_candidate(syllable, fixture_inventory) is Eligibility.VIETNAMESE_CANDIDATE


@pytest.mark.parametrize("syllable", ["đi", "đó", "đường", "ăn", "cân", "ơn", "ưu", "êm", "hoà", "học"])
def test_letter_diacritic_syllables_are_eligible(syllable, fixture_inventory):
    """Covers đ, ă, â, ê, ô, ơ, ư -- every Vietnamese letter diacritic."""
    assert classify_candidate(syllable, fixture_inventory) is Eligibility.VIETNAMESE_CANDIDATE


@pytest.mark.parametrize("form", ["BAN", "Ban", "BÀN", "ĐƯỜNG", "Đường"])
def test_uppercase_is_eligible(form, fixture_inventory):
    assert classify_candidate(form, fixture_inventory) is Eligibility.VIETNAMESE_CANDIDATE


def test_nfc_and_nfd_classify_identically(fixture_inventory):
    for text in ["bàn", "đường", "học"]:
        assert classify_candidate(unicodedata.normalize("NFC", text), fixture_inventory) is classify_candidate(
            unicodedata.normalize("NFD", text), fixture_inventory
        )


def test_stripped_forms_of_valid_syllables_are_eligible(fixture_inventory):
    for stripped in ["ban", "ma", "duong", "hoc", "hoa"]:
        assert classify_candidate(stripped, fixture_inventory) is Eligibility.VIETNAMESE_CANDIDATE


@pytest.mark.parametrize("text", ["machine", "learning", "javascript", "qwerty", "zzzz", "strength"])
def test_long_latin_strings_are_not_applicable(text, fixture_inventory):
    assert classify_candidate(text, fixture_inventory) is Eligibility.NOT_APPLICABLE


@pytest.mark.parametrize("text", ["123", "!!!", "  ", "", "😄", "42%", "a@b.com", "https://x.vn"])
def test_non_alphabetic_input_is_not_applicable(text, fixture_inventory):
    assert classify_candidate(text, fixture_inventory) is Eligibility.NOT_APPLICABLE


def test_ambiguous_ascii_resolves_towards_vietnamese(fixture_inventory):
    """Proposal 4.3's known and deliberate error mode, asserted not hidden."""
    for text in ["ban", "the", "an", "co", "ta", "va", "em"]:
        assert classify_candidate(text, fixture_inventory) is Eligibility.VIETNAMESE_CANDIDATE


def test_no_inventory_means_undecided_not_not_applicable():
    """UNDECIDED must mean 'cannot resolve', never 'resolved as non-Vietnamese'."""
    assert classify_candidate("ban", None) is Eligibility.UNDECIDED
    assert classify_candidate("machine", None) is Eligibility.UNDECIDED


def test_is_vietnamese_candidate_helper(fixture_inventory):
    assert is_vietnamese_candidate("bàn", fixture_inventory)
    assert not is_vietnamese_candidate("machine", fixture_inventory)


def test_classifier_uses_no_context_frequency_or_language_detection():
    source = (REPO_ROOT / "unmark" / "linguistics" / "classify.py").read_text(encoding="utf-8")
    for forbidden in ("langdetect", "frequency", "stopword", "ENGLISH_WORDS", "context", "neighbour"):
        assert forbidden not in source.split('"""')[2], f"classify.py body references {forbidden}"


# ---------------------------------------------------------------------------
# B1A integration
# ---------------------------------------------------------------------------
def test_decompose_without_a_classifier_stays_undecided():
    assert all(s.eligibility is Eligibility.UNDECIDED for s in decompose("Tôi đang học").syllables)


def test_decompose_with_a_classifier_resolves(fixture_inventory):
    classifier = make_classifier(fixture_inventory)
    spans = decompose("hoa ban machine", eligibility_classifier=classifier).syllables
    assert [s.eligibility for s in spans] == [
        Eligibility.VIETNAMESE_CANDIDATE,
        Eligibility.VIETNAMESE_CANDIDATE,
        Eligibility.NOT_APPLICABLE,
    ]


@pytest.mark.parametrize(
    "text", ["Tôi đang học", "đường ăn", "café ngon", "a@b.com 2026 😄", "", "   ", "hòa"]
)
def test_round_trip_is_unaffected_by_eligibility(text, fixture_inventory):
    classifier = make_classifier(fixture_inventory)
    assert recompose(decompose(text, eligibility_classifier=classifier)) == canon(text)
    assert recompose(decompose(text)) == canon(text)


def test_eligibility_does_not_change_the_base_or_tone_channels(fixture_inventory):
    classifier = make_classifier(fixture_inventory)
    for text in ["Tôi đang học", "đường ăn cân"]:
        plain = decompose(text)
        resolved = decompose(text, eligibility_classifier=classifier)
        assert plain.base_text == resolved.base_text
        assert plain.letter_channel == resolved.letter_channel
        assert plain.observed_tone_channel == resolved.observed_tone_channel


# ---------------------------------------------------------------------------
# Real pinned inventory (skipped when the cache is absent)
# ---------------------------------------------------------------------------
def test_real_inventory_matches_the_pinned_shape():
    inventory = _real_inventory_or_skip()
    assert inventory.raw_entry_count == 17974
    assert inventory.unique_canonical_entry_count == 17954
    assert inventory.unique_stripped_form_count == 2489
    assert inventory.collisions_after_stripping == 15465


def test_real_inventory_accepts_known_vietnamese_and_rejects_known_foreign():
    inventory = _real_inventory_or_skip()
    for text in ["tôi", "đang", "nghiên", "cứu", "học", "đường", "phở", "người", "được", "nguyễn"]:
        assert classify_candidate(text, inventory) is Eligibility.VIETNAMESE_CANDIDATE, text
    for text in ["machine", "learning", "python", "pytorch", "café", "google", "server", "email"]:
        assert classify_candidate(text, inventory) is Eligibility.NOT_APPLICABLE, text


def test_real_inventory_ambiguous_ascii_examples():
    inventory = _real_inventory_or_skip()
    for text in ["ban", "the", "com", "on", "in", "an", "la", "co", "nam", "ma"]:
        assert classify_candidate(text, inventory) is Eligibility.VIETNAMESE_CANDIDATE, text


def test_fetch_script_verify_only_succeeds_when_cached():
    _real_inventory_or_skip()
    result = subprocess.run(
        [sys.executable, "scripts/fetch_vietnamese_syllable_inventory.py", "--verify-only"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"},
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "verified" in result.stdout.lower()


def test_full_test_suite_needs_no_network():
    """The fetch script is the only module allowed to open a socket."""
    import ast

    for package in ("unmark/linguistics", "unmark/corruption", "unmark/orthography"):
        for path in (REPO_ROOT / package).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        assert alias.name.split(".")[0] not in ("urllib", "requests", "socket", "http"), (
                            f"{path.name} imports {alias.name}"
                        )
                elif isinstance(node, ast.ImportFrom) and node.module:
                    assert node.module.split(".")[0] not in ("urllib", "requests", "socket", "http"), (
                        f"{path.name} imports {node.module}"
                    )
