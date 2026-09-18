import json
import pathlib
import py_compile

ROOT = pathlib.Path(__file__).resolve().parents[1]

FORBIDDEN = (
    "UNMARK-A",
    "V2-SCF",
    "R4-W",
    "R6-U",
    "R6-W",
    "D1",
    "D2",
    "D3",
    "D4",
    "COMP-D1",
    "COMP-D2",
    "SYS1",
    "SYS2-1",
    "SYS2-2",
    "OPT1",
    "OPT2",
    "OPT3",
    "preg1",
    "/content/drive",
    "MyDrive",
    "wandb",
)


def allowed(path: pathlib.Path) -> bool:
    rel = path.relative_to(ROOT).as_posix()
    return (
        rel.startswith("artifacts/historical/")
        or rel == "artifacts/provenance/historical_aliases.json"
        or rel == "docs/repository-migration-manifest.json"
        or rel == "docs/provenance.md"
        or rel == ".gitignore"
        or rel == "tests/test_public_surface.py"
    )


def test_public_surface_has_no_research_identifiers_outside_provenance():
    hits = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix in {".pyc"} or allowed(path):
            continue
        text = path.read_text(encoding="utf-8")
        for token in FORBIDDEN:
            if token in text:
                hits.append((path.relative_to(ROOT).as_posix(), token))
    assert hits == []


def test_no_parent_repo_imports_or_symlinks():
    for path in list((ROOT / "src").rglob("*.py")) + list((ROOT / "scripts").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert "from unmark" not in text
        assert "import unmark" not in text
    assert not [p for p in ROOT.rglob("*") if p.is_symlink()]


def test_public_scripts_compile_and_notebook_is_valid_json():
    for path in list((ROOT / "src").rglob("*.py")) + list((ROOT / "scripts").rglob("*.py")):
        py_compile.compile(str(path), doraise=True)
    payload = json.loads((ROOT / "notebooks/ViUnMark_Reproduction.ipynb").read_text(encoding="utf-8"))
    assert payload["nbformat"] == 4


def test_notebook_path_hygiene_and_public_orchestration():
    notebook = (ROOT / "notebooks/ViUnMark_Reproduction.ipynb").read_text(encoding="utf-8")
    assert "/content/drive" not in notebook
    assert "MyDrive" not in notebook
    assert "DATA_ROOT" in notebook
    assert "ASSET_ROOT" in notebook
    assert "OUTPUT_ROOT" in notebook
    assert "scripts/verify_assets.py" in notebook
    assert "scripts/reproduce_uit_vsfc.py" in notebook
