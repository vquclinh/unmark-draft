"""Stage-1 device ownership -- runtime half (Audit 030 §Y).

Executes the contract its torch-free companion `test_stage1_device_contract.py`
asserts structurally. The cross-device tests need a CUDA host and skip elsewhere;
they are the authoritative evidence for the fourth-smoke failure and run in Colab.

No PhoBERT: a tiny recording encoder stands in, and it mirrors the production
interface exactly by **not** moving its inputs.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

REPO = pathlib.Path(__file__).resolve().parents[1]

torch = pytest.importorskip("torch", reason="the runtime half needs torch")

from unmark.stage1.data import batch_to_device, module_device  # noqa: E402
from unmark.stage1.protocol import VALIDATION_CONDITIONS  # noqa: E402
from unmark.stage1.validation import evaluate  # noqa: E402


class RecordingCore(torch.nn.Module):
    """Frozen-encoder stand-in that records the device of every tensor it sees."""

    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.config = type("Config", (), {"hidden_size": dim})()
        self.embedding = torch.nn.Embedding(64, dim)
        self.seen: list[tuple[str, torch.device]] = []
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def forward(self, ids, mask=None):
        self.seen.append(("input_ids", ids.device))
        if mask is not None:
            self.seen.append(("attention_mask", mask.device))
        return self.embedding(ids)


class RecordingUnmarkEncoder(torch.nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.encoder = RecordingCore(dim)
        self.adapter = torch.nn.Linear(dim, dim)


class RecordingObjective(torch.nn.Module):
    """Mirrors the production interface exactly: it does NOT move its inputs."""

    def __init__(self, unmark_encoder, weights=None) -> None:
        super().__init__()
        self.unmark_encoder = unmark_encoder
        self.weights = weights

    def reference_representation(self, ids, mask, special):
        return self.unmark_encoder.encoder(ids.long(), mask).mean(dim=1)

    def adapted_representation(self, ids, mask, special, *channels):
        pooled = self.unmark_encoder.encoder(ids.long(), mask).mean(dim=1)
        return self.unmark_encoder.adapter(pooled)


def tiny_prepared():
    """Real prepared examples through the real preparation path."""
    sys.path.insert(0, str(REPO / "tests"))
    from test_stage1_validation_preparation import StubTokenizer

    from unmark.linguistics import load_inventory, make_classifier
    from unmark.stage1.execute import TRUNCATION
    from unmark.stage1.validation import HeldOutExample, prepare_condition_batch

    classifier = make_classifier(load_inventory())
    text = " ".join(["tiếng", "việt", "không", "dấu"] * 3)
    held = [HeldOutExample(f"doc-{i:04d}#0", text) for i in range(4)]
    return {
        condition: prepare_condition_batch(
            held, StubTokenizer(), condition, truncation=TRUNCATION, classifier=classifier
        )
        for condition in VALIDATION_CONDITIONS
    }


provisioned = pytest.mark.skipif(
    not (REPO / ".resources-cache/vietnamese-syllables/all-vietnamese-syllables.txt").is_file(),
    reason="the git-ignored inventory cache is not provisioned in this runtime",
)
needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the cross-device half needs a CUDA host"
)


# -- the two helpers ---------------------------------------------------------
def test_module_device_reads_the_parameters():
    model = RecordingUnmarkEncoder()
    assert module_device(model) == next(model.parameters()).device


def test_module_device_falls_back_to_cpu_without_parameters():
    assert module_device(torch.nn.Module()).type == "cpu"


def test_batch_to_device_moves_tensors_and_passes_non_tensors_through():
    batch = {
        "base_input_ids": torch.ones(2, 3, dtype=torch.long),
        "base_attention_mask": torch.ones(2, 3, dtype=torch.bool),
        "sample_ids": ["a", "b"],
        "corruption_rates": [0.5, 0.5],
    }
    moved = batch_to_device(batch, torch.device("cpu"))
    assert moved["sample_ids"] == ["a", "b"]
    assert moved["corruption_rates"] == [0.5, 0.5]
    assert moved["base_input_ids"].device.type == "cpu"
    assert moved["base_attention_mask"].dtype is torch.bool, "dtype must survive"


# -- the contract, end to end through the real evaluator ---------------------
@provisioned
def test_cpu_objective_with_cpu_batch_works():
    objective = RecordingObjective(RecordingUnmarkEncoder())
    point = evaluate(objective, tiny_prepared(), pad_token_id=1, batch_size=2)
    assert sorted(point.distances) == sorted(VALIDATION_CONDITIONS)
    assert objective.unmark_encoder.encoder.seen, "the encoder must have run"


@provisioned
def test_every_tensor_the_encoder_sees_is_on_the_models_device():
    """Covers input_ids AND attention_mask, on both the reference and adapted paths."""
    objective = RecordingObjective(RecordingUnmarkEncoder())
    evaluate(objective, tiny_prepared(), pad_token_id=1, batch_size=2)
    expected = module_device(objective)
    seen = objective.unmark_encoder.encoder.seen
    assert {name for name, _ in seen} == {"input_ids", "attention_mask"}
    for name, device in seen:
        assert device == expected, f"{name} arrived on {device}, model is on {expected}"


@provisioned
@needs_cuda
def test_a_cuda_objective_receives_cuda_tensors_from_a_cpu_prepared_batch():
    """The fourth-smoke failure, inverted into a passing contract.

    `prepare_condition_batch` produces CPU tensors; the objective is on CUDA.
    Without the shared boundary this raises
    "Expected all tensors to be on the same device".
    """
    objective = RecordingObjective(RecordingUnmarkEncoder()).to("cuda")
    point = evaluate(objective, tiny_prepared(), pad_token_id=1, batch_size=2)

    assert sorted(point.distances) == sorted(VALIDATION_CONDITIONS)
    seen = objective.unmark_encoder.encoder.seen
    assert seen, "the encoder must have run"
    for name, device in seen:
        assert device.type == "cuda", f"{name} reached a CUDA model on {device}"


@provisioned
@needs_cuda
def test_removing_the_transfer_really_would_fail():
    """Proves the test above is not vacuous: the mismatch is a real error here."""
    objective = RecordingObjective(RecordingUnmarkEncoder()).to("cuda")
    prepared = tiny_prepared()
    from unmark.stage1.data import collate_stage1_batch

    cpu_batch = collate_stage1_batch(list(prepared["FULL"])[:2], 1)
    with pytest.raises(RuntimeError, match="same device"):
        objective.reference_representation(
            cpu_batch["reference_input_ids"],
            cpu_batch["reference_attention_mask"],
            cpu_batch["reference_special_tokens_mask"],
        )
