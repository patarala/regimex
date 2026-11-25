"""Keep padded patches and neighboring instruments out of Transformer predictions."""

import importlib
from pathlib import Path

import pytest


def test_paired_comparison_requires_labeled_steps_in_every_draw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("torch")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    comparison = importlib.import_module("compare_temporal_architectures")
    monkeypatch.setattr(comparison, "DRAWS", 100)
    counts = np.zeros((2, 20), dtype=np.int64)
    counts[1, 13] = 12  # All labeled sideways steps were missed in this experiment.
    assert comparison.summary(counts[:1])["point_macro_f1"] is None
    assert comparison.summary(counts)["point_macro_f1"] == 0
    cohort = (["unlabeled", "missed"], np.array([256, 256]), counts)
    paired = comparison.paired(cohort, cohort)
    assert 0 < paired["point_macro_f1"]["defined_draws"] < 100
    assert paired["point_macro_f1"]["interval_95pct"] is None


def test_patch_predictions_preserve_partial_patches_and_sequence_isolation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    candidates = importlib.import_module("temporal_candidates")
    with torch.random.fork_rng(devices=[]), torch.inference_mode():
        torch.manual_seed(23)
        net = candidates.PatchEncoder(6, 32, dropout=0.15).eval()
        observations = torch.randn(3, 25, 6)
        lengths = torch.tensor([1, 13, 25])
        together = net(observations, lengths)
        assert together.shape == (3, 25, 3)
        for index, steps in enumerate(lengths.tolist()):
            expected = net(observations[index : index + 1, :steps], lengths[index : index + 1])
            torch.testing.assert_close(together[index, :steps], expected[0], rtol=1e-5, atol=2e-6)
            assert not together[index, steps:].count_nonzero()
