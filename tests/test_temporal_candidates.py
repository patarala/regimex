"""Keep padded patches and neighboring instruments out of Transformer predictions."""

import importlib
from pathlib import Path

import pytest


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
