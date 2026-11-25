import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")


def test_slope_survives_an_intercept_change_and_missing_observations() -> None:
    from regimex.backends.segment_moe_features import _piece_evidence

    values = np.linspace(0, 0.03, 65)
    values[29:] += 0.4
    values[[4, 11, 48]] = np.nan
    movement, fit, simple_movement, simple_fit = _piece_evidence(values)
    assert movement == pytest.approx(3.0)
    assert fit == pytest.approx(1.0)
    assert simple_movement > 3 and simple_fit < fit
    assert _piece_evidence(np.array([np.nan, 0.0])) == (0, 0, 0, 0)


def test_geometry_preserves_original_steps_and_resets_at_unavailable_blocks() -> None:
    from test_segmentation import experiment

    from regimex.backends.segment_moe_features import segment_features
    from regimex.directions import AnalysisConfig

    prices = (100 * np.exp(0.001 * np.arange(96))).tolist()
    config = AnalysisConfig(version="2.2")
    rows = experiment(prices + [None] * 40 + prices[::-1]).observations
    spans, context, block = segment_features(rows, config)
    assert spans.tolist() == [[0, 96], [136, 232]]
    assert np.count_nonzero(context[:, [0, 1, 2, 3, 5, 6, 7, 8]]) == 0
    changed = experiment(prices + [None] * 40 + [110.0] * 96).observations
    _, changed_context, changed_block = segment_features(changed, config)
    np.testing.assert_array_equal(context[0], changed_context[0])
    np.testing.assert_array_equal(block[0], changed_block[0])
    np.testing.assert_allclose(block[0], block[1], atol=1e-6)
    blank_spans, blank_context, blank_block = segment_features(
        experiment([None] * 64).observations, config
    )
    assert (blank_spans.shape, blank_context.shape, blank_block.shape) == (
        (0, 2),
        (0, 9, 4),
        (0, 28),
    )


def test_accumulation_joins_consecutive_evidence_and_reflects_direction() -> None:
    from regimex.backends.segment_moe_network import AccumulationExpert, BlockGate

    net = AccumulationExpert()
    data = torch.zeros(3, 9, 4)
    data[:, 4] = torch.tensor([0.4, 1.0, 0.4, 1.0])
    data[1:, 6] = torch.tensor([1.5, 1.0, 1.5, 1.0])
    data[2, 5] = torch.tensor([0.4, 1.0, 0.4, 1.0])
    with torch.inference_mode():
        probabilities = net(data)
        negative = data.clone()
        negative[:, :, [0, 2]] *= -1
        torch.testing.assert_close(net(negative), probabilities[:, [0, 2, 1]], rtol=0, atol=0)
        torch.testing.assert_close(probabilities[0], probabilities[1], rtol=0, atol=0)
        assert probabilities[2, 1] > probabilities[0, 1]
        torch.testing.assert_close(probabilities.sum(-1), torch.ones(3))
        assert BlockGate()(torch.ones(1, 28)).shape == (1, 1)
