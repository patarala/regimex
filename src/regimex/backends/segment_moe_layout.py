"""Segment MoE dimensions shared by feature extraction, the network and discovery."""

NEIGHBOR_RADIUS = 4
CONTEXT_LENGTH = 2 * NEIGHBOR_RADIUS + 1
SEGMENT_FEATURE_COUNT = 4
RETURN_LAGS = (1, 4, 16, 32)
BLOCK_FEATURE_COUNT = 7 * len(RETURN_LAGS)

ACCUMULATION_INPUTS = 6
ACCUMULATION_WIDTH = 16
GATE_WIDTH = 24
CLASS_COUNT = 3


def trainable_parameter_count(context_width: int) -> int:
    def linear(inputs: int, outputs: int) -> int:
        return (inputs + 1) * outputs

    # Two scalar thresholds and a gain for each accompany the correction layers.
    accumulation = (
        4 + linear(ACCUMULATION_INPUTS, ACCUMULATION_WIDTH) + linear(ACCUMULATION_WIDTH, 1)
    )
    context = linear(CONTEXT_LENGTH * SEGMENT_FEATURE_COUNT, context_width) + linear(
        context_width, CLASS_COUNT
    )
    gate = linear(BLOCK_FEATURE_COUNT, GATE_WIDTH) + linear(GATE_WIDTH, 1)
    return accumulation + context + gate
