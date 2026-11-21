"""Inspect actual training observations with a pointer, keyboard, and touchscreen."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from test_offline_report_browser import browser as browser
from test_theme_browser import assert_contrast, rendered_colors
from test_training_workspace_browser import ORIGIN, RUN_ID, TrainingAPI


@contextmanager
def metric_page(browser: Any, history: dict[str, Any], **options: Any) -> Iterator[Any]:
    api = TrainingAPI()
    api.runs.append(api.run)
    api.metrics["history"] = history
    page = browser.new_page(locale="en-US", **options)
    page.route(ORIGIN + "/**", api.dispatch)
    try:
        page.goto(f"{ORIGIN}/training/{RUN_ID}")
        yield page
    finally:
        page.close()


def chart_figure(page: Any, label: str) -> Any:
    return page.locator("#training-charts figure").filter(has=page.get_by_role("img", name=label))


def plot_position(figure: Any, fraction: float) -> tuple[float, float]:
    graph = figure.locator('svg[role="img"]')
    graph.scroll_into_view_if_needed()
    x, y = graph.evaluate(
        """(svg, fraction) => {
          const lines = [...svg.querySelectorAll('.training-grid')]
            .map(line => line.getBoundingClientRect());
          const rows = lines.map(line => line.top);
          const top = Math.min(...rows), bottom = Math.max(...rows);
          return [lines[0].left + fraction * lines[0].width, bottom - (bottom - top) * 0.07];
        }""",
        fraction,
    )
    return x, y


def expect_readout(figure: Any, observation: str, metric: str, value: str) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    tip = figure.locator(".training-chart-tooltip")
    expect(tip).to_be_visible()
    expect(tip.locator(":scope > *")).to_have_text([value, observation])
    description = figure.get_by_role("slider").get_attribute("aria-valuetext")
    assert description and observation in description and f"{metric}: {value}" in description


def test_hover_snaps_to_measured_updates_and_marks_first_best_and_last(browser: Any) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    history = {
        "updates": [
            {"update": step, "training_loss": loss, "validation_segment_macro_f1": score}
            for step, loss, score in (
                (1, 0.95, 0.2),
                (8, 0.83, 0.8),
                (21, 0.72, None),
                (75, 0.65, 0.8),
                (100, 0.54, 0.45),
            )
        ]
    }
    with metric_page(browser, history) as page:
        loss = chart_figure(page, "Training loss by update")
        score = chart_figure(page, "Validation interval macro-F1 by update")
        expect(loss.locator("circle.training-point")).to_have_count(1)
        expect(loss.locator("[data-last] title")).to_have_text("Update 100: 0.54")
        expect(score.locator("circle.training-point")).to_have_count(2)
        expect(score.locator("[data-best] title")).to_have_text("Update 8: 0.8")
        expect(score.locator("[data-last] title")).to_have_text("Update 100: 0.45")

        # At update 21 loss is recorded, but the nearest measured validation score is at 8.
        page.mouse.move(*plot_position(loss, 20 / 99))
        expect_readout(loss, "Update 21", "Training loss", "0.72")
        assert "Validation" not in loss.get_by_role("slider").get_attribute("aria-valuetext")
        page.mouse.move(*plot_position(score, 20 / 99))
        expect_readout(score, "Update 8", "Validation interval macro-F1", "0.8")
        description = score.get_by_role("slider").get_attribute("aria-valuetext")
        assert "Training loss: 0.83" in description and "Best validation score" in description
        expected_x, _ = plot_position(score, 7 / 99)
        guide = score.locator(".training-crosshair").evaluate(
            "line => line.getBoundingClientRect().toJSON()"
        )
        assert guide["x"] == pytest.approx(expected_x, abs=1)
        assert guide["width"] <= 1 and guide["height"] > 50
        marker = score.locator(".training-hover-point").bounding_box()
        assert marker and marker["x"] + marker["width"] / 2 == pytest.approx(expected_x, abs=1)
        page.get_by_role("heading", name="Training metrics").hover()
        expect(score.locator(".training-chart-tooltip")).to_be_hidden()


def test_regularization_inspector_uses_the_log_axis_and_measured_keyboard_steps(
    browser: Any,
) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    history = {
        "candidates": [
            {"regularization": value, "validation_segment_macro_f1": score}
            for value, score in ((1, 0.87), (100, 0.62), (0.01, 0.15), (0.1, 0.87))
        ]
    }
    with metric_page(browser, history) as page:
        metric = "Validation interval macro-F1"
        score = chart_figure(page, metric + " by regularization")
        expect(score.locator("circle.training-point")).to_have_count(2)
        expect(score.locator("[data-best] title")).to_have_text("Regularization 1: 0.87")
        expect(score.locator("[data-last] title")).to_have_text("Regularization 0.1: 0.87")
        inspector = score.get_by_role("slider")
        inspector.focus()
        expect_readout(score, "Regularization 0.1", metric, "0.87")
        inspector.press("Home")
        expect_readout(score, "Regularization 0.01", metric, "0.15")
        inspector.press("ArrowRight")
        expect_readout(score, "Regularization 0.1", metric, "0.87")
        expect(inspector).to_have_attribute("aria-valuenow", "0.1")
        inspector.press("End")
        expect_readout(score, "Regularization 100", metric, "0.62")
        inspector.press("ArrowLeft")
        expect_readout(score, "Regularization 1", metric, "0.87")
        inspector.press("Escape")
        expect(score.locator(".training-chart-tooltip")).to_be_hidden()
        expect(inspector).to_be_focused()
        # 60% along four decades is closer to 1 on screen than to 100.
        page.mouse.move(*plot_position(score, 0.60))
        expect_readout(score, "Regularization 1", metric, "0.87")


def test_hover_reads_an_observation_omitted_from_a_long_polyline(browser: Any) -> None:
    history = {
        "updates": [
            {"update": index, "training_loss": (900 - index) / 1000} for index in range(1, 642)
        ]
    }
    with metric_page(browser, history, viewport={"width": 1600, "height": 900}) as page:
        loss = chart_figure(page, "Training loss by update")
        drawn = loss.locator("polyline").evaluate("line => Array.from(line.points, p => p.x)")
        fraction = (323 - 1) / (641 - 1)
        assert len(drawn) < 641 and all(abs(x - fraction * 100) > 0.001 for x in drawn)
        page.mouse.move(*plot_position(loss, fraction))
        expect_readout(loss, "Update 323", "Training loss", "0.577")


def test_new_epochs_preserve_the_keyboard_inspection_point(browser: Any) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    api.run["job"].update(state="running", stage="training", attempt=1)
    api.runs.append(api.run)
    api.progress_rows = [
        {"seed": 11, "epoch": 1, "training_loss": 0.81},
        {"seed": 11, "epoch": 2, "training_loss": 0.74},
    ]
    page = browser.new_page()
    page.route(ORIGIN + "/**", api.dispatch)
    try:
        page.clock.install()
        page.goto(f"{ORIGIN}/training/{RUN_ID}")
        loss = chart_figure(page, "Training loss by epoch")
        inspector = loss.get_by_role("slider")
        inspector.press("Home")
        inspector.press("ArrowRight")
        expect_readout(loss, "Epoch 2", "Training loss", "0.74")
        api.progress_rows.append({"seed": 11, "epoch": 3, "training_loss": 0.65})
        page.clock.fast_forward(3000)
        expect(loss.locator("[data-last] title")).to_have_text("Epoch 3: 0.65")
        expect(inspector).to_be_focused()
        expect(inspector).to_have_attribute("aria-valuenow", "2")
        expect_readout(loss, "Epoch 2", "Training loss", "0.74")
    finally:
        page.close()


@pytest.mark.parametrize("scheme", ["dark", "light"])
def test_tapping_both_chart_edges_keeps_the_readout_visible_on_a_phone(
    browser: Any, scheme: str
) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    history = {
        "epochs": [
            {"epoch": 1, "training_loss": 0.92, "validation_segment_macro_f1": 0.61},
            {"epoch": 2, "training_loss": 0.74, "validation_segment_macro_f1": 0.73},
        ]
    }
    with metric_page(
        browser,
        history,
        viewport={"width": 320, "height": 820},
        has_touch=True,
        color_scheme=scheme,
    ) as page:
        loss = chart_figure(page, "Training loss by epoch")
        score = chart_figure(page, "Validation interval macro-F1 by epoch")
        expect(score.locator("circle.training-point")).to_have_count(1)
        expect(score.locator("[data-last][data-best] title")).to_have_text("Epoch 2: 0.73")
        tooltip = loss.locator(".training-chart-tooltip")
        for fraction, epoch, value in ((0.005, 1, "0.92"), (0.995, 2, "0.74")):
            page.touchscreen.tap(*plot_position(loss, fraction))
            expect_readout(loss, f"Epoch {epoch}", "Training loss", value)
            assert tooltip.evaluate("""tip => {
              const box = tip.getBoundingClientRect();
              return box.left >= 0 && box.right <= innerWidth &&
                box.top >= 0 && box.bottom <= innerHeight;
            }""")
            value_text, axis_text = tooltip.locator("strong"), tooltip.locator("span")
            assert_contrast(value_text, 4.5)
            assert_contrast(axis_text, 4.5)
            assert rendered_colors(value_text)[1] != rendered_colors(axis_text)[1]
        page.get_by_role("heading", name="Training metrics").tap()
        expect(tooltip).to_be_hidden()
