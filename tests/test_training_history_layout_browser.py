"""Run history remains readable at the narrowest sidebar and phone layouts."""

import re
from typing import Any

import pytest
from test_offline_report_browser import browser as browser
from test_training_workspace_browser import ORIGIN, RUN_ID, TrainingAPI, completed_run


@pytest.mark.parametrize("width", [821, 320])
def test_run_history_shows_full_labels_and_keeps_time_together(browser: Any, width: int) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    name = "سوق العملات " + "W" * 84
    dataset = "market_" + "momentum" * 12
    api.runs = [api.run, completed_run("00000000-0000-0000-0000-000000000002", "training")]
    for run in api.runs:
        run["display_name"] = name
        run["training"]["dataset_id"] = dataset
        run["trainable_parameters"] = 120000
        run["job"]["created_at"] = "2026-09-30T23:59:00Z"
        run["job"]["state"] = "cancelled"
        run["archived_at"] = "2026-10-01T12:00:00Z"
    short = completed_run("00000000-0000-0000-0000-000000000003", "training")
    short["display_name"] = "Trial"
    short["job"].update(state="queued", created_at="2027-01-01T01:00:00Z")
    api.runs.append(short)

    page = browser.new_page(
        viewport={"width": width, "height": 900}, locale="en-US", timezone_id="UTC"
    )
    page.route(ORIGIN + "/**", api.dispatch)
    try:
        # Older dates include the year even in the narrowest selection layout.
        page.clock.set_fixed_time("2027-01-01T13:00:00Z")
        page.goto(f"{ORIGIN}/training/{RUN_ID}?view=all")
        expect(page.locator("#training-state")).to_have_text("Cancelled")
        expect(page.locator("#training-archived")).to_be_visible()
        if width == 320:
            created = page.locator("#training-created time")
            assert created.evaluate("""time => {
                const range = document.createRange();
                range.selectNodeContents(time);
                return range.getClientRects().length === 1;
            }""")
            page.get_by_role("button", name="Run history", exact=True).click()
        row = page.locator(f'#training-runs button[data-run-id="{RUN_ID}"]')
        expect(row.locator("strong")).to_have_text(name)
        expect(row.locator("strong")).to_have_css("direction", "rtl")
        expect(row.locator("[data-state]")).to_have_text("Cancelled")
        expect(row).to_have_accessible_name(re.compile(r"\bArchived\b"))
        expect(row.get_by_role("img", name="Archived")).to_be_visible()
        expect(row.locator(".training-run-id")).to_have_text("#00000001")
        expect(row.locator("time")).to_contain_text("PM")
        expect(row.locator("time")).to_contain_text("2026")
        page.locator(f'[data-run-info="{RUN_ID}"]').click()
        expect(page.get_by_role("tooltip")).to_contain_text("Archived Oct 1, 2026")
        page.keyboard.press("Escape")
        for selecting in (False, True):
            if selecting:
                page.get_by_role("button", name="Select runs", exact=True).click()
                page.locator(f'#training-runs input[data-select-run="{RUN_ID}"]').check()
            layout = row.evaluate(
                """button => {
                  const bounds = button.getBoundingClientRect();
                  return ['strong', 'time', '[data-state]'].map(selector => {
                    const element = button.querySelector(selector);
                    const range = document.createRange();
                    range.selectNodeContents(element);
                    const lines = [...range.getClientRects()].filter(rect => rect.width);
                    const box = element.getBoundingClientRect();
                    return {
                      lines: new Set(lines.map(rect => rect.top)).size,
                      fits: lines.every(rect =>
                        rect.left >= bounds.left && rect.right <= bounds.right &&
                        rect.top >= box.top - 1 && rect.bottom <= box.bottom + 1)
                    };
                  });
                }"""
            )
            assert all(item["fits"] for item in layout), layout
            title, timestamp, state = layout
            assert title["lines"] > 2
            assert timestamp["lines"] == state["lines"] == 1
            for run_id in (RUN_ID, short["run_id"]):
                control = page.locator(f'button[data-run-id="{run_id}"]')
                alignment = control.evaluate(
                    """button => {
                      const row = button.parentElement;
                      const box = selector => row.querySelector(selector)?.getBoundingClientRect();
                      const meta = box(".training-run-meta"), info = box("[data-run-info]");
                      const suffix = box(".training-run-id");
                      function textBox(selector) {
                        const range = document.createRange();
                        range.selectNodeContents(button.querySelector(selector));
                        return range.getBoundingClientRect();
                      }
                      const time = textBox("time"), state = textBox("[data-state]");
                      return {
                        timeInset: time.left - meta.left, stateInset: meta.right - state.right,
                        wrapped: state.top >= time.bottom,
                        clearsInfo: state.top >= info.bottom || state.right <= info.left,
                        clearsSuffix: !suffix || suffix.top >= Math.max(time.bottom, state.bottom)
                      };
                    }"""
                )
                assert abs(alignment["timeInset"]) <= 1 and abs(alignment["stateInset"]) <= 1
                assert alignment["clearsInfo"] and alignment["clearsSuffix"]
                if width == 320:
                    assert not alignment["wrapped"]
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    finally:
        page.close()


def word_bounds(element: Any, word: str) -> dict[str, float]:
    return dict(
        element.evaluate(
            """(element, word) => {
        const nodes = document.createTreeWalker(element, NodeFilter.SHOW_TEXT);
        while (nodes.nextNode()) {
            const node = nodes.currentNode, start = node.textContent.indexOf(word);
            if (start < 0) continue;
            const range = document.createRange();
            range.setStart(node, start);
            range.setEnd(node, start + word.length);
            if (range.getClientRects().length !== 1) throw Error(word + " wraps");
            return range.getBoundingClientRect().toJSON();
        }
        throw Error(word + " is missing");
    }""",
            word,
        )
    )


@pytest.mark.parametrize("width", [320, 821, 1280])
def test_unnamed_archived_baseline_keeps_its_dataset_id_with_its_name(
    browser: Any, width: int
) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    prefix = "f6805c13"
    dataset = f"dataset-{prefix}" + "0" * 24
    api.run.update(display_name=None, archived_at="2026-10-01T12:00:00Z", trainable_parameters=9315)
    api.run["training"].update(architecture_id="baseline", dataset_id=dataset)
    api.run["job"]["state"] = "cancelled"
    api.runs.append(api.run)
    page = browser.new_page(viewport={"width": width, "height": 900})
    page.route(ORIGIN + "/**", api.dispatch)
    try:
        page.goto(f"{ORIGIN}/training/{RUN_ID}?view=all")
        heading = page.locator("#training-detail-title")
        expect(heading).to_contain_text(prefix)
        assert word_bounds(heading, "Dataset")["top"] == word_bounds(heading, prefix)["top"]
        if width == 320:
            page.get_by_role("button", name="Run history", exact=True).click()
        row = page.locator(f'button[data-run-id="{RUN_ID}"]')
        info = page.locator(f'[data-run-info="{RUN_ID}"]')
        expect(row.locator("[data-state]")).to_have_text("Cancelled")
        for selecting in (False, True):
            if selecting:
                page.get_by_role("button", name="Select runs", exact=True).click()
                page.locator(f'input[data-select-run="{RUN_ID}"]').check()
            name = word_bounds(row.locator("strong"), "Baseline")
            identifier = word_bounds(row.locator("strong"), prefix)
            archive = row.get_by_role("img", name="Archived").bounding_box()
            details = info.bounding_box()
            assert archive and details
            assert name["top"] < (identifier["top"] + identifier["bottom"]) / 2 < name["bottom"]
            assert archive["x"] + archive["width"] <= name["left"]
            assert name["right"] <= identifier["left"] < identifier["right"] <= details["x"]
        info.click()
        expect(page.get_by_role("tooltip").locator(".run-info-dataset")).to_have_text(dataset)
    finally:
        page.close()


@pytest.mark.parametrize("width", [1280, 375, 320])
def test_history_tooltip_keeps_parameter_counts_and_dataset_ids_readable(
    browser: Any, width: int
) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    canonical = "dataset-12345678abcdef0012345678abcdef00"
    long_id = "dataset-" + "WideIdentifier" * 8
    api.runs = [api.run, completed_run("00000000-0000-0000-0000-000000000002", "training")]
    for run, dataset in zip(api.runs, (canonical, long_id), strict=True):
        run.update(display_name=dataset[:18], trainable_parameters=135555)
        run["training"]["dataset_id"] = dataset
    page = browser.new_page(viewport={"width": width, "height": 800}, locale="en-US")
    page.route(ORIGIN + "/**", api.dispatch)
    try:
        page.goto(f"{ORIGIN}/training/new")
        if width < 821:
            page.get_by_role("button", name="Run history", exact=True).click()
        for run in api.runs:
            dataset = run["training"]["dataset_id"]
            page.locator(f'[data-run-info="{run["run_id"]}"]').click()
            tooltip = page.get_by_role("tooltip")
            expect(tooltip.locator(".run-info-size")).to_have_text("135,555 parameters")
            expect(tooltip.locator(".run-info-dataset")).to_have_text(dataset)
            layout = tooltip.evaluate(
                """tip => {
                  function fragments(selector, end) {
                    const node = tip.querySelector(selector).firstChild;
                    const range = document.createRange();
                    range.setStart(node, 0);
                    range.setEnd(node, end ?? node.length);
                    return [...range.getClientRects()];
                  }
                  const id = fragments('.run-info-dataset');
                  const prefix = fragments('.run-info-dataset', 'dataset-'.length + 1);
                  const size = fragments('.run-info-size');
                  const bounds = tip.getBoundingClientRect();
                  return {
                    sizeLines: size.length,
                    idLines: id.length,
                    prefixTogether: prefix.length === 1,
                    inBounds: bounds.left >= 0 && bounds.right <= innerWidth &&
                      tip.scrollWidth <= tip.clientWidth &&
                      [...id, ...size].every(r => r.left >= bounds.left && r.right <= bounds.right)
                  };
                }"""
            )
            assert layout["inBounds"] and layout["sizeLines"] == 1, layout
            assert layout["prefixTogether"], layout
            if dataset == long_id:
                assert layout["idLines"] > 1, layout
            else:
                assert layout["idLines"] == 1, layout
            page.keyboard.press("Escape")
    finally:
        page.close()
