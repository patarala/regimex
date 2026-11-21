"""Exercise appearance preferences and the rendered workspace palette in Chromium."""

import json
from typing import Any

import pytest
from test_offline_report_browser import browser as browser
from test_training_workspace_browser import ORIGIN, RUN_ID, TrainingAPI


def luminance(rgb: list[int]) -> float:
    def linear(channel: int) -> float:
        value = channel / 255
        return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4

    red, green, blue = map(linear, rgb)
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def rendered_colors(locator: Any, foreground: str = "color") -> tuple[list[int], list[int]]:
    background, ink = locator.evaluate(
        """(element, foreground) => {
          // Let the browser resolve CSS color syntax and blend translucent ancestor backgrounds.
          const canvas = document.createElement('canvas');
          canvas.width = 2;
          canvas.height = 1;
          const ctx = canvas.getContext('2d');
          ctx.fillStyle = 'white';
          ctx.fillRect(0, 0, 2, 1);
          const ancestors = [];
          for (let node = element; node; node = node.parentElement) ancestors.push(node);
          for (const node of ancestors.reverse()) {
            ctx.fillStyle = getComputedStyle(node).backgroundColor;
            ctx.fillRect(0, 0, 2, 1);
          }
          ctx.fillStyle = getComputedStyle(element)[foreground];
          ctx.fillRect(1, 0, 1, 1);
          const pixels = ctx.getImageData(0, 0, 2, 1).data;
          return [Array.from(pixels.slice(0, 3)), Array.from(pixels.slice(4, 7))];
        }""",
        foreground,
    )
    return background, ink


def assert_contrast(locator: Any, minimum: float, foreground: str = "color") -> None:
    background, ink = rendered_colors(locator, foreground)
    low, high = sorted((luminance(background), luminance(ink)))
    ratio = (high + 0.05) / (low + 0.05)
    assert ratio >= minimum, f"{locator}: {ink} on {background} gives {ratio:.2f}:1"


def theme_button(page: Any) -> Any:
    return page.get_by_role("button", name="Dark theme", exact=True)


def assert_scheme(page: Any, scheme: str) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    button = theme_button(page)
    expect(button).to_have_attribute("aria-pressed", "true" if scheme == "dark" else "false")
    expect(button.locator(".theme-icon [data-mode]:visible")).to_have_attribute("data-mode", scheme)
    background, _ = rendered_colors(page.locator("html"))
    assert (luminance(background) < 0.18) == (scheme == "dark"), (scheme, background)


def assert_override(page: Any, value: str | None) -> None:
    assert page.locator("html").get_attribute("data-theme") == value
    assert page.evaluate("localStorage.getItem('regimex-theme')") == value


@pytest.mark.parametrize("width", [320, 1280], ids=["phone", "desktop"])
def test_theme_follows_os_or_saved_choice_across_sessions(browser: Any, width: int) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    page = browser.new_page(viewport={"width": width, "height": 800}, color_scheme="light")
    page.route(ORIGIN + "/**", api.dispatch)
    signed_in = False

    def session(route: Any) -> None:
        nonlocal signed_in
        if route.request.method == "POST":
            signed_in = True
        if route.request.method == "DELETE":
            signed_in = False
            route.fulfill(status=204)
        elif signed_in:
            route.fulfill(
                json={"principal": {"owner_id": "operator", "role": "admin"}, "csrf_token": "test"}
            )
        else:
            route.fulfill(status=401, json={})

    page.route(ORIGIN + "/api/v1/session", session)
    try:
        page.goto(ORIGIN)
        theme = theme_button(page)
        expect(page.get_by_role("heading", name="Sign in")).to_be_visible()
        assert_override(page, None)
        assert_scheme(page, "light")
        page.emulate_media(color_scheme="dark")
        assert_scheme(page, "dark")
        theme.click()
        assert_scheme(page, "light")
        assert_override(page, "light")
        page.reload()
        assert_scheme(page, "light")
        theme.click()
        assert_scheme(page, "dark")
        assert_override(page, None)
        page.emulate_media(color_scheme="light")
        assert_scheme(page, "light")
        theme.click()
        assert_scheme(page, "dark")
        assert_override(page, "dark")
        page.reload()
        assert_scheme(page, "dark")
        for selector in ("html", "#token"):
            background, _ = rendered_colors(page.locator(selector))
            red, green, blue = background
            assert green <= max(red, blue), f"Green cast on {selector}: {background}"
        for selector in ("#login-submit", "#token-hint", "#token"):
            assert_contrast(page.locator(selector), 4.5)
        assert_contrast(theme.locator(".theme-icon"), 3, "stroke")

        page.get_by_label("Access token").fill("test-session")
        page.get_by_role("button", name="Sign in").click()
        expect(page.get_by_role("link", name="Training", exact=True)).to_be_visible()
        expect(theme).to_be_in_viewport()
        assert_scheme(page, "dark")
        geometry = page.get_by_role("banner").evaluate("""header => {
          const logout = header.querySelector('#logout').getBoundingClientRect();
          const toggle = header.querySelector('#theme').getBoundingClientRect();
          const others = [...header.querySelectorAll('a, button:not(#logout)')]
            .map(control => control.getBoundingClientRect()).filter(box => box.width);
          return {gap: logout.left - toggle.right, clear: others.every(box =>
            box.right <= logout.right && (box.right <= logout.left ||
              box.bottom <= logout.top || box.top >= logout.bottom))};
        }""")
        assert geometry["gap"] >= 12 and geometry["clear"], geometry
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        theme.press("Space")
        expect(theme).to_be_focused()
        assert_scheme(page, "light")
        assert_override(page, None)
        theme.press("Enter")
        assert_scheme(page, "dark")
        theme.press("Enter")
        assert_scheme(page, "light")
        theme.press("Tab")
        expect(page.get_by_role("button", name="Sign out")).to_be_focused()
        page.emulate_media(color_scheme="dark")
        assert_scheme(page, "dark")
        page.get_by_role("button", name="Sign out").click()
        expect(page.get_by_role("heading", name="Sign in")).to_be_visible()
        expect(theme).to_be_in_viewport()
        page.reload()
        assert_override(page, None)
        page.emulate_media(color_scheme="light")
        assert_scheme(page, "light")
        for selector in ("#login-submit", "#token-hint", "#token"):
            assert_contrast(page.locator(selector), 4.5)
        assert_contrast(theme.locator(".theme-icon"), 3, "stroke")
    finally:
        page.close()


def test_training_chart_text_curves_and_markers_contrast_in_both_themes(browser: Any) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    api = TrainingAPI()
    api.runs.append(api.run)
    api.metrics["history"]["epochs"].append(
        {"seed": 23, "epoch": 2, "training_loss": 0.41, "validation_segment_macro_f1": 0.73}
    )
    page = browser.new_page(color_scheme="light")
    page.route(ORIGIN + "/**", api.dispatch)
    try:
        page.goto(f"{ORIGIN}/training/{RUN_ID}")
        loss = page.get_by_role("img", name="Training loss by epoch")
        score = page.get_by_role("img", name="Validation interval macro-F1 by epoch")
        expect(loss).to_be_visible()
        for scheme in ("dark", "light"):
            theme_button(page).click()
            assert_scheme(page, scheme)
            for chart in (loss, score):
                assert_contrast(chart.locator("text").first, 4.5, "fill")
                assert_contrast(chart.locator("polyline"), 3, "stroke")
                assert_contrast(chart.locator("circle").first, 3, "fill")
            assert_contrast(page.locator("#training-runs [aria-current=true] strong"), 4.5)
            assert_contrast(page.locator("#training-runs [aria-current=true] time"), 4.5)
    finally:
        page.close()


def test_theme_syncs_between_tabs_and_sends_only_the_scheme_to_sandboxed_previews(
    browser: Any,
) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    context = browser.new_context(color_scheme="light")
    context.route(ORIGIN + "/**", TrainingAPI().dispatch)
    context.route(
        ORIGIN + "/test-preview-*",
        lambda route: route.fulfill(
            content_type="text/html",
            body="""<body><script>
              addEventListener("message", event => {
                document.body.dataset.theme = event.data.theme;
                document.body.textContent = JSON.stringify(event.data);
              });
            </script></body>""",
        ),
    )
    try:
        workspace = context.new_page()
        preferences = context.new_page()
        workspace.goto(ORIGIN)
        preferences.goto(ORIGIN)
        frames = ("report-preview", "comparison-preview")

        def expect_previews(scheme: str) -> None:
            for name in frames:
                body = workspace.frame_locator("#" + name).locator("body")
                expect(body).to_have_attribute("data-theme", scheme)
                assert json.loads(body.text_content()) == {"type": "regimex-theme", "theme": scheme}

        for name in frames:
            workspace.locator("#" + name).evaluate(
                "(frame, url) => { frame.src = url; }", ORIGIN + "/test-preview-" + name
            )
        expect_previews("light")
        other = theme_button(preferences)
        other.click()
        assert_scheme(workspace, "dark")
        expect_previews("dark")
        other.click()
        assert_scheme(workspace, "light")
        expect_previews("light")
        workspace.emulate_media(color_scheme="dark")
        expect_previews("dark")
    finally:
        context.close()


def test_switching_theme_still_works_when_browser_storage_is_blocked(browser: Any) -> None:
    expect = pytest.importorskip("playwright.sync_api").expect
    page = browser.new_page(color_scheme="light")
    page.route(ORIGIN + "/**", TrainingAPI().dispatch)
    page.add_init_script("""Object.defineProperty(window, 'localStorage', {
        get() { throw new DOMException('Storage unavailable', 'SecurityError'); }
    });""")
    try:
        page.goto(ORIGIN)
        expect(page.get_by_role("link", name="Training", exact=True)).to_be_visible()
        assert_scheme(page, "light")
        theme_button(page).click()
        assert_scheme(page, "dark")
        page.reload()
        assert_scheme(page, "light")
    finally:
        page.close()


@pytest.mark.parametrize(("saved", "other"), [("light", "dark"), ("dark", "light")])
def test_existing_explicit_preference_survives_an_os_change(
    browser: Any, saved: str, other: str
) -> None:
    context = browser.new_context(
        color_scheme=saved,
        storage_state={
            "cookies": [],
            "origins": [
                {"origin": ORIGIN, "localStorage": [{"name": "regimex-theme", "value": saved}]}
            ],
        },
    )
    context.route(ORIGIN + "/**", TrainingAPI().dispatch)
    try:
        page = context.new_page()
        page.goto(ORIGIN)
        assert_scheme(page, saved)
        page.emulate_media(color_scheme=other)
        assert_scheme(page, saved)
        assert_override(page, saved)
        theme_button(page).click()
        assert_scheme(page, other)
        assert_override(page, None)
    finally:
        context.close()
