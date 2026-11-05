"""Preserve installed software notices in portable report exports."""

from importlib.metadata import PackageNotFoundError, distribution

from regimex.model_registry import RegistryError


def _license(package: str, filename: str) -> str:
    try:
        text = distribution(package).read_text(filename)
    except PackageNotFoundError:
        text = None
    if text is None or not text.strip():
        raise RegistryError(f"The installed {package} distribution has no license notice.")
    if "--" in text or "\x00" in text:
        raise RegistryError(f"The installed {package} license is unsafe for an HTML comment.")
    return text


def license_notices(*, include_plotly: bool = False) -> str:
    """Read exact installed texts lazily; comparison reports need only Regimex."""
    notices = "# Regimex\n\n" + _license("regimex", "licenses/LICENSE")
    if include_plotly:
        notices += "\n\n# Plotly\n\n" + _license("plotly", "licenses/LICENSE.txt")
    return notices
