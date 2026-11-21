"""Assemble local styles and scripts for the offline report renderers."""

from importlib.resources import files


def report_assets(*names: str) -> str:
    templates = files("regimex").joinpath("templates")
    return "\n".join(templates.joinpath(name).read_text(encoding="utf-8") for name in names)
