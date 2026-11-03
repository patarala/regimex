"""Release assembly must bind source identity and keep unrelated local data out."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

import pytest

BUILDER = Path(__file__).resolve().parents[1] / "scripts" / "build_release.py"


def run(*arguments: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(arguments, cwd=cwd, text=True, capture_output=True, check=True)


@pytest.fixture
def source(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    package = root / "src" / "regimex"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('"""Release test package."""\n')
    (package / "__main__.py").write_text('print("installed-release-ok")\n')
    (package / "web").mkdir()
    (package / "web" / "index.html").write_text("<h1>Bundled resource</h1>\n")
    (root / "README.md").write_text("# Release test\n")
    (root / "pyproject.toml").write_text(
        '[build-system]\nrequires = ["hatchling==1.32.4"]\n'
        'build-backend = "hatchling.build"\n'
        '[project]\nname = "regimex"\nversion = "1.2.3"\n'
        'readme = "README.md"\nrequires-python = ">=3.14"\n'
        '[tool.hatch.build.targets.wheel]\npackages = ["src/regimex"]\n'
    )
    run("git", "init", "-q", cwd=root)
    run("git", "add", ".", cwd=root)
    run(
        "git",
        "-c",
        "user.name=Release test",
        "-c",
        "user.email=release-test@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-qm",
        "Fixture",
        cwd=root,
    )
    return root


def build(source: Path, output: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(BUILDER),
            "--source",
            str(source),
            "--output-dir",
            str(output),
            *arguments,
        ],
        cwd=source,
        env={**os.environ, "UV_OFFLINE": "true", "UV_CACHE_DIR": str(output.parent / "uv-cache")},
        text=True,
        capture_output=True,
    )


def asset_pack(root: Path, files: dict[str, bytes]) -> Path:
    root.mkdir()
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    (root / "SHA256SUMS").write_text(
        "".join(f"{hashlib.sha256(data).hexdigest()}  {name}\n" for name, data in files.items())
    )
    return root


def test_release_builds_installable_packages_with_stable_hashes(
    source: Path, tmp_path: Path
) -> None:
    (source / ".env").write_text("PRIVATE_VALUE=must-not-be-shipped\n")
    (source / "untracked.json").write_text('{"private": true}\n')
    output = tmp_path / "output"
    result = build(source, output)
    assert result.returncode == 0, result.stderr
    archive_path = output / "regimex-1.2.3.zip"
    original = archive_path.read_bytes()
    with ZipFile(archive_path) as archive:
        names = archive.namelist()
        assert all(
            ".env" not in name and "untracked" not in name and "/.git/" not in name
            for name in names
        )
        prefix = "regimex-1.2.3/"
        manifest = json.loads(archive.read(prefix + "BUILD.json"))
        assert (
            manifest["source"]["commit"]
            == run("git", "rev-parse", "HEAD", cwd=source).stdout.strip()
        )
        assert manifest["source"]["working_tree_preview"] is False
        assert manifest["assets"]["included"] is False
        checksums = archive.read(prefix + "SHA256SUMS").decode().splitlines()
        for line in checksums:
            digest, name = line.split("  ", 1)
            assert hashlib.sha256(archive.read(prefix + name)).hexdigest() == digest
        wheel_name = next(name for name in names if name.endswith(".whl"))
        wheel = output / Path(wheel_name).name
        wheel.write_bytes(archive.read(wheel_name))
    assert (
        (output / "regimex-1.2.3.zip.sha256")
        .read_text()
        .startswith(hashlib.sha256(original).hexdigest())
    )
    environment = tmp_path / "installed"
    run(sys.executable, "-m", "venv", "--without-pip", str(environment), cwd=tmp_path)
    python = environment / "bin" / "python"
    run("uv", "pip", "install", "--python", str(python), "--no-deps", str(wheel), cwd=tmp_path)
    assert run(str(python), "-m", "regimex", cwd=tmp_path).stdout.strip() == "installed-release-ok"
    repeated = build(source, output)
    assert repeated.returncode == 0, repeated.stderr
    assert archive_path.read_bytes() == original


def test_dirty_source_requires_explicit_preview(source: Path, tmp_path: Path) -> None:
    (source / "README.md").write_text("# Changed source\n")
    output = tmp_path / "output"
    result = build(source, output)
    assert result.returncode != 0
    assert "Commit tracked changes" in result.stderr
    assert not output.exists()
    preview = build(source, output, "--allow-dirty")
    assert preview.returncode == 0, preview.stderr
    with ZipFile(output / "regimex-1.2.3.zip") as archive:
        manifest = json.loads(archive.read("regimex-1.2.3/BUILD.json"))
        assert manifest["source"]["working_tree_preview"] is True
        assert archive.read("regimex-1.2.3/README.md") == b"# Changed source\n"


def test_assets_are_verified_and_old_evidence_is_identified(source: Path, tmp_path: Path) -> None:
    assets = asset_pack(
        tmp_path / "assets",
        {"documents/guide.docx": b"editable-document", "verification.json": b'{"source": "old"}'},
    )
    (assets / "private.key").write_text("unlisted private data")
    output = tmp_path / "output"
    result = build(source, output, "--assets", str(assets))
    assert result.returncode == 0, result.stderr
    with ZipFile(output / "regimex-1.2.3.zip") as archive:
        assert not any("private.key" in name for name in archive.namelist())
        assert archive.read("regimex-1.2.3/release/documents/guide.docx") == b"editable-document"
        manifest = json.loads(archive.read("regimex-1.2.3/BUILD.json"))
        assert "do not certify this source revision" in manifest["assets"]["evidence_scope"]
    prior = (output / "regimex-1.2.3.zip").read_bytes()
    (assets / "documents" / "guide.docx").write_bytes(b"changed-document")
    failed = build(source, output, "--assets", str(assets))
    assert failed.returncode != 0
    assert "Asset checksum mismatch" in failed.stderr
    assert (output / "regimex-1.2.3.zip").read_bytes() == prior


@pytest.mark.parametrize(
    "name",
    [
        "../private.json",
        "/tmp/private.json",
        "models/../../private.json",
        ".env",
        "documents/private.key",
    ],
)
def test_unsafe_asset_entries_fail_before_build(source: Path, tmp_path: Path, name: str) -> None:
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "SHA256SUMS").write_text(f"{'0' * 64}  {name}\n")
    output = tmp_path / "output"
    result = build(source, output, "--assets", str(assets))
    assert result.returncode != 0
    assert not output.exists()


def test_asset_symlinks_and_missing_files_are_rejected(source: Path, tmp_path: Path) -> None:
    assets = asset_pack(tmp_path / "assets", {"models/model.json": b"{}"})
    model = assets / "models" / "model.json"
    model.unlink()
    output = tmp_path / "output"
    missing = build(source, output, "--assets", str(assets))
    assert missing.returncode != 0
    assert "missing or not regular" in missing.stderr
    outside = tmp_path / "private.json"
    outside.write_bytes(b"{}")
    model.symlink_to(outside)
    linked = build(source, output, "--assets", str(assets))
    assert linked.returncode != 0
    assert "Symbolic links are not allowed" in linked.stderr
    assert not output.exists()


def test_linked_asset_directory_is_rejected(source: Path, tmp_path: Path) -> None:
    assets = asset_pack(tmp_path / "assets", {"models/model.json": b"{}"})
    linked_assets = tmp_path / "linked-assets"
    linked_assets.symlink_to(assets, target_is_directory=True)
    result = build(source, tmp_path / "output", "--assets", str(linked_assets))
    assert result.returncode != 0
    assert "Symbolic links are not allowed" in result.stderr
