"""Build a source, wheel and optional asset bundle without reading runtime data."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import tomllib
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

SOURCE_DIRECTORIES = {".github", "deploy", "docs", "schemas", "scripts", "src", "tests"}
SOURCE_FILES = {
    ".dockerignore",
    ".editorconfig",
    ".gitignore",
    ".pre-commit-config.yaml",
    ".python-version",
    "AGENTS.md",
    "Dockerfile",
    "LICENSE",
    "LICENSE.md",
    "README.md",
    "compose.yaml",
    "compose.s3.yaml",
    "pyproject.toml",
    "uv.lock",
}
ASSET_SUFFIXES = {
    "comparison": {".html", ".css", ".js", ".json", ".md"},
    "documents": {".docx", ".pdf", ".json", ".md"},
    "evaluation": {".json", ".npy", ".md"},
    "examples": {".html", ".css", ".js", ".json", ".md"},
    "models": {".json", ".npy", ".npz", ".pt", ".md"},
}


def git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *arguments], text=True)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def safe_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if (
        not name
        or path.is_absolute()
        or path.as_posix() != name
        or any(part in {".", ".."} for part in path.parts)
        or any(ord(character) < 32 or character in "\\:" for character in name)
    ):
        raise ValueError(f"Unsafe bundle path: {name!r}")
    return path


def regular_file(root: Path, name: str) -> Path:
    if root.is_symlink():
        raise ValueError("Symbolic links are not allowed for an input directory.")
    path = root
    for part in safe_path(name).parts:
        path /= part
        if path.is_symlink():
            raise ValueError(f"Symbolic links are not allowed: {name}")
    if not path.is_file():
        raise ValueError(f"Required file is missing or not regular: {name}")
    return path


def source_snapshot(
    root: Path, commit: str, *, allow_dirty: bool
) -> tuple[dict[str, bytes], dict[str, int]]:
    if git(root, "rev-parse", "--show-toplevel").strip() != str(root):
        raise ValueError("The source directory must be the Git repository root.")
    dirty = bool(git(root, "status", "--porcelain", "--untracked-files=no"))
    if dirty and not allow_dirty:
        raise ValueError(
            "Commit tracked changes before building, or use --allow-dirty for a preview."
        )
    files: dict[str, bytes] = {}
    modes: dict[str, int] = {}
    for entry in git(root, "ls-files", "--stage", "-z").split("\0"):
        if not entry:
            continue
        metadata, name = entry.split("\t", 1)
        mode, _object_id, stage = metadata.split()
        path = safe_path(name)
        if stage != "0" or mode not in {"100644", "100755"}:
            raise ValueError(f"Unsupported unmerged file, submodule or symbolic link: {name}")
        if path.parts[0] not in SOURCE_DIRECTORIES and name not in SOURCE_FILES:
            raise ValueError(f"Tracked file is outside the source allowlist: {name}")
        if any(part.startswith(".") for part in path.parts[1:]) or "__pycache__" in path.parts:
            raise ValueError(f"Hidden source files and caches are not allowed: {name}")
        files[name] = regular_file(root, name).read_bytes()
        modes[name] = 0o755 if mode == "100755" else 0o644
    if not allow_dirty:
        archive = subprocess.check_output(["git", "-C", str(root), "archive", commit])
        committed: dict[str, bytes] = {}
        with tarfile.open(fileobj=io.BytesIO(archive)) as source:
            for name in files:
                reader = source.extractfile(name)
                if reader is None:
                    raise ValueError(f"Committed source file is missing: {name}")
                committed[name] = reader.read()
        # A concurrent edit cannot silently turn a normal release into a preview.
        if committed != files:
            raise ValueError("Source changed while capturing the release. Retry after committing.")
        files = committed
    return files, modes


def asset_snapshot(root: Path) -> dict[str, bytes]:
    checksums = regular_file(root, "SHA256SUMS").read_bytes()
    files = {"SHA256SUMS": checksums}
    for line in checksums.decode("utf-8").splitlines():
        match = re.fullmatch(r"([a-f0-9]{64})  (.+)", line)
        if match is None:
            raise ValueError("Asset SHA256SUMS requires SHA-256, two spaces and a relative path.")
        expected, name = match.groups()
        path = safe_path(name)
        allowed = (
            name in {"README.md", "verification.json", "evaluation/reproduce.py"}
            or (path.name == "SHA256SUMS" and path.parts[0] in ASSET_SUFFIXES)
            or path.suffix in ASSET_SUFFIXES.get(path.parts[0], set())
        )
        if not allowed or any(part.startswith(".") for part in path.parts):
            raise ValueError(f"Asset file is outside the asset allowlist: {name}")
        if name in files:
            raise ValueError(f"Duplicate checksum path: {name}")
        content = regular_file(root, name).read_bytes()
        if sha256(content) != expected:
            raise ValueError(f"Asset checksum mismatch: {name}")
        files[name] = content
    if len(files) == 1:
        raise ValueError("Asset SHA256SUMS contains no files.")
    return files


def write_snapshot(root: Path, files: dict[str, bytes], modes: dict[str, int]) -> None:
    for name, content in files.items():
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        destination.chmod(modes.get(name, 0o644))


def build_distributions(root: Path, timestamp: int) -> dict[str, bytes]:
    output = root / "dist"
    subprocess.run(
        [
            "uv",
            "build",
            "--no-build-isolation",
            "--python",
            sys.executable,
            "--no-create-gitignore",
            "--out-dir",
            str(output),
            str(root),
        ],
        check=True,
        env={**os.environ, "SOURCE_DATE_EPOCH": str(timestamp)},
    )
    wheels = list(output.glob("*.whl"))
    sdists = list(output.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError("The build must produce exactly one wheel and one source distribution.")
    with ZipFile(wheels[0]) as wheel:
        for path in (root / "src" / "regimex").rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                name = path.relative_to(root / "src").as_posix()
                if wheel.read(name) != path.read_bytes():
                    raise ValueError(f"Wheel resource differs from source: {name}")
    with tarfile.open(sdists[0]) as sdist:
        if any(member.issym() or member.islnk() for member in sdist.getmembers()):
            raise ValueError("The source distribution contains a symbolic or hard link.")
    return {f"dist/{path.name}": path.read_bytes() for path in (*wheels, *sdists)}


def write_archive(
    target: Path, prefix: str, files: dict[str, bytes], modes: dict[str, int], timestamp: int
) -> None:
    date = datetime.fromtimestamp(max(timestamp, 315532800), UTC)
    with ZipFile(target, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for name, content in sorted(files.items()):
            info = ZipInfo(f"{prefix}/{name}", date.timetuple()[:6])
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | modes.get(name, 0o644)) << 16
            info.compress_type = ZIP_DEFLATED
            archive.writestr(info, content, compresslevel=9)


def build_release(root: Path, output: Path, assets: Path | None, *, allow_dirty: bool) -> Path:
    commit = git(root, "rev-parse", "HEAD").strip()
    source, modes = source_snapshot(root, commit, allow_dirty=allow_dirty)
    configuration = tomllib.loads(source["pyproject.toml"].decode("utf-8"))
    version = configuration["project"]["version"]
    if not isinstance(version, str) or re.fullmatch(r"[0-9][A-Za-z0-9.+-]*", version) is None:
        raise ValueError("The project version is not a safe release filename.")
    timestamp = int(git(root, "show", "-s", "--format=%ct", commit))
    source_hashes = {name: sha256(content) for name, content in sorted(source.items())}
    imported = asset_snapshot(assets) if assets is not None else {}
    output.mkdir(parents=True, exist_ok=True)
    prefix = f"regimex-{version}"
    target = output / f"{prefix}.zip"
    checksum_target = output / f"{prefix}.zip.sha256"
    if target.is_symlink() or checksum_target.is_symlink():
        raise ValueError("Release output files must not be symbolic links.")
    # All packages use this immutable snapshot, even if the working tree changes later.
    with tempfile.TemporaryDirectory(prefix="regimex-release-") as temporary:
        staging = Path(temporary) / prefix
        write_snapshot(staging, source, modes)
        distributions = build_distributions(staging, timestamp)
        files = {
            **source,
            **distributions,
            **{f"release/{name}": content for name, content in imported.items()},
        }
        files["BUILD.json"] = json_bytes(
            {
                "schema_version": "1",
                "version": version,
                "source": {
                    "commit": commit,
                    "working_tree_preview": allow_dirty,
                    "files": source_hashes,
                    "sha256": sha256(
                        json.dumps(source_hashes, sort_keys=True, separators=(",", ":")).encode()
                    ),
                },
                "checks": {"package_build": "passed", "wheel_resources": "passed"},
                "assets": {
                    "included": bool(imported),
                    "files": {name: sha256(content) for name, content in sorted(imported.items())},
                    "evidence_scope": (
                        "Imported reports, documents and verification describe the assets. "
                        "They do not certify this source revision; no training or tests were rerun."
                    ),
                },
            }
        )
        files["SHA256SUMS"] = "".join(
            f"{sha256(content)}  {name}\n" for name, content in sorted(files.items())
        ).encode()
        # Publish only a complete archive; an interrupted build leaves the previous output intact.
        with tempfile.NamedTemporaryFile(dir=output, prefix=f".{prefix}-", delete=False) as handle:
            pending = Path(handle.name)
        try:
            write_archive(pending, prefix, files, modes, timestamp)
            archive_digest = sha256(pending.read_bytes())
            pending.replace(target)
        finally:
            pending.unlink(missing_ok=True)
        checksum_target.write_text(f"{archive_digest}  {target.name}\n", encoding="utf-8")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output-dir", type=Path, help="Output directory (default: release/build)."
    )
    parser.add_argument("--assets", type=Path, help="Asset directory containing SHA256SUMS.")
    parser.add_argument("--allow-dirty", action="store_true", help="Build a working-tree preview.")
    args = parser.parse_args()
    root = args.source.resolve()
    output = args.output_dir.resolve() if args.output_dir else root / "release" / "build"
    assets = args.assets.absolute() if args.assets else None
    if shutil.which("uv") is None:
        parser.error("uv must be available on PATH.")
    try:
        target = build_release(root, output, assets, allow_dirty=args.allow_dirty)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print(f"Release build failed: {error}", file=sys.stderr)
        return 1
    print(f"Built {target}")
    print(f"Verify with: cd {output} && sha256sum -c {target.name}.sha256")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
