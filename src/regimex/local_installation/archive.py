"""Validate bounded release archives before executing or extracting their contents."""

import hashlib
import json
import os
import re
import stat
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

MAX_ARCHIVE_BYTES = 512 * 1024**2
MAX_EXPANDED_BYTES = 2 * 1024**3
MAX_FILES = 20000


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024**2):
            checksum.update(block)
    return checksum.hexdigest()


@contextmanager
def extraction_scratch(directory: Path) -> Iterator[Path]:
    """Keep one recoverable draft outside the archive tree under the install lock."""
    if directory.parent.is_symlink():
        raise ValueError("The extraction scratch parent cannot be a symbolic link.")
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = directory.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError("The extraction scratch directory must be private and owned by this user.")
    draft = directory / "member.partial"
    try:
        info = draft.lstat()
    except FileNotFoundError:
        pass
    else:
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ValueError("The extraction draft must be a private, owned regular file.")
        draft.unlink()
    try:
        yield draft
    finally:
        draft.unlink(missing_ok=True)


def _publish_member(
    archive: ZipFile, member: str, target: Path, expected: str, partial: Path
) -> None:
    descriptor = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as output, archive.open(member) as source:
        checksum = hashlib.sha256()
        while block := source.read(1024**2):
            output.write(block)
            checksum.update(block)
        if checksum.hexdigest() != expected:
            raise ValueError("A release member failed checksum verification.")
        output.flush()
        os.fsync(output.fileno())
    os.link(partial, target)
    descriptor = os.open(target.parent, os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    partial.unlink()


def relative(name: str) -> str:
    path = PurePosixPath(name)
    if (
        not name
        or len(name) > 1024
        or path.is_absolute()
        or path.as_posix() != name
        or any(part in {".", ".."} for part in path.parts)
        or any(ord(char) < 32 or char in "\\:" for char in name)
    ):
        raise ValueError("Release contains an unsafe path.")
    return name


def checksums(content: bytes) -> dict[str, str]:
    result = {}
    for line in content.decode("utf-8").splitlines():
        match = re.fullmatch(r"([a-f0-9]{64})  (.+)", line)
        if match is None or match[2] in result:
            raise ValueError("Release checksums are invalid or duplicated.")
        result[relative(match[2])] = match[1]
    if not result:
        raise ValueError("Release checksums are empty.")
    return result


class Release:
    def __init__(self, path: Path, expected: str) -> None:
        if re.fullmatch(r"[a-f0-9]{64}", expected) is None:
            raise ValueError("--sha256 requires the trusted archive SHA-256 digest.")
        if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_ARCHIVE_BYTES:
            raise ValueError("Release must be a bounded regular ZIP archive.")
        if digest(path) != expected:
            raise ValueError("Release archive checksum differs from --sha256.")
        self.path = path
        self.sha256 = expected
        with ZipFile(path) as archive:
            members = archive.infolist()
            if not members or len(members) > MAX_FILES:
                raise ValueError("Release exceeds its file-count limit.")
            self.members: dict[str, str] = {}
            total = 0
            prefixes = set()
            for member in members:
                name = relative(member.filename)
                parts = PurePosixPath(name).parts
                mode = member.external_attr >> 16
                if len(parts) < 2 or not stat.S_ISREG(mode) or member.flag_bits & 1:
                    raise ValueError("Release must contain ordinary unencrypted files.")
                prefixes.add(parts[0])
                name = relative("/".join(parts[1:]))
                if name in self.members:
                    raise ValueError("Release contains duplicate paths.")
                total += member.file_size
                if total > MAX_EXPANDED_BYTES:
                    raise ValueError("Release exceeds its expanded-size limit.")
                self.members[name] = member.filename
            if len(prefixes) != 1 or "SHA256SUMS" not in self.members:
                raise ValueError("Release requires one root directory and SHA256SUMS.")
            if archive.getinfo(self.members["SHA256SUMS"]).file_size > 4 * 1024**2:
                raise ValueError("Release checksum manifest exceeds its size limit.")
            self.hashes = checksums(archive.read(self.members["SHA256SUMS"]))
            if set(self.hashes) != set(self.members) - {"SHA256SUMS"}:
                raise ValueError("Release checksums must cover every payload file exactly once.")
            for name, checksum in self.hashes.items():
                with archive.open(self.members[name]) as source:
                    actual = hashlib.sha256()
                    while block := source.read(1024**2):
                        actual.update(block)
                    if actual.hexdigest() != checksum:
                        raise ValueError("A release member failed checksum verification.")
            if archive.getinfo(self.members["BUILD.json"]).file_size > 8 * 1024**2:
                raise ValueError("Release build metadata exceeds its size limit.")
            build = json.loads(archive.read(self.members["BUILD.json"]))
            if build["schema_version"] != "1" or build["source"]["working_tree_preview"]:
                raise ValueError(
                    "Install requires a committed release, not a working-tree preview."
                )
            self.commit = str(build["source"]["commit"])
            self.version = str(build["version"])
            if re.fullmatch(r"[a-f0-9]{40,64}", self.commit) is None:
                raise ValueError("Release source commit is invalid.")
            if not build["assets"]["included"]:
                raise ValueError("Installation requires a release containing saved model assets.")
            required = {
                "pyproject.toml",
                "uv.lock",
                ".python-version",
                "release/models/baseline/manifest.json",
                "release/models/tcn/manifest.json",
            }
            if not required <= self.members.keys():
                raise ValueError("Release lacks locked dependencies or required model bundles.")
            wheels = [
                name for name in self.members if name.startswith("dist/") and name.endswith(".whl")
            ]
            if len(wheels) != 1:
                raise ValueError("Release must contain exactly one wheel.")
            self.wheel = wheels[0]

    def extract(self, destination: Path, *, draft: Path | None = None) -> None:
        """Write once; existing release directories must still match immutable input."""
        if digest(self.path) != self.sha256:
            raise ValueError("Release archive changed after verification.")
        destination.mkdir(parents=True, exist_ok=True, mode=0o700)
        scratch = (
            extraction_scratch(destination.parent / ".regimex-unpack")
            if draft is None
            else nullcontext(draft)
        )
        with scratch as partial, ZipFile(self.path) as archive:
            for name, member in self.members.items():
                target = destination / name
                parent = destination
                for part in PurePosixPath(name).parts[:-1]:
                    parent /= part
                    if parent.is_symlink():
                        raise ValueError("Staged release contains a symbolic link.")
                    parent.mkdir(exist_ok=True, mode=0o700)
                expected = (
                    hashlib.sha256(archive.read(member)).hexdigest()
                    if name == "SHA256SUMS"
                    else self.hashes[name]
                )
                if target.exists() or target.is_symlink():
                    if target.is_symlink() or not target.is_file() or digest(target) != expected:
                        raise ValueError(
                            "Existing staged release differs from its verified archive."
                        )
                else:
                    _publish_member(archive, member, target, expected, partial)
