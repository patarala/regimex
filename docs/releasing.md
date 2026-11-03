# Building a release

From a committed checkout, run:

```sh
uv run --locked python scripts/build_release.py
```

The command builds a wheel and source distribution, checks that every runtime
resource is present in the wheel, and packages them with the tracked source in
`release/build/regimex-<version>.zip`. It also writes a `.zip.sha256` checksum file.
Git, the project's Python version and its pinned uv version must be installed.
`uv run --locked` supplies the locked build dependencies.

The archive contains a `regimex-<version>/` directory with source, documentation,
`dist/`, `BUILD.json` and `SHA256SUMS`. It excludes Git history, untracked files,
runtime data and credentials. `BUILD.json` records the source commit, individual
file hashes, a source content digest and the checks actually performed during
assembly. A package build does not replace the test suite or deployment checks.

## Include saved models, examples and editable documents

Generated assets are separate from Git. To include an existing asset directory:

```sh
uv run --locked python scripts/build_release.py --assets release
```

The input directory must contain `SHA256SUMS` in the standard
`<sha256><two spaces><relative filename>` format. Every listed file is required
and must match its checksum. Supported content includes models, examples, model
comparison reports, evaluation data and documentation in DOCX/PDF. Files absent
from the checksum list are not included. Links, unsafe paths and unapproved file
types are rejected. A partial pack is supported: the builder verifies exactly
the listed content and does not claim that missing models or documents are
present. Do not put credentials in an asset pack.

The verified assets appear under `release/` in the archive. Their existing
reports, measurements and verification evidence retain their original scope;
they do not certify the new source revision. This command copies documents and
saved models unchanged. It does not train models, rewrite editable documents or
rerun scientific evaluation. Revalidate affected assets separately when model
or report behavior changes.

## Preview and verify

Tracked changes must be committed for a normal release. For a local preview:

```sh
uv run --locked python scripts/build_release.py --allow-dirty --output-dir release/preview
```

Add new intended source files to Git before a preview; untracked files are never
included. The manifest marks uncommitted changes. Package construction uses a
temporary source snapshot, so generated build files do not modify the checkout.
Repeated builds with the same source, assets and toolchain use stable archive
ordering and timestamps. Rebuilding replaces only the matching archive and its
checksum in the output directory; asset inputs are not changed.

Verify the archive before unpacking, then verify its contents:

```sh
cd release/build
sha256sum -c regimex-0.1.0.zip.sha256
unzip regimex-0.1.0.zip
cd regimex-0.1.0
sha256sum -c SHA256SUMS
```

Choose another destination with `--output-dir`. The builder requires a Git
checkout to establish source membership and identity; use the wheel or source
distribution in `dist/` to install a downloaded release.
