# Building a release

Use a clean Git checkout with the Python and uv versions pinned in the project.

```sh
uv run --locked python scripts/build_release.py
```

The builder puts the tracked source, documentation, wheel and source distribution
in `release/build/regimex-<version>.zip`, next to a `.zip.sha256` file. Inside
the archive, `BUILD.json` records the source commit and build checks.
`SHA256SUMS` lists the payload files and their hashes.

## Include saved models, examples and editable documents

Pass `--assets` to include a verified pack in the release ZIP. A pack can
contain one or more model bundles, or just examples, evaluation results and
documents in DOCX or PDF format.

```sh
uv run --locked python scripts/build_release.py --assets release
```

The builder reads `SHA256SUMS` from the supplied directory. Each line
contains a SHA-256 digest, two spaces and a relative filename. List the files
to ship under `models/`, `examples/`, `comparison/`, `evaluation/` or
`documents/`. Include each model's manifest and all of its members when
shipping weights. The ZIP contains the verified files under `release/`.
The Python wheel and source distribution contain the package built from the
Git checkout.

Saved models, reports and measurements describe the versions that produced
them. Rebuild the relevant assets separately if their inputs or behavior change.

## Preview and verify

To package the current tracked changes locally, run:

```sh
uv run --locked python scripts/build_release.py --allow-dirty --output-dir release/preview
```

A preview includes edits to tracked files. Stage new files that belong in the
preview.

For version 0.1.0, verify the archive and its contents with:

```sh
cd release/build
sha256sum -c regimex-0.1.0.zip.sha256
unzip regimex-0.1.0.zip
cd regimex-0.1.0
sha256sum -c SHA256SUMS
```

When distributing an archive, give recipients its SHA-256 through a trusted
channel so they can verify their copy before use.
