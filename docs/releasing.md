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

To include ready-to-register models, supply an asset pack containing the
baseline and TCN bundles.

```sh
uv run --locked python scripts/build_release.py --assets release
```

The builder copies the files listed in `release/SHA256SUMS` after verifying
each digest. Each line contains a SHA-256 digest, two spaces and a relative
filename. The pack can also include example exports and reports, comparison
results, evaluation data, and documents in DOCX or PDF format. For a source
distribution with selected examples, list just the assets you want to ship.

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
