FROM python:3.14.4-slim-bookworm@sha256:fc74d22ffd0d5ac395a4b7bdda75a4539758862c49ebf3005647084631e63789 AS builder
COPY --from=ghcr.io/astral-sh/uv:0.13.0@sha256:cdc6093146eb3ff6a40107b38f008b789e050e77ad87865e381d9917da55a168 /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_NO_CACHE=1
WORKDIR /build
COPY pyproject.toml uv.lock README.md .python-version ./
RUN uv sync --locked --only-group build --no-install-project
COPY src ./src
RUN uv build --no-build-isolation --wheel

FROM python:3.14.4-slim-bookworm@sha256:fc74d22ffd0d5ac395a4b7bdda75a4539758862c49ebf3005647084631e63789 AS locked-environment
COPY --from=ghcr.io/astral-sh/uv:0.13.0@sha256:cdc6093146eb3ff6a40107b38f008b789e050e77ad87865e381d9917da55a168 /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_NO_CACHE=1 UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /build
COPY pyproject.toml uv.lock README.md .python-version ./

FROM locked-environment AS api-environment
RUN uv sync --locked --no-default-groups --extra service --extra s3 --no-install-project
COPY --from=builder /build/dist /tmp/wheels
RUN uv pip install --python /opt/venv/bin/python --no-deps /tmp/wheels/*.whl

FROM locked-environment AS cpu-environment
RUN uv sync --locked --no-default-groups --extra service --extra s3 --extra ml --extra viz --no-install-project
COPY --from=builder /build/dist /tmp/wheels
RUN uv pip install --python /opt/venv/bin/python --no-deps /tmp/wheels/*.whl

FROM python:3.14.4-slim-bookworm@sha256:fc74d22ffd0d5ac395a4b7bdda75a4539758862c49ebf3005647084631e63789 AS service-base
ENV PATH="/opt/venv/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN useradd --uid 10001 --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin regimex \
    && mkdir -p /data/objects /work /coordination /operator-output \
    && chown -R regimex:regimex /data /work /coordination /operator-output
USER 10001:10001
WORKDIR /work
ENTRYPOINT ["regimex"]
CMD ["--help"]

FROM service-base AS api
COPY --from=api-environment /opt/venv /opt/venv

FROM service-base AS worker
COPY --from=cpu-environment /opt/venv /opt/venv

# Matching-major native clients come from the exact deployed PostgreSQL image.
# Their private non-glibc libraries affect only the native tool subprocess.
FROM postgres:16.15-bookworm@sha256:d81f7b51697014367236d07ec58b54153fc28f828b72bf247fe5c7487520bcb0 AS postgres-clients
RUN mkdir -p /opt/pg16/bin /opt/pg16/lib \
    && cp /usr/lib/postgresql/16/bin/pg_dump /usr/lib/postgresql/16/bin/pg_restore /opt/pg16/bin/ \
    && for client in /opt/pg16/bin/*; do \
         for library in $(ldd "$client" | awk '/=> \// {print $3}'); do \
           case "$library" in \
             */libc.so.*|*/libm.so.*|*/libpthread.so.*|*/libresolv.so.*|*/librt.so.*|*/libdl.so.*) ;; \
             *) cp -L "$library" /opt/pg16/lib/ ;; \
           esac; \
         done; \
       done

FROM worker AS ops
COPY --from=postgres-clients /opt/pg16 /opt/pg16
COPY --chmod=755 deploy/pg_dump deploy/pg_restore /opt/pg16/
ENV PGDUMP_BIN=/opt/pg16/pg_dump PGRESTORE_BIN=/opt/pg16/pg_restore
