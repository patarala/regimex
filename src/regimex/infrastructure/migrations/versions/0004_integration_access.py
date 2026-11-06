"""Manage integration identities independently of their credentials."""

from typing import cast

from alembic import op

revision = "0004_integration_access"
down_revision = "0003_experiment_names"
branch_labels = None
depends_on = None

_OWNER_OVERLAP = """SELECT EXISTS (
            SELECT 1 FROM regimex_tokens integration
            JOIN regimex_tokens local ON local.owner_id=integration.owner_id
            WHERE integration.role='integration' AND local.role<>'integration'
        ) OR EXISTS (
            SELECT 1 FROM regimex_integration_subjects subject JOIN regimex_tokens token
            ON token.owner_id=subject.effective_owner_id
        ) OR EXISTS (
            SELECT 1 FROM regimex_integration_subjects subject
            JOIN regimex_integration_subjects account
            ON subject.effective_owner_id=account.integration_owner_id
        ) OR EXISTS (
            SELECT 1 FROM regimex_integration_subjects subject
            JOIN regimex_tokens local ON local.owner_id=subject.integration_owner_id
            WHERE local.role<>'integration'
        )"""

STATEMENTS = (
    """CREATE TABLE regimex_integrations (
        integration_id uuid PRIMARY KEY,
        owner_id text NOT NULL, source_namespace text NOT NULL,
        display_name text NOT NULL CHECK (char_length(display_name) BETWEEN 1 AND 120),
        enabled boolean NOT NULL DEFAULT true,
        revision bigint NOT NULL DEFAULT 0 CHECK (revision >= 0),
        created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
        updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
        UNIQUE (owner_id, source_namespace),
        UNIQUE (integration_id, owner_id, source_namespace))""",
    """INSERT INTO regimex_integrations
        (integration_id,owner_id,source_namespace,display_name,created_at,updated_at)
        SELECT gen_random_uuid(),owner_id,source_namespace,left(source_namespace,120),
               min(created_at),clock_timestamp()
        FROM (
            SELECT owner_id,source_namespace,created_at FROM regimex_tokens
            WHERE role='integration'
            UNION ALL
            SELECT integration_owner_id,source_namespace,created_at
            FROM regimex_integration_subjects
        ) identities GROUP BY owner_id,source_namespace""",
    """ALTER TABLE regimex_tokens
        ADD COLUMN integration_id uuid,
        ADD COLUMN name text,
        ADD COLUMN expires_at timestamptz,
        ADD COLUMN last_used_at timestamptz,
        ADD COLUMN scopes text[] NOT NULL DEFAULT '{}'""",
    """UPDATE regimex_tokens t SET integration_id=i.integration_id,
        name='Provisioned key',scopes=ARRAY['read','upload','analyze']
        FROM regimex_integrations i WHERE t.role='integration'
        AND t.owner_id=i.owner_id AND t.source_namespace=i.source_namespace""",
    """ALTER TABLE regimex_tokens
        ADD CONSTRAINT regimex_token_integration_identity
            FOREIGN KEY (integration_id,owner_id,source_namespace)
            REFERENCES regimex_integrations(integration_id,owner_id,source_namespace),
        ADD CONSTRAINT regimex_token_integration_role
            CHECK ((role='integration') = (integration_id IS NOT NULL)),
        ADD CONSTRAINT regimex_token_name
            CHECK ((role='integration' AND name IS NOT NULL
                    AND char_length(name) BETWEEN 1 AND 120)
                   OR (role<>'integration' AND name IS NULL)),
        ADD CONSTRAINT regimex_token_expiry
            CHECK (expires_at IS NULL OR expires_at > created_at),
        ADD CONSTRAINT regimex_token_scopes
            CHECK (array_position(scopes,NULL) IS NULL
                   AND scopes <@ ARRAY['read','upload','analyze']::text[]
                   AND ((role='integration' AND cardinality(scopes) BETWEEN 1 AND 3)
                        OR (role<>'integration' AND cardinality(scopes)=0)))""",
    """CREATE INDEX regimex_tokens_integration
        ON regimex_tokens(integration_id,created_at DESC,token_id)""",
    """CREATE TABLE regimex_access_events (
        event_id uuid PRIMARY KEY,
        integration_id uuid NOT NULL REFERENCES regimex_integrations(integration_id),
        created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
        actor_owner_id text, actor_token_id uuid,
        action text NOT NULL CHECK (action IN
            ('integration.created','integration.updated','key.created','key.revoked')),
        target_token_id uuid, enabled boolean NOT NULL,
        display_name text NOT NULL CHECK (char_length(display_name) BETWEEN 1 AND 120),
        CHECK (actor_owner_id IS NULL OR char_length(actor_owner_id) BETWEEN 1 AND 128))""",
    """CREATE INDEX regimex_access_events_recent
        ON regimex_access_events(integration_id,created_at DESC,event_id DESC)""",
)


def upgrade() -> None:
    overlap = cast(bool, op.get_bind().exec_driver_sql(_OWNER_OVERLAP).scalar_one())
    if overlap:
        raise RuntimeError(
            "Local, integration and delegated account identities overlap; "
            "separate their credentials and ownership before upgrading."
        )
    for statement in STATEMENTS:
        op.execute(statement)


def downgrade() -> None:
    raise RuntimeError("Database downgrades are not supported.")
