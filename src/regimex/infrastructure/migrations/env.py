"""Packaged migrations run only with the connection supplied by the operator."""

from alembic import context

connection = context.config.attributes["connection"]
schema = context.config.attributes["schema"]
context.configure(connection=connection, version_table_schema=schema)
with context.begin_transaction():
    context.run_migrations()
