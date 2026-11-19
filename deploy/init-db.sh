#!/bin/sh
set -eu
# psql reads this transient environment value without placing a password in argv.
REGIMEX_INITIAL_APP_PASSWORD=$(cat /run/secrets/application_password)
if [ -z "$REGIMEX_INITIAL_APP_PASSWORD" ] || [ "$REGIMEX_INITIAL_APP_PASSWORD" = "$POSTGRES_PASSWORD" ]; then
    echo 'Provide a nonempty application password distinct from the database administrator password.' >&2
    exit 1
fi
export REGIMEX_INITIAL_APP_PASSWORD
psql --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" --no-psqlrc --set=ON_ERROR_STOP=1 <<'SQL'
\getenv app_password REGIMEX_INITIAL_APP_PASSWORD
BEGIN;
CREATE ROLE regimex LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION PASSWORD :'app_password';
ALTER DATABASE regimex OWNER TO regimex;
-- PostgreSQL 16's pg_database_owner gives the app control of public and leaves
-- the fresh database eligible for an offline restore before migrations run.
REVOKE ALL ON DATABASE regimex FROM PUBLIC;
REVOKE CREATE ON SCHEMA public FROM PUBLIC;
COMMIT;
SQL
unset REGIMEX_INITIAL_APP_PASSWORD
