#!/bin/sh
set -e
# psql reads this transient environment value without placing a password in argv.
REGIMEX_INITIAL_APP_PASSWORD=$(cat /run/secrets/application_password)
export REGIMEX_INITIAL_APP_PASSWORD
psql --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" --no-psqlrc --set=ON_ERROR_STOP=1 <<'SQL'
\getenv app_password REGIMEX_INITIAL_APP_PASSWORD
CREATE ROLE regimex LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION PASSWORD :'app_password';
ALTER DATABASE regimex OWNER TO regimex;
ALTER SCHEMA public OWNER TO regimex;
REVOKE ALL ON DATABASE regimex FROM PUBLIC;
REVOKE CREATE ON SCHEMA public FROM PUBLIC;
SQL
unset REGIMEX_INITIAL_APP_PASSWORD
