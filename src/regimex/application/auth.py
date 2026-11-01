"""HTTP-independent session and delegated-credential decisions."""

import hashlib
import hmac
from typing import Literal, cast

from pydantic import ValidationError

from regimex.auth_contracts import (
    AuthPolicy,
    BrowserPrincipal,
    BrowserSession,
    RequestAuth,
    SessionRecord,
    SessionView,
    canonical_origin,
)
from regimex.infrastructure.auth_repository import SESSION, AuthRepository, session_hash
from regimex.service_contracts import Forbidden, Principal, ServiceError, Unauthorized

_SESSION_NAMES = {"__Host-regimex_session", "regimex_session"}
_SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
_ASSERTION_HEADERS = ("x-regimex-subject", "x-regimex-run-id")
_UNTRUSTED_OWNER_HEADERS = (
    "x-external-user-id",
    "x-external-owner-id",
    "x-owner-id",
    "x-source-namespace",
)


def csrf_token(raw_session: str) -> str:
    session_hash(raw_session)
    return hmac.new(
        raw_session.encode("ascii"), b"regimex-session-csrf/1.0", hashlib.sha256
    ).hexdigest()


class AuthService:
    def __init__(self, repository: AuthRepository, policy: AuthPolicy) -> None:
        self.repository = repository
        self.policy = AuthPolicy.model_validate_json(policy.model_dump_json())

    @property
    def cookie_name(self) -> str:
        return self.policy.cookie_name

    @property
    def cookie_secure(self) -> bool:
        return self.policy.secure

    @property
    def session_ttl_seconds(self) -> int:
        return self.policy.session_ttl_seconds

    def _transport(self, request: RequestAuth, *, browser: bool) -> None:
        if len(request.headers) > 100 or sum(len(k) + len(v) for k, v in request.headers) > 8192:
            raise ServiceError(
                "capacity_limit", "Authentication headers exceed their size bound.", 431
            )
        for header in (
            "authorization",
            "origin",
            "x-csrf-token",
            "sec-fetch-site",
            *_ASSERTION_HEADERS,
        ):
            request.one(header)
        try:
            origin = canonical_origin(request.scheme + "://" + request.authority)
        except ValueError as exc:
            raise Forbidden("Authentication requires an approved HTTPS origin.") from exc
        if request.scheme != "https" and not (
            self.policy.allow_loopback_http and origin == self.policy.origin
        ):
            raise Forbidden("HTTPS is required outside explicitly enabled loopback development.")
        if browser and origin != self.policy.origin:
            raise Forbidden("Browser sessions are bound to the configured public origin.")
        for header in _UNTRUSTED_OWNER_HEADERS:
            if request.values(header):
                raise Forbidden(
                    "External ownership assertions require the dedicated integration API."
                )

    def _origin(self, request: RequestAuth, *, required: bool) -> None:
        supplied = request.one("origin")
        if supplied is None:
            if required:
                raise Forbidden("Cookie mutations require the configured Origin header.")
        else:
            try:
                actual = canonical_origin(supplied)
            except ValueError as exc:
                raise Forbidden("The request origin is not allowed.") from exc
            if actual != self.policy.origin:
                raise Forbidden("The request origin is not allowed.")
        fetch_site = request.one("sec-fetch-site")
        if fetch_site is not None and fetch_site not in {"same-origin", "none"}:
            raise Forbidden("Cross-origin browser authentication is unsupported.")

    @staticmethod
    def _cookie(request: RequestAuth) -> tuple[str, str] | None:
        credentials: list[tuple[str, str]] = []
        for header in request.values("cookie"):
            for item in header.split(";"):
                name, delimiter, value = item.strip().partition("=")
                if name in _SESSION_NAMES:
                    credentials.append((name, value if delimiter else ""))
        if len(credentials) > 1:
            raise ServiceError("invalid_request", "Ambiguous session cookies are unsupported.", 400)
        return credentials[0] if credentials else None

    @staticmethod
    def _bearer(value: str) -> str:
        parts = value.split(" ")
        if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1]:
            raise Unauthorized()
        if any(character.isspace() for character in parts[1]) or len(parts[1]) > 1024:
            raise Unauthorized()
        return parts[1]

    def _credentials(self, request: RequestAuth) -> tuple[str | None, str | None]:
        authorization = request.one("authorization")
        cookie = self._cookie(request)
        if authorization is not None and cookie is not None:
            raise ServiceError("invalid_request", "Use exactly one authentication credential.", 400)
        if cookie is not None:
            name, value = cookie
            if name != self.cookie_name or SESSION.fullmatch(value) is None:
                raise Unauthorized()
            return None, value
        if authorization is not None:
            return self._bearer(authorization), None
        raise Unauthorized()

    @staticmethod
    def _assertion_free(request: RequestAuth) -> None:
        if any(request.values(header) for header in _ASSERTION_HEADERS):
            raise Forbidden("Subject and run assertions require the dedicated integration API.")

    def _cookie_session(self, request: RequestAuth) -> tuple[str, SessionRecord]:
        self._transport(request, browser=True)
        self._assertion_free(request)
        raw_token, raw_session = self._credentials(request)
        if raw_token is not None or raw_session is None:
            raise Unauthorized()
        self._origin(request, required=request.method.upper() not in _SAFE_METHODS)
        if request.method.upper() not in _SAFE_METHODS:
            supplied = request.one("x-csrf-token")
            expected = csrf_token(raw_session)
            if (
                supplied is None
                or not supplied.isascii()
                or len(supplied) != 64
                or not hmac.compare_digest(supplied, expected)
            ):
                raise Forbidden("A matching session CSRF token is required.")
        return raw_session, self.repository.authenticate_session(raw_session)

    @staticmethod
    def _view(raw_session: str, record: SessionRecord) -> SessionView:
        if record.principal.role not in {"user", "admin"}:
            raise Unauthorized()
        return SessionView(
            principal=BrowserPrincipal(
                owner_id=record.principal.owner_id,
                role=cast(Literal["user", "admin"], record.principal.role),
            ),
            csrf_token=csrf_token(raw_session),
            expires_at=record.expires_at,
        )

    def login(self, raw_token: str, request: RequestAuth) -> BrowserSession:
        if request.method.upper() != "POST":
            raise ServiceError("invalid_request", "Session exchange requires POST.", 405)
        self._transport(request, browser=True)
        self._origin(request, required=True)
        self._assertion_free(request)
        if request.values("authorization") or self._cookie(request) is not None:
            raise ServiceError(
                "invalid_request", "Login accepts exactly one token from its JSON body.", 400
            )
        issued = self.repository.issue_session(
            raw_token,
            ttl_seconds=self.policy.session_ttl_seconds,
            maximum_sessions=self.policy.max_sessions_per_token,
        )
        return BrowserSession(
            issued.cookie_value,
            self._view(issued.cookie_value.get_secret_value(), issued.record),
        )

    def session(self, request: RequestAuth) -> SessionView:
        if request.method.upper() not in {"GET", "HEAD"}:
            raise ServiceError("invalid_request", "Session inspection requires GET.", 405)
        raw_session, record = self._cookie_session(request)
        return self._view(raw_session, record)

    def logout(self, request: RequestAuth) -> None:
        if request.method.upper() != "DELETE":
            raise ServiceError("invalid_request", "Session logout requires DELETE.", 405)
        raw_session, _record = self._cookie_session(request)
        self.repository.revoke_session(raw_session)

    def resolve(self, request: RequestAuth, *, delegated: bool = False) -> Principal:
        self._transport(request, browser=False)
        raw_token, raw_session = self._credentials(request)
        if delegated:
            if raw_session is not None or raw_token is None:
                raise Forbidden("Delegation requires the integration backend's bearer credential.")
            if request.one("origin") is not None or request.one("sec-fetch-site") is not None:
                raise Forbidden("The delegated integration API accepts backend requests only.")
            subject = request.one("x-regimex-subject")
            run_id = request.one("x-regimex-run-id")
            if subject is None or run_id is None:
                raise Forbidden(
                    "Delegation requires the verified external subject and run identity."
                )
            try:
                return self.repository.bind_subject(
                    raw_token,
                    subject,
                    run_id,
                    maximum_subjects=self.policy.max_subjects_per_integration,
                )
            except ValidationError as exc:
                raise ServiceError(
                    "invalid_request", "The asserted subject/run format is invalid.", 422
                ) from exc
        self._assertion_free(request)
        if raw_session is not None:
            return self._cookie_session(request)[1].principal
        assert raw_token is not None
        self._origin(request, required=False)
        return self.repository.database.authenticate(raw_token)
