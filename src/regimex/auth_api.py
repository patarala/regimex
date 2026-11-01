"""Optional session routes; no global application mutation on import."""

import json
from datetime import UTC

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from pydantic import Field, SecretStr, ValidationError
from starlette.concurrency import run_in_threadpool

from regimex.application.auth import AuthService
from regimex.auth_contracts import RequestAuth, SessionView
from regimex.contracts import StrictModel
from regimex.service_contracts import ServiceError, Unauthorized

_PRIVATE = {"Cache-Control": "private, no-store", "Pragma": "no-cache"}


class TokenExchange(StrictModel):
    token: SecretStr = Field(min_length=47, max_length=47)


def _unique_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ServiceError(
                "invalid_request", "Ambiguous token exchange fields are unsupported.", 400
            )
        result[key] = value
    return result


def request_auth(request: Request) -> RequestAuth:
    return RequestAuth(
        method=request.method,
        scheme=request.url.scheme,
        authority=request.url.netloc,
        headers=tuple(request.headers.raw),
    )


def _clear(response: Response, service: AuthService) -> None:
    response.delete_cookie(
        key=service.cookie_name,
        path="/",
        secure=service.cookie_secure,
        httponly=True,
        samesite="strict",
    )


def _error(error: ServiceError) -> JSONResponse:
    return JSONResponse(
        {"error": {"code": error.code, "message": error.message}},
        status_code=error.status_code,
        headers=_PRIVATE,
    )


def install_session_routes(app: FastAPI, service: AuthService) -> None:
    if hasattr(app.state, "request_models"):
        app.state.request_models += (TokenExchange,)
    request_body = {
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "schema": TokenExchange.model_json_schema(
                        ref_template="#/components/schemas/{model}"
                    )
                }
            },
        }
    }

    @app.post("/api/v1/session", response_model=SessionView, openapi_extra=request_body)
    async def login(request: Request) -> Response:
        if request.headers.get("content-type", "").split(";", 1)[0].lower() != "application/json":
            return _error(ServiceError("invalid_request", "Login requires application/json.", 415))
        data = bytearray()
        async for block in request.stream():
            data.extend(block)
            if len(data) > 1024:
                return _error(
                    ServiceError("capacity_limit", "Login JSON exceeds its size bound.", 413)
                )
        try:
            exchange = TokenExchange.model_validate(
                json.loads(data, object_pairs_hook=_unique_fields), strict=True
            )
            issued = await run_in_threadpool(
                service.login, exchange.token.get_secret_value(), request_auth(request)
            )
        except ValidationError, json.JSONDecodeError, UnicodeDecodeError:
            return _error(
                ServiceError("invalid_data", "A bounded provisioned token is required.", 422)
            )
        except ServiceError as error:
            return _error(error)
        response = JSONResponse(issued.view.model_dump(mode="json"), headers=_PRIVATE)
        response.set_cookie(
            key=service.cookie_name,
            value=issued.cookie_value.get_secret_value(),
            path="/",
            max_age=service.session_ttl_seconds,
            expires=issued.view.expires_at.astimezone(UTC),
            httponly=True,
            secure=service.cookie_secure,
            samesite="strict",
        )
        return response

    @app.get("/api/v1/session", response_model=SessionView)
    def session(request: Request) -> Response:
        try:
            view = service.session(request_auth(request))
        except ServiceError as error:
            response = _error(error)
            if isinstance(error, Unauthorized):
                _clear(response, service)
            return response
        return JSONResponse(view.model_dump(mode="json"), headers=_PRIVATE)

    @app.delete("/api/v1/session", status_code=204)
    def logout(request: Request) -> Response:
        response: Response
        try:
            service.logout(request_auth(request))
        except ServiceError as error:
            response = _error(error)
            if isinstance(error, Unauthorized):
                _clear(response, service)
            return response
        response = Response(status_code=204, headers=_PRIVATE)
        _clear(response, service)
        return response
