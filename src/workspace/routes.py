"""HTTP routes for accounts and private workspaces."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NoReturn, TypeVar

from fastapi import APIRouter, Header, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field

from src.accounts.actor import Actor
from src.workspace.errors import WorkspaceError
from src.workspace.runtime import runtime
from src.workspace.service import (
    apply_session_cookie,
    cancel_job,
    clear_session_cookie,
    delete_dataset,
    enqueue_starter,
    get_dataset,
    get_job,
    get_prediction,
    limits_for,
    list_configs,
    list_datasets,
    list_jobs,
    list_models,
    list_predictions,
    list_starter_cards,
    login,
    login_owner,
    logout,
    predict_model,
    predict_starter,
    replace_dataset,
    resolve_actor,
    signup,
    update_config,
    upload_dataset,
)

router = APIRouter()
T = TypeVar("T")


class SignupRequest(BaseModel):
    email: str
    password: str = Field(min_length=1)


class ConfigUpdate(BaseModel):
    body: str = Field(min_length=1)


class PredictBody(BaseModel):
    model_id: str
    sequence: list[list[float]]


def _raise(exc: WorkspaceError) -> NoReturn:
    headers = {"WWW-Authenticate": "Bearer"} if exc.status == 401 else None
    raise HTTPException(
        status_code=exc.status, detail=exc.detail, headers=headers
    ) from exc


def _actor(request: Request, authorization: str | None) -> Actor:
    try:
        return resolve_actor(request, authorization)
    except WorkspaceError as exc:
        _raise(exc)


def _call(action: Callable[[], T]) -> T:
    try:
        return action()
    except WorkspaceError as exc:
        _raise(exc)


@router.post("/auth/signup", status_code=201, tags=["accounts"])
def auth_signup(
    body: SignupRequest, request: Request, response: Response
) -> dict[str, str]:
    try:
        actor, token = signup(body.email, body.password)
    except WorkspaceError as exc:
        _raise(exc)
    apply_session_cookie(response, request, token)
    return actor.public()


@router.post("/auth/login", tags=["accounts"])
def auth_login(
    body: SignupRequest, request: Request, response: Response
) -> dict[str, str]:
    try:
        actor, token = login(body.email, body.password)
    except WorkspaceError as exc:
        _raise(exc)
    apply_session_cookie(response, request, token)
    return actor.public()


@router.post("/auth/owner", tags=["accounts"])
def auth_owner(
    request: Request,
    response: Response,
    authorization: str | None = Header(default=None),
) -> dict[str, str]:
    actor = _actor(request, authorization)
    if not actor.is_owner or actor.via != "bearer":
        _raise(WorkspaceError(401, "Invalid or missing API token."))
    try:
        session_actor, token = login_owner()
    except WorkspaceError as exc:
        _raise(exc)
    apply_session_cookie(response, request, token)
    return session_actor.public()


@router.post("/auth/logout", tags=["accounts"])
def auth_logout(request: Request, response: Response) -> dict[str, str]:
    logout(request)
    clear_session_cookie(response, request)
    return {"status": "signed-out"}


@router.get("/auth/me", tags=["accounts"])
def auth_me(
    request: Request,
    authorization: str | None = Header(default=None),
) -> dict[str, str]:
    actor = _actor(request, authorization)
    return actor.public()


@router.get("/workspace/limits", tags=["workspace"])
def workspace_limits(
    request: Request,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    _actor(request, authorization)
    return {"limits": limits_for(runtime.config).as_dict()}


@router.get("/workspace/starters", tags=["workspace"])
def workspace_starters(
    request: Request,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    actor = _actor(request, authorization)
    return _call(lambda: list_starter_cards(actor))


@router.post("/workspace/starters/{starter_id}/predict", tags=["workspace"])
def workspace_starter_predict(
    starter_id: str,
    request: Request,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    actor = _actor(request, authorization)
    config = _app_config()
    return _call(lambda: predict_starter(actor, starter_id, config))


@router.post("/workspace/starters/{starter_id}/train", tags=["workspace"])
def workspace_starter_train(
    starter_id: str,
    request: Request,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    actor = _actor(request, authorization)
    config = _app_config()
    return _call(lambda: enqueue_starter(actor, starter_id, config))


@router.get("/workspace/datasets", tags=["workspace"])
def workspace_datasets(
    request: Request,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    actor = _actor(request, authorization)
    return _call(lambda: list_datasets(actor))


@router.post("/workspace/datasets", status_code=201, tags=["workspace"])
async def workspace_upload_dataset(
    request: Request,
    name: str = Query(default="dataset.csv"),
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    actor = _actor(request, authorization)
    data = await _read_upload(request)
    config = _app_config()
    return _call(lambda: upload_dataset(actor, name, data, config))


@router.get("/workspace/datasets/{dataset_id}", tags=["workspace"])
def workspace_get_dataset(
    dataset_id: str,
    request: Request,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    actor = _actor(request, authorization)
    return _call(lambda: get_dataset(actor, dataset_id))


@router.put("/workspace/datasets/{dataset_id}", tags=["workspace"])
async def workspace_replace_dataset(
    dataset_id: str,
    request: Request,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    actor = _actor(request, authorization)
    data = await _read_upload(request)
    return _call(lambda: replace_dataset(actor, dataset_id, data, _app_config()))


@router.delete("/workspace/datasets/{dataset_id}", tags=["workspace"])
def workspace_delete_dataset(
    dataset_id: str,
    request: Request,
    authorization: str | None = Header(default=None),
) -> dict[str, str]:
    actor = _actor(request, authorization)
    _call(lambda: delete_dataset(actor, dataset_id))
    return {"status": "deleted"}


@router.get("/workspace/configs", tags=["workspace"])
def workspace_configs(
    request: Request,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    actor = _actor(request, authorization)
    return _call(lambda: list_configs(actor))


@router.put("/workspace/configs/{config_id}", tags=["workspace"])
def workspace_update_config(
    config_id: str,
    body: ConfigUpdate,
    request: Request,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    actor = _actor(request, authorization)
    return _call(lambda: update_config(actor, config_id, body.body, _app_config()))


@router.get("/workspace/models", tags=["workspace"])
def workspace_models(
    request: Request,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    actor = _actor(request, authorization)
    return _call(lambda: list_models(actor))


@router.get("/workspace/jobs", tags=["workspace"])
def workspace_jobs(
    request: Request,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    actor = _actor(request, authorization)
    return _call(lambda: list_jobs(actor))


@router.get("/workspace/jobs/{job_id}", tags=["workspace"])
def workspace_get_job(
    job_id: str,
    request: Request,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    actor = _actor(request, authorization)
    return _call(lambda: get_job(actor, job_id))


@router.delete("/workspace/jobs/{job_id}", tags=["workspace"])
def workspace_cancel_job(
    job_id: str,
    request: Request,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    actor = _actor(request, authorization)
    return _call(lambda: cancel_job(actor, job_id))


@router.get("/workspace/predictions", tags=["workspace"])
def workspace_predictions(
    request: Request,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    actor = _actor(request, authorization)
    return _call(lambda: list_predictions(actor))


@router.get("/workspace/predictions/{prediction_id}", tags=["workspace"])
def workspace_get_prediction(
    prediction_id: str,
    request: Request,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    actor = _actor(request, authorization)
    return _call(lambda: get_prediction(actor, prediction_id))


@router.post("/workspace/predictions", tags=["workspace"])
def workspace_predict(
    body: PredictBody,
    request: Request,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    actor = _actor(request, authorization)
    return _call(
        lambda: predict_model(actor, body.model_id, body.sequence, _app_config())
    )


async def _read_upload(request: Request) -> bytes:
    # One byte past the hard ceiling so the configurable limit can still say 413.
    return await request.body()


def _app_config() -> dict[str, Any] | None:
    config = runtime.config
    return config if isinstance(config, dict) else None
