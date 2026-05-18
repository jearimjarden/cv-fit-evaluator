from typing import Annotated
from fastapi import APIRouter, Depends, status
from ...IO.artifact_manager import ArtifactManager
from ..dependencies import (
    get_artifact_manager,
    get_public_config,
    get_public_health,
    run_api_security,
    get_telemetry,
)
from ...tools.schemas import (
    CandidateList,
    Metadata,
    PublicConfig,
    PublicHealth,
    Telemetry,
)

dev_router = APIRouter()


@dev_router.get(
    "/health",
    status_code=status.HTTP_200_OK,
)
def get_health(
    public_health: Annotated[PublicHealth, Depends(get_public_health)],
) -> PublicHealth:

    return public_health


@dev_router.get("/telemetry", status_code=status.HTTP_200_OK)
def telemetry(
    telemetry: Annotated[Telemetry, Depends(get_telemetry)],
    _: Annotated[None, Depends(run_api_security)],
) -> Telemetry:
    return telemetry


@dev_router.get("/candidates", status_code=status.HTTP_200_OK)
def get_candidates_list(
    artifact_manager: Annotated[ArtifactManager, Depends(get_artifact_manager)],
    _: Annotated[None, Depends(run_api_security)],
) -> CandidateList:
    return artifact_manager.list_candidates()


@dev_router.get("/candidates/{candidate_name}/metadata", status_code=status.HTTP_200_OK)
def get_candidate_metadata(
    candidate_name: str,
    artifact_manager: Annotated[ArtifactManager, Depends(get_artifact_manager)],
    _: Annotated[None, Depends(run_api_security)],
) -> Metadata:
    return artifact_manager.get_candidate_metadata(candidate_name=candidate_name)


@dev_router.get("/config", status_code=status.HTTP_200_OK)
def get_config(
    public_config: Annotated[PublicConfig, Depends(get_public_config)],
    _: Annotated[None, Depends(run_api_security)],
) -> PublicConfig:
    return public_config
