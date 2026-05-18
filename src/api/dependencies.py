from fastapi import Request, Header
import time
import logging
from ..tools.exceptions_schemas import MissingAPIKey
from ..tools.schemas import APIStage, LoggerLayer, PublicConfig, PublicHealth, Telemetry
from ..IO.artifact_manager import ArtifactManager
from ..pipelines.inference_pipeline import InferencePipeline
from ..pipelines.preprocess_pipeline import PreprocessPipeline

logger = logging.getLogger(__name__)


def get_inference_pipeline(
    request: Request,
) -> InferencePipeline:
    return request.app.state.inference_pipeline


def get_preprocess_pipeline(request: Request) -> PreprocessPipeline:
    return request.app.state.preprocess_pipeline


def get_artifact_manager(request: Request) -> ArtifactManager:
    return request.app.state.artifact_manager


def get_public_config(request: Request) -> PublicConfig:
    return PublicConfig(
        embedding_device=request.app.state.config.embedding.device,
        embedding_model=request.app.state.config.embedding.model,
        llm_model=request.app.state.config.llm.model,
        query_top_k=request.app.state.config.retrieval.query_top_k,
        component_top_k=request.app.state.config.retrieval.component_top_k,
        retrieval_threshold=request.app.state.config.retrieval.threshold,
        filter_below_threshold=request.app.state.config.retrieval.filter_below_threshold,
    )


def run_api_security(request: Request, api_key: str = Header(...)) -> str:
    if api_key is None:
        logger.error(
            "Missing API key",
            extra={"layer": LoggerLayer.MIDDLEWARE, "stage": APIStage.LIFESPAN},
        )
        raise MissingAPIKey(("Missing API Key"))
    request.app.state.api_security.run(api_key=api_key, request_url=request.url.path)
    return api_key


def run_llm_abuse_protection(request: Request, api_key: str = Header(...)) -> None:
    request.app.state.llm_abuse_protection.check(api_key=api_key)


def get_public_health(request: Request) -> PublicHealth:
    return PublicHealth(status="ok", environment=request.app.state.settings.environment)


def get_telemetry(request: Request) -> Telemetry:
    return Telemetry(
        uptime=round((time.perf_counter() - request.app.state.start_time), 2),
        token_summary=request.app.state.track_token.get_all(),
        total_request=request.app.state.total_request,
    )
