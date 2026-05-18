import os

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
from typing import AsyncIterator
from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
import time
import logging
from fastapi.responses import JSONResponse
from ..IO.artifact_manager import ArtifactManager
from ..services.llm_client import LLMClient
from ..services.embedder import EmbeddingService
from ..services.evaluator import EvaluatorService
from ..tools.security import APISecurity, LLMAbuseProtection
from ..tools.resilience import CircuitBreaker, ConcurrencyLimiter
from ..tools.config_loader import load_auth_config, load_config, load_env
from ..tools.logging_setup import setup_bootstrap_logger, setup_logger
from ..tools.schemas import (
    APIStage,
    ErrorResponse,
    ErrorResponseError,
    LoggerLayer,
)
from ..tools.exceptions_schemas import (
    BaseAppError,
    ConfigurationError,
)
from ..tools.observabillity import TrackLatency, TrackToken
from ..pipelines.inference_pipeline import InferencePipeline
from ..pipelines.preprocess_pipeline import PreprocessPipeline
from ..api.routes.user_route import user_router
from ..api.routes.dev_route import dev_router
from ..api.middleware import BaseMiddleWare
from dataclasses import asdict

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    try:
        # bootstrap logger
        bootstrap_logger = setup_bootstrap_logger()

        # configuration
        app.state.settings = load_env()
        app.state.config = load_config()
        app.state.auth_config = load_auth_config()

        # logger setup
        setup_logger(
            level=app.state.config.logger.level,
            environment=app.state.settings.environment,
            save_log=app.state.config.logger.save_log,
            pipeline_name=APIStage.LIFESPAN,
        )

        # security and resilience
        app.state.api_security = APISecurity(auth_config=app.state.auth_config)
        app.state.circuit_breaker = CircuitBreaker(
            threshold=app.state.config.resilience.circuit_breaker.threshold_s,
            window_time=app.state.config.resilience.circuit_breaker.window_time_s,
        )
        app.state.llm_abuse_protection = LLMAbuseProtection(
            auth_config=app.state.auth_config,
            window_time=app.state.config.llm_protection.window_time_s,
            threshold=app.state.config.llm_protection.threshold_s,
            suspend_time=app.state.config.llm_protection.suspend_s,
        )
        app.state.concurrency_limiter = ConcurrencyLimiter(config=app.state.config)

        # telemetry
        app.state.track_latency = TrackLatency()
        app.state.track_token = TrackToken(llm_config=app.state.config.llm)
        app.state.start_time = time.perf_counter()
        app.state.total_request = 0

        # services
        app.state.llm_client = LLMClient(
            api_key=app.state.settings.oa_api_key,
            track_token=app.state.track_token,
            config=app.state.config,
            model=app.state.config.llm.model,
            circuit_breaker=app.state.circuit_breaker,
            concurrency_limiter=app.state.concurrency_limiter,
            llm_abuse_protection=app.state.llm_abuse_protection,
        )
        app.state.embedding_service = EmbeddingService(
            latency_store=app.state.track_latency,
            model_name=app.state.config.embedding.model,
            device=app.state.config.embedding.device,
        )
        app.state.evaluator_service = EvaluatorService(
            llm_client=app.state.llm_client,
            evaluation=app.state.config.evaluation,
            latency_store=app.state.track_latency,
        )

        # IO
        app.state.artifact_manager = ArtifactManager()

        # pipelines
        app.state.preprocess_pipeline = PreprocessPipeline(
            config=app.state.config,
            settings=app.state.settings,
            track_token=app.state.track_token,
            latency_store=app.state.track_latency,
            embedding_service=app.state.embedding_service,
            llm_client=app.state.llm_client,
            artifact_manager=app.state.artifact_manager,
            llm_abuse_protection=app.state.llm_abuse_protection,
        )
        app.state.inference_pipeline = InferencePipeline(
            config=app.state.config,
            settings=app.state.settings,
            track_latency=app.state.track_latency,
            track_token=app.state.track_token,
            llm_client=app.state.llm_client,
            evaluator_service=app.state.evaluator_service,
            embedding_service=app.state.embedding_service,
            preprocess_pipeline=app.state.preprocess_pipeline,
            artifact_manager=app.state.artifact_manager,
            llm_abuse_protection=app.state.llm_abuse_protection,
        )

        bootstrap_logger.warning(
            "Starting FastAPI Server",
            extra={
                "layer": LoggerLayer.PIPELINE,
                "environment": app.state.settings.environment,
                "logging_level": app.state.config.logger.level.value,
            },
        )

    except ConfigurationError as e:
        bootstrap_logger.error(str(e))
        raise

    except Exception:
        bootstrap_logger.critical(
            "Unexpected error occured",
            exc_info=True,
            extra={"layer": LoggerLayer.EXCEPTION, "stage": APIStage.LIFESPAN},
        )
        raise

    yield

    logger.warning(
        "Application Shutdown Summary",
        extra={
            "layer": LoggerLayer.PIPELINE,
            "stage": APIStage.LIFESPAN,
            "total_uptime": f"{round((time.perf_counter()-app.state.start_time),2)} s",
            "token_summary": asdict(app.state.track_token.get_all()),
        },
    )


app = FastAPI(lifespan=lifespan)
app.include_router(router=user_router)
app.include_router(router=dev_router)
app.add_middleware(BaseMiddleWare)


@app.exception_handler(BaseAppError)
async def base_app_error(request: Request, exc: BaseAppError) -> JSONResponse:
    error_response = ErrorResponse(
        status="error",
        request_id=request.state.request_id,
        error=ErrorResponseError(
            error_type=exc.error_type, code=exc.code, message=str(exc)
        ),
    )
    return JSONResponse(
        content=error_response.model_dump(), status_code=exc.status_code
    )


@app.exception_handler(Exception)
async def unknown_error(request: Request, exc: Exception):
    logger.critical(
        "Unexpected Error Occured",
        exc_info=True,
        extra={"layer": LoggerLayer.EXCEPTION, "stage": APIStage.LIFESPAN},
    )
    error_response = ErrorResponse(
        status="error",
        request_id=request.state.request_id,
        error=ErrorResponseError(
            error_type=exc.__class__.__name__,
            code="InternalError",
            message="Internal server error",
        ),
    )
    return JSONResponse(content=error_response.model_dump(), status_code=500)
