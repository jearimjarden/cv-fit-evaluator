import os
import logging
import sys
import asyncio

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
from ..IO.artifact_manager import ArtifactManager
from ..pipelines.inference_pipeline import InferencePipeline
from ..pipelines.preprocess_pipeline import PreprocessPipeline
from ..services.llm_client import LLMClient
from ..services.embedder import EmbeddingService
from ..services.evaluator import EvaluatorService
from ..tools.security import LLMAbuseProtection
from ..tools.resilience import CircuitBreaker, ConcurrencyLimiter
from ..tools.exceptions_schemas import (
    BaseAppError,
    ConfigurationError,
    LoggedPipelineError,
)
from ..tools.observabillity import TrackLatency, TrackToken
from ..tools.logging_setup import setup_bootstrap_logger, setup_logger
from ..tools.config_loader import load_auth_config, load_config, load_env
from ..tools.schemas import AuthConfig, Config, Env, JRInput, LoggerLayer, PipelineStage

# Tester
test = {
    "text": "1. Strong proficiency in Python\n2. Ability to design and develop backend services\n3. Familiarity with containerization tools (e.g., Docker)\n4. Experience applying machine learning techniques to real problems\n5. Knowledge of relational databases and SQL"
}
test2 = {
    "text": "Develop scalable backend systems for production use\n2. Work with data-driven models and analytics pipelines\n- Collaborate using modern development tools and workflows\n4. Ensure system deployment and environment consistency\n5. Handle structured data storage and querying"
}


async def main(
    logger: logging.Logger, config: Config, settings: Env, auth_config: AuthConfig
) -> None:
    try:
        # telemetry
        track_latency = TrackLatency()
        track_token = TrackToken(llm_config=config.llm)

        # security and resilience
        llm_abuse_protection = LLMAbuseProtection(
            auth_config=auth_config,
            window_time=config.llm_protection.window_time_s,
            threshold=config.llm_protection.threshold_s,
            suspend_time=config.llm_protection.suspend_s,
        )
        circuit_breaker = CircuitBreaker(
            threshold=config.resilience.circuit_breaker.threshold_s,
            window_time=config.resilience.circuit_breaker.window_time_s,
        )
        concurrency_limiter = ConcurrencyLimiter(config=config)

        # services
        llm_client = LLMClient(
            api_key=settings.oa_api_key,
            track_token=track_token,
            config=config,
            model=config.llm.model,
            circuit_breaker=circuit_breaker,
            concurrency_limiter=concurrency_limiter,
            llm_abuse_protection=llm_abuse_protection,
        )
        embedding_service = EmbeddingService(
            device=config.embedding.device, latency_store=track_latency
        )
        evaluator_service = EvaluatorService(
            llm_client=llm_client,
            evaluation=config.evaluation,
            latency_store=track_latency,
        )

        # IO
        artifact_manager = ArtifactManager()

        # pipelines
        preprocess_pipeline = PreprocessPipeline(
            config=config,
            settings=settings,
            track_token=track_token,
            latency_store=track_latency,
            embedding_service=embedding_service,
            llm_client=llm_client,
            artifact_manager=artifact_manager,
            llm_abuse_protection=llm_abuse_protection,
        )
        pipeline = InferencePipeline.load_from_config(
            config=config,
            settings=settings,
            track_latency=track_latency,
            track_token=track_token,
            llm_client=llm_client,
            embedding_service=embedding_service,
            evaluator_service=evaluator_service,
            preprocess_pipeline=preprocess_pipeline,
            artifact_manager=artifact_manager,
            llm_abuse_protection=llm_abuse_protection,
        )

        logger.info(
            "Inference Pipeline created",
            extra={
                "layer": LoggerLayer.PIPELINE,
                "stage": PipelineStage.INFERENCE,
                "config": {
                    "input_mode": config.input_mode,
                    "cv_name": config.file_service.cv.file_name,
                    "jr_name": config.file_service.jr.file_name,
                    "query_top_k": config.retrieval.query_top_k,
                    "component_top_k": config.retrieval.component_top_k,
                    "retrieval_filter": config.retrieval.filter_below_threshold,
                    "evidence_mul": config.evaluation.evidence_mul,
                    "capability_mul": config.evaluation.capability_mul,
                    "responsibility_mul": config.evaluation.responsibility_mul,
                },
            },
        )

        report = await pipeline.run(
            candidate_name="ardi_pratama", jr_input=JRInput(**test), api_key="test"
        )
        # print(report)

        sys.exit(0)

    except LoggedPipelineError:
        logger.error(
            "Exiting Inference Pipeline",
            extra={"layer": LoggerLayer.EXCEPTION, "stage": PipelineStage.INFERENCE},
        )
        sys.exit(1)

    except BaseAppError as e:
        logger.error(
            str(e),
            extra={
                "layer": LoggerLayer.EXCEPTION,
                "stage": e.stage,
                "error_type": e.error_type,
                "code": e.code,
            },
        )

    except Exception:
        logger.critical(
            "Unexpected error occured",
            exc_info=True,
            extra={"layer": LoggerLayer.EXCEPTION, "stage": PipelineStage.INFERENCE},
        )
        sys.exit(2)


if __name__ == "__main__":
    try:
        bootstrap_logger = setup_bootstrap_logger()

        auth_config = load_auth_config()
        config = load_config()
        env = load_env()
        setup_logger(
            level=config.logger.level,
            environment=env.environment,
            pipeline_name=PipelineStage.INFERENCE,
            save_log=config.logger.save_log,
        )

        bootstrap_logger.info(
            "Starting Inference Pipeline",
            extra={
                "environment": env.environment,
                "logging_level": config.logger.level.value,
            },
        )
        logger = logging.getLogger(__name__)

        asyncio.run(
            main(logger=logger, config=config, settings=env, auth_config=auth_config)
        )

    except ConfigurationError as e:
        bootstrap_logger.error(str(e))
        sys.exit(1)

    except Exception:
        bootstrap_logger.critical("Unexpected error occured", exc_info=True)
        sys.exit(2)
