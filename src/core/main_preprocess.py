import asyncio
import os
import sys
import logging

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
from ..IO.artifact_manager import ArtifactManager
from ..pipelines.preprocess_pipeline import PreprocessPipeline
from ..services.llm_client import LLMClient
from ..services.embedder import EmbeddingService
from ..tools.exceptions_schemas import (
    BaseAppError,
    ConfigurationError,
    LoggedPipelineError,
)
from ..tools.observabillity import TrackToken
from ..tools.security import LLMAbuseProtection
from ..tools.resilience import CircuitBreaker, ConcurrencyLimiter
from ..tools.observabillity import TrackLatency
from ..tools.logging_setup import setup_logger
from ..tools.schemas import AuthConfig, CVInput, Config, Env, LoggerLayer, PipelineStage
from ..tools.logging_setup import setup_bootstrap_logger
from ..tools.config_loader import load_auth_config, load_config, load_env

# Tester
test = {
    "text": "Name: Ardi Pratama\n\nProfessional Summary:\nBackend-oriented software engineer with experience in API development, automation, and machine learning projects. Interested in scalable systems, data processing, and practical AI applications. Comfortable working with Python-based tools and Linux environments.\n\nTechnical Skills:\nProgramming Languages:\n- Python\n- C++\n- SQL (basic)\n\nFrameworks / Backend:\n- FastAPI\n- Flask\n- REST API development\n\nMachine Learning & Data:\n- scikit-learn\n- TensorFlow (basic understanding)\n- Pandas\n- NumPy\n- Data preprocessing\n- Feature engineering\n\nTools & Platforms:\n- Git\n- Linux\n- JSON / CSV processing\n- Postman\n\nSoft Skills:\n- Problem-solving\n- Analytical thinking\n- Team collaboration\n- Communication\n- Adaptability\n- Time management\n- Attention to detail\n- Continuous learning\n\nLanguages:\n- Indonesian (Native)\n- English (Professional Working Proficiency)\n\nExperience:\nBackend Developer Intern\nPT Solusi Data Nusantara\nJan 2024 – Jun 2024\n\nResponsibilities:\n- Developed internal REST APIs using FastAPI\n- Created endpoints for data retrieval, filtering, and processing\n- Worked with JSON request/response workflows\n- Assisted debugging backend service issues\n- Collaborated with senior developers for feature integration\n\nAchievements:\n- Reduced manual reporting process by automating API-based data retrieval\n- Improved response structure consistency across several endpoints\n\nProjects:\n\nPersonal ML Classification Project\n- Built a machine learning classification pipeline using scikit-learn\n- Performed preprocessing, feature cleaning, and label preparation\n- Evaluated model performance using precision, recall, and F1-score\n- Compared Logistic Regression and Random Forest models\n\nAutomation & Data Processing Scripts\n- Created Python scripts for automated data collection and cleaning\n- Processed CSV and JSON datasets from multiple sources\n- Built reusable scripts for formatting and validation\n\nMini API Service Project\n- Built a small Flask-based API service for testing prediction requests\n- Implemented basic request validation and logging\n- Tested endpoints using Postman\n\nEducation:\nBachelor of Electrical Engineering\n\nAdditional Information:\n- Familiar with Linux command line workflows\n- Interested in AI engineering and backend systems\n- Experience using Git for version control"
}
test2 = {
    "text": "Name: Ardi Pratama\n\nProfessional Summary:\nBackend-oriented software engineer with experience in API development, automation, and machine learning projects. Interested in scalable systems, data processing, and practical AI applications. Comfortable working with Python-based tools and Linux environments."
}


async def main(
    logger: logging.Logger, config: Config, settings: Env, auth_config: AuthConfig
):
    try:
        # telemetry
        track_token = TrackToken(llm_config=config.llm)
        latency_store = TrackLatency()

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
        embedding_service = EmbeddingService(
            latency_store=latency_store, device=config.embedding.device
        )
        llm_client = LLMClient(
            api_key=settings.oa_api_key,
            track_token=track_token,
            config=config,
            model=config.llm.model,
            circuit_breaker=circuit_breaker,
            concurrency_limiter=concurrency_limiter,
            llm_abuse_protection=llm_abuse_protection,
        )

        # IO
        artifact_manager = ArtifactManager()

        # pipelines
        preprocess_pipeline = PreprocessPipeline.start_from_config(
            config=config,
            settings=settings,
            track_token=track_token,
            latency_store=latency_store,
            embedding_service=embedding_service,
            llm_client=llm_client,
            artifact_manager=artifact_manager,
            llm_abuse_protection=llm_abuse_protection,
        )

        logger.info(
            "Preprocess Pipeline created",
            extra={"layer": LoggerLayer.PIPELINE, "stage": PipelineStage.PREPROCESS},
        )

        validated_test = CVInput(**test)

        await preprocess_pipeline.run(
            cv_input=validated_test, candidate_name="ardi_pratama", api_key="test"
        )

        sys.exit(0)

    except LoggedPipelineError:
        logger.error(
            "Exiting Preprocess Pipeline",
            extra={"layer": LoggerLayer.EXCEPTION, "stage": PipelineStage.PREPROCESS},
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
            extra={"layer": LoggerLayer.EXCEPTION, "stage": PipelineStage.PREPROCESS},
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
            pipeline_name=PipelineStage.PREPROCESS,
            save_log=config.logger.save_log,
        )

        bootstrap_logger.info(
            "Starting Preprocess Pipeline",
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
