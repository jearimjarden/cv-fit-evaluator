from typing import TypeVar, Type
from pydantic import ValidationError
import logging
from dataclasses import asdict
from ..IO.artifact_manager import ArtifactManager
from ..services.embedder import EmbeddingService
from ..services.chunker import chunk_cv_semantic
from ..services.llm_client import LLMClient
from ..services.parser import parse_cv_llm
from ..tools.security import LLMAbuseProtection
from ..tools.exceptions_schemas import (
    InvalidParsedCV,
    LLMInvalidSchemas,
    PreprocessError,
    InvalidCVLength,
    LLMError,
    LoggedPipelineError,
    ArtifactError,
)
from ..tools.observabillity import TrackLatency, TrackToken
from ..tools.schemas import (
    CVChunk,
    CVEmbedding,
    CVInput,
    Env,
    Config,
    LatencyStored,
    LoggerLayer,
    PreprocessStage,
    PipelineStage,
    StructuredCV,
    TokenSummary,
)
from ..tools.observabillity import track_latency

logger = logging.getLogger(__name__)


class PreprocessPipeline:
    def __init__(
        self,
        config: Config,
        settings: Env,
        track_token: TrackToken,
        latency_store: TrackLatency,
        embedding_service: EmbeddingService,
        llm_client: LLMClient,
        artifact_manager: ArtifactManager,
        llm_abuse_protection: LLMAbuseProtection,
    ) -> None:
        self.settings = settings
        self.config = config
        self.track_token = track_token
        self.latency_store = latency_store
        self.llm_client = llm_client
        self.embedding_service = embedding_service
        self.artifact_manager = artifact_manager
        self.llm_abuse_protection = llm_abuse_protection

    async def run(self, cv_input: CVInput, candidate_name: str, api_key: str) -> None:
        self.api_key = api_key
        self.request_track_token = TrackToken(llm_config=self.config.llm)

        try:
            await self.preprocess_cv(cv_input=cv_input, candidate_name=candidate_name)
            logger.info(
                "CV Created",
                extra={
                    "layer": LoggerLayer.PIPELINE,
                    "stage": PipelineStage.PREPROCESS,
                    "latencies": self.latency_stored.latencies_ms,
                },
            )

            logger.info(
                "Token Tracked",
                extra={
                    "layer": LoggerLayer.PIPELINE,
                    "stage": PipelineStage.PREPROCESS,
                    "summary": asdict(self.request_track_token.get_all()),
                },
            )

        except (PreprocessError, LLMError, ArtifactError) as e:
            logger.error(
                str(e),
                extra={
                    "layer": LoggerLayer.EXCEPTION,
                    "stage": e.stage,
                    "error_type": e.error_type,
                    "code": e.code,
                },
            )

            raise LoggedPipelineError(
                str(e),
                stage=e.stage,
                error_type=e.error_type,
                code=e.code,
                status_code=e.status_code,
            ) from e

    @track_latency(stage=PipelineStage.PREPROCESS)
    async def preprocess_cv(self, cv_input: CVInput, candidate_name: str) -> None:
        logger.info(
            "CV input accepted",
            extra={
                "layer": LoggerLayer.PIPELINE,
                "stage": PreprocessStage.INPUT,
                "cv_text_n": len(cv_input.text),
                "candidate_name": candidate_name,
            },
        )
        self.artifact_manager.check_cv_name(cv_name_str=candidate_name)

        cv_parsed = await self.parse_cv(
            cv_input=cv_input,
            request_track_token=self.request_track_token,
            api_key=self.api_key,
        )
        logger.info(
            "Parsed CV",
            extra={
                "layer": LoggerLayer.PIPELINE,
                "stage": PreprocessStage.PARSE,
                "cv_parsed_n": len(cv_parsed),
            },
        )

        cv_chunks = self.chunk_cv(cv_parsed=cv_parsed)
        logger.info(
            "Chunked parsed CV",
            extra={
                "layer": LoggerLayer.PIPELINE,
                "stage": PreprocessStage.CHUNK,
                "cv_chunks(n)": len(cv_chunks),
            },
        )

        cv_embedding = self.embed_cv(cv_chunks=cv_chunks)
        logger.info(
            "Embedded CV chunks",
            extra={
                "layer": LoggerLayer.PIPELINE,
                "stage": PreprocessStage.EMBED,
            },
        )

        self.latency_stored = self.latency_store.get_all()

        self.save_cv(
            cv_parsed=cv_parsed,
            cv_chunk=cv_chunks,
            cv_embedding=cv_embedding,
            candidate_name=candidate_name,
            token_summary=self.track_token.get_all(),
            latency_stored=self.latency_stored,
        )

    @track_latency(stage=PreprocessStage.PARSE)
    async def parse_cv(
        self,
        cv_input: CVInput,
        request_track_token: TrackToken,
        api_key: str,
    ) -> StructuredCV:
        try:
            if len(cv_input.text) < 500:
                self.llm_abuse_protection.record_failure(api_key=self.api_key)
                raise InvalidCVLength(
                    f"Not Enough CV Information, cv_text_n: {len(cv_input.text)}",
                )

            structured_cv = await parse_cv_llm(
                request_track_token=request_track_token,
                cv_text=cv_input.text,
                llm_client=self.llm_client,
                llm_abuse_protection=self.llm_abuse_protection,
                api_key=api_key,
            )

            if len(structured_cv) < 10:
                self.llm_abuse_protection.record_failure(api_key=self.api_key)
                raise InvalidParsedCV(
                    f"Not Enough CV Information, cv_parse_n: {len(structured_cv)}",
                )

            return structured_cv

        except ValidationError as e:
            raise LLMInvalidSchemas(str(e)) from e

    def chunk_cv(self, cv_parsed: StructuredCV) -> list[CVChunk]:
        return chunk_cv_semantic(
            technical_skills=cv_parsed.technical_skills,
            work_experiences=cv_parsed.work_experience,
            projects=cv_parsed.project,
            languages=cv_parsed.languages,
            soft_skills=cv_parsed.soft_skills,
        )

    def embed_cv(self, cv_chunks: list[CVChunk]) -> list[CVEmbedding]:
        return self.embedding_service.embed_cv(
            cv_chunks=cv_chunks, batch_size=self.config.embedding.batch_size
        )

    def save_cv(
        self,
        cv_parsed: StructuredCV,
        cv_chunk: list[CVChunk],
        cv_embedding: list[CVEmbedding],
        candidate_name: str,
        token_summary: TokenSummary,
        latency_stored: LatencyStored,
    ) -> None:
        cv_name_str = candidate_name.strip().lower()

        self.artifact_manager.save_all_cv(
            cv_parsed=cv_parsed,
            cv_chunk=cv_chunk,
            cv_embedding=cv_embedding,
            cv_name_str=cv_name_str,
            token_summary=token_summary,
            latency_stored=latency_stored,
        )

    T = TypeVar("T", bound="PreprocessPipeline")

    @classmethod
    def start_from_config(
        cls: Type[T],
        config: Config,
        settings: Env,
        track_token: TrackToken,
        latency_store: TrackLatency,
        embedding_service: EmbeddingService,
        llm_client: LLMClient,
        artifact_manager: ArtifactManager,
        llm_abuse_protection: LLMAbuseProtection,
    ) -> T:
        return cls(
            config=config,
            settings=settings,
            track_token=track_token,
            latency_store=latency_store,
            embedding_service=embedding_service,
            llm_client=llm_client,
            artifact_manager=artifact_manager,
            llm_abuse_protection=llm_abuse_protection,
        )
