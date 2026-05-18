import logging
from typing import Type, TypeVar
from pydantic import ValidationError
from dataclasses import asdict
from .preprocess_pipeline import PreprocessPipeline
from ..IO.artifact_manager import ArtifactManager
from ..IO.json_loader import load_json
from ..services.evaluator import EvaluatorService
from ..services.llm_client import LLMClient
from ..services.parser import parse_normalize_jr
from ..services.embedder import EmbeddingService
from ..services.retriever import (
    faiss_ip_search,
    retrieve_base_chunk,
)
from ..services.chunker import decompose_and_validate_jr
from ..tools.security import LLMAbuseProtection
from ..tools.exceptions_schemas import (
    InferenceError,
    InvalidJRError,
    LLMInvalidSchemas,
    PreprocessError,
    InvalidFileError,
    LLMError,
    ArtifactError,
    LoggedPipelineError,
)
from ..tools.observabillity import TrackLatency, TrackToken, track_latency
from ..tools.schemas import (
    BaseRetrieval,
    CVChunk,
    CVEmbedding,
    CVInput,
    Config,
    Env,
    InferenceStage,
    InputMode,
    JRChunks,
    JREmbedding,
    JRInput,
    LoggerLayer,
    Report,
    PipelineStage,
)

logger = logging.getLogger(__name__)


class InferencePipeline:
    def __init__(
        self,
        config: Config,
        settings: Env,
        track_latency: TrackLatency,
        track_token: TrackToken,
        llm_client: LLMClient,
        evaluator_service: EvaluatorService,
        embedding_service: EmbeddingService,
        preprocess_pipeline: PreprocessPipeline,
        artifact_manager: ArtifactManager,
        llm_abuse_protection: LLMAbuseProtection,
    ) -> None:
        self.settings = settings
        self.config = config
        self.latency_store = track_latency
        self.track_token = track_token
        self.llm_client = llm_client
        self.embedding_service = embedding_service
        self.evaluator_service = evaluator_service
        self.preprocess_pipeline = preprocess_pipeline
        self.artifact_manager = artifact_manager
        self.llm_abuse_protection = llm_abuse_protection

    async def run(self, candidate_name: str, jr_input: JRInput, api_key: str) -> Report:
        self.api_key = api_key
        self.request_track_token = TrackToken(llm_config=self.config.llm)

        try:
            if self.config.input_mode == InputMode.FILE:
                report = await self.predict_file()
                logger.info(
                    "File predicted",
                    extra={
                        "layer": LoggerLayer.PIPELINE,
                        "stage": PipelineStage.INFERENCE,
                        "latencies": self.latency_store.get_all().latencies_ms,
                    },
                )

                logger.info(
                    "Token Tracked",
                    extra={
                        "layer": LoggerLayer.PIPELINE,
                        "stage": PipelineStage.INFERENCE,
                        "summary": asdict(self.request_track_token.get_all()),
                    },
                )

                return report

            elif self.config.input_mode == InputMode.API:
                report = await self.predict_api(
                    candidate_name=candidate_name, jr_input=jr_input
                )
                logger.info(
                    "API predicted",
                    extra={
                        "layer": LoggerLayer.PIPELINE,
                        "stage": PipelineStage.INFERENCE,
                        "latencies": self.latency_store.get_all().latencies_ms,
                    },
                )

                logger.info(
                    "Token Tracked",
                    extra={
                        "layer": LoggerLayer.PIPELINE,
                        "stage": PipelineStage.INFERENCE,
                        "summary": asdict(self.request_track_token.get_all()),
                    },
                )

                return report

            else:
                raise Exception

        except (
            LLMError,
            InferenceError,
            PreprocessError,
            ArtifactError,
        ) as e:
            logger.error(
                str(e),
                extra={
                    "layer": LoggerLayer.EXCEPTION,
                    "stage": e.stage,
                    "error_type": e.error_type,
                    "code": e.code,
                },
            )
            logger.error(
                "Error occured in inference pipeline",
                extra={
                    "layer": LoggerLayer.PIPELINE,
                    "stage": PipelineStage.INFERENCE,
                    "latencies": self.latency_store.get_all().latencies_ms,
                },
            )

            raise LoggedPipelineError(
                str(e),
                stage=e.stage,
                error_type=e.error_type,
                code=e.code,
                status_code=e.status_code,
            ) from e

    @track_latency(stage=InferenceStage.PREDICTFILE)
    async def predict_file(self) -> Report:

        cv_input, jr_input = self.load_cv_jr_file()
        logger.info(
            "Retrieved raw CV and JR input",
            extra={
                "layer": LoggerLayer.PIPELINE,
                "stage": InferenceStage.FILEINPUT,
                "cv_length(n)": len(cv_input.text),
                "jr_length(n)": len(jr_input.text),
            },
        )

        cv_parsed = await self.preprocess_pipeline.parse_cv(
            cv_input=cv_input,
            request_track_token=self.request_track_token,
            api_key=self.api_key,
        )
        jr_parsed = self.parse_jr(jr_input=jr_input)
        logger.info(
            "Parsed raw CV and JR",
            extra={
                "layer": LoggerLayer.PIPELINE,
                "stage": InferenceStage.PARSE,
                "cv_parsed(n)": len(cv_parsed),
                "jr_parsed(n)": len(jr_parsed),
            },
        )

        cv_chunks = self.preprocess_pipeline.chunk_cv(cv_parsed=cv_parsed)
        jr_chunks = await self.chunk_jr(jr_parsed_text=jr_parsed)
        logger.info(
            "Chunked parsed CV and JR",
            extra={
                "layer": LoggerLayer.PIPELINE,
                "stage": InferenceStage.CHUNK,
                "cv_chunks(n)": len(cv_chunks),
                "jr_chunks(n)": sum(len(chunk) for chunk in jr_chunks),
            },
        )

        jr_embedding = self.embedding_service.embed_jr(
            jr_chunks=jr_chunks, batch_size=self.config.embedding.batch_size
        )
        logger.info(
            "Embedded JR chunk",
            extra={
                "layer": LoggerLayer.PIPELINE,
                "stage": InferenceStage.EMBED,
            },
        )

        cv_embedding = self.preprocess_pipeline.embed_cv(cv_chunks=cv_chunks)
        logger.info(
            "Embedded CV chunk",
            extra={
                "layer": LoggerLayer.PIPELINE,
                "stage": InferenceStage.EMBED,
            },
        )

        retrieved_base = self.retrieve_base(
            cv_embedding=cv_embedding, jr_embedding=jr_embedding, cv_chunks=cv_chunks
        )
        logger.info(
            "Retrieved similar semantic chunk",
            extra={
                "layer": LoggerLayer.PIPELINE,
                "stage": InferenceStage.RETRIEVAL,
                "base_retrieval(n)": sum(
                    len(retrieved) for retrieved in retrieved_base
                ),
            },
        )

        evaluations = await self.evaluator_service.generate_evaluation(
            base_retrieval=retrieved_base, request_track_token=self.request_track_token
        )
        logger.info(
            "Successfully generated evaluation",
            extra={
                "layer": LoggerLayer.PIPELINE,
                "stage": InferenceStage.EVALUATION,
                "evaluations(n)": len(evaluations),
            },
        )

        scores = self.evaluator_service.generate_score(evaluations=evaluations)
        logger.info(
            "Successfully generated score",
            extra={
                "layer": LoggerLayer.PIPELINE,
                "stage": InferenceStage.SCORING,
                "scores": [score.score for score in scores],
            },
        )

        reports = await self.evaluator_service.generate_report(
            scores=scores,
            candidate_name=cv_parsed.person_name,
            request_track_token=self.request_track_token,
        )

        logger.info(
            "Successfully generated report",
            extra={
                "layer": LoggerLayer.PIPELINE,
                "stage": InferenceStage.REPORT,
                "final_score": reports.report_score,
            },
        )

        return reports

    @track_latency(stage=InferenceStage.PREDICTAPI)
    async def predict_api(self, candidate_name: str, jr_input: JRInput) -> Report:
        metadata, cv_chunks, cv_embedding = self.load_cv_artifact(
            candidate_name=candidate_name
        )
        logger.info(
            "CV acquired",
            extra={
                "layer": LoggerLayer.PIPELINE,
                "stage": InferenceStage.ARTIFACT,
                "person_name": metadata.cv_name,
                "cv_chunks_n": metadata.cv_chunked_n,
            },
        )
        jr_parsed = self.parse_jr(jr_input=jr_input)
        logger.info(
            "Parsed raw JR",
            extra={
                "layer": LoggerLayer.PIPELINE,
                "stage": InferenceStage.PARSE,
                "jr_parsed(n)": len(jr_parsed),
            },
        )

        jr_chunks = await self.chunk_jr(jr_parsed_text=jr_parsed)
        logger.info(
            "Chunked parsed JR",
            extra={
                "layer": LoggerLayer.PIPELINE,
                "stage": InferenceStage.CHUNK,
                "jr_chunks(n)": sum(len(chunk) for chunk in jr_chunks),
            },
        )
        jr_embedding = self.embedding_service.embed_jr(
            jr_chunks=jr_chunks, batch_size=self.config.embedding.batch_size
        )
        logger.info(
            "Embedded chunked JR",
            extra={
                "layer": LoggerLayer.PIPELINE,
                "stage": InferenceStage.EMBED,
            },
        )

        retrieved_base = self.retrieve_base(
            cv_embedding=cv_embedding, jr_embedding=jr_embedding, cv_chunks=cv_chunks
        )
        logger.info(
            "Retrieved similar semantic chunk",
            extra={
                "layer": LoggerLayer.PIPELINE,
                "stage": InferenceStage.RETRIEVAL,
                "base_retrieval(n)": sum(
                    len(retrieved) for retrieved in retrieved_base
                ),
            },
        )

        evaluations = await self.evaluator_service.generate_evaluation(
            base_retrieval=retrieved_base, request_track_token=self.request_track_token
        )
        logger.info(
            "Successfully generated evaluation",
            extra={
                "layer": LoggerLayer.PIPELINE,
                "stage": InferenceStage.EVALUATION,
                "evaluations(n)": len(evaluations),
            },
        )

        scores = self.evaluator_service.generate_score(evaluations=evaluations)
        logger.info(
            "Successfully generated score",
            extra={
                "layer": LoggerLayer.PIPELINE,
                "stage": InferenceStage.SCORING,
                "scores": [score.score for score in scores],
            },
        )

        reports = await self.evaluator_service.generate_report(
            scores=scores,
            candidate_name=metadata.cv_name,
            request_track_token=self.request_track_token,
        )

        logger.info(
            "Successfully generated report",
            extra={
                "layer": LoggerLayer.PIPELINE,
                "stage": InferenceStage.REPORT,
                "final_score": reports.report_score,
            },
        )

        return reports

    def load_cv_artifact(self, candidate_name: str):
        return self.artifact_manager.load_cv_all(candidate_name=candidate_name)

    @track_latency(stage=InferenceStage.CHUNK)
    async def chunk_jr(
        self,
        jr_parsed_text: list[str],
    ) -> list[JRChunks]:
        try:
            return await decompose_and_validate_jr(
                jr_parsed_text=jr_parsed_text,
                llm_client=self.llm_client,
                llm_abuse_protection=self.llm_abuse_protection,
                api_key=self.api_key,
                request_track_token=self.request_track_token,
            )

        except ValidationError as e:
            self.llm_abuse_protection.record_failure(api_key=self.api_key)
            raise LLMInvalidSchemas(str(e)) from e

    def parse_jr(self, jr_input: JRInput) -> list[str]:
        """Default Job Requirement Parser using New Line and Normalization"""
        if len(jr_input.text) < 20:
            raise InvalidJRError("JR input could not have less than 20 characters")

        parsed_normalized_jr = parse_normalize_jr(
            text=jr_input.text,
            chunk_size=self.config.jr_chunk.chunk_size,
            stride=self.config.jr_chunk.stride,
        )
        return parsed_normalized_jr

    def retrieve_base(
        self,
        cv_embedding: list[CVEmbedding],
        jr_embedding: list[JREmbedding],
        cv_chunks: list[CVChunk],
    ) -> list[BaseRetrieval]:
        search_result = faiss_ip_search(
            cv_embedding=cv_embedding,
            jr_embedding=jr_embedding,
            query_top_k=self.config.retrieval.query_top_k,
            component_top_k=self.config.retrieval.component_top_k,
        )

        idx_to_chunk = {item.idx: item.chunk for item in cv_chunks}

        retrieved_chunks = retrieve_base_chunk(
            search_result=search_result,
            idx_to_chunk=idx_to_chunk,
            threshold=self.config.retrieval.threshold,
            filter_below_threshold=self.config.retrieval.filter_below_threshold,
        )

        logger.debug(
            "Retrieved Chunk",
            extra={
                "layer": LoggerLayer.PIPELINE,
                "stage": InferenceStage.RETRIEVAL,
                "result": retrieved_chunks,
            },
        )
        return retrieved_chunks

    def load_cv_jr_file(self) -> tuple[CVInput, JRInput]:
        cv_input = self._load_CV()
        jr_input = self._load_JR()

        return cv_input, jr_input

    def _load_JR(self) -> JRInput:
        file_name = self.config.file_service.jr.file_name
        folder_path = self.config.file_service.jr.folder_path

        if file_name is None or folder_path is None:
            raise InvalidFileError(
                "file_name or folder_path can not be empty when using File Service"
            )

        jr_loaded = load_json(
            file_name=file_name,
            folder_path=folder_path,
        )

        validated_jr = JRInput(**jr_loaded)
        return validated_jr

    def _load_CV(self) -> CVInput:
        file_name = self.config.file_service.cv.file_name
        folder_path = self.config.file_service.cv.folder_path

        if file_name is None or folder_path is None:
            raise InvalidFileError(
                "file_name or folder_path can not be empty when using File Service"
            )

        cv_loaded = load_json(
            file_name=file_name,
            folder_path=folder_path,
        )

        validated_cv = CVInput(**cv_loaded)
        return validated_cv

    T = TypeVar("T", bound="InferencePipeline")

    @classmethod
    def load_from_config(
        cls: Type[T],
        config: Config,
        settings: Env,
        track_latency: TrackLatency,
        track_token: TrackToken,
        llm_client: LLMClient,
        evaluator_service: EvaluatorService,
        embedding_service: EmbeddingService,
        preprocess_pipeline: PreprocessPipeline,
        artifact_manager: ArtifactManager,
        llm_abuse_protection: LLMAbuseProtection,
    ) -> T:
        return cls(
            config=config,
            settings=settings,
            track_latency=track_latency,
            track_token=track_token,
            llm_client=llm_client,
            evaluator_service=evaluator_service,
            embedding_service=embedding_service,
            preprocess_pipeline=preprocess_pipeline,
            artifact_manager=artifact_manager,
            llm_abuse_protection=llm_abuse_protection,
        )
