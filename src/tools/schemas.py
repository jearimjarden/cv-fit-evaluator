from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import SettingsConfigDict, BaseSettings
from enum import Enum
from dataclasses import dataclass
import numpy as np


# ===============================================================================
# Pipeline Stage
# ===============================================================================
class LoggerLayer(str, Enum):
    PIPELINE = "pipeline"
    MIDDLEWARE = "middleware"
    EXCEPTION = "exception"
    RESILIENCE = "resilience"
    SECURITY = "security"


# ===============================================================================
# ===============================================================================
# Pipeline Stage
# ===============================================================================
class APIStage(str, Enum):
    LIFESPAN = "api_lifespan"
    ARTIFACT = "api_artifact"


class PipelineStage(str, Enum):
    LLM = "llm_client"
    LLMRepair = "llm_repair"
    PREPROCESS = "preprocess_service"
    INFERENCE = "inference_service"
    EMBED = "embed_service"
    CONFIG = "configuration"


class InferenceStage(str, Enum):
    FILEINPUT = "inf_file_input"
    ARTIFACT = "inf_artifact"
    PREDICTFILE = "inf_predict_file"
    PREDICTAPI = "inf_predict_api"
    PARSE = "inf_parse"
    CHUNK = "inf_chunk"
    CHUNKREPAIR = "inf_chunk_repair"
    EMBED = "inf_embed"
    RETRIEVAL = "inf_retrieval"
    EVALUATION = "inf_evaluation"
    SCORING = "inf_scoring"
    REPORT = "inf_report"


class PreprocessStage(str, Enum):
    PARSE = "pre_parse"
    INPUT = "pre_input"
    CHUNK = "pre_chunk"
    EMBED = "pre_embed_cv"
    SAVE = "pre_save_cv"


# ===============================================================================


# ===============================================================================
# ENV SCHEMAS
# ===============================================================================
class Env(BaseSettings):
    environment: str = Field(...)
    oa_api_key: str = Field(...)
    hf_api_key: str = Field(...)
    model_config = SettingsConfigDict(extra="forbid", env_file=".env")


# ===============================================================================


# ===============================================================================
# CONFIG SCHEMAS
# ===============================================================================
class EmbeddingDevice(str, Enum):
    CUDA = "cuda"
    CPU = "cpu"


class EmbeddingModel(str, Enum):
    MINI_LLM = "sentence-transformers/all-MiniLM-L6-v2"


class LoggerLevel(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class InputMode(str, Enum):
    FILE = "file"
    API = "api"


class LLMModel(str, Enum):
    GPT4O = "gpt-4o-mini"


class ConfigLogger(BaseModel):
    level: LoggerLevel = Field(...)
    save_log: bool = Field(False)
    model_config = ConfigDict(extra="forbid")


class ConfigFileServiceItem(BaseModel):
    file_name: str | None = Field(..., min_length=3, max_length=40)
    folder_path: str | None = Field(..., min_length=3, max_length=40)
    model_config = ConfigDict(extra="forbid")


class ConfigFileService(BaseModel):
    cv: ConfigFileServiceItem = Field(...)
    jr: ConfigFileServiceItem = Field(...)
    model_config = ConfigDict(extra="forbid")


class ConfigJRChunk(BaseModel):
    chunk_size: int = Field(..., ge=1, le=4)
    stride: int = Field(..., ge=1, le=4)
    model_config = ConfigDict(extra="forbid")


class ConfigEmbedding(BaseModel):
    device: EmbeddingDevice = Field(...)
    batch_size: int = Field(4096, ge=4, le=16384)
    model: EmbeddingModel = Field(...)
    model_config = ConfigDict(extra="forbid")


class ConfigRetrieval(BaseModel):
    query_top_k: int = Field(..., ge=1, le=4)
    component_top_k: int = Field(..., ge=1, le=4)
    threshold: float = Field(..., ge=0, le=1)
    filter_below_threshold: bool = Field(False)
    model_config = ConfigDict(extra="forbid")


class ConfigEvaluation(BaseModel):
    evidence_mul: float = Field(1.0)
    capability_mul: float = Field(1.0)
    responsibility_mul: float = Field(1.0)
    model_config = ConfigDict(extra="forbid")


class ConfigLLM(BaseModel):
    model: LLMModel = Field(...)
    max_retry: int = Field(2, ge=1, le=5)
    timeout: int = Field(..., ge=5, le=30)
    prompt_tokens_per_1M: float = Field(..., ge=0, le=10)
    completions_tokens_per_1M: float = Field(..., ge=0, le=10)
    usd_to_idr: int = Field(..., ge=0)
    model_config = ConfigDict(extra="forbid")


class ConfigLLMProtection(BaseModel):
    threshold_s: int = Field(...)
    window_time_s: int = Field(...)
    suspend_s: int = Field(...)
    model_config = ConfigDict(extra="forbid")


class ConfigResilienceCircuitBreaker(BaseModel):
    threshold_s: int = Field(..., ge=1)
    window_time_s: int = Field(..., ge=5)
    model_config = ConfigDict(extra="forbid")


class ConfigResilienceConcurrencyLimiter(BaseModel):
    preprocess: int = Field(..., ge=1)
    chunking: int = Field(..., ge=1)
    evaluation: int = Field(..., ge=1)
    report: int = Field(..., ge=1)
    model_config = ConfigDict(extra="forbid")


class ConfigResilienceConcurrencyTimeout(BaseModel):
    preprocess: int = Field(..., ge=1)
    chunking: int = Field(..., ge=1)
    evaluation: int = Field(..., ge=1)
    report: int = Field(..., ge=1)
    model_config = ConfigDict(extra="forbid")


class ConfigResilience(BaseModel):
    circuit_breaker: ConfigResilienceCircuitBreaker = Field(...)
    concurrency_limiter: ConfigResilienceConcurrencyLimiter = Field(...)
    concurrency_timeout: ConfigResilienceConcurrencyTimeout = Field(...)
    model_config = ConfigDict(extra="forbid")


class Config(BaseModel):
    input_mode: InputMode = Field(...)
    logger: ConfigLogger = Field(...)
    file_service: ConfigFileService = Field(...)
    jr_chunk: ConfigJRChunk = Field(...)
    embedding: ConfigEmbedding = Field(...)
    retrieval: ConfigRetrieval = Field(...)
    evaluation: ConfigEvaluation = Field(...)
    llm: ConfigLLM = Field(...)
    llm_protection: ConfigLLMProtection = Field(...)
    resilience: ConfigResilience = Field(...)
    model_config = ConfigDict(extra="forbid")


# ===============================================================================


# ===============================================================================
# CV Schemas
# ===============================================================================
class StructuredCVItem(BaseModel):
    name: str = Field(...)
    item: list = Field(...)
    model_config = ConfigDict(extra="forbid")


class StructuredCVLanguage(BaseModel):
    name: str = Field(...)
    level: str = Field(...)


class StructuredCV(BaseModel):
    person_name: str = Field("")
    education: list = Field(default_factory=list)
    technical_skills: list[StructuredCVItem] = Field(default_factory=list)
    work_experience: list[StructuredCVItem] = Field(default_factory=list)
    project: list[StructuredCVItem] = Field(default_factory=list)
    soft_skills: list = Field(default_factory=list)
    languages: list[StructuredCVLanguage] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")

    def __len__(self):
        return (
            int(bool(self.person_name))
            + len(self.education)
            + len(self.technical_skills)
            + len(self.work_experience)
            + len(self.project)
            + len(self.soft_skills)
            + len(self.languages)
        )


class CVChunk(BaseModel):
    idx: int
    type: str
    chunk: str


class CVEmbedding(BaseModel):
    idx: int
    embedding: np.ndarray
    model_config = ConfigDict(arbitrary_types_allowed=True)


# ===============================================================================


# ===============================================================================
# Job Requirement Schemas
# ===============================================================================
class JRInput(BaseModel):
    text: str = Field(...)


class JRChunks(BaseModel):
    idx: int = Field(...)
    job_requirement: str = Field(...)
    components: list[str] = Field(...)
    reason: str = Field(...)

    def __len__(self):
        return len(self.components) + 1


class JREmbedding(BaseModel):
    idx: int = Field(...)
    job_requirement: str = Field(...)
    components: list[str] = Field(...)
    job_requirement_embedding: np.ndarray = Field(...)
    components_embedding: np.ndarray = Field(...)

    class Config:
        arbitrary_types_allowed = True


# ===============================================================================


# ===============================================================================
# Retrieval Schemas
# ===============================================================================
class BaseSearchQuery(BaseModel):
    query: str
    distances: list
    indices: list


class BaseSearchComponents(BaseModel):
    component: str
    distances: list
    indices: list


class BaseSearch(BaseModel):
    idx: int
    query_search: BaseSearchQuery
    components_search: list[BaseSearchComponents]


class BaseRetrievalQuery(BaseModel):
    query: str
    distances: list
    chunks: list

    def __len__(self):
        return len(self.chunks)


class BaseRetrievalComponent(BaseModel):
    component: str = Field(...)
    distances: list = Field(...)
    chunks: list = Field(...)

    def __len__(self):
        return len(self.chunks)


class BaseRetrieval(BaseModel):
    idx: int = Field(...)
    query_retrieval: BaseRetrievalQuery = Field(...)
    components_retrieval: list[BaseRetrievalComponent] = Field(...)

    def __len__(self):
        total_components_retrieval = sum(
            len(component) for component in self.components_retrieval
        )
        return len(self.query_retrieval) + total_components_retrieval


# ===============================================================================


# ===============================================================================
# Evaluation Schemas
# ===============================================================================


class EvidenceQuery(BaseModel):
    query: str
    evidence: list


class EvidenceComponent(BaseModel):
    component: str
    evidence: list


class Evidence(BaseModel):
    idx: int
    query: EvidenceQuery
    component: list[EvidenceComponent]


class EvaluationResult(BaseModel):
    components: str
    evidence_score: float
    responsible_multiplier: float
    capability_level: str
    reason: str


class Evaluation(BaseModel):
    query: str
    result: list[EvaluationResult]


class Score(BaseModel):
    query: str = Field(...)
    score: float = Field(...)
    reason: list[str] = Field(...)


class ReportInput(BaseModel):
    result: list[str] = Field(...)


class ReportScore(BaseModel):
    query: str
    score: float
    reason: str


class Report(BaseModel):
    datetime: str
    name: str
    report: list[ReportScore]
    report_score: float


class CapabilityLevel(str, Enum):
    EXPLICIT_STRONG = "explicit_strong"
    EXPLICIT_WEAK = "explicit_weak"
    IMPLICIT_STRONG = "implicit_strong"
    IMPLICIT_WEAK = "implicit_weak"
    MISSING = "missing"


@dataclass
class Capability:
    capability_level: CapabilityLevel

    def __post_init__(self):
        if isinstance(self.capability_level, str):
            self.capability_level = CapabilityLevel(self.capability_level)

    def weight(self) -> float:
        mapping = {
            CapabilityLevel.EXPLICIT_STRONG: 1.1,
            CapabilityLevel.EXPLICIT_WEAK: 0.8,
            CapabilityLevel.IMPLICIT_STRONG: 0.6,
            CapabilityLevel.IMPLICIT_WEAK: 0.4,
            CapabilityLevel.MISSING: 0.0,
        }

        return mapping[self.capability_level]


# ===============================================================================


# ===============================================================================
# Telemetry Schemas
# ===============================================================================


@dataclass
class TokenSummary:
    total_prompt_tokens: int
    total_completion_tokens: int
    total_cost_idr: float
    tokens_history: dict[str, int]


@dataclass
class LatencyStored:
    latencies_ms: dict[str, float]


@dataclass
class Metadata:
    cv_name: str
    created_date: str
    cv_parsed_n: int
    cv_chunked_n: int
    cv_embedding_n: int
    token_summary: TokenSummary
    latencies_ms: LatencyStored


# ===============================================================================


# ===============================================================================
# FastAPI Schemas
# ===============================================================================
class CVInput(BaseModel):
    text: str = Field(...)
    model_config = ConfigDict(extra="forbid")


class CandidateList(BaseModel):
    candidates_list: list[str] = Field(...)
    model_config = ConfigDict(extra="forbid")


class PublicConfig(BaseModel):
    embedding_model: EmbeddingModel
    embedding_device: EmbeddingDevice
    llm_model: LLMModel
    query_top_k: int
    component_top_k: int
    retrieval_threshold: float
    filter_below_threshold: bool


class PublicHealth(BaseModel):
    status: str
    environment: str


class ErrorResponseError(BaseModel):
    error_type: str
    code: str
    message: str


class ErrorResponse(BaseModel):
    status: str
    request_id: str
    error: ErrorResponseError


class Telemetry(BaseModel):
    uptime: float
    token_summary: TokenSummary
    total_request: int


class BaseResponse(BaseModel):
    status: str
    candidate_name: str
    model_config = ConfigDict(extra="allow")


# ===============================================================================


# ===============================================================================
# Auth Config Schemas
# ===============================================================================
class AuthConfigKey(BaseModel):
    key: str
    rate_limit: int
    allowed_routes: list[str] = []
    model_config = ConfigDict(extra="forbid")


class AuthConfig(BaseModel):
    api_keys: dict[str, AuthConfigKey]
    rate_limit_window: int
    model_config = ConfigDict(extra="forbid")


# ===============================================================================
