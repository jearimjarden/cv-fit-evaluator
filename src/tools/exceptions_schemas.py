from .schemas import InferenceStage, PreprocessStage, PipelineStage


class BaseAppError(Exception):
    def __init__(
        self, message: str, stage: str, error_type: str, code: str, status_code: int
    ):
        self.message = message
        self.error_type = error_type
        self.code = code
        self.stage = stage
        self.status_code = status_code
        super().__init__(message)


class LoggedPipelineError(BaseAppError):
    def __init__(
        self, message: str, stage: str, error_type: str, code: str, status_code: int
    ):
        self.message = message
        self.error_type = error_type
        self.code = code
        self.stage = stage
        self.status_code = status_code
        super().__init__(
            message,
            stage=self.stage,
            error_type=self.error_type,
            code=self.code,
            status_code=self.status_code,
        )


# ===============================================================================
# Configuration Exceptions
# ===============================================================================
class ConfigurationError(BaseAppError):
    def __init__(self, message, stage: str, status_code: int):
        self.error_type = self.__class__.__bases__[0].__name__
        self.code = self.__class__.__name__
        self.stage = stage
        self.status_code = status_code
        super().__init__(
            message,
            stage=self.stage,
            error_type=self.error_type,
            code=self.code,
            status_code=self.status_code,
        )


class InvalidConfig(ConfigurationError):
    def __init__(self, message: str):
        super().__init__(message, stage=PipelineStage.CONFIG, status_code=500)


class InvalidSettings(ConfigurationError):
    def __init__(self, message: str):
        super().__init__(message, stage=PipelineStage.CONFIG, status_code=500)


# ===============================================================================


# ===============================================================================
# Preprocess Exceptions
# ===============================================================================
class PreprocessError(BaseAppError):
    def __init__(self, message: str, stage: str, status_code: int):
        self.error_type = self.__class__.__bases__[0].__name__
        self.code = self.__class__.__name__
        self.stage = stage
        self.status_code = status_code
        super().__init__(
            message,
            stage=self.stage,
            error_type=self.error_type,
            code=self.code,
            status_code=self.status_code,
        )


class InvalidCVLength(PreprocessError):
    def __init__(self, message):
        super().__init__(message, stage=PreprocessStage.PARSE, status_code=400)


class InvalidParsedCV(PreprocessError):
    def __init__(self, message):
        super().__init__(message, stage=PreprocessStage.PARSE, status_code=400)


# ===============================================================================


# ===============================================================================
# Inference Exceptions
# ===============================================================================
class InferenceError(BaseAppError):
    def __init__(self, message: str, stage: str, status_code: int):
        self.error_type = self.__class__.__bases__[0].__name__
        self.code = self.__class__.__name__
        self.stage = stage
        self.status_code = status_code
        super().__init__(
            message,
            stage=self.stage,
            error_type=self.error_type,
            code=self.code,
            status_code=self.status_code,
        )


class InvalidJRError(InferenceError):
    def __init__(self, message):
        super().__init__(message, stage=PreprocessStage.PARSE, status_code=400)


class InvalidFileError(InferenceError):
    def __init__(self, message):
        super().__init__(message, stage=InferenceStage.FILEINPUT, status_code=500)


# ===============================================================================


# ===============================================================================
# Artifacts Exceptions
# ===============================================================================
class ArtifactError(BaseAppError):
    def __init__(self, message: str, stage: str, status_code: int):
        self.error_type = self.__class__.__bases__[0].__name__
        self.code = self.__class__.__name__
        self.stage = stage
        self.status_code = status_code
        super().__init__(
            message,
            stage=self.stage,
            error_type=self.error_type,
            code=self.code,
            status_code=self.status_code,
        )


class CorruptedArtifact(ArtifactError):
    def __init__(self, message):
        super().__init__(message, stage=InferenceStage.ARTIFACT, status_code=500)


class ExistingArtifact(ArtifactError):
    def __init__(self, message):
        super().__init__(message, stage=InferenceStage.ARTIFACT, status_code=409)


class ArtifactNotFound(ArtifactError):
    def __init__(self, message):
        super().__init__(message, stage=InferenceStage.ARTIFACT, status_code=404)


# ===============================================================================


# ===============================================================================
# LLM Client Exceptions
# ===============================================================================
class LLMError(BaseAppError):
    def __init__(self, message: str, stage: str, status_code: int):
        self.error_type = self.__class__.__bases__[0].__name__
        self.code = self.__class__.__name__
        self.stage = stage
        self.status_code = status_code
        super().__init__(
            message,
            stage=self.stage,
            error_type=self.error_type,
            code=self.code,
            status_code=self.status_code,
        )


class InvalidJSON(LLMError):
    def __init__(self, message):
        super().__init__(message, stage=PipelineStage.LLM, status_code=502)


class InvalidResponse(LLMError):
    def __init__(self, message):
        super().__init__(message, stage=PipelineStage.LLM, status_code=502)


class LLMTimeoutError(LLMError):
    def __init__(self, message):
        super().__init__(message, stage=PipelineStage.LLM, status_code=504)


class LLMConnectionError(LLMError):
    def __init__(self, message):
        super().__init__(message, stage=PipelineStage.LLM, status_code=502)


class LLMAuthenticationError(LLMError):
    def __init__(self, message):
        super().__init__(message, stage=PipelineStage.LLM, status_code=503)


class LLMInvalidSchemas(LLMError):
    def __init__(self, message):
        super().__init__(message, stage=PipelineStage.LLM, status_code=502)


class LLMQuotaExceeded(LLMError):
    def __init__(self, message):
        super().__init__(message, stage=PipelineStage.LLM, status_code=503)


class LLMRateLimitExceeded(LLMError):
    def __init__(self, message):
        super().__init__(message, stage=PipelineStage.LLM, status_code=503)


# ===============================================================================


# ===============================================================================
# Security Exceptions
# ===============================================================================
class SecurityError(BaseAppError):
    def __init__(self, message: str, stage: str, status_code: int):
        self.error_type = self.__class__.__bases__[0].__name__
        self.code = self.__class__.__name__
        self.stage = stage
        self.status_code = status_code
        super().__init__(
            message,
            stage=self.stage,
            error_type=self.error_type,
            code=self.code,
            status_code=self.status_code,
        )


class MissingAPIKey(SecurityError):
    def __init__(self, message):
        super().__init__(message, stage=PipelineStage.LLM, status_code=401)


class InvalidAPIKey(SecurityError):
    def __init__(self, message):
        super().__init__(message, stage=PipelineStage.LLM, status_code=401)


class RateLimitExceeded(SecurityError):
    def __init__(self, message):
        super().__init__(message, stage=PipelineStage.LLM, status_code=429)


class UnauthorizedRoute(SecurityError):
    def __init__(self, message):
        super().__init__(message, stage=PipelineStage.LLM, status_code=403)


class LLMAbusedError(SecurityError):
    def __init__(self, message):
        super().__init__(message, stage=PipelineStage.LLM, status_code=429)


# ===============================================================================


# ===============================================================================
# Resilience Exceptions
# ===============================================================================
class ResilienceError(BaseAppError):
    def __init__(self, message: str, stage: str, status_code: int):
        self.error_type = self.__class__.__bases__[0].__name__
        self.code = self.__class__.__name__
        self.stage = stage
        self.status_code = status_code
        super().__init__(
            message,
            stage=self.stage,
            error_type=self.error_type,
            code=self.code,
            status_code=self.status_code,
        )


class CircuitBreakerOpen(ResilienceError):
    def __init__(self, message):
        super().__init__(message, stage=PipelineStage.LLM, status_code=503)


class ConcurrencyLimitError(ResilienceError):
    def __init__(self, message):
        super().__init__(message, stage=PipelineStage.LLM, status_code=503)


# ===============================================================================
