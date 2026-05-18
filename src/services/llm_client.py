from openai import (
    AuthenticationError,
    AsyncOpenAI,
    APITimeoutError,
    APIConnectionError,
    RateLimitError,
)
import json
import logging
from ..services.prompt_builder import create_fix_json_prompt
from ..tools.security import LLMAbuseProtection
from ..tools.resilience import CircuitBreaker, ConcurrencyLimiter
from ..tools.exceptions_schemas import (
    InvalidJSON,
    InvalidResponse,
    LLMAuthenticationError,
    LLMConnectionError,
    LLMTimeoutError,
    LLMQuotaExceeded,
    LLMRateLimitExceeded,
)
from ..tools.schemas import (
    Config,
    InferenceStage,
    LoggerLayer,
    PipelineStage,
    PreprocessStage,
)
from ..tools.observabillity import TrackToken

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(
        self,
        api_key: str,
        track_token: TrackToken,
        config: Config,
        circuit_breaker: CircuitBreaker,
        concurrency_limiter: ConcurrencyLimiter,
        llm_abuse_protection: LLMAbuseProtection,
        model: str = "gpt-4o-mini",
    ) -> None:
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.track_token = track_token
        self.config = config
        self.circuit_breaker = circuit_breaker
        self.concurrency_limiter = concurrency_limiter
        self.llm_abuse_protection = llm_abuse_protection

    async def generate(self, prompt: str, stage: InferenceStage | PreprocessStage, request_track_token: TrackToken) -> str:  # type: ignore

        for attempt in range(self.config.llm.max_retry):
            self.circuit_breaker.check(stage=stage)
            async with self.concurrency_limiter.limit(stage=stage):
                try:
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0,
                        timeout=self.config.llm.timeout,
                    )
                    if response.usage:
                        request_track_token.add(
                            completion_tokens=response.usage.completion_tokens,
                            prompt_token=response.usage.prompt_tokens,
                            stage=stage.value,
                        )
                        self.track_token.add(
                            completion_tokens=response.usage.completion_tokens,
                            prompt_token=response.usage.prompt_tokens,
                            stage=stage.value,
                        )

                    content = response.choices[0].message.content

                    if isinstance(content, str):
                        self.circuit_breaker.record_success()
                        return content

                    else:
                        raise InvalidResponse("LLM response content is not a string")

                except AuthenticationError:
                    raise LLMAuthenticationError("Invalid OpenAI API key")

                except APITimeoutError:
                    logger.warning(
                        f"LLM timeout on retry, {attempt + 1}/{self.config.llm.max_retry}",
                        extra={
                            "layer": LoggerLayer.PIPELINE,
                            "stage": PipelineStage.LLM,
                        },
                    )

                    if attempt == self.config.llm.max_retry - 1:
                        self.circuit_breaker.record_failure()
                        raise LLMTimeoutError(
                            f"LLM request timed out after {self.config.llm.max_retry} attempts"
                        )

                except APIConnectionError:
                    self.circuit_breaker.record_failure()
                    raise LLMConnectionError("Failed to connect to OpenAI API")

                except RateLimitError as e:
                    error_message = str(e)
                    if "insufficient_quota" in error_message:
                        logger.warning(
                            "OpenAI quota exceeded",
                            extra={
                                "layer": LoggerLayer.PIPELINE,
                                "stage": PipelineStage.LLM,
                            },
                        )
                        raise LLMQuotaExceeded("OpenAI quota exceeded")

                    self.circuit_breaker.record_failure()
                    raise LLMRateLimitExceeded("OpenAI rate limit exceeded")

    async def json_repair(self, context: str, request_track_token: TrackToken, api_key: str = None) -> dict:  # type: ignore
        prompt = create_fix_json_prompt(context=context)
        if api_key:
            self.llm_abuse_protection.record_failure(api_key=api_key)

        for attempt in range(self.config.llm.max_retry):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    timeout=self.config.llm.timeout,
                )

                if response.usage:
                    self.track_token.add(
                        completion_tokens=response.usage.completion_tokens,
                        prompt_token=response.usage.prompt_tokens,
                        stage=PipelineStage.LLMRepair,
                    )
                    request_track_token.add(
                        completion_tokens=response.usage.completion_tokens,
                        prompt_token=response.usage.prompt_tokens,
                        stage=PipelineStage.LLMRepair,
                    )
                content = response.choices[0].message.content

                if isinstance(content, str):
                    dict_content = json.loads(content)
                    return dict_content
                else:
                    raise InvalidResponse("LLM response content is not a string")

            except json.JSONDecodeError:
                raise InvalidJSON("Failed to repair invalid JSON response")

            except APITimeoutError:
                logger.warning(
                    f"LLM timeout on retry, {attempt + 1}/{self.config.llm.max_retry}",
                    extra={"layer": LoggerLayer.PIPELINE, "stage": PipelineStage.LLM},
                )

                if attempt == self.config.llm.max_retry - 1:
                    raise LLMTimeoutError(
                        f"LLM request timed out after {self.config.llm.max_retry} attempts"
                    )

            except APIConnectionError:
                raise LLMConnectionError("Failed to connect to OpenAI API")
