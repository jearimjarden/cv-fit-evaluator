import asyncio
import time
import logging
from .schemas import Config, InferenceStage, PreprocessStage, LoggerLayer
from .exceptions_schemas import CircuitBreakerOpen, ConcurrencyLimitError

logger = logging.getLogger(__name__)


class ConcurrencyLimiterCM:
    def __init__(
        self,
        stage: InferenceStage | PreprocessStage,
        semaphores: dict[
            InferenceStage | PreprocessStage,
            asyncio.Semaphore,
        ],
        timeouts: dict[
            InferenceStage | PreprocessStage,
            int,
        ],
        limits: dict[
            InferenceStage | PreprocessStage,
            int,
        ],
    ) -> None:

        self._stage = stage
        self._semaphores = semaphores
        self._timeouts = timeouts
        self._limits = limits
        self._acquired = False

    async def __aenter__(self) -> None:
        semaphore = self._semaphores[self._stage]
        timeout_s = self._timeouts[self._stage]
        max_concurrency = self._limits[self._stage]
        current_in_use = max_concurrency - semaphore._value
        queued_tasks = max(
            0,
            len(semaphore._waiters) if semaphore._waiters is not None else 0,
        )

        usage_percent = int((current_in_use / max_concurrency) * 100)
        threshold = None

        if usage_percent == 100:
            threshold = 100

        if threshold:
            logger.warning(
                "Concurrency threshold limit reached",
                extra={
                    "layer": LoggerLayer.RESILIENCE,
                    "stage": self._stage.value,
                    "timeout_s": timeout_s,
                    "usage_percent": usage_percent,
                    "current_in_use": current_in_use,
                    "max_concurrency": max_concurrency,
                    "queued_tasks": queued_tasks,
                },
            )

        try:
            await asyncio.wait_for(
                semaphore.acquire(),
                timeout=timeout_s,
            )
            self._acquired = True

            all_stage_usage = {}
            for stage, stage_semaphore in self._semaphores.items():
                stage_max = self._limits[stage]
                stage_in_use = stage_max - stage_semaphore._value
                stage_usage_percent = int((stage_in_use / stage_max) * 100)
                all_stage_usage[stage.value] = {
                    "used": stage_in_use,
                    "max": stage_max,
                    "usage_percent": stage_usage_percent,
                    "queued_tasks": max(
                        0,
                        (
                            len(stage_semaphore._waiters)
                            if stage_semaphore._waiters is not None
                            else 0
                        ),
                    ),
                }

            logger.debug(
                "Concurrency debug:",
                extra={
                    "layer": LoggerLayer.RESILIENCE,
                    "stage": self._stage.value,
                    "threshold": threshold,
                    "stages": all_stage_usage,
                },
            )

        except asyncio.TimeoutError:
            logger.error(
                "Concurrency limit timeout exceeded",
                extra={
                    "layer": LoggerLayer.RESILIENCE,
                    "stage": self._stage.value,
                    "timeout_s": timeout_s,
                    "usage_percent": usage_percent,
                    "current_in_use": current_in_use,
                    "max_concurrency": max_concurrency,
                    "queued_tasks": queued_tasks,
                },
            )

            raise ConcurrencyLimitError(f"{self._stage.value} concurrency overloaded")

    async def __aexit__(
        self,
        exc_type,
        exc,
        tb,
    ) -> None:

        if self._acquired:
            semaphore = self._semaphores[self._stage]
            semaphore.release()


class ConcurrencyLimiter:
    def __init__(
        self,
        config: Config,
    ) -> None:
        limiter = config.resilience.concurrency_limiter
        timeout = config.resilience.concurrency_timeout

        self._limits = {
            PreprocessStage.PARSE: limiter.preprocess,
            InferenceStage.CHUNK: limiter.chunking,
            InferenceStage.EVALUATION: limiter.evaluation,
            InferenceStage.REPORT: limiter.report,
            InferenceStage.CHUNKREPAIR: 10,
        }
        self._semaphores = {
            PreprocessStage.PARSE: asyncio.Semaphore(limiter.preprocess),
            InferenceStage.CHUNK: asyncio.Semaphore(limiter.chunking),
            InferenceStage.EVALUATION: asyncio.Semaphore(limiter.evaluation),
            InferenceStage.REPORT: asyncio.Semaphore(limiter.report),
            InferenceStage.CHUNKREPAIR: asyncio.Semaphore(10),
        }
        self._timeouts = {
            PreprocessStage.PARSE: timeout.preprocess,
            InferenceStage.CHUNK: timeout.chunking,
            InferenceStage.EVALUATION: timeout.evaluation,
            InferenceStage.REPORT: timeout.report,
            InferenceStage.CHUNKREPAIR: 10,
        }

    def limit(
        self,
        stage: InferenceStage | PreprocessStage,
    ) -> ConcurrencyLimiterCM:
        return ConcurrencyLimiterCM(
            stage=stage,
            semaphores=self._semaphores,
            timeouts=self._timeouts,
            limits=self._limits,
        )


class CircuitBreaker:
    def __init__(self, threshold: int, window_time: int):
        self.threshold = threshold
        self.window_time = window_time
        self._total_failure = 0
        self._state_last_failure = 0.0
        self._state = "CLOSED"

    def check(self, stage: InferenceStage | PreprocessStage) -> None:
        if self._state == "CLOSED":
            return

        if self._state == "OPEN":
            if (time.perf_counter() - self._state_last_failure) > self.window_time:
                self._state = "HALF_OPEN"
                return
            else:
                logger.error(
                    "Circuit breaker opened",
                    extra={
                        "layer": LoggerLayer.SECURITY,
                        "stage": stage,
                        "time_left_s": self.window_time
                        - (time.perf_counter() - self._state_last_failure),
                    },
                )
                raise CircuitBreakerOpen("Circuit breaker opened")

    def record_failure(self):
        self._total_failure += 1

        if self._total_failure >= self.threshold:
            self._state = "OPEN"
            self._state_last_failure = time.perf_counter()

    def record_success(self):
        self._total_failure = 0

        if self._state == "HALF_OPEN":
            self._state = "CLOSED"
