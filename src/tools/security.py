import time
import logging
from .exceptions_schemas import (
    InvalidAPIKey,
    LLMAbusedError,
    RateLimitExceeded,
    UnauthorizedRoute,
)
from .schemas import APIStage, AuthConfig, LoggerLayer

logger = logging.getLogger(__name__)


class APISecurity:
    def __init__(self, auth_config: AuthConfig):
        self.auth_config = auth_config
        self.api_keys = {}
        for api_name, api_detail in self.auth_config.api_keys.items():
            self.api_keys[api_detail.key] = {
                "api_name": api_name,
                "rate_limit": api_detail.rate_limit,
                "usage": [],
                "allowed_routes": api_detail.allowed_routes,
            }

    def run(self, api_key: str, request_url: str):
        self._check_api_key(api_key=api_key)
        self._check_usage_rate(api_key=api_key)
        self._check_route(api_key=api_key, request_url=request_url)

    def _check_api_key(self, api_key: str) -> None:
        if api_key in self.api_keys:
            pass
        else:
            raise InvalidAPIKey("Invalid API Key")

    def _check_usage_rate(self, api_key: str) -> None:
        rate_limit = self.api_keys[api_key]["rate_limit"]
        usage = self.api_keys[api_key]["usage"]
        time_now = time.perf_counter()

        self.api_keys[api_key]["usage"] = [
            u for u in usage if (time_now - u) < self.auth_config.rate_limit_window
        ]

        if len(self.api_keys[api_key]["usage"]) < rate_limit:
            self.api_keys[api_key]["usage"].append(time.perf_counter())

        else:
            logger.error(
                "Rate limit per API key exceeded",
                extra={"layer": LoggerLayer.SECURITY, "stage": APIStage.LIFESPAN},
            )
            raise RateLimitExceeded("Rate Limit has Exceeded")

    def _check_route(self, api_key: str, request_url: str):
        allowed_routes = self.api_keys[api_key]["allowed_routes"]
        parts = request_url.split("/")
        first_url = f"/{parts[1]}" if len(parts) > 1 else "/"

        if "*" in allowed_routes:
            return

        if first_url not in allowed_routes:
            raise UnauthorizedRoute("This route is forbidden")


class LLMAbuseProtection:
    def __init__(
        self,
        auth_config: AuthConfig,
        window_time: int,
        threshold: int,
        suspend_time: int,
    ):
        self.auth_config = auth_config
        self.window_time = window_time
        self.suspend_time = suspend_time
        self.threshold = threshold

        self.api_keys = {}

        for _, api_detail in self.auth_config.api_keys.items():
            self.api_keys[api_detail.key] = {
                "failures": [],
                "state": "CLOSED",
                "state_last_failure": 0.0,
            }

    def check(self, api_key: str) -> None:
        state = self.api_keys[api_key]["state"]

        if state == "CLOSED":
            return

        if state == "OPEN":
            elapsed_time = (
                time.perf_counter() - self.api_keys[api_key]["state_last_failure"]
            )

            if elapsed_time > self.suspend_time:
                self.api_keys[api_key]["state"] = "CLOSED"
                self.api_keys[api_key]["failures"] = []
                return

            remaining_time = round(
                self.suspend_time - elapsed_time,
                2,
            )

            logger.error(
                "LLM protector triggered",
                extra={"layer": LoggerLayer.SECURITY, "stage": APIStage.LIFESPAN},
            )
            raise LLMAbusedError(
                f"This api_key has been temporarily blocked for "
                f"{remaining_time} more seconds"
            )

    def record_failure(self, api_key: str) -> None:
        current_time = time.perf_counter()

        failures = self.api_keys[api_key]["failures"]

        self.api_keys[api_key]["failures"] = [
            failure_time
            for failure_time in failures
            if (current_time - failure_time) < self.window_time
        ]

        self.api_keys[api_key]["failures"].append(current_time)

        if len(self.api_keys[api_key]["failures"]) >= self.threshold:
            self.api_keys[api_key]["state"] = "OPEN"
            self.api_keys[api_key]["state_last_failure"] = current_time
