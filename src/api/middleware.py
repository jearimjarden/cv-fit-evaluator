from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from typing import Callable, Awaitable
import logging
import time
import uuid
from ..tools.schemas import LoggerLayer

logger = logging.getLogger(__name__)


class BaseMiddleWare(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        start_time = time.perf_counter()

        parts = request.url.path.split("/")
        first_url = f"/{parts[1]}" if len(parts) > 1 else "/"

        if first_url in ["/inference", "/preprocess"]:
            request.app.state.total_request += 1

        request.state.request_id = str(uuid.uuid4())

        response = await call_next(request)

        latency_ms = round((time.perf_counter() - start_time) * 1000, 3)

        logger.info(
            "Request completed",
            extra={
                "layer": LoggerLayer.MIDDLEWARE,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "latency_ms": latency_ms,
                "request_id": request.state.request_id,
            },
        )

        return response
