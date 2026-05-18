import logging
import traceback
import json
import os
from datetime import datetime
from pathlib import Path
from .schemas import APIStage, PipelineStage


class JSONFormatter(logging.Formatter):
    def __init__(self, environment: str):
        self.environment = environment

    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "levelname": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
            "timestamp": datetime.now().strftime("%d-%m-%Y_%H:%M"),
            "environment": self.environment,
        }

        if record.exc_info:
            log_record["traceback"] = "".join(
                traceback.format_exception(*record.exc_info)
            )

        for key, value in record.__dict__.items():
            if key not in (
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
            ):
                if value is not None:
                    log_record[key] = value

        return json.dumps(log_record, default=str)


def setup_logger(
    level: str,
    environment: str,
    save_log: bool,
    pipeline_name: PipelineStage | APIStage,
) -> None:
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper()))

    streamer = logging.StreamHandler()
    streamer.setFormatter(JSONFormatter(environment=environment))
    streamer.setLevel(getattr(logging, level.upper()))

    for handler in root.handlers[:]:
        root.removeHandler(handler)

    root.addHandler(streamer)

    if save_log:
        os.makedirs("logs", exist_ok=True)

        file_path = (
            Path("logs/")
            / f"{pipeline_name.value}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
        )
        file = logging.FileHandler(filename=file_path)
        file.setFormatter(JSONFormatter(environment=environment))
        file.setLevel(getattr(logging, level.upper()))

        root.addHandler(file)


def setup_bootstrap_logger() -> logging.Logger:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s: %(message)s",
        level=logging.INFO,
    )

    # Reduce verbose third-party library logs
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)

    return logging.getLogger("bootstrap")
