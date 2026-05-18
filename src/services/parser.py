import json
import re
import logging
from nltk.tokenize import sent_tokenize
from .llm_client import LLMClient
from .prompt_builder import create_cv_parser_prompt
from ..tools.observabillity import TrackToken
from ..tools.security import LLMAbuseProtection
from ..tools.schemas import InferenceStage, LoggerLayer, PreprocessStage, StructuredCV
from ..tools.exceptions_schemas import InvalidJRError

logger = logging.getLogger(__name__)


async def parse_cv_llm(
    cv_text: str,
    llm_client: LLMClient,
    request_track_token: TrackToken,
    llm_abuse_protection: LLMAbuseProtection,
    api_key: str,
) -> StructuredCV:
    try:
        prompt = create_cv_parser_prompt(cv_text=cv_text)
        response = await llm_client.generate(
            prompt, stage=PreprocessStage.PARSE, request_track_token=request_track_token
        )
        dict_response = json.loads(response)
        structured_cv = StructuredCV(**dict_response)

        logger.debug(
            "Parsed CV",
            extra={
                "layer": LoggerLayer.PIPELINE,
                "stage": PreprocessStage.PARSE,
                "result": structured_cv,
            },
        )
        return structured_cv

    except json.JSONDecodeError:
        llm_abuse_protection.record_failure(api_key=api_key)
        logger.warning(
            "Invalid JSON output detected, attempting JSON repair",
            extra={"layer": LoggerLayer.PIPELINE, "stage": PreprocessStage.PARSE},
        )

        repaired_response = await llm_client.json_repair(
            context=response, request_track_token=request_track_token, api_key=api_key
        )
        structured_cv = StructuredCV(**repaired_response)

        return structured_cv


def parse_normalize_jr(text: str, chunk_size: int, stride: int) -> list[str]:
    """Simple parser separating text by new line"""

    sentences = text.split("\n")

    all_chunks = []
    if len(sentences) < chunk_size:
        return sentences

    for x in range(0, len(sentences) - chunk_size + 1, stride):
        chunk = ". ".join(sentences[x : x + chunk_size])
        all_chunks.append(chunk)

    all_chunks = [x for x in all_chunks if x]

    normalized_chunk = _normalize_jr_text(parsed_text=all_chunks)

    if len(normalized_chunk) > 25:
        raise InvalidJRError("Parsed JR could not exceed 25 parses")

    logger.debug(
        "Parsed JR",
        extra={
            "layer": LoggerLayer.PIPELINE,
            "stage": InferenceStage.PARSE,
            "result": normalized_chunk,
        },
    )

    return normalized_chunk


def _normalize_jr_text(parsed_text: list[str]) -> list[str]:
    normalized_text = []

    for text in parsed_text:
        parts = re.split(r"(?:^|\n)(?:\d+[.)]|[-•·])\s+", text)
        parts = [p.strip() for p in parts if p.strip()]
        normalized_text.extend(parts)

    return normalized_text


def _legacy_parse_text_regex(text: str):
    """Legacy parser using regex that separated by:
    - Field name structure: Word that ends with ':' followed by optional whitespace and new line
    - Value: text separated by each Field name or end of the text

    Status:
    - Not used in current active pipeline

    Notes:
    - Hard to separated each parsed text category"""

    all_chunks = []
    result = re.findall(r"(\w+):\s*\n([\S\s]+?)(?:\n\n|$)", text)
    for field, value in result:
        all_chunks.append(f"{field}: {value}")
    return all_chunks


def _legacy_parse_structured_text(text: str) -> dict:
    """Legacy parser separating field name:
    - Name
    - Summary
    - Skills
    - Experience
    - Education

    Status:
    - active"""

    pattern = r"(Name|Summary|Skills|Experience|Education):\s*\n"
    splits = re.split(pattern, text)

    all_split = {}
    for x in range(1, len(splits), 2):
        key = splits[x]
        value = splits[x + 1]
        all_split[key] = value

    return all_split


def _legacy_parse_text_nltk(text: str, chunk_size: int, stride: int) -> list[str]:
    """Legacy parser using NLTK sentence tokenization.

    Current Status:
    - Not used in the active pipeline.

    Notes:
    - Failed to separate Index Number followed by '.'"""

    sentences = sent_tokenize(text)
    all_chunks = []

    for x in range(0, len(sentences) - chunk_size + 1, stride):
        chunk = ". ".join(sentences[x : x + chunk_size])
        all_chunks.append(chunk)

    return all_chunks
