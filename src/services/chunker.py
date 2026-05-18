import json
import logging
import asyncio
from .prompt_builder import create_component_prompt, create_correction_prompt
from .llm_client import LLMClient
from ..tools.observabillity import TrackToken
from ..tools.security import LLMAbuseProtection
from ..tools.schemas import (
    CVChunk,
    InferenceStage,
    LoggerLayer,
    PreprocessStage,
    StructuredCVItem,
    StructuredCVLanguage,
    JRChunks,
)

logger = logging.getLogger(__name__)


async def decompose_and_validate_jr(
    jr_parsed_text: list,
    llm_client: LLMClient,
    llm_abuse_protection: LLMAbuseProtection,
    request_track_token: TrackToken,
    api_key: str,
) -> list[JRChunks]:
    all_chunks = []

    async def process_parse(idx: int, jr_text: str):
        prompt = create_component_prompt(jr_text=jr_text)
        response = await llm_client.generate(
            prompt=prompt,
            stage=InferenceStage.CHUNK,
            request_track_token=request_track_token,
        )

        try:
            dict_response = json.loads(response)

        except json.JSONDecodeError:
            llm_abuse_protection.record_failure(api_key=api_key)
            logger.warning(
                "Invalid JSON output detected, attempting JSON repair",
                extra={"layer": LoggerLayer.PIPELINE, "stage": InferenceStage.CHUNK},
            )
            dict_response = await llm_client.json_repair(
                context=response,
                request_track_token=request_track_token,
                api_key=api_key,
            )

        dict_response["idx"] = idx
        all_chunks.append(JRChunks(**dict_response))

    tasks = [
        asyncio.create_task(
            process_parse(idx=idx, jr_text=jr_text), name=f"task_chunker_{idx}"
        )
        for idx, jr_text in enumerate(jr_parsed_text)
    ]

    await asyncio.gather(*tasks)

    validated_chunks = await _validate_jr_chunks(
        jr_chunks=all_chunks,
        llm_client=llm_client,
        request_track_token=request_track_token,
        api_key=api_key,
    )
    logger.debug(
        "Chunked JR",
        extra={
            "layer": LoggerLayer.PIPELINE,
            "stage": InferenceStage.CHUNK,
            "result": all_chunks,
        },
    )
    return validated_chunks


async def _validate_jr_chunks(
    jr_chunks: list[JRChunks],
    llm_client: LLMClient,
    request_track_token: TrackToken,
    api_key: str,
) -> list[JRChunks]:
    async def process_chunk(jr_decom: JRChunks) -> JRChunks:

        invalid_components = []

        for component in jr_decom.components:

            c = component.lower().strip()

            if len(c.split()) < 2 or c.startswith(
                ("for ", "in ", "with ", "using ", "to ")
            ):
                invalid_components.append(c)

        if invalid_components:

            logger.warning(
                "Repairing invalid JR requirement",
                extra={
                    "layer": LoggerLayer.PIPELINE,
                    "stage": InferenceStage.CHUNKREPAIR,
                    "invalid_components": invalid_components,
                },
            )

            prompt = create_correction_prompt(
                jr_text=jr_decom.job_requirement,
                invalid_components=invalid_components,
                jr_components=jr_decom.components,
            )

            response = await llm_client.generate(
                prompt=prompt,
                stage=InferenceStage.CHUNKREPAIR,
                request_track_token=request_track_token,
            )

            try:
                dict_answer = json.loads(response)

            except json.JSONDecodeError:

                logger.warning(
                    "Invalid JSON output detected, attempting JSON repair",
                    extra={
                        "layer": LoggerLayer.PIPELINE,
                        "stage": InferenceStage.CHUNK,
                    },
                )

                dict_answer = await llm_client.json_repair(
                    context=response,
                    request_track_token=request_track_token,
                    api_key=api_key,
                )

            return JRChunks(
                idx=jr_decom.idx,
                job_requirement=jr_decom.job_requirement,
                components=dict_answer["components"],
                reason=jr_decom.reason,
            )

        return jr_decom

    tasks = [
        asyncio.create_task(
            process_chunk(jr_decom),
            name=f"jr_validation_{jr_decom.idx}",
        )
        for jr_decom in jr_chunks
    ]

    validated_jr_components = await asyncio.gather(*tasks)

    return validated_jr_components


def chunk_cv_semantic(
    technical_skills: list[StructuredCVItem],
    work_experiences: list[StructuredCVItem],
    projects: list[StructuredCVItem],
    languages: list[StructuredCVLanguage],
    soft_skills: list[str],
) -> list[CVChunk]:
    chunk_idx = 0
    all_chunk = []

    for skill in technical_skills:
        skills = ", ".join(skill.item)
        if skills:
            chunk = f"Technical Skills ({skill.name}): {skills}"
        else:
            chunk = f"Technical Skills: {skill.name}"
        all_chunk.append(CVChunk(idx=chunk_idx, type="Technical Skill", chunk=chunk))
        chunk_idx += 1

    for experience in work_experiences:
        for item in experience.item:
            if item:
                chunk = f"Work Experience ({experience.name}): {item}"
            else:
                chunk = f"Work Experience: {experience.name}"

            all_chunk.append(
                CVChunk(idx=chunk_idx, type="Work Experience", chunk=chunk)
            )
            chunk_idx += 1

    for project in projects:
        for item in project.item:
            if item:
                chunk = f"Project ({project.name}): {item}"
            else:
                chunk = f"Project: {project.name}"

            all_chunk.append(CVChunk(idx=chunk_idx, type="Project", chunk=chunk))
            chunk_idx += 1

    for language in languages:
        if language.level:
            chunk = f"Language: {language.name} ({language.level})"
        else:
            chunk = f"Language: {language.name}"

        all_chunk.append(CVChunk(idx=chunk_idx, type="Language", chunk=chunk))
        chunk_idx += 1

    for idx in range(0, len(soft_skills), 2):
        soft_skill = ", ".join(soft_skills[idx : idx + 3])
        chunk = f"Soft Skills: {soft_skill}"
        all_chunk.append(CVChunk(idx=chunk_idx, type="Soft Skills", chunk=chunk))
        chunk_idx += 1

    logger.debug(
        "Chunked CV",
        extra={
            "layer": LoggerLayer.PIPELINE,
            "stage": PreprocessStage.CHUNK,
            "result": all_chunk,
        },
    )
    return all_chunk


def _legacy_chunk_cv(text_experience: str, text_skills: str) -> list:
    """Legacy custom CV chunking:
    - Added title/category name for each chunk
    - Added subtile name (project name) for experience

    Notes: Unused for active pipeline"""

    experience_chunks = []
    experiences_splitted = text_experience.split("\n")
    experiences_normalized = [x for x in experiences_splitted[:] if x]

    context = ""
    subtitle = ""
    for experience in experiences_normalized:
        if experience.strip().startswith("-"):
            context = experience
        elif not experience.strip().startswith("-"):
            subtitle = experience
        if subtitle and context:
            experience_chunks.append(f"Experience: {subtitle} {context}")
            context = ""

    skills_chunks = []
    skills_splitted = text_skills.split("\n")
    skills_normalized = [x for x in skills_splitted if x]

    for skill in skills_normalized:
        skill = skill.replace("-", "")
        skill.strip()
        skills_chunks.append(f"Skills:{skill}")

    return experience_chunks + skills_chunks
