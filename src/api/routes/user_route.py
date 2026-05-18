from fastapi import APIRouter, Depends, status
from typing import Annotated
import asyncio
from ...IO.artifact_manager import ArtifactManager
from ...pipelines.inference_pipeline import InferencePipeline
from ...pipelines.preprocess_pipeline import PreprocessPipeline
from ...api.dependencies import (
    get_artifact_manager,
    get_inference_pipeline,
    get_preprocess_pipeline,
    run_api_security,
    run_llm_abuse_protection,
)
from ...tools.schemas import CVInput, BaseResponse, Report, JRInput

user_router = APIRouter()


@user_router.post(
    "/inference/{candidate_name}",
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Successful inference"},
        401: {"description": "Missing or invalid API key"},
        403: {"description": "Forbidden route access"},
        429: {"description": "Too many requests"},
        503: {"description": "Service temporarily unavailable"},
        504: {"description": "LLM service timeout"},
    },
)
async def predict(
    candidate_name: str,
    jr_input: JRInput,
    inference_pipeline: Annotated[
        InferencePipeline,
        Depends(get_inference_pipeline),
    ],
    api_key: Annotated[str, Depends(run_api_security)],
    _: Annotated[None, Depends(run_llm_abuse_protection)],
) -> Report:
    task = asyncio.create_task(
        inference_pipeline.run(
            candidate_name=candidate_name,
            jr_input=jr_input,
            api_key=api_key,
        ),
        name=f"inference-{candidate_name}",
    )
    return await task


@user_router.post(
    "/preprocess/{candidate_name}",
    status_code=status.HTTP_201_CREATED,
    responses={
        201: {"description": "Successful preprocess"},
        401: {"description": "Missing or invalid API key"},
        403: {"description": "Forbidden route access"},
        429: {"description": "Too many requests"},
        503: {"description": "Service temporarily unavailable"},
        504: {"description": "LLM service timeout"},
    },
)
async def preprocess(
    cv_input: CVInput,
    candidate_name: str,
    preprocess_pipeline: Annotated[
        PreprocessPipeline, Depends(get_preprocess_pipeline)
    ],
    api_key: Annotated[str, Depends(run_api_security)],
    _: Annotated[None, Depends(run_llm_abuse_protection)],
) -> BaseResponse:
    task = asyncio.create_task(
        preprocess_pipeline.run(
            cv_input=cv_input, candidate_name=candidate_name, api_key=api_key
        ),
        name=f"preprocess-{candidate_name}",
    )
    await task
    return BaseResponse(status="completed", candidate_name=candidate_name)


@user_router.delete("/delete/{candidate_name}", status_code=status.HTTP_204_NO_CONTENT)
def delete(
    candidate_name: str,
    artifact_manager: Annotated[ArtifactManager, Depends(get_artifact_manager)],
    _: Annotated[None, Depends(run_llm_abuse_protection)],
) -> None:
    artifact_manager.delete_candidate(candidate_name=candidate_name)
