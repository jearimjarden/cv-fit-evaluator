from pathlib import Path
import json
import numpy as np
import re
from dataclasses import asdict
from datetime import datetime
import logging
import shutil
from pydantic import ValidationError
from ..tools.exceptions_schemas import (
    ArtifactNotFound,
    CorruptedArtifact,
    ExistingArtifact,
)
from ..tools.schemas import (
    APIStage,
    CVChunk,
    CVEmbedding,
    CandidateList,
    LoggerLayer,
    Metadata,
    TokenSummary,
    StructuredCV,
    PreprocessStage,
    LatencyStored,
)

logger = logging.getLogger(__name__)


class ArtifactManager:
    def __init__(self):
        pass

    def load_cv_all(
        self, candidate_name: str
    ) -> tuple[Metadata, list[CVChunk], list[CVEmbedding]]:
        load_path = Path("storage/candidates") / candidate_name
        if not load_path.exists():
            raise ArtifactNotFound(f"CV data for '{candidate_name}' not found")

        try:
            metadata = self._load_cv_metadata(load_path=load_path)
            cv_chunks = self._load_cv_chunks(load_path=load_path)
            cv_embedding = self._load_cv_embedding(load_path=load_path)

            if metadata.cv_embedding_n != metadata.cv_chunked_n:
                raise CorruptedArtifact(
                    "Invalid Metadata (cv_chunked_n and cv_embedding_n does not match)"
                )

            if metadata.cv_embedding_n != len(cv_embedding):
                raise CorruptedArtifact("Metadata and cv_embedding does not match)")

            if len(cv_chunks) != len(cv_embedding):
                raise CorruptedArtifact(
                    f"CV chunks and embedding does not match (cv_chunks_n: {len(cv_chunks)}, cv_embedding_n: {len(cv_embedding)})"
                )

            return metadata, cv_chunks, cv_embedding

        except FileNotFoundError as e:
            match = re.search(r"'([^']+)'", str(e))

            if match:
                raise ArtifactNotFound(f"File was not found for '{match.group(1)}'")

            else:
                raise ArtifactNotFound(str(e))

        except TypeError as e:
            raise CorruptedArtifact(f"Invalid metadata schema: {e}") from e

        except ValidationError as e:
            messages = []

            for err in e.errors():
                field = ".".join(str(x) for x in err["loc"])

                if err["type"] == "missing":
                    messages.append(f"Missing artifact parameter: '{field}'")

                elif err["type"] == "extra_forbidden":
                    messages.append(f"Forbidden extra artifact parameter: '{field}'")

                else:
                    messages.append(
                        f"Invalid artifact value for '{field}': {err['msg']}"
                    )

            raise CorruptedArtifact(" | ".join(messages)) from e

    def check_cv_name(self, cv_name_str: str) -> None:
        storage_dir = Path("storage")
        storage_dir.mkdir(exist_ok=True)

        candidates_dir = storage_dir / "candidates"
        candidates_dir.mkdir(exist_ok=True)

        candidate_dir = candidates_dir / cv_name_str

        if candidate_dir.exists():
            # future add overwrite logic
            # logger.warning(
            #     f"Rewriting existed CV for '{cv_name_str}'",
            #     extra={"layer": LoggerLayer.PIPELINE, "stage": PreprocessStage.SAVE},
            # )
            logger.error(
                f"Cant rewrite existed cv: '{cv_name_str}'",
                extra={"layer": LoggerLayer.PIPELINE, "stage": PreprocessStage.SAVE},
            )
            raise ExistingArtifact("Cannot rewrite existing artifact")

    def save_all_cv(
        self,
        cv_parsed: StructuredCV,
        cv_chunk: list[CVChunk],
        cv_embedding: list[CVEmbedding],
        cv_name_str: str,
        token_summary: TokenSummary,
        latency_stored: LatencyStored,
    ) -> None:
        storage_dir = Path("storage")
        storage_dir.mkdir(exist_ok=True)

        candidates_dir = storage_dir / "candidates"
        candidates_dir.mkdir(exist_ok=True)

        candidate_dir = candidates_dir / cv_name_str

        candidate_dir.mkdir(exist_ok=True)

        self._save_cv_parsed(cv_parsed=cv_parsed, save_path=candidate_dir)
        self._save_cv_chunk(cv_chunks=cv_chunk, save_path=candidate_dir)
        cv_embedding_n = self._save_cv_embedding(
            cv_embedding=cv_embedding, save_path=candidate_dir
        )
        self._save_cv_metadata(
            name=cv_parsed.person_name,
            cv_embedding_n=cv_embedding_n,
            cv_parsed_n=len(cv_parsed),
            cv_chunks_n=len(cv_chunk),
            token_summary=token_summary,
            save_path=candidate_dir,
            latency_stored=latency_stored,
        )

        logger.info(
            f"CV artifacts saved for '{cv_name_str}'",
            extra={
                "layer": LoggerLayer.PIPELINE,
                "stage": PreprocessStage.SAVE,
                "save_path": "/" + "/".join(candidate_dir.parts[-2:]),
            },
        )

    def delete_candidate(self, candidate_name: str):
        delete_path = Path("storage/candidates") / candidate_name

        if not delete_path.exists():
            logger.error(
                f"CV deletion for '{candidate_name}' not found",
                extra={
                    "layer": LoggerLayer.EXCEPTION,
                    "stage": APIStage.ARTIFACT,
                    "error_type": "ArtifactError",
                    "code": "ArtifactNotFound",
                },
            )
            raise ArtifactNotFound(f"CV deletion for '{candidate_name}' not found")

        shutil.rmtree(delete_path)

    def list_candidates(self) -> CandidateList:
        list_path = Path("storage/candidates")

        if not list_path.exists():
            logger.error(
                "There is no candidate's artifact",
                extra={
                    "layer": LoggerLayer.EXCEPTION,
                    "stage": APIStage.ARTIFACT,
                    "error_type": "ArtifactError",
                    "code": "ArtifactNotFound",
                },
            )
            raise ArtifactNotFound("There is not any canidate's artifact found")
        candidates_list = [p.name for p in list_path.iterdir() if p.is_dir()]

        return CandidateList(candidates_list=candidates_list)

    def get_candidate_metadata(self, candidate_name: str) -> Metadata:
        candidate_path = Path("storage/candidates") / candidate_name

        if not candidate_path.exists():
            logger.error(
                f"CV metadata for '{candidate_name}' not found",
                extra={
                    "layer": LoggerLayer.EXCEPTION,
                    "stage": APIStage.ARTIFACT,
                    "error_type": "ArtifactError",
                    "code": "ArtifactNotFound",
                },
            )
            raise ArtifactNotFound(f"CV metadata for '{candidate_name}' not found")

        return self._load_cv_metadata(load_path=candidate_path)

    def _load_cv_metadata(self, load_path: Path) -> Metadata:
        with open(load_path / "metadata.json", "r", encoding="utf-8") as f:
            data = json.load(f)

            return Metadata(**data)

    def _load_cv_chunks(self, load_path: Path) -> list[CVChunk]:
        with open(load_path / "cv_chunks.json", "r", encoding="utf-8") as f:
            datas = json.load(f)

        cv_chunks = [CVChunk(**data) for data in datas]
        return cv_chunks

    def _load_cv_embedding(self, load_path: Path) -> list[CVEmbedding]:
        loaded_embeddings = np.load(load_path / "cv_embedding.npy")
        cv_embedding = []

        for idx, embedding in enumerate(loaded_embeddings):
            cv_embedding.append(CVEmbedding(idx=idx, embedding=embedding))

        return cv_embedding

    def _save_cv_parsed(self, cv_parsed: StructuredCV, save_path: Path) -> None:
        data = cv_parsed.model_dump()
        save_file = save_path / "cv_parsed.json"

        with open(save_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        logger.debug(
            "Parsed CV saved",
            extra={
                "layer": LoggerLayer.PIPELINE,
                "stage": PreprocessStage.SAVE,
                "save_path": save_file,
            },
        )

    def _save_cv_chunk(self, cv_chunks: list[CVChunk], save_path: Path) -> None:
        data = [chunk.model_dump() for chunk in cv_chunks]
        save_file = save_path / "cv_chunks.json"

        with open(save_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        logger.debug(
            "CV chunk saved",
            extra={
                "layer": LoggerLayer.PIPELINE,
                "stage": PreprocessStage.SAVE,
                "save_path": save_file,
            },
        )

    def _save_cv_embedding(
        self, cv_embedding: list[CVEmbedding], save_path: Path
    ) -> int:
        data = [chunk.embedding for chunk in cv_embedding]
        save_file = save_path / "cv_embedding.npy"

        np.save(save_file, data)

        logger.debug(
            "CV embedding saved",
            extra={
                "layer": LoggerLayer.PIPELINE,
                "stage": PreprocessStage.SAVE,
                "save_path": save_file,
            },
        )

        return len(data)

    def _save_cv_metadata(
        self,
        name: str,
        cv_embedding_n: int,
        cv_parsed_n: int,
        cv_chunks_n: int,
        token_summary: TokenSummary,
        save_path: Path,
        latency_stored: LatencyStored,
    ) -> None:
        data = {
            "cv_name": name,
            "created_date": datetime.now().strftime("%d-%m-%Y_%H:%M"),
            "cv_parsed_n": cv_parsed_n,
            "cv_chunked_n": cv_chunks_n,
            "cv_embedding_n": cv_embedding_n,
            "token_summary": asdict(token_summary),
            "latencies_ms": asdict(latency_stored),
        }
        save_file = save_path / "metadata.json"

        with open(save_file, "w") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        logger.debug(
            "CV metadata saved",
            extra={
                "layer": LoggerLayer.PIPELINE,
                "stage": PreprocessStage.SAVE,
                "save_path": save_file,
            },
        )
