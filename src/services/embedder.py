from sentence_transformers import SentenceTransformer
import shutil
import logging
from ..tools.schemas import (
    CVChunk,
    CVEmbedding,
    EmbeddingDevice,
    InferenceStage,
    JRChunks,
    JREmbedding,
    EmbeddingModel,
    LoggerLayer,
    PipelineStage,
    PreprocessStage,
)
from ..tools.observabillity import TrackLatency, track_latency

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(
        self,
        latency_store: TrackLatency,
        model_name: EmbeddingModel = EmbeddingModel.MINI_LLM,
        device: EmbeddingDevice = EmbeddingDevice.CPU,
    ) -> None:
        self.latency_store = latency_store

        cuda_available = self._check_cuda()

        if device.value == "cuda":
            if cuda_available:
                self.model = SentenceTransformer(
                    model_name.value, device=device.value, local_files_only=True
                )

            elif not cuda_available:
                logger.warning(
                    "CUDA not available for embedding model switching to CPU",
                    extra={"layer": LoggerLayer.PIPELINE, "stage": PipelineStage.EMBED},
                )
                self.model = SentenceTransformer(
                    model_name.value,
                    device=EmbeddingDevice.CPU.value,
                    local_files_only=True,
                )

        elif device.value == "cpu":
            self.model = SentenceTransformer(model_name.value, device=device.value)

    def _check_cuda(self) -> bool:
        return shutil.which("nvidia-smi") is not None

    @track_latency(stage=PreprocessStage.EMBED)
    def embed_cv(
        self,
        cv_chunks: list[CVChunk],
        batch_size: int,
    ) -> list[CVEmbedding]:
        all_cv_embedding = []

        flat_cv_chunks = [cv_chunk.chunk for cv_chunk in cv_chunks]

        cv_embeddings = self.model.encode(
            flat_cv_chunks,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        all_cv_embedding = [
            CVEmbedding(idx=idx, embedding=cv_embedding)
            for idx, cv_embedding in enumerate(cv_embeddings)
        ]

        logger.debug(
            "Embedded CV",
            extra={
                "layer": LoggerLayer.PIPELINE,
                "stage": PreprocessStage.EMBED,
                "CV_embedding": len(all_cv_embedding),
            },
        )

        return all_cv_embedding

    @track_latency(stage=InferenceStage.EMBED)
    def embed_jr(
        self,
        jr_chunks: list[JRChunks],
        batch_size: int,
    ) -> list[JREmbedding]:

        all_components = []
        all_queries = []

        component_lengths = []

        for jr_chunk in jr_chunks:
            all_components.extend(jr_chunk.components)
            all_queries.append(jr_chunk.job_requirement)
            component_lengths.append(len(jr_chunk.components))

        all_components_embeddings = self.model.encode(
            all_components,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        all_query_embeddings = self.model.encode(
            all_queries,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        all_jr_embedding = []
        component_cursor = 0

        for idx, jr_chunk in enumerate(jr_chunks):

            component_count = component_lengths[idx]

            component_embeddings = all_components_embeddings[
                component_cursor : component_cursor + component_count
            ]

            component_cursor += component_count

            jr_embedding = JREmbedding(
                idx=jr_chunk.idx,
                job_requirement=jr_chunk.job_requirement,
                components=jr_chunk.components,
                components_embedding=component_embeddings,
                job_requirement_embedding=all_query_embeddings[idx],
            )

            all_jr_embedding.append(jr_embedding)

        logger.debug(
            "Embedded JR",
            extra={
                "layer": LoggerLayer.PIPELINE,
                "stage": InferenceStage.EMBED,
                "JR_embedding": len(all_jr_embedding),
            },
        )

        return all_jr_embedding
