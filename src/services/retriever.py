import faiss
import numpy as np
import logging
from ..tools.schemas import (
    BaseRetrieval,
    BaseRetrievalComponent,
    BaseRetrievalQuery,
    BaseSearch,
    BaseSearchComponents,
    BaseSearchQuery,
    CVEmbedding,
    InferenceStage,
    JREmbedding,
    LoggerLayer,
)

logger = logging.getLogger(__name__)


def faiss_ip_search(
    cv_embedding: list[CVEmbedding],
    jr_embedding: list[JREmbedding],
    query_top_k: int,
    component_top_k: int,
) -> list[BaseSearch]:
    flat_cv_embedding = []

    for embedding in cv_embedding:
        flat_cv_embedding.append(embedding.embedding)

    flat_cv_embedding = np.array(flat_cv_embedding, dtype=np.float32)

    dim = flat_cv_embedding.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(flat_cv_embedding)  # type: ignore

    all_base_search = []
    for jr in jr_embedding:
        components_retrieval = []

        for idx, component in enumerate(jr.components):
            component_embedding = jr.components_embedding[idx].astype("float32")
            component_embedding = component_embedding.reshape(1, -1)
            faiss.normalize_L2(component_embedding)

            distances, indices = index.search(component_embedding, component_top_k)  # type: ignore
            components_retrieval.append(
                BaseSearchComponents(
                    component=component, distances=distances[0], indices=indices[0]
                )
            )

        query_embedding = jr.job_requirement_embedding.astype("float32")
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        distances, indices = index.search(query_embedding, query_top_k)  # type: ignore
        query_retrieval = BaseSearchQuery(
            query=jr.job_requirement, distances=distances[0], indices=indices[0]
        )

        all_base_search.append(
            BaseSearch(
                idx=jr.idx,
                query_search=query_retrieval,
                components_search=components_retrieval,
            )
        )

    logger.debug(
        "Vector searched",
        extra={
            "layer": LoggerLayer.PIPELINE,
            "stage": InferenceStage.RETRIEVAL,
            "result": all_base_search,
        },
    )
    return all_base_search


def retrieve_base_chunk(
    search_result: list[BaseSearch],
    idx_to_chunk: dict[int, str],
    threshold: float,
    filter_below_threshold: bool,
) -> list[BaseRetrieval]:
    all_retrieved_chunks = []

    for result in search_result:
        query_retrieved_chunks = []
        query_retrieved_distances = []

        for idx, indice in enumerate(result.query_search.indices):
            distance = result.query_search.distances[idx]
            if distance < threshold:
                if filter_below_threshold:
                    logger.info(
                        "Skipping low semantic retrieved chunk",
                        extra={
                            "layer": LoggerLayer.PIPELINE,
                            "stage": InferenceStage.RETRIEVAL,
                            "query": result.query_search.query,
                            "chunk": idx_to_chunk[indice],
                            "score": distance,
                        },
                    )
                    continue

                logger.warning(
                    "Allowing low semantic retrieved chunk",
                    extra={
                        "layer": LoggerLayer.PIPELINE,
                        "stage": InferenceStage.RETRIEVAL,
                        "query": result.query_search.query,
                        "chunk": idx_to_chunk[indice],
                        "score": distance,
                    },
                )

            query_retrieved_chunks.append(idx_to_chunk[indice])
            query_retrieved_distances.append(result.query_search.distances[idx])

        retrieved_query = BaseRetrievalQuery(
            query=result.query_search.query,
            distances=query_retrieved_distances,
            chunks=query_retrieved_chunks,
        )

        retrieved_components = []
        for component in result.components_search:
            components_retrieved_chunks = []
            components_retrieved_distances = []
            for idx, indice in enumerate(component.indices):
                distance = component.distances[idx]

                if distance < threshold:
                    if filter_below_threshold:
                        logger.info(
                            "Skipping low semantic retrieved chunk",
                            extra={
                                "layer": LoggerLayer.PIPELINE,
                                "stage": InferenceStage.RETRIEVAL,
                                "component": component.component,
                                "chunk": idx_to_chunk[indice],
                                "score": distance,
                            },
                        )
                        continue

                    logger.warning(
                        "Allowing low semantic retrieved chunk",
                        extra={
                            "layer": LoggerLayer.PIPELINE,
                            "stage": InferenceStage.RETRIEVAL,
                            "component": component.component,
                            "chunk": idx_to_chunk[indice],
                            "score": distance,
                        },
                    )

                components_retrieved_chunks.append(idx_to_chunk[indice])
                components_retrieved_distances.append(distance)

            retrieved_components.append(
                BaseRetrievalComponent(
                    component=component.component,
                    distances=components_retrieved_distances,
                    chunks=components_retrieved_chunks,
                )
            )

        all_retrieved_chunks.append(
            BaseRetrieval(
                idx=result.idx,
                query_retrieval=retrieved_query,
                components_retrieval=retrieved_components,
            )
        )

    logger.debug(
        "Chunk retrieved",
        extra={
            "layer": LoggerLayer.PIPELINE,
            "stage": InferenceStage.RETRIEVAL,
            "result": all_retrieved_chunks,
        },
    )

    return all_retrieved_chunks
