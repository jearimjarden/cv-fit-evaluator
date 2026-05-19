# Developer Notes
- Documents architecture decisions, implementation reasoning, tradeoffs, limitations, and observed challenges across development stages
- Provides technical notes and system-level context for future development and maintenance

## Problem
Most CV–job matching systems rely heavily on keyword-based filtering (e.g., matching “Python” or “Docker”).

This approach has several limitations:
- Lack of context — Keywords do not capture how skills are applied (e.g., “FastAPI” may imply REST API experience without explicitly stating it)
- Implicit requirements — Job requirements are often described indirectly, making exact keyword matching insufficient
- Limited explainability — Traditional systems often return scores without grounded reasoning or supporting evidence
- No prioritization control — Systems usually lack distinction between required and nice-to-have qualifications

## Objective
Build a system that:
- Evaluates job requirements individually using evidence-based analysis
- Retrieves explicit and implicit supporting evidence from the candidate’s CV
- Generates grounded fit scoring and reasoning
- Identifies unsupported or missing requirements explicitly
- Reduces reliance on keyword-only matching
- Supports future requirement weighting (must-have vs nice-to-have)

## CV-fit RAG System v1.0 (6 May 2026)
### a. Job Requirement Parser and Chunking
Decision:
- Job Requirements input should be structured
- Job Requirements parsed using "\n" (new line)
- Parsed Job Requirement straight used as chunks for embedding

Reason:
- Job Requirements posting in real-world usually structured and could be parsed by new line
- Total Job Requirement usually short so we straight use parsed Job Requirement as chunk. If we combine Job Requirement, the semantic meaning of each Job Requirement will be bias because of pooling dillution 
- This a evaluation system so Job Requirement's chunks need to have a strong semantic meaning that can be compared to evidence (evidence from CV usually has indirect/implicit meaning)

Notes:
- We could implement OCR and parsing using strict LLM to get structured Job Requirement (Need to be strict so LLM would not change exact word from Job Requirement)
- Need normalization for each parsed text

### b. Regex CV Parser
Decision:
- CV Parsed using Regex for words ("Name", "Summary", "Skills", "Experience", "Education")
- Text that separated by Regex will parsed again using "\n" (new line)

Reason:
- CV usually has a semi structured that we can separate by using regex
- This is not recommend for long term (because of probability of messy CV input) but for RAG testing we will use this as CV parser

Notes:
- Implement text normalization before searching with Regex (e.g "Name" to "name", " name " to "name")
- We could implement OCR and parser using strict LLM to get structutured CV

### c. Custom CV Chunking
Decision:
- Augmented CV chunks with contextual labels (e.g., section titles like Experience, Skills, Projects) and associated subtitles when available.

    Example:
    CV: 
    "Experience:
    Backend Developer Intern
    - Built REST APIs using FastAPI for internal tools"

    CV Chunk:
    "Experience: Backend Developer Intern Built REST APIs using FastAPI for internal tools"

Reason:
- Preserved contextual labels (e.g., Experience, Skills, Projects) to improve semantic alignment with similarly structured job requirements.
- Strengthens downstream evaluation (e.g., evidence scoring and capability classification) by providing clearer intent and role framing.

Notes:
- This Chunking require prestructured semi structured CV, random CV structured will fails resulting in random or incorrect chunk

### d. Job Requirement Decomposer
Decision:
- Decompose (if able) each Job Requirement into components using LLM
- Semi strict prompt are used
    
    Example:
        Job Requirement:
            "Experience applying machine learning techniques to real problems"

        Components:
            - "Experience applying machine learning techniques"
            - "Solve real problem using machine learning"

Reason:
- Job requirements define the target evidence, so decomposing them into specific, granular components enables more precise matching and evaluation,
- Improved Evaluation scoring by decomposing Job Requirement into evaluatable components,
- Used an LLM to decompose job requirements, as their complexity and variability cannot be reliably handled with rule-based systems alone.
- Very strict prompt usually decompose Job Requirement per word not context, while loose prompt usually added concept outside the Job Requirement's context.

Limitation:
- Job Requirement that is short, combined level of proficiency and the skills sometimes return uncontextualized components
- Job Requirement with many verbs return uncontextual components

Observed Issues:
- A sentence with many verbs return uncontextual components
    
    Example:
        Job Requirement:
            Collaborate using modern development tools and workflows

        Components:
            - Collaborate
            - using modern development tools
            - using modern development workflows

- A sentence that combined level of proficiency and the skills

        Example:
        Job Requirement:
            Strong proficiency in Python

        Components:
            - Strong proficiency
            - in Python

### e. Decomposer LLM failure handling
Decision:
- Catch invalid components using rule-based system
- Generate new contextualized components using LLM

    Example:
        Invalid Components:
            - Knowledge of relational databases
            - SQL

        Fixed Components:
            - Knowledge of relational databases
            - Knowledge of relational SQL

Reason:
- Decomposing using LLM still has probability of hallucinating, we need to handle a invalid components
- Rule-based invalid components catcher are used to ensure the reliability of the system
- Invalid components and original Job Requirement are used for LLM to fix the invalid components

Limitation:
- Rule-based classification of components has limited flexibility, making it difficult to capture diverse and nuanced requirement structures.

Notes:
- Continuously analyze decomposed outputs to refine and expand the rule-based classification logic.
- If possible use ML's classification method to catch bad component

### f. Vector Search Using FAISS
Decision:
- Used FAISS IndexFlatIP for vector similarity search

Reason:
- The system does not require large-scale vector retrieval because retrieval is only performed against a relatively small number of CV chunks (~20–30 chunks)
- More advanced FAISS indexes such as IVF or HNSW are unnecessary for the current retrieval scale and would introduce additional complexity without meaningful performance benefits
- Exact vector search using IndexFlatIP is sufficiently fast and simpler to maintain for the current system architecture

### g. Custom Retrieval or evidence preparation
Decision:
- Combined component-level retrieval with global retrieval using the original job requirement.
- Configurable Top-K for both component and global retrieval.

Reason:
- Component-level retrieval provides high precision but limited recall.
- Global retrieval (original job requirement) improves recall, enabling detection of indirect or implicit evidence.
- Avoided deduplication to preserve evidence diversity, ensuring each component retains sufficient supporting context.

Notes:
- Balanced precision–recall trade-off by separating local (component) and global (requirement-level) retrieval, improving robustness in noisy semantic matching scenarios.

### h. Job Requirement vs Evidence LLM
Decision:
- Performed LLM-based evaluation per individual job requirement (including its decomposed components)

Reason:
- Isolating evaluation at the requirement level reduces context noise and improves scoring accuracy from LLM.
- Enables more precise evidence-to-requirement alignment and clearer attribution of results.

Observed Issue:
- Avoided including pre-scored values in the prompt, as they can bias the LLM and lead it to treat those scores as evidence during evaluation. (e.g added Retrieval Score)
- Increasing the number of components beyond 3 per job requirement reduces the LLM’s ability to clearly separate components and their corresponding evidence, leading to degraded focus and evaluation quality.

### i. CV fit evaluation logic
Decision:
- Evaluation score is composed of:
  - Capability Level
  - Evidence Score
  - Responsibility Multiplier

Reason:
- Structured scoring constrains LLM reasoning into defined dimensions, reducing uncontrolled inference.
- Separating evaluation into three independent factors (capability, evidence strength, responsibility) prevents the LLM from collapsing all judgment into a single ambiguous score.
- Capability Level captures semantic alignment (does the evidence match the requirement).
- Evidence Score captures strength and context (how strong and reliable the evidence is).
- Responsibility Multiplier captures level of ownership or involvement, preventing overestimation from passive participation.
- Final score is computed outside the LLM to enforce deterministic aggregation and avoid arithmetic hallucination.
- Forcing structured reasoning output (capability + evidence + responsibility + reason) reduces free-form hallucination and improves consistency.

Limitation:
- The LLM may still misclassify one of the scoring dimensions (e.g., overestimate capability or misinterpret evidence strength).
- Errors in one dimension can propagate into the final score despite structured separation.
- Multiplicative scoring reduces the impact of overestimation in a single dimension but does not eliminate systematic bias.

Prompt:
"2. Determine capability level:
  - explicit_strong:
    - The capability is directly and clearly demonstrated
    - Evidence shows real implementation or usage
  - explicit_weak:
    - The capability is directly mentioned but shallow or weak
    - Evidence lacks depth, scale, or clear impact
  - implicit_strong:
    - Capability is not directly stated but strongly implied
    - Multiple signals support the inference
  - implicit_weak:
    - Capability is only loosely related
    - Weak or indirect signals
  - missing:
    - No meaningful evidence supports the component
    - Evidence is irrelevant or absent
3. Determine Evidence score:
  - Do NOT assume capabilities not stated in the evidence
  - Presence alone does NOT imply strength
  - If evidence type is not explicitly stated then default to the LOWER category.
  Evidence Score Mapping:
    - 1.0 → deployed system / real users / production impact
    - 0.8 → professional experience (internship, job, real company work)
    - 0.65 → solid project with clear implementation (non-trivial, end-to-end)
    - 0.5 → basic project
    - 0.4 → coursework / structured learning
    - 0.3 → listed skill, no proof
    - 0.2 → weak indirect relevance with some concrete supporting action
    - 0.0 → none
4. Determine Responsible Multiplier Score:
  - If ownership or leadership is not explicitly stated,
  - DO NOT assign "ownership" or "led", default to 1.0 .
  Responsible Multiplier:
    - 1.2 → led (led or managed others)
    - 1.1 → ownership (explicit OR clearly implied independent work) 
    - 1.0 → contributed (worked on or developed without ownership signals) 
    - 0.9 → assisted (helped, supported, or assisted)"

## CV-fit RAG System v1.1 (7 May 2026)
### a. LLM-based structured CV parser and CV Semantic Chunk
Decision:
- Replaced combined regex and split parser with LLM-based 
- Convert messy CV input into structured CV
- Created new parser category that will be use by CV semantic chunk (Technical Skills, Work Experience, Project, Language, Soft Skills)

Reason:
- Previous Parser struggled in messy CV, this isnt align with real world CV that messy and unstructured
- Previous CV Semantic Chunk has a limited sections that limit number of information category that can be used as evidence for job requirement
- This version of the parser enables the system to handle CV inputs that are closer to real-world formats.

Limitation:
- Unknown category tend to be inputted as Project by LLM 

  Fix:
  - Forcing LLM not to add (e.g. Organizational) to project category
  - Added fallback classification handling to prevent the LLM from incorrectly assigning unknown categories to semantically similar existing categories

  Note:
  - I havent observe with larger testing if this fix is reliable or not

Note:
- Category: ("Language", "Project", "Work Experience", "Technical Skills") has inside structured introducing "name", "item"
- If not enough evidence for "item" then llm return [], this [] output change the CV semantic chunk format.

Example:
"[CVChunk(idx=0, type='Technical Skill', chunk='Technical Skills (Programming Languages): Python, C++, SQL (basic)'),CVChunk(idx=10, type='Work Experience', chunk='Work Experience (Backend Developer Intern): Improved response structure consistency across several endpoints'), CVChunk(idx=22, type='Language', chunk='Language: English (Professional Working Proficiency)')]"

### b. Final Scoring Logic
Decision:
- Use multiplication scoring for "Evidence Score", "Responsibility Multiplier", "Capability Score" as Final Score for each Job Requirements
- Introduced configuration to edit each scoring multiplier

Reason:
- LLM-generated scoring requires constrained and structured evaluation dimensions to reduce inconsistent or ungrounded scoring behavior
- Separating scoring into multiple dimensions (capability, evidence strength, responsibility) prevents the evaluation from collapsing into a single ambiguous score.
- Multiplicative scoring helps reduce overestimation / underestimation from a single incorrect dimension by requiring all dimensions to contribute consistently to the final score.

Limitation:
- Multiplicative scoring tends to produce conservative final scores, especially when one dimension is weak.
- Scoring multipliers require calibration using real evaluation cases and failure analysis
- Final scoring quality is dependent on retrieval quality and evidence relevance.

### c. Report Generator
Decision:
- Conclude each Job Reqiurement's components score reasoning into 1 job requirement reasoning using LLM-based.
- Change systematic explanation into more user friendly explanation while adding some suggestion.

Reason:
- Job requirement evaluation's reason are separated into components level of Job Requirement, user should not analyze the report per component but per job requirement
- User appriciate more friendly / mentor-ish explanation than systematic explanation
- CV improvement suggestion is added, adding value for user to add missing context in their CV or understand their weakness

Note:
- I havent try to include final score of each job's requirement as context for LLM to conclude reasoning, I assume this could improve the reasoning if the score is perfectly align and hurt the reason if the score is not correct.

Example:
"Query: Experience applying machine learning techniques to real problems
Score: 0.525
Reason: Your experience in building a machine learning classification pipeline using scikit-learn shows that you have a solid understanding of machine learning techniques. However, it would strengthen your profile if you could provide examples of how your work addressed specific real-world problems.

Query: Familiarity with containerization tools (e.g., Docker)
Score: 0.0
Reason: Unfortunately, there is no evidence in your CV that demonstrates your familiarity with containerization tools like Docker. Gaining experience in this area could enhance your skill set and make you a more competitive candidate for roles that require containerization knowledge.

## CV-fit Multi-Step RAG System v1.2 (12 May 2026)
### a. CV Preprocess Pipeline
Decision:
- Create a pipeline for CV preprocess (parsing, chunking, embedding, artifact generation).
- Pipeline input consist of cv_name and cv_input (raw CV text).
- Reject CV input with fewer than 500 characters.
- Reject parsed CV output with fewer than 10 structured parses.

Reason:
- The system is designed for candidates to evaluate CV fitness against specific job requirements.
- Persisting reusable CV artifacts improves: inference convenience, token efficiency, inference latency.
- Detailed benchmarking is available in "docs/token_usage_analysis" and "docs/latency_analysis".
- Validation rules help prevent low-quality evaluation, insufficient CV information, invalid/non-CV user input.

Limitation:
- Parsed CV validation still naive should condisering the number of specific parsed category than the whole parse.

### b. CV Artifact persistence system
Decision:
- Store each CV preprocessing artifact separately (cv_chunks.json, cv_embedding.npy, cv_parsed.json, metadata.json)
- Load and validate (cv_emebdding.npy, cv_chunks.json, metadata.json)

Reason:
- Separating preprocess artifact improves (debugging, observability, future pipeline improvement)
- Artifact integrity validation includes:
  - consistency between CV chunks and embedding
  - metadata validation for preprocessing configruation
  - schema validation using Pydantic and Dataclass
- NumPy `.npy` format is used for embedding persistence
  - NumPy `.npy` format is optimized for numerical array storage
  - More storage-efficient than text-based JSON serialization
  - Avoids expensive list-to-NumPy conversion during loading

Limitation:
- Artifact update/versioning workflow is not yet implemented.
- Existing CV artifacts with the same name are overwritten with warning
- Artifact storage structure is currently rigid and not yet configurable

### c. Observability & Telemetry Module
Decision:
- Created a latency tracking wrapper to measure pipeline stage execution time
- Added LLM token usage tracking and cost calculation for each request

Reason:
- Observability is important for identifying:
  - performance bottlenecks
  - latency instability
  - LLM cost tradeoffs
  - optimization opportunities
- LLM requests are centralized inside `llm_client`, allowing:
  - prompt token tracking
  - completion token tracking
  - request cost calculation
  - configurable currency conversion
- Wrapper-based latency tracking improves reusability

Limitation:
- Current implementation only supports per-process runtime tracking and is not yet compatible with FastAPI request lifecycle management
- Function-level latency tracking cannot measure internal execution steps
- Observability logs are currently mixed with application logs and do not yet use a dedicated telemetry logging pipeline

### d. Centralized LLM Failure Handling
Decision:
- Added centralized LLM failure handling for invalid structured JSON generation
- Added automatic timeout retry mechanism with configurable:
  - timeout duration
  - maximum retry attempts

Reason:
- LLM structured generation may occasionally return invalid JSON output
- Automatic JSON repair improves pipeline robustness for strict JSON workflows
- JSON repair attempts are tracked through telemetry and token usage monitoring
- LLM API communication may experience unstable network conditions or timeout failures, requiring retry handling

Limitation:
- Retry mechanism currently only handles timeout-related failures
- General API connection errors are treated as terminal failures and are not retried
- Connection error classification is still limited and does not yet distinguish recoverable vs non-recoverable connection failures

### e. Retrieval failure handling
Decision:
- Added retrieval filtering with configurable similarity threshold
- If no retrieval passes the threshold and filtering is enabled, the evidence is replaced with "no evidence"
- If filtering is disabled, logger will show warning for using low similarity retrieval as evidence

Reason:
- In a CV-fit evaluation system, moderate or weak semantic retrieval can still provide useful signals for capability evaluation
- The goal of the system is not exact keyword matching, but evaluating evidence relevance and capability alignment
- However, extremely low-semantic retrieval results may introduce unrelated evidence and increase hallucination risk during evaluation
- Filtering low-confidence retrieval results helps reduce forced reasoning on irrelevant context

Observed Issue:
- When retrieval filtering removes all component-level evidence, the system falls back to "no evidence"
- In some cases, this causes the evaluator to ignore relevant information that still exists in global evidence retrieval

Notes:
- Future Improvement: If component-level retrieval is empty, allow the evaluator to analyze using global evidence retrieval as fallback context

### f. JSON Structured logger and bootstrap logger
Decision:
- Added bootstrap logger initialization before structured logger configuration
- Implemented structured JSON logging
- Added optional log persistence using file handler

Reason:
- Bootstrap logger is required to capture initialization and configuration-stage failures before the structured logger is fully configured
- Structured JSON logging improves:
  - readability
  - observability
  - log parsing consistency
  - debugging workflow
- Standardized JSON log format includes:
  - levelname
  - logger name
  - message
  - timestamp
  - environment
  - pipeline stage
- Additional contextual metadata can also be attached for debugging and telemetry analysis

Example:
{
  "levelname": "INFO",
  "name": "src.pipelines.inference_pipeline",
  "message": "CV acquired",
  "timestamp": "12/05/2026_15:01",
  "environment": "v.1.2",
  "stage": "inf_api_input",
  "person_name": "Ardi Pratama",
  "cv_chunks_n": 27
}

Limitation:
- Logger file path and file naming configuration are not yet configurable


## CV-fit Multi-Step RAG System v1.3 (18 May 2026)
### a. Integrated FastAPI Architecture
Decision:
- Migrated the v1.2 pipeline system into a FastAPI backend architecture
- Added dependency injection
- Added lifespan-managed application state
- Added request middleware
- Added route-based endpoint separation
- Added centralized FastAPI exception handling

Reason:
- Lifespan management is used to initialize and maintain shared application-scoped services during runtime
- Dependency injection improves service access, maintainability, and future extensibility
- Middleware enables request-level observability, authentication, and request processing control
- Route separation improves endpoint responsibility isolation
- Centralized exception handling improves API-level robustness and consistent error responses

Limitation:
- The current architecture still experiences instability when deployed using multiple workers
- Shared runtime state and orchestration behavior are not yet fully designed for multi-worker execution environments

### b. Built Artifact Manager
Decision:
- Implemented a dedicated artifact manager service for:
  - CV artifact persistence
  - artifact loading
  - artifact existence validation
  - artifact integrity validation
  - artifact deletion

Reason:
- The system requires reusable persistence and IO management for preprocessed CV artifacts
- Separating artifact operations into a dedicated service improves maintainability, validation consistency, and pipeline reusability
- Persistent artifacts reduce repeated preprocessing, embedding generation, token usage, and inference latency

Limitation:
- The current artifact manager does not yet support overwriting existing CV artifacts
- Artifact persistence currently relies on local storage and is not yet designed for distributed storage environments

### c. Implemented Asynchronous Multi-Stage Orchestration
Decision:
- Added asynchronous orchestration for preprocess and inference pipelines
- Implemented fanout concurrency for job requirement decomposition and evaluation stages

Reason:
- FastAPI already supports asynchronous request handling at the API level
- The preprocess and inference pipelines rely heavily on upstream OpenAI requests, making asynchronous execution beneficial for reducing stage latency
- Fanout concurrency was introduced to parallelize job requirement decomposition and evaluation workloads

Limitation:
- The current orchestration model uses batch-stage synchronization instead of streaming execution
- Downstream stages must wait until all asynchronous tasks from the previous stage are completed
- This limits the ability to fully utilize upstream OpenAI concurrency capacity under sustained workloads before saturation occurs

### d. Added Circuit Breaker
Decision:
- Added a circuit breaker mechanism for upstream LLM generation requests

Reason:
- LLM generation depends on external upstream services used throughout the system
- External failures such as:
  - timeout errors
  - OpenAI rate limiting
  - insufficient API credit balance
  - connection instability
  can propagate and destabilize the inference pipeline
- The circuit breaker temporarily blocks upstream generation requests after repeated failures to reduce cascading errors and improve system resilience

Limitation:
- Circuit breaker behavior currently uses static threshold and recovery timing configuration
- Upstream service degradation can still increase latency before the breaker is triggered

### e. Added LLM Abuse Protection
Decision:
- Added API-key-based abuse protection for repeated malformed LLM generation behavior
- Implemented temporary API-key suspension using time-window-based failure tracking
- Protection is triggered by:
  - repeated JSON repair failures
  - repeated malformed structured outputs
  - business-logic validation violations caused by a specific API key

Reason:
- Users may deliberately attempt to manipulate prompts to force invalid structured LLM responses
- Repeated malformed generations can destabilize orchestration pipelines and increase unnecessary retry/repair workloads
- The protection mechanism reduces repeated abusive generation behavior and protects upstream orchestration stability

Limitation:
- The current system does not yet validate whether user inputs are genuinely job requirements or curriculum vitae content
- Abuse classification currently relies primarily on malformed generation behavior rather than semantic intent validation

### f. Implemented Stage-Level Concurrency Limiter and Timeout Strategy
Decision:
- Implemented stage-level concurrency limiters for:
  - CV preprocessing
  - job requirement decomposition
  - evaluation
  - report generation
- Added timeout strategies for upstream LLM generation stages

Reason:
- The inference pipeline uses asynchronous fanout concurrency for decomposition and evaluation stages
- Each stage has different workload characteristics, prompt complexity, and upstream latency behavior
- Uncontrolled asynchronous fanout can amplify latency and trigger upstream OpenAI saturation under sustained concurrent workloads
- Stage-level concurrency governance improves orchestration stability and upstream survivability

Limitation:
- Concurrency limiters primarily protect upstream LLM generation services from saturation
- Current orchestration still uses batch-stage synchronization instead of streaming execution
- Upstream OpenAI degradation and traffic variability can dynamically change effective saturation limits over time

Future:
- Implement adaptive concurrency governance based on upstream degradation behavior and runtime latency conditions

## CV-fit Multi-Step RAG System v1.4 (19 May 2026)
## CV-fit Multi-Step RAG System v1.4 (19 May 2026)

### Implemented Stage-Level Concurrency Limiter and Timeout Strategy
Decision:
- Dockerized FastAPI deployment workflow
- Created Docker volume mounts for logs, storage, `config.yaml`, and `auth_config.yaml`
- Added `.dockerignore` for non-runtime and non-persistent files
- Introduced Docker Compose for reproducible runtime orchestration

Reason:
- deployment environments require reproducible runtime behavior
- runtime configuration and persistent artifacts should remain mutable outside the container lifecycle
- Docker Compose improves deployment consistency and reduces manual startup complexity

Limitation:
- GPU-oriented container deployment is not yet integrated because the current embedding workload uses lightweight models with relatively small batch sizes

### b. Used ONNX Runtime for CPU Embedding Inference
Decision:
- Changed SentenceTransformer CPU embedding inference backend from the default PyTorch runtime to ONNX Runtime.

Reason:
- ONNX Runtime is optimized for inference deployment and provides lower runtime overhead for CPU-based transformer inference workloads.
- ONNX also provides better deployment portability across different inference environments.

Observed Result:
- Average job requirement embedding latency decreased from ~579 ms to ~38 ms during local benchmarking (~15x improvement).

## Author
Jearim Jarden


