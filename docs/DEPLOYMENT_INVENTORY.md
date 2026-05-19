# Required Runtime Files
- config.yaml
    - main application configuration
    - startup 
- auth_config.yaml
    - API authentication for fastAPI configuration
    - startup

# Environment Variable
- environment
    - Required
    - Environment runtime name
- oa_api_key
    - Required
    - OpenAI upstream authentication
- hf_api_key
    - Optional
    - Huggingface upstream authentication

# Runtime Storage
- storage/candidates
    - load and save cv
    - persistance
- logs/ 
    - save logs
    - optional through config (persistance / not)

# Startup Initialization
- bootstrap_logger: setup for bootstrap logger
- load_env: loading for .env via pydantic settings
- load_config: loading configuration for system base config (config.yaml)
- load_auth_config: loading configuration for fastapi API config (auth_config.yaml)
- APISecurity instance: Instance for fastapi api security, checking for allowed_api key, routes, rate limiter
- CircuitBreaker instance: circuit breaker intance for protection of llm_client request to uptream OpenAI that triggered cascading error
- LLMAbuseProtection: instance for suspend api key in a certain time if specific api key caused a llm reponse error in specific range and time window
- ConcurrencyLimiter: limit concurrency that used for upstream llm call for specific stages
- TrackLatency: latency tracker for important step in pipeline
- LLMClient: instance that handle llm request to openAI
- EmbeddingService: instance for embedding and load embedding model
- EvaluatorService: instance for evaluator service
- ArtifactManager: instance for handling IO of CV artifact
- PreprocessPipeline: instance for preprocess pipeline
- InferencePipeline: isntance for inference pipeline
- include user_router and dev_router
- adding middleware

# External Dependency:
- openAI API
    - LLM inference
    - required
- Internet Access
    - upstream communication
    - required
- Local filesystem
    - artifact persistance
    - required
- NVIDIA GPU
    - faster embedding generation
    - optional

# Network Ports
- 8000 
    -FastAPI HTTP API

# Known Error
| Failure                   | Cause                             | Operational Impact |
| Missing config            | Invalid/missing YAML config       | Service boot failure |
| OpenAI timeout            | Upstream latency                  | Delayed inference |
| OpenAI rate limit         | Upstream throttling        | Request rejection |
| Circuit breaker open      | Upstream instability      | Temporary service degradation |
| Concurrency saturation    | Excessive parallel requests | Temporary request rejection |
| Corrupted artifact        | Invalid persisted CV artifact   | Inference failure |
| Invalid API key           | Authentication failure    | Access denied |
| Invalid JSON from LLM     | Malformed upstream response   | Parsing failure |