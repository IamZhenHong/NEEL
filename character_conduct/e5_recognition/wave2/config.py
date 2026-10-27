"""CC-E5 wave 2 generation settings, shared by the arm-text judge (arms.py) and the continuations (episode.py)."""
URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "qwen/qwen3.6-27b"
PROVIDER = "Alibaba"        # Qwen's own host: tools + reasoning, $0.45/M in, $2.7/M out (OpenRouter endpoints, 10-28).
                            # DeepInfra (first choice, fp8) was rate-limited upstream on 10-28 (HTTP 429 through
                            # OpenRouter's shared pool); its 8 arm-(a) judgements made before that are kept.
                            # A real continuation request: Alibaba 12 s, SiliconFlow 69 s, both proper tool calls.
TEMPERATURE = 0.7           # the harness's documented sampled setting (docker-compose.yml); the release is greedy
