"""OpenAI-compatible Flask server that fronts a local GGUF or MLX model."""
from __future__ import annotations

import logging
import os
import platform
import time
import uuid
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, jsonify, request

LOGGER = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

# Local LLM runtime
LLM_RUNTIME = "gguf"  # supported: gguf, mlx

# Llama.cpp
LLAMA_MODEL_PATH = "Qwen2.5-7B-Instruct-1M-Q5_K_M.gguf"
LLAMA_CTX = 65_536  # Qwen = 65536/101000
LLAMA_N_THREADS = max(1, (os.cpu_count() or 4) - 1)
LLAMA_N_GPU_LAYERS = 20  # -1 offloads all layers when GPU backend is available
LLAMA_N_BATCH = 256  # prompt processing batch
LLAMA_N_UBATCH: Optional[int] = 256  # physical micro-batch; None lets llama.cpp choose
LLAMA_LOW_VRAM = True  # reduce Metal VRAM usage

# MLX local model directory. Relative paths are resolved under ~/models.
MLX_MODEL_PATH = "Qwen2.5-7B-Instruct-4bit"

# External LLM constants kept here for consistency with the rest of the app.
# This server module currently serves the local model directly.
LLM_SERVER_URL = "http://127.0.0.1:8001/v1"
LLM_SERVER_API_KEY = "local-llm"
LLM_SERVER_MODEL = "local-llm"
EXTERNAL_BASE_URL = "https://inference.do-ai.run/v1/chat/completions"
EXTERNAL_MODEL = "llama3-8b-instruct"

# Named entity recognition service
NER_URL = "http://127.0.0.1:8000/ner"
NER_TIMEOUT_SECS = 5.0

# Server
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000


# -----------------------------------------------------------------------------
# Runtime helpers
# -----------------------------------------------------------------------------

def _is_apple_silicon() -> bool:
    """Return True when running on Apple Silicon hardware."""

    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _resolve_gpu_layers() -> int:
    """Pick an appropriate GPU offload value for the local runtime."""

    if _is_apple_silicon() and LLAMA_N_GPU_LAYERS == 0:
        return -1
    return LLAMA_N_GPU_LAYERS


def _resolve_local_model_path(model_path: str) -> str:
    """Resolve a local model path, defaulting relative paths under ~/models."""

    expanded = os.path.expanduser(model_path)
    if os.path.isabs(expanded):
        return expanded
    return os.path.join(os.path.expanduser("~/models"), expanded)


@lru_cache(maxsize=1)
def _load_local_llm() -> Any:
    """Load and cache the configured local model for serving."""

    if LLM_RUNTIME == "mlx":
        return _load_local_mlx_llm()
    if LLM_RUNTIME == "gguf":
        return _load_local_gguf_llm()
    raise RuntimeError(f"Unsupported LLM_RUNTIME={LLM_RUNTIME!r}; expected 'gguf' or 'mlx'.")


def _load_local_gguf_llm() -> Any:
    """Load and cache the local llama.cpp model for serving."""

    from llama_cpp import Llama

    model_path = _resolve_local_model_path(LLAMA_MODEL_PATH)
    LOGGER.info("Loading GGUF model from %s", model_path)

    n_gpu_layers = _resolve_gpu_layers()
    if _is_apple_silicon() and n_gpu_layers != 0:
        LOGGER.info(
            "Apple Silicon detected; using Metal GPU offload with n_gpu_layers=%s.",
            n_gpu_layers,
        )

    kwargs: Dict[str, Any] = {
        "model_path": model_path,
        "n_ctx": LLAMA_CTX,
        "n_threads": LLAMA_N_THREADS,
        "n_gpu_layers": n_gpu_layers,
        "n_batch": LLAMA_N_BATCH,
        "chat_format": "chatml",
        "verbose": False,
    }

    if LLAMA_N_UBATCH is not None:
        kwargs["n_ubatch"] = LLAMA_N_UBATCH
    if LLAMA_LOW_VRAM:
        kwargs["low_vram"] = True

    return Llama(**kwargs)


def _load_local_mlx_llm() -> Tuple[Any, Any]:
    """Load and cache the local MLX model and tokenizer for serving."""

    if not _is_apple_silicon():
        raise RuntimeError("LLM_RUNTIME=mlx is only supported on Apple Silicon.")

    from mlx_lm import load

    model_path = _resolve_local_model_path(MLX_MODEL_PATH)
    LOGGER.info("Loading MLX model from %s", model_path)
    return load(model_path)


def _error(status: int, message: str) -> tuple[Dict[str, Any], int]:
    """Return a JSON API error payload."""

    return {"error": {"message": message, "type": "invalid_request_error"}}, status


def _normalize_messages(payload: Dict[str, Any]) -> List[Dict[str, str]]:
    """Validate and normalize chat messages from the request body."""

    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("Expected non-empty 'messages' list.")

    normalized: List[Dict[str, str]] = []
    for item in messages:
        if not isinstance(item, dict):
            raise ValueError("Each message must be a JSON object.")

        role = item.get("role")
        content = item.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError("Each message requires string 'role' and 'content'.")

        normalized.append({"role": role, "content": content})

    return normalized


def _build_mlx_prompt(tokenizer: Any, messages: List[Dict[str, str]]) -> str:
    """Render chat messages into a prompt string for MLX generation."""

    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        return str(
            apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

    prompt_lines = [f"{message['role']}: {message['content']}" for message in messages]
    prompt_lines.append("assistant:")
    return "\n".join(prompt_lines)


def _count_tokens(tokenizer: Any, text: str) -> int:
    """Best-effort token counting for usage reporting."""

    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        return 0

    try:
        return int(len(encode(text)))
    except Exception:
        return 0


def _build_chat_response(
    *,
    model: str,
    content: str,
    usage: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Format the response payload to match the OpenAI chat completion schema."""

    payload: Dict[str, Any] = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }

    if usage:
        payload["usage"] = {
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
            "total_tokens": int(usage.get("total_tokens", 0)),
        }

    return payload


def _run_gguf_chat_completion(
    *,
    messages: List[Dict[str, str]],
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Generate a chat completion with llama.cpp."""

    llm = _load_local_llm()
    response = llm.create_chat_completion(
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    content = ""
    if isinstance(response, dict):
        choices = response.get("choices") or []
        if choices and isinstance(choices[0], dict):
            message = choices[0].get("message") or {}
            content = str(message.get("content") or "")

    usage = response.get("usage") if isinstance(response, dict) else None
    return content, usage


def _run_mlx_chat_completion(
    *,
    messages: List[Dict[str, str]],
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> Tuple[str, Dict[str, Any]]:
    """Generate a chat completion with mlx-lm."""

    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = _load_local_llm()
    prompt = _build_mlx_prompt(tokenizer, messages)
    sampler = make_sampler(temp=temperature, top_p=top_p)
    content = str(
        generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            verbose=False,
        )
    )

    prompt_tokens = _count_tokens(tokenizer, prompt)
    completion_tokens = _count_tokens(tokenizer, content)
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    return content, usage


# -----------------------------------------------------------------------------
# Flask app
# -----------------------------------------------------------------------------

def create_app() -> Flask:
    """Create the Flask app that serves OpenAI-compatible endpoints."""

    app = Flask(__name__)
    _load_local_llm()

    @app.route("/health", methods=["GET"])
    def health() -> tuple[Any, int]:
        return jsonify(
            {
                "status": "ok",
                "model": LLM_SERVER_MODEL,
                "runtime": LLM_RUNTIME,
                "server": {
                    "host": SERVER_HOST,
                    "port": SERVER_PORT,
                },
            }
        ), 200

    @app.route("/v1/models", methods=["GET"])
    def models() -> tuple[Any, int]:
        return jsonify(
            {
                "object": "list",
                "data": [
                    {
                        "id": LLM_SERVER_MODEL,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "local",
                    }
                ],
            }
        ), 200

    @app.route("/v1/chat/completions", methods=["POST"])
    def chat_completions() -> tuple[Any, int]:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return _error(400, "Expected JSON object payload.")

        if payload.get("stream") is True:
            return _error(400, "Streaming responses are not supported.")

        try:
            messages = _normalize_messages(payload)
        except ValueError as exc:
            return _error(400, str(exc))

        temperature = float(payload.get("temperature", 0.2))
        top_p = float(payload.get("top_p", 0.9))
        max_tokens = int(payload.get("max_tokens", 512))
        model = str(payload.get("model") or LLM_SERVER_MODEL)

        if LLM_RUNTIME == "mlx":
            content, usage = _run_mlx_chat_completion(
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
        else:
            content, usage = _run_gguf_chat_completion(
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )

        return jsonify(_build_chat_response(model=model, content=content, usage=usage)), 200

    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = create_app()
    app.run(
        host=SERVER_HOST,
        port=SERVER_PORT,
        debug=False,
    )
