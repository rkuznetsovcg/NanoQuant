# Copyright (c) 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Text-only OpenAI-compatible server for a saved NanoQuant checkpoint."""

from __future__ import annotations

import argparse
import hmac
import html
import json
import logging
import re
import threading
import time
import uuid
from typing import Any

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .modules.linear import NanoQuantLinear
from .utils.load_utils import load_compressed_model, load_tokenizer


LOGGER = logging.getLogger("nanoquant.openai_server")
_TOOL_BLOCK_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_FUNCTION_RE = re.compile(r"<function\s*=\s*([^>\s]+)\s*>", re.DOTALL)
_PARAMETER_RE = re.compile(r"<parameter\s*=\s*([^>\s]+)\s*>(.*?)</parameter>", re.DOTALL)
_THINK_RE = re.compile(r"<think(?:ing)?>(.*?)</think(?:ing)?>(.*)", re.DOTALL)


def _jsonish_value(value: str) -> Any:
    value = html.unescape(value).strip()
    if not value:
        return ""
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def parse_qwen_tool_calls(text: str) -> tuple[list[dict[str, Any]], str]:
    """Turn Qwen's native XML tool-call form (or JSON form) into OpenAI calls."""
    calls: list[dict[str, Any]] = []
    consumed: list[tuple[int, int]] = []

    for match in _TOOL_BLOCK_RE.finditer(text):
        block = match.group(1).strip()
        parsed_json = None
        try:
            parsed_json = json.loads(block)
        except json.JSONDecodeError:
            pass

        if isinstance(parsed_json, dict) and isinstance(parsed_json.get("name"), str):
            name = parsed_json["name"]
            arguments = parsed_json.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"input": arguments}
            if not isinstance(arguments, dict):
                arguments = {"input": arguments}
        else:
            name_match = _FUNCTION_RE.search(block)
            if not name_match:
                continue
            name = name_match.group(1)
            arguments = {
                key: _jsonish_value(value)
                for key, value in _PARAMETER_RE.findall(block)
            }

        calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
            }
        )
        consumed.append(match.span())

    if not calls:
        think_match = _THINK_RE.search(text)
        if think_match:
            text = think_match.group(2)
        elif text.startswith("<think>") or text.startswith("<thinking>"):
            text = ""
        return [], text.strip()

    visible = text
    for start, end in reversed(consumed):
        visible = visible[:start] + visible[end:]
    think_match = _THINK_RE.search(visible)
    if think_match:
        visible = think_match.group(2)
    visible = re.sub(r"</?(?:function|tool_call)>", "", visible)
    return calls, visible.strip() or None


def _text_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        chunks: list[str] = []
        for part in value:
            if not isinstance(part, dict) or part.get("type") not in {"text", "input_text"}:
                raise HTTPException(
                    status_code=400,
                    detail="This NanoQuant checkpoint is text-only; image and video inputs are unsupported.",
                )
            chunks.append(str(part.get("text", "")))
        return "".join(chunks)
    raise HTTPException(status_code=400, detail="Message content must be text.")


def _normalize_messages(raw_messages: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_messages, list) or not raw_messages:
        raise HTTPException(status_code=400, detail="messages must be a non-empty array.")

    messages: list[dict[str, Any]] = []
    for raw in raw_messages:
        if not isinstance(raw, dict):
            raise HTTPException(status_code=400, detail="Each message must be an object.")
        role = raw.get("role")
        if role == "developer":
            role = "system"
        if role not in {"system", "user", "assistant", "tool"}:
            raise HTTPException(status_code=400, detail=f"Unsupported message role: {role!r}.")

        message: dict[str, Any] = {"role": role, "content": _text_content(raw.get("content"))}
        if role == "tool":
            if not raw.get("tool_call_id"):
                raise HTTPException(status_code=400, detail="Tool messages require tool_call_id.")
            message["tool_call_id"] = raw["tool_call_id"]
            if raw.get("name"):
                message["name"] = raw["name"]
        elif role == "assistant" and raw.get("tool_calls"):
            tool_calls = []
            for tool_call in raw["tool_calls"]:
                function = tool_call.get("function", {})
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {"input": arguments}
                tool_calls.append(
                    {
                        "id": tool_call.get("id", f"call_{uuid.uuid4().hex[:12]}"),
                        "type": "function",
                        "function": {
                            "name": function.get("name", ""),
                            "arguments": arguments if isinstance(arguments, dict) else {"input": arguments},
                        },
                    }
                )
            message["tool_calls"] = tool_calls
        messages.append(message)
    return messages


def _require_api_key(request: Request, api_key: str) -> None:
    supplied = request.headers.get("authorization", "")
    if supplied.lower().startswith("bearer "):
        supplied = supplied[7:].strip()
    else:
        supplied = ""
    if not hmac.compare_digest(supplied, api_key):
        raise HTTPException(status_code=401, detail="Invalid or missing bearer token.")


def create_app(args: argparse.Namespace) -> FastAPI:
    if not torch.cuda.is_available():
        raise RuntimeError("NanoQuant inference requires a CUDA GPU.")
    if not args.api_key:
        raise ValueError("Pass a random --api-key; the OpenAI endpoint is bearer-token protected.")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("This optimized server currently requires a CUDA device.")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    started = time.monotonic()
    LOGGER.info("Loading tokenizer for %s", args.model_id)
    tokenizer = load_tokenizer(args.model_id, revision=args.revision)

    LOGGER.info("Loading NanoQuant checkpoint %s", args.checkpoint)
    model = load_compressed_model(
        model_name_or_path=args.model_id,
        checkpoint_path=args.checkpoint,
        seqlen=args.seqlen,
        device=str(device),
        dtype=dtype,
        revision=args.revision,
    )

    move_started = time.monotonic()
    model.to(device)
    model.eval()
    model.config.use_cache = True
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = True
    torch.cuda.synchronize(device)
    move_seconds = time.monotonic() - move_started

    modules = [module for module in model.modules() if isinstance(module, NanoQuantLinear)]
    if not modules:
        raise RuntimeError("Checkpoint loaded but contains no NanoQuantLinear layers.")

    prepare_seconds = 0.0
    if args.kernel != "none":
        LOGGER.info("Preparing %s kernel for %d NanoQuant layers", args.kernel, len(modules))
        prepare_started = time.monotonic()
        for index, module in enumerate(modules, start=1):
            module._prepare_kernel(kernel_type=args.kernel, dtype=dtype)
            if index % 32 == 0 or index == len(modules):
                LOGGER.info("Prepared %d/%d NanoQuant layers", index, len(modules))
        torch.cuda.synchronize(device)
        prepare_seconds = time.monotonic() - prepare_started

    startup_seconds = time.monotonic() - started
    LOGGER.info(
        "Ready: kernel=%s, layers=%d, device=%s, model_move_s=%.1f, kernel_prepare_s=%.1f, total_startup_s=%.1f",
        args.kernel,
        len(modules),
        torch.cuda.get_device_name(device),
        move_seconds,
        prepare_seconds,
        startup_seconds,
    )

    app = FastAPI(title="NanoQuant OpenAI-compatible API", version="1.0.0")
    inference_lock = threading.Lock()
    app.state.model = model
    app.state.tokenizer = tokenizer
    app.state.ready_info = {
        "model": args.model_name,
        "kernel": args.kernel,
        "layers": len(modules),
        "device": torch.cuda.get_device_name(device),
        "dtype": args.dtype,
        "startup_seconds": round(startup_seconds, 3),
        "model_move_seconds": round(move_seconds, 3),
        "kernel_prepare_seconds": round(prepare_seconds, 3),
        "max_tokens": args.max_tokens,
    }

    @app.get("/health")
    def health(request: Request):
        _require_api_key(request, args.api_key)
        return {"status": "ok", **app.state.ready_info}

    @app.get("/v1/models")
    def models(request: Request):
        _require_api_key(request, args.api_key)
        return {"object": "list", "data": [{"id": args.model_name, "object": "model", "owned_by": "nanoquant"}]}

    @app.post("/v1/chat/completions")
    def chat_completions(request: Request, payload: dict[str, Any]):
        _require_api_key(request, args.api_key)
        if payload.get("stream"):
            raise HTTPException(status_code=400, detail="Streaming responses are not enabled in this server.")
        if int(payload.get("n", 1)) != 1:
            raise HTTPException(status_code=400, detail="Only n=1 is supported.")

        messages = _normalize_messages(payload.get("messages"))
        raw_tools = payload.get("tools") or []
        if not isinstance(raw_tools, list):
            raise HTTPException(status_code=400, detail="tools must be an array.")
        tool_choice = payload.get("tool_choice", "auto")
        tools = [] if tool_choice == "none" else raw_tools
        template_options = payload.get("chat_template_kwargs") or {}
        extra_body = payload.get("extra_body") or {}
        enable_thinking = template_options.get(
            "enable_thinking",
            payload.get("enable_thinking", extra_body.get("enable_thinking", True)),
        )
        template_kwargs: dict[str, Any] = {"enable_thinking": bool(enable_thinking)}
        if tools:
            template_kwargs["tools"] = tools

        try:
            encoded = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                **template_kwargs,
            )
        except Exception as error:
            LOGGER.exception("Qwen chat template rejected the request")
            raise HTTPException(status_code=400, detail=f"Could not format messages/tools: {error}") from error

        input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded
        attention_mask = encoded.get("attention_mask") if isinstance(encoded, dict) else None
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        requested_tokens = payload.get("max_completion_tokens", payload.get("max_tokens", args.max_tokens))
        max_new_tokens = int(requested_tokens)
        if max_new_tokens < 1 or max_new_tokens > args.max_tokens:
            raise HTTPException(
                status_code=400,
                detail=f"max_tokens must be between 1 and {args.max_tokens} for this server.",
            )
        stop = payload.get("stop")
        if stop not in (None, [], ""):
            raise HTTPException(status_code=400, detail="Custom stop strings are not supported; use the model EOS token.")

        temperature = float(payload.get("temperature", 1.0))
        top_p = float(payload.get("top_p", 0.95))
        top_k = int(payload.get("top_k", 20))
        if temperature < 0 or not 0 < top_p <= 1 or top_k < 1:
            raise HTTPException(status_code=400, detail="temperature must be non-negative and top_p in (0, 1].")

        generation: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "use_cache": True,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            generation.update(temperature=temperature, top_p=top_p, top_k=top_k)
        seed = payload.get("seed")

        try:
            with inference_lock, torch.inference_mode():
                if seed is not None:
                    torch.manual_seed(int(seed))
                    torch.cuda.manual_seed_all(int(seed))
                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **generation,
                )
                completion_ids = output_ids[0, input_ids.shape[1] :]
                generated = tokenizer.decode(completion_ids, skip_special_tokens=False)
        except HTTPException:
            raise
        except Exception as error:
            LOGGER.exception("NanoQuant generation failed")
            raise HTTPException(status_code=500, detail=f"Generation failed: {type(error).__name__}: {error}") from error

        calls, content = parse_qwen_tool_calls(generated)
        message: dict[str, Any] = {"role": "assistant", "content": content}
        if calls:
            message["tool_calls"] = calls
        finish_reason = "tool_calls" if calls else ("length" if len(completion_ids) >= max_new_tokens else "stop")
        completion_tokens = int(completion_ids.numel())
        response = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": args.model_name,
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": {
                "prompt_tokens": int(input_ids.shape[1]),
                "completion_tokens": completion_tokens,
                "total_tokens": int(input_ids.shape[1]) + completion_tokens,
            },
        }
        return JSONResponse(response)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to the saved NanoQuant .pt state dict.")
    parser.add_argument("--model-id", default="Qwen/Qwen3.8-27B", help="HF config/tokenizer model ID.")
    parser.add_argument("--model-name", default="nanoquant-qwen38-27b", help="ID returned by /v1/models.")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--seqlen", type=int, default=-1)
    parser.add_argument("--kernel", choices=("none", "gemv", "gemlite", "gemm"), default="gemlite")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--api-key", required=True, help="Bearer token required on every API endpoint.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-level", choices=("debug", "info", "warning", "error"), default="info")
    return parser.parse_args()


def main() -> None:
    import uvicorn

    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(message)s")
    app = create_app(args)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
