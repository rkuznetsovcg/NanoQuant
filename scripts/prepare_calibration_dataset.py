#!/usr/bin/env python3
"""Build a small, task-balanced NanoQuant calibration set from public HF data.

The script materializes 128 fixed-length sequences without downloading full
source datasets. Stack v3 is scanned in streaming mode to fill language quotas;
benchmark test sets are never read.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer


MODEL_ID = "Qwen/Qwen3.8-27B"
SEQUENCE_LENGTH = 2048
SEED = 42

NATURAL_LANGUAGES = {
    "english": "English",
    "russian": "Russian",
    "chinese": "Chinese",
    "spanish": "Spanish",
    "german": "German",
    "french": "French",
    "korean": "Korean",
}

AYA_COUNTS = {
    "english": 7,
    "russian": 7,
    "chinese": 7,
    "spanish": 7,
    "german": 7,
    "french": 7,
    "korean": 6,
}

STACK_LANGUAGES = {
    "Python": "Python",
    "JavaScript": "JavaScript",
    "TypeScript": "TypeScript",
    "Java": "Java",
    "C++": "C++",
    "Go": "Go",
    "Rust": "Rust",
    "Shell": "Shell",
}

HERMES_CONFIGS = (
    ("func_calling_singleturn", 6),
    ("func_calling", 6),
    ("json_mode_agentic", 2),
    ("json_mode_singleturn", 2),
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_output = (
        repo_root
        / "data"
        / "calibration"
        / "generated"
        / "qwen3.8-27b-multidomain-128x2048"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--sequence-length", type=int, default=SEQUENCE_LENGTH)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output while preserving it in a .previous sibling directory.",
    )
    parser.add_argument(
        "--stack-max-rows",
        type=int,
        default=25000,
        help="Maximum Stack v3 repositories to scan before reporting missing languages.",
    )
    parser.add_argument(
        "--stack-shuffle-buffer",
        type=int,
        default=32,
        help="Small streaming shuffle buffer; avoids loading Stack v3 locally.",
    )
    return parser.parse_args()


def encode_chat(tokenizer: Any, messages: list[dict[str, str]], sequence_length: int) -> list[int]:
    """Render with the target model's own chat template and cap long records."""
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        truncation=True,
        max_length=sequence_length,
    )
    if hasattr(token_ids, "keys"):
        token_ids = token_ids["input_ids"]
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return [int(token_id) for token_id in token_ids]


def normalized_messages(turns: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    role_map = {
        "human": "user",
        "user": "user",
        "gpt": "assistant",
        "assistant": "assistant",
        "system": "system",
        "tool": "tool",
        "observation": "tool",
    }
    messages = []
    for turn in turns:
        content = turn.get("content", turn.get("value", ""))
        if content is None or not str(content).strip():
            continue
        role = role_map.get(str(turn.get("role", turn.get("from", "user"))).lower(), "user")
        messages.append({"role": role, "content": str(content)})
    return messages


def pack_from_stream(
    *,
    repo_id: str,
    config: str,
    count: int,
    source: str,
    domain: str,
    language: str,
    tokenizer: Any,
    sequence_length: int,
    seed: int,
    formatter: Callable[[dict[str, Any]], list[dict[str, str]]],
    prompt_language: str | None = None,
    predicate: Callable[[dict[str, Any]], bool] | None = None,
    shuffle_buffer: int = 128,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if count == 0:
        return [], {"repo_id": repo_id, "config": config, "rows_scanned": 0, "blocks": 0}

    stream = load_dataset(repo_id, name=config, split="train", streaming=True)
    if shuffle_buffer > 1:
        stream = stream.shuffle(seed=seed, buffer_size=shuffle_buffer)

    blocks: list[dict[str, Any]] = []
    pending: list[int] = []
    rows_scanned = 0
    for row in stream:
        rows_scanned += 1
        if predicate is not None and not predicate(row):
            continue

        messages = formatter(row)
        if not messages:
            continue
        pending.extend(encode_chat(tokenizer, messages, sequence_length))
        while len(pending) >= sequence_length and len(blocks) < count:
            block = pending[:sequence_length]
            del pending[:sequence_length]
            blocks.append(
                {
                    "input_ids": block,
                    "attention_mask": [1] * sequence_length,
                    "source": source,
                    "domain": domain,
                    "language": language,
                    "prompt_language": prompt_language or language,
                }
            )
        if len(blocks) >= count:
            break

    if len(blocks) != count:
        raise RuntimeError(
            f"{repo_id}/{config}: requested {count} blocks, got {len(blocks)} after "
            f"scanning {rows_scanned} rows."
        )
    return blocks, {
        "repo_id": repo_id,
        "config": config,
        "split": "train",
        "rows_scanned": rows_scanned,
        "blocks": len(blocks),
    }


def aya_formatter(row: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": str(row.get("inputs", ""))},
        {"role": "assistant", "content": str(row.get("targets", ""))},
    ]


def hermes_formatter(row: dict[str, Any]) -> list[dict[str, str]]:
    messages = normalized_messages(row.get("conversations", []))
    tools = row.get("tools")
    if tools:
        system_message = next((item for item in messages if item["role"] == "system"), None)
        if system_message and "<tools>" not in system_message["content"]:
            system_message["content"] += f"\n<tools>\n{tools}\n</tools>"
    return messages


def mimo_formatter(row: dict[str, Any]) -> list[dict[str, str]]:
    return normalized_messages(row.get("prompt", []))


def korean_agent_formatter(row: dict[str, Any]) -> list[dict[str, str]]:
    """Serialize the Korean train trajectory and preserve its tool-call events."""
    messages = []
    for turn in row.get("messages", []):
        role = str(turn.get("role", "user")).lower()
        if role not in {"system", "user", "assistant", "tool"}:
            role = "user"
        content = str(turn.get("content") or "")

        if role == "system" and turn.get("tools"):
            tools = json.dumps(turn["tools"], ensure_ascii=False, separators=(",", ":"))
            content += f"\n<tools>\n{tools}\n</tools>"

        for call in turn.get("tool_calls") or []:
            function = call.get("function") or {}
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    pass
            payload = {
                "name": function.get("name", ""),
                "arguments": arguments,
            }
            tool_call = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            tool_call = f"<tool_call>\n{tool_call}\n</tool_call>"
            content = f"{content}\n{tool_call}" if content else tool_call

        if role == "tool":
            content = f"<tool_response>\n{content}\n</tool_response>"
        if content.strip():
            message = {"role": role, "content": content}
            if role == "tool" and turn.get("name"):
                message["name"] = str(turn["name"])
            messages.append(message)
    return messages


def build_stack_blocks(
    *,
    tokenizer: Any,
    sequence_length: int,
    seed: int,
    max_rows: int,
    shuffle_buffer: int,
    per_language: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    repo_id = "HuggingFaceCode/stack-v3-train"
    stream = load_dataset(repo_id, split="train", streaming=True)
    if shuffle_buffer > 1:
        stream = stream.shuffle(seed=seed, buffer_size=shuffle_buffer)

    quotas = {language: per_language for language in STACK_LANGUAGES}
    blocks_by_language: dict[str, list[dict[str, Any]]] = {
        language: [] for language in STACK_LANGUAGES
    }
    pending_by_language: dict[str, list[int]] = {
        language: [] for language in STACK_LANGUAGES
    }
    korean_pending_by_language: dict[str, list[int]] = {
        language: [] for language in STACK_LANGUAGES
    }
    korean_prompt_languages = {"Python", "JavaScript", "TypeScript", "Rust"}
    korean_prompt_done: set[str] = set()
    korean_language_names = {
        "Python": "파이썬",
        "JavaScript": "자바스크립트",
        "TypeScript": "타입스크립트",
        "Java": "자바",
        "C++": "C++",
        "Go": "Go",
        "Rust": "Rust",
        "Shell": "셸",
    }
    rows_scanned = 0

    for row in stream:
        rows_scanned += 1
        files_by_language: dict[str, list[dict[str, Any]]] = {
            language: [] for language in STACK_LANGUAGES
        }
        for file_record in row.get("files", []):
            language = file_record.get("language")
            if language not in STACK_LANGUAGES:
                continue
            if file_record.get("license_type") != "permissive":
                continue
            if file_record.get("is_vendor", False):
                continue
            content = file_record.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            if file_record.get("size_bytes", len(content.encode("utf-8"))) > 1_000_000:
                continue
            files_by_language[language].append(file_record)

        for language, file_records in files_by_language.items():
            if len(blocks_by_language[language]) >= quotas[language] or not file_records:
                continue
            code_parts = []
            for file_record in file_records[:3]:
                code_parts.append(
                    f"### {file_record.get('file_path', 'source')}\n"
                    f"```{language.lower()}\n{file_record['content']}\n```"
                )
            use_korean_prompt = (
                language in korean_prompt_languages and language not in korean_prompt_done
            )
            if use_korean_prompt:
                prompt_language = "Korean"
                prompt = (
                    f"다음 {korean_language_names[language]} 코드를 검토하고, "
                    "무엇을 구현하는지와 잠재적인 오류를 한국어로 설명해 주세요.\n\n"
                    + "\n\n".join(code_parts)
                )
                pending = korean_pending_by_language[language]
            else:
                prompt_language = "English"
                prompt = (
                    f"Review this {language} source code. Explain its purpose and identify "
                    "important implementation details.\n\n"
                    + "\n\n".join(code_parts)
                )
                pending = pending_by_language[language]
            messages = [{"role": "user", "content": prompt}]
            pending.extend(encode_chat(tokenizer, messages, sequence_length))
            while (
                len(pending) >= sequence_length
                and len(blocks_by_language[language]) < quotas[language]
            ):
                block = pending[:sequence_length]
                del pending[:sequence_length]
                blocks_by_language[language].append(
                    {
                        "input_ids": block,
                        "attention_mask": [1] * sequence_length,
                        "source": "stack-v3-train",
                        "domain": "code",
                        "language": language,
                        "prompt_language": prompt_language,
                    }
                )
                if use_korean_prompt:
                    korean_prompt_done.add(language)
                    break

        if all(len(blocks_by_language[language]) >= quotas[language] for language in quotas):
            break
        if rows_scanned >= max_rows:
            break
        if rows_scanned % 1000 == 0:
            counts = ", ".join(
                f"{language}={len(blocks_by_language[language])}/{quotas[language]}"
                for language in quotas
            )
            print(f"Stack v3: scanned {rows_scanned} repositories ({counts})", flush=True)

    missing = {
        language: quotas[language] - len(blocks_by_language[language])
        for language in quotas
        if len(blocks_by_language[language]) < quotas[language]
    }
    if missing:
        raise RuntimeError(
            f"Stack v3 did not supply enough permissively licensed code blocks: {missing}. "
            f"Scanned {rows_scanned} repositories; raise --stack-max-rows to continue."
        )

    blocks = [block for language_blocks in blocks_by_language.values() for block in language_blocks]
    return blocks, {
        "repo_id": repo_id,
        "config": "default",
        "split": "train",
        "license_filter": "permissive and not vendor",
        "rows_scanned": rows_scanned,
        "blocks_by_language": {
            language: len(language_blocks)
            for language, language_blocks in blocks_by_language.items()
        },
        "blocks": len(blocks),
    }


def main() -> None:
    args = parse_args()
    if args.sequence_length <= 0:
        raise ValueError("--sequence-length must be positive")
    if args.stack_max_rows <= 0:
        raise ValueError("--stack-max-rows must be positive")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {args.output}. Pass --overwrite to replace it safely."
        )

    print(f"Loading tokenizer: {args.model_id}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        use_fast=True,
        trust_remote_code=True,
    )
    if not tokenizer.chat_template:
        raise RuntimeError(f"Tokenizer {args.model_id} has no chat template")

    blocks: list[dict[str, Any]] = []
    source_manifest: list[dict[str, Any]] = []

    # Preserve a 48-window multilingual budget across the original six languages and Korean.
    for index, (config, language) in enumerate(NATURAL_LANGUAGES.items()):
        count = AYA_COUNTS[config]
        print(f"Aya: {language}, {count} blocks", flush=True)
        language_blocks, stats = pack_from_stream(
            repo_id="CohereLabs/aya_collection_language_split",
            config=config,
            count=count,
            source="aya_collection_language_split",
            domain="multilingual-instruction",
            language=language,
            tokenizer=tokenizer,
            sequence_length=args.sequence_length,
            seed=args.seed + index,
            formatter=aya_formatter,
            prompt_language=language,
        )
        blocks.extend(language_blocks)
        source_manifest.append(stats)

    # Function calls and structured JSON output (train configs only).
    for index, (config, count) in enumerate(HERMES_CONFIGS):
        print(f"Hermes: {config}, {count} blocks", flush=True)
        hermes_blocks, stats = pack_from_stream(
            repo_id="NousResearch/hermes-function-calling-v1",
            config=config,
            count=count,
            source="hermes-function-calling-v1",
            domain="tool-use-and-structured-output",
            language="English",
            tokenizer=tokenizer,
            sequence_length=args.sequence_length,
            seed=args.seed + 100 + index,
            formatter=hermes_formatter,
        )
        blocks.extend(hermes_blocks)
        source_manifest.append(stats)

    print("MiMo: code-agent tasks, 20 blocks", flush=True)
    mimo_code, stats = pack_from_stream(
        repo_id="XiaomiMiMo/MiMo-V2.6-RL-oss",
        config="code",
        count=20,
        source="mimo-v2.6-rl-oss",
        domain="agentic-coding",
        language="mixed",
        tokenizer=tokenizer,
        sequence_length=args.sequence_length,
        seed=args.seed + 200,
        formatter=mimo_formatter,
    )
    blocks.extend(mimo_code)
    source_manifest.append(stats)

    print("Korean: agent/tool trajectories from train split, 4 blocks", flush=True)
    korean_agent, stats = pack_from_stream(
        repo_id="taejoon89/Ko-Agent-Trajectories-1.0",
        config="train",
        count=4,
        source="ko-agent-trajectories-1.0",
        domain="korean-tool-agent",
        language="Korean",
        tokenizer=tokenizer,
        sequence_length=args.sequence_length,
        seed=args.seed + 150,
        formatter=korean_agent_formatter,
        prompt_language="Korean",
        shuffle_buffer=16,
    )
    blocks.extend(korean_agent)
    source_manifest.append(
        {
            **stats,
            "license": "CC-BY-4.0",
            "attribution": "Imjin Ahn, Young-Hak Kim, Sanghyun Park, and Tae Joon Jun",
        }
    )

    print("MiMo: general-agent tasks, 8 blocks (Terminal-Bench filtered out)", flush=True)

    def general_agent_only(row: dict[str, Any]) -> bool:
        if row.get("data_source") != "mimoagent/general_agent":
            return False
        extra_info = row.get("extra_info") or {}
        return extra_info.get("dataset_type") != "terminal_bench"

    mimo_general, stats = pack_from_stream(
        repo_id="XiaomiMiMo/MiMo-V2.6-RL-oss",
        config="general",
        count=8,
        source="mimo-v2.6-rl-oss",
        domain="general-agent",
        language="mixed",
        tokenizer=tokenizer,
        sequence_length=args.sequence_length,
        seed=args.seed + 201,
        formatter=mimo_formatter,
        predicate=general_agent_only,
    )
    blocks.extend(mimo_general)
    source_manifest.append({**stats, "filter": "data_source=mimoagent/general_agent; not terminal_bench"})

    print(
        "Stack v3: permissively licensed code in 8 languages, including Korean review prompts",
        flush=True,
    )
    stack_blocks, stats = build_stack_blocks(
        tokenizer=tokenizer,
        sequence_length=args.sequence_length,
        seed=args.seed + 300,
        max_rows=args.stack_max_rows,
        shuffle_buffer=args.stack_shuffle_buffer,
        per_language=4,
    )
    blocks.extend(stack_blocks)
    source_manifest.append(stats)

    expected_count = 128
    if len(blocks) != expected_count:
        raise RuntimeError(f"Expected {expected_count} calibration blocks; built {len(blocks)}")
    for index, block in enumerate(blocks):
        if len(block["input_ids"]) != args.sequence_length:
            raise RuntimeError(f"Block {index} has an unexpected token length")

    random.Random(args.seed).shuffle(blocks)
    dataset = Dataset.from_list(blocks)
    source_counts = Counter(block["source"] for block in blocks)
    domain_counts = Counter(block["domain"] for block in blocks)
    language_counts = Counter(block["language"] for block in blocks)
    prompt_language_counts = Counter(block["prompt_language"] for block in blocks)
    manifest = {
        "model_id": args.model_id,
        "tokenizer_name_or_path": tokenizer.name_or_path,
        "sequence_length": args.sequence_length,
        "num_samples": len(blocks),
        "tokens_total": len(blocks) * args.sequence_length,
        "seed": args.seed,
        "dataset_path": str(args.output / "dataset"),
        "schema": {
            "input_ids": f"list[int] of length {args.sequence_length}",
            "attention_mask": f"list[int] of length {args.sequence_length}, all ones",
            "source": "string",
            "domain": "string",
            "language": "string",
            "prompt_language": "string",
        },
        "counts": {
            "source": dict(source_counts),
            "domain": dict(domain_counts),
            "language": dict(language_counts),
            "prompt_language": dict(prompt_language_counts),
        },
        "sources": source_manifest,
        "attribution": [
            {
                "dataset": "taejoon89/Ko-Agent-Trajectories-1.0",
                "license": "CC-BY-4.0",
                "authors": [
                    "Imjin Ahn",
                    "Young-Hak Kim",
                    "Sanghyun Park",
                    "Tae Joon Jun",
                ],
                "config": "train",
            }
        ],
        "benchmark_exclusions": [
            "datacurve/deep-swe",
            "harborframework/terminal-bench-2.1",
            "harborframework/terminal-bench",
            "hkust-nlp/Toolathlon",
            "mimoagent/terminal_bench examples in the MiMo general config",
            "taejoon89/Ko-Agent-Trajectories-1.0 eval split (train split only)",
        ],
    }

    previous_output = None
    if args.output.exists():
        previous_output = args.output.with_name(f"{args.output.name}.previous")
        if previous_output.exists():
            raise FileExistsError(
                f"Cannot preserve old output: backup path already exists: {previous_output}"
            )
        os.replace(args.output, previous_output)
    try:
        args.output.mkdir(parents=True)
        dataset.save_to_disk(str(args.output / "dataset"))
        manifest_path = args.output / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    except Exception:
        if args.output.exists():
            shutil.rmtree(args.output)
        if previous_output is not None:
            os.replace(previous_output, args.output)
        raise

    print(f"Saved {len(dataset)} sequences × {args.sequence_length} tokens to {args.output}")
    print(f"Source counts: {dict(source_counts)}")
    print(f"Content languages: {dict(language_counts)}")
    print(f"Prompt languages: {dict(prompt_language_counts)}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDataset preparation cancelled.", file=sys.stderr)
        raise
