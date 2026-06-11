"""External LLM clients for CICL causal judging.

The project keeps LLM access behind the small ``LLMClient`` protocol from
``llm_judge.py``. This module provides production-facing adapters without
making external SDKs mandatory for local tests.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cicl_agent.causal.llm_judge import LLMCausalContextJudge


DEFAULT_QWEN_MODEL = "qwen3.6-plus"
DEFAULT_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_ANTHROPIC_MODEL = "claude-opus-4-7"
DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com"
DEFAULT_ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_QWEN_LOCAL_BASE = "Qwen/Qwen3.5-9B"
DEFAULT_QWEN_LOCAL_ADAPTER = "XinyuGuan/CICL"
DEFAULT_CODEX_MODEL = "gpt-5.5"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class QwenOpenAICompatibleClient:
    """Minimal DashScope/Qwen client using the OpenAI-compatible chat endpoint."""

    api_key: str
    model: str = DEFAULT_QWEN_MODEL
    base_url: str = DEFAULT_DASHSCOPE_BASE_URL
    timeout_seconds: float = 60.0
    temperature: float = 0.0
    max_tokens: int = 700
    enable_thinking: bool = False

    @classmethod
    def from_env(
        cls,
        api_key_env: str = "DASHSCOPE_API_KEY",
        model: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 60.0,
    ) -> "QwenOpenAICompatibleClient":
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing Qwen/DashScope API key. Set {api_key_env} before using --llm-provider qwen."
            )
        return cls(
            api_key=api_key,
            model=model or os.getenv("QWEN_MODEL", DEFAULT_QWEN_MODEL),
            base_url=base_url
            or os.getenv("DASHSCOPE_BASE_URL")
            or os.getenv("QWEN_BASE_URL")
            or DEFAULT_DASHSCOPE_BASE_URL,
            timeout_seconds=timeout_seconds,
            temperature=float(os.getenv("QWEN_TEMPERATURE", "0")),
            enable_thinking=_env_bool("QWEN_ENABLE_THINKING", False),
        )

    def complete(self, prompt: str) -> str:
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a precise causal critic. Return valid JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
            "enable_thinking": self.enable_thinking,
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Qwen request failed with HTTP {error.code}: {body[:600]}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"Qwen request failed: {error}") from error

        data = json.loads(body)
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError(f"Unexpected Qwen response shape: {body[:600]}") from error
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(f"Qwen returned empty content: {body[:600]}")
        return content


@dataclass(slots=True)
class CodexCliClient:
    """Use the local Codex CLI as a JSON-only judge.

    Each call starts a non-interactive Codex run using the user's existing login
    state. This is suitable for small GPT-family label subsets; it is heavier
    than a normal HTTP completion endpoint and should not be used as a 150-way
    batch client.
    """

    model: str = DEFAULT_CODEX_MODEL
    timeout_seconds: float = 180.0
    executable: str = "codex"
    system_prompt: str = "You are a precise causal critic. Return valid JSON only."

    @classmethod
    def from_env(
        cls,
        model: str | None = None,
        timeout_seconds: float = 180.0,
    ) -> "CodexCliClient":
        return cls(
            model=model or os.getenv("CODEX_MODEL", DEFAULT_CODEX_MODEL),
            timeout_seconds=timeout_seconds,
            executable=os.getenv("CODEX_EXECUTABLE", "codex"),
        )

    def complete(self, prompt: str) -> str:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            output_path = tmp / "last_message.json"
            schema_path = tmp / "judgment_schema.json"
            schema_path.write_text(json.dumps(_judgment_output_schema()), encoding="utf-8")
            completed = subprocess.run(
                [
                    self.executable,
                    "exec",
                    "--ephemeral",
                    "--ignore-rules",
                    "--sandbox",
                    "read-only",
                    "--model",
                    self.model,
                    "--output-schema",
                    str(schema_path),
                    "--output-last-message",
                    str(output_path),
                    "-",
                ],
                input=f"{self.system_prompt}\n\n{prompt}",
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                cwd=os.getcwd(),
            )
            if completed.returncode != 0:
                stderr = completed.stderr.strip()[-800:]
                stdout = completed.stdout.strip()[-800:]
                raise RuntimeError(
                    f"Codex CLI request failed with exit {completed.returncode}: "
                    f"stdout={stdout!r} stderr={stderr!r}"
                )
            if not output_path.exists():
                raise RuntimeError("Codex CLI did not write --output-last-message")
            content = output_path.read_text(encoding="utf-8").strip()
            if not content:
                raise RuntimeError("Codex CLI returned empty content")
            return content


@dataclass(slots=True)
class AnthropicMessagesClient:
    """Minimal Anthropic Messages API client for Claude/Opus judging.

    Compatible with both api.anthropic.com and the Claude-Code-style proxies
    (e.g., idealab) that gate Opus access by inspecting user-agent.
    """

    api_key: str
    model: str = DEFAULT_ANTHROPIC_MODEL
    base_url: str = DEFAULT_ANTHROPIC_BASE_URL
    timeout_seconds: float = 60.0
    max_tokens: int = 700
    anthropic_version: str = DEFAULT_ANTHROPIC_VERSION
    user_agent: str = "claude-cli/1.0.51 (external, cli)"

    @classmethod
    def from_env(
        cls,
        api_key_env: str = "ANTHROPIC_API_KEY",
        model: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 60.0,
    ) -> "AnthropicMessagesClient":
        api_key = (
            os.getenv(api_key_env)
            or os.getenv("ANTHROPIC_AUTH_TOKEN")
            or os.getenv("ANTHROPIC_API_KEY")
        )
        if not api_key:
            raise RuntimeError(
                f"Missing Anthropic API key. Set {api_key_env} or ANTHROPIC_AUTH_TOKEN "
                "before using --llm-provider anthropic."
            )
        return cls(
            api_key=api_key,
            model=model or os.getenv("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL),
            base_url=base_url or os.getenv("ANTHROPIC_BASE_URL", DEFAULT_ANTHROPIC_BASE_URL),
            timeout_seconds=timeout_seconds,
        )

    def complete(self, prompt: str) -> str:
        url = f"{self.base_url.rstrip('/')}/v1/messages"
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": "You are a precise causal critic. Return valid JSON only.",
            "messages": [{"role": "user", "content": prompt}],
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": self.anthropic_version,
                "content-type": "application/json",
                "user-agent": self.user_agent,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Anthropic request failed with HTTP {error.code}: {body[:600]}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"Anthropic request failed: {error}") from error

        data = json.loads(body)
        content_blocks = data.get("content", [])
        texts = [
            block.get("text", "")
            for block in content_blocks
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        content = "\n".join(text for text in texts if text).strip()
        if not content:
            raise RuntimeError(f"Anthropic returned empty content: {body[:600]}")
        return content


@dataclass(slots=True)
class QwenLocalAdapterClient:
    """Run inference against a Qwen base + LoRA adapter.

    Mirrors the recast + quant logic in train_qwen_judge.py / eval_qwen_judge.py
    so the 16 GB V100 budget holds. ``base_path`` and ``adapter_path`` may be
    local paths or Hugging Face repo ids. Model is loaded lazily on first
    ``complete`` so import doesn't pay the cost.
    """

    base_path: str = DEFAULT_QWEN_LOCAL_BASE
    adapter_path: str = DEFAULT_QWEN_LOCAL_ADAPTER
    max_new_tokens: int = 320
    max_input_tokens: int = 1024
    no_quant: bool = False
    system_prompt: str = "You are a precise causal critic. Return valid JSON only."
    _torch: Any = field(default=None, init=False, repr=False)
    _model: Any = field(default=None, init=False, repr=False)
    _tokenizer: Any = field(default=None, init=False, repr=False)

    @classmethod
    def from_env(
        cls,
        base_path: str | None = None,
        adapter_path: str | None = None,
        max_new_tokens: int = 320,
        max_input_tokens: int = 1024,
        no_quant: bool = False,
    ) -> "QwenLocalAdapterClient":
        return cls(
            base_path=base_path or os.getenv("QWEN_LOCAL_BASE", DEFAULT_QWEN_LOCAL_BASE),
            adapter_path=adapter_path or os.getenv("QWEN_LOCAL_ADAPTER", DEFAULT_QWEN_LOCAL_ADAPTER),
            max_new_tokens=int(os.getenv("QWEN_LOCAL_MAX_NEW_TOKENS", max_new_tokens)),
            max_input_tokens=int(os.getenv("QWEN_LOCAL_MAX_INPUT_TOKENS", max_input_tokens)),
            no_quant=no_quant,
        )

    def _ensure_loaded(self) -> None:
        if getattr(self, "_model", None) is not None:
            return
        import torch
        from peft import PeftModel
        from transformers import AutoConfig, AutoTokenizer, BitsAndBytesConfig

        quant_config = None
        if not self.no_quant:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        cfg = AutoConfig.from_pretrained(self.base_path, trust_remote_code=False)
        architectures = getattr(cfg, "architectures", None) or []
        kwargs: dict = dict(
            trust_remote_code=False,
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
        if quant_config is not None:
            kwargs["quantization_config"] = quant_config
            kwargs["device_map"] = {"": 0}

        if any("Qwen3_5" in a for a in architectures):
            from transformers.models.qwen3_5 import Qwen3_5ForConditionalGeneration
            base = Qwen3_5ForConditionalGeneration.from_pretrained(self.base_path, **kwargs)
        else:
            from transformers import AutoModelForCausalLM
            base = AutoModelForCausalLM.from_pretrained(self.base_path, **kwargs)

        if quant_config is not None:
            for name, p in base.named_parameters():
                if p.dtype != torch.float32:
                    continue
                is_ln = (
                    "norm" in name.lower()
                    or "layernorm" in name.lower()
                    or "rmsnorm" in name.lower()
                )
                if is_ln and p.numel() < 100_000:
                    continue
                p.data = p.data.to(torch.float16)

        model = PeftModel.from_pretrained(base, self.adapter_path, is_trainable=False)
        model.eval()
        model.config.use_cache = True

        tokenizer = AutoTokenizer.from_pretrained(self.base_path, trust_remote_code=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        self._torch = torch
        self._model = model
        self._tokenizer = tokenizer

    def complete(self, prompt: str) -> str:
        self._ensure_loaded()
        torch = self._torch
        tokenizer = self._tokenizer
        model = self._model
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        chat = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
        enc = tokenizer(
            [chat],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_input_tokens,
            add_special_tokens=False,
        )
        input_ids = enc.input_ids.to(model.device)
        attention_mask = enc.attention_mask.to(model.device)
        with torch.inference_mode():
            out = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        gen = out[:, input_ids.shape[-1]:]
        text = tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
        if not text.strip():
            raise RuntimeError("QwenLocalAdapterClient returned empty content")
        return text


def normalize_provider(provider: str) -> str:
    normalized = provider.lower().strip()
    if normalized in {"claude", "opus"}:
        return "anthropic"
    if normalized in {"qwen_local", "qwen-local", "local_qwen", "local-qwen"}:
        return "qwen_local"
    if normalized in {"codex", "codex_cli", "codex-cli", "gpt5", "gpt-5", "gpt5.5", "gpt-5.5"}:
        return "codex"
    return normalized


def resolve_llm_model(provider: str, model: str | None = None) -> str:
    provider = normalize_provider(provider)
    if provider == "qwen":
        return model or os.getenv("QWEN_MODEL", DEFAULT_QWEN_MODEL)
    if provider == "anthropic":
        return model or os.getenv("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL)
    if provider == "qwen_local":
        return model or os.getenv("QWEN_LOCAL_ADAPTER", DEFAULT_QWEN_LOCAL_ADAPTER)
    if provider == "codex":
        return model or os.getenv("CODEX_MODEL", DEFAULT_CODEX_MODEL)
    return model or "simulator"


def resolve_llm_api_key_env(provider: str, api_key_env: str | None = None) -> str:
    provider = normalize_provider(provider)
    if api_key_env and api_key_env != "DASHSCOPE_API_KEY":
        return api_key_env
    if provider == "anthropic":
        # Prefer ANTHROPIC_AUTH_TOKEN (claude-code style) but from_env will fall
        # back to ANTHROPIC_API_KEY if the named one is empty.
        return "ANTHROPIC_AUTH_TOKEN" if os.getenv("ANTHROPIC_AUTH_TOKEN") else "ANTHROPIC_API_KEY"
    return api_key_env or "DASHSCOPE_API_KEY"


def build_llm_judge(
    provider: str = "simulator",
    model: str | None = None,
    base_url: str | None = None,
    api_key_env: str = "DASHSCOPE_API_KEY",
    timeout_seconds: float = 60.0,
    max_tokens: int | None = None,
) -> LLMCausalContextJudge:
    """Build the causal judge used by LLM-CICL and distillation."""

    provider = normalize_provider(provider)
    if provider == "simulator":
        return LLMCausalContextJudge()
    if provider == "qwen":
        client = QwenOpenAICompatibleClient.from_env(
            api_key_env=resolve_llm_api_key_env(provider, api_key_env),
            model=resolve_llm_model(provider, model),
            base_url=base_url,
            timeout_seconds=timeout_seconds,
        )
        if max_tokens is not None:
            client.max_tokens = max_tokens
        return LLMCausalContextJudge(client=client)
    if provider == "anthropic":
        client = AnthropicMessagesClient.from_env(
            api_key_env=resolve_llm_api_key_env(provider, api_key_env),
            model=resolve_llm_model(provider, model),
            base_url=base_url,
            timeout_seconds=timeout_seconds,
        )
        if max_tokens is not None:
            client.max_tokens = max_tokens
        return LLMCausalContextJudge(client=client)
    if provider == "codex":
        client = CodexCliClient.from_env(
            model=resolve_llm_model(provider, model),
            timeout_seconds=timeout_seconds,
        )
        return LLMCausalContextJudge(client=client)
    if provider == "qwen_local":
        client = QwenLocalAdapterClient.from_env(
            base_path=base_url,
            adapter_path=model,
        )
        return LLMCausalContextJudge(client=client)
    raise ValueError(f"Unsupported LLM provider: {provider}")


def _judgment_output_schema() -> dict[str, Any]:
    numeric = {"type": "number", "minimum": 0.0, "maximum": 1.0}
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "no_context_action": {"type": "string"},
            "with_context_action": {"type": "string"},
            "action_shift": numeric,
            "necessity": numeric,
            "expected_outcome_uplift": numeric,
            "negative_transfer_risk": numeric,
            "reason": {"type": "string"},
            "confidence": numeric,
        },
        "required": [
            "no_context_action",
            "with_context_action",
            "action_shift",
            "necessity",
            "expected_outcome_uplift",
            "negative_transfer_risk",
            "reason",
            "confidence",
        ],
    }
