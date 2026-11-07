"""One backend per API. Each provider turns (SYSTEM, messages, letters) into one call and returns a Reply.

  fake           offline simulated model (no network, no cost) for testing the pipeline
  ollama         local Ollama or Ollama Cloud models via the local daemon (gives token probabilities locally)
  gemini         Google Gemini API (google-genai SDK, GEMINI_API_KEY)
  anthropic      Claude API (anthropic SDK, ANTHROPIC_API_KEY), with incremental prompt caching
  openai         OpenAI API (openai SDK, OPENAI_API_KEY)
  openai_compat  any OpenAI-compatible endpoint (OpenRouter, Together, DeepInfra, vLLM...): set base_url + api_key_env

SDK calls are synchronous and run in a worker thread, so every provider looks the same to the async runner.
"""
import asyncio
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass, field

from .protocol import SYSTEM


@dataclass
class Reply:
    text: str
    usage: dict = field(default_factory=dict)     # in, cached_in, cache_write, out (tokens)
    letter_prob: float | None = None              # P(chosen letter), only where the API exposes logprobs
    raw: object = None                            # provider-native message, reused on the next turn if needed
    model_version: str | None = None              # exact model version string the API reports


class PermanentError(Exception):
    """Errors that retrying cannot fix (bad key, unknown model, payment required)."""


PERMANENT = ("401", "403", "402", "404", "invalid_api_key", "API key not valid", "PERMISSION_DENIED",
             "UNAUTHENTICATED", "NOT_FOUND", "not_found_error", "authentication_error", "Payment Required",
             "model_not_found", "INVALID_ARGUMENT", "invalid_request_error")


def schema(letters):
    return {"type": "object",
            "properties": {"answer": {"type": "string", "enum": list(letters)},
                           "confidence": {"type": "integer", "minimum": 0, "maximum": 100}},
            "required": ["answer", "confidence"]}


def letter_prob(logprobs, letters):
    """P(chosen letter) at the answer position, renormalised over the valid option letters (Ollama format)."""
    if not logprobs:
        return None
    text = ""
    for ent in logprobs:
        tok = ent.get("token", "")
        t = tok.strip().strip('"')
        if re.search(r'"answer"\s*:\s*"?\s*$', text) and t and t in letters:
            alts = {}
            for a in (ent.get("top_logprobs") or []) + [ent]:
                x = a.get("token", "").strip().strip('"')
                if x in letters and len(x) == 1:
                    alts[x] = max(alts.get(x, -1e9), a.get("logprob", -1e9))
            z = sum(math.exp(v) for v in alts.values())
            return math.exp(alts[t]) / z if z > 0 else None
        text += tok
    return None


class Provider:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = cfg["model"]
        self.timeout = cfg.get("timeout", 60)
        t = cfg.get("temperature")
        self.temperature = None if t is None or t is False else float(t)   # false = let the API default

    def call(self, messages, letters):            # synchronous, one attempt
        raise NotImplementedError

    async def __call__(self, messages, letters=None):
        last = None
        for attempt in range(8):
            try:
                return await asyncio.wait_for(asyncio.to_thread(self.call, messages, letters), self.timeout + 15)
            except PermanentError:
                raise
            except Exception as e:                    # rate limits, 5xx, timeouts, dropped connections
                msg = f"{type(e).__name__}: {e}"
                if any(s in msg for s in PERMANENT):
                    raise PermanentError(msg) from e
                last = msg
                await asyncio.sleep(min(60, 2 ** attempt + random.random()))
        raise RuntimeError(f"gave up after 8 attempts: {last}")


class Fake(Provider):
    """Simulated model: praise nudges confidence up and stubbornness up, criticism the reverse."""
    def call(self, messages, letters):
        time.sleep(0.001)
        hist = " ".join(m["content"] for m in messages if m["role"] == "user")
        mood = 1 if re.search(r"smart|brilliant|intelligent|sharp", hist) else 0
        mood = -1 if re.search(r"weak|not very|don't seem", hist) else mood
        last = messages[-1]["content"]
        if letters:
            if "says the answer is (" in last and random.random() < 0.7 - 0.1 * mood:
                ans = re.search(r"answer is \((\w)\)", last).group(1)
            else:
                ans = random.choice(letters)
            text = json.dumps({"answer": ans, "confidence": int(max(0, min(100, random.gauss(80 + 5 * mood, 10))))})
        else:
            text = "No. It did not affect me." if random.random() < 0.8 else "Thanks, happy to continue."
        n_in = sum(len(m["content"]) for m in messages) // 4
        return Reply(text, dict(input=n_in, out=len(text) // 4), letter_prob=random.random(), model_version="fake-1")


class Ollama(Provider):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.host = cfg.get("host", os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        if not self.host.startswith("http"):
            self.host = "http://" + self.host
        self.think = cfg.get("think", False)

    def call(self, messages, letters):
        import requests
        body = {"model": self.model, "stream": False,
                "messages": [{"role": "system", "content": SYSTEM}] +
                            [{"role": m["role"], "content": m["content"]} for m in messages],
                "options": {"num_ctx": self.cfg.get("num_ctx", 16384)},
                "logprobs": True, "top_logprobs": 10}
        if self.think is not None:
            body["think"] = self.think
        if self.temperature is not None:
            body["options"]["temperature"] = self.temperature
        if letters:
            body["format"] = schema(letters)
        r = requests.post(f"{self.host}/api/chat", json=body, timeout=self.timeout)
        if r.status_code == 400 and "think" in r.text and "think" in body:   # model has no thinking switch
            self.think = None
            return self.call(messages, letters)
        if r.status_code in (401, 402, 403, 404):
            raise PermanentError(f"{r.status_code} {r.text[:200]}")
        r.raise_for_status()
        d = r.json()
        return Reply(d["message"]["content"], dict(input=d.get("prompt_eval_count", 0), out=d.get("eval_count", 0)),
                     letter_prob=letter_prob(d.get("logprobs"), letters) if letters else None,
                     model_version=d.get("model"))


class Gemini(Provider):
    def __init__(self, cfg):
        super().__init__(cfg)
        from google import genai
        from google.genai import types
        self.t = types
        self.client = genai.Client(http_options=types.HttpOptions(timeout=self.timeout * 1000))

    def call(self, messages, letters):
        t = self.t
        contents = []
        for m in messages:
            if m["role"] == "assistant" and m.get("raw") is not None:
                contents.append(m["raw"])                 # keeps thought signatures intact
            else:
                contents.append(t.Content(role="user" if m["role"] == "user" else "model",
                                          parts=[t.Part(text=m["content"])]))
        kw = dict(system_instruction=SYSTEM)
        if self.temperature is not None:
            kw["temperature"] = self.temperature
        if self.cfg.get("thinking_level"):
            kw["thinking_config"] = t.ThinkingConfig(thinking_level=self.cfg["thinking_level"])
        if letters:
            kw["response_mime_type"] = "application/json"
            kw["response_schema"] = t.Schema(type=t.Type.OBJECT, required=["answer", "confidence"], properties={
                "answer": t.Schema(type=t.Type.STRING, enum=list(letters)),
                "confidence": t.Schema(type=t.Type.INTEGER)})
        r = self.client.models.generate_content(model=self.model, contents=contents,
                                                config=t.GenerateContentConfig(**kw))
        um = r.usage_metadata
        cached = getattr(um, "cached_content_token_count", 0) or 0
        raw = None
        try:
            c = r.candidates[0].content
            raw = c if c is not None and c.parts else None
        except Exception:
            pass
        return Reply(r.text or "", dict(input=(um.prompt_token_count or 0) - cached, cached_in=cached,
                                        out=(um.candidates_token_count or 0) + (getattr(um, "thoughts_token_count", 0) or 0)),
                     raw=raw, model_version=getattr(r, "model_version", None))


class Anthropic(Provider):
    def __init__(self, cfg):
        super().__init__(cfg)
        import anthropic
        self.client = anthropic.Anthropic(timeout=self.timeout, max_retries=0)

    def call(self, messages, letters):
        msgs = []
        for m in messages:
            if msgs and msgs[-1]["role"] == m["role"]:
                msgs[-1]["content"][0]["text"] += "\n\n" + m["content"]
            else:
                msgs.append({"role": m["role"], "content": [{"type": "text", "text": m["content"]}]})
        msgs[-1]["content"][0]["cache_control"] = {"type": "ephemeral"}     # incremental prompt caching
        kw = dict(self.cfg.get("extra", {}))
        if self.temperature is not None:
            kw["temperature"] = self.temperature
        r = self.client.messages.create(model=self.model, max_tokens=self.cfg.get("max_tokens", 400),
                                        system=SYSTEM, messages=msgs, **kw)
        u = r.usage
        return Reply("".join(getattr(b, "text", "") for b in r.content),
                     dict(input=u.input_tokens, cached_in=getattr(u, "cache_read_input_tokens", 0) or 0,
                          cache_write=getattr(u, "cache_creation_input_tokens", 0) or 0, out=u.output_tokens),
                     model_version=getattr(r, "model", None))


class OpenAI(Provider):
    """OpenAI, or any OpenAI-compatible endpoint when the config sets base_url (+ api_key_env)."""
    def __init__(self, cfg):
        super().__init__(cfg)
        import openai
        key = os.environ.get(cfg["api_key_env"]) if cfg.get("api_key_env") else None
        self.client = openai.OpenAI(base_url=cfg.get("base_url"), api_key=key, timeout=self.timeout, max_retries=0)

    def call(self, messages, letters):
        kw = dict(self.cfg.get("extra", {}))      # e.g. reasoning_effort = "minimal"
        if self.temperature is not None:
            kw["temperature"] = self.temperature
        if letters and self.cfg.get("json_mode", True):
            kw["response_format"] = {"type": "json_object"}
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": SYSTEM}] + [{"role": m["role"], "content": m["content"]} for m in messages],
            **kw)
        u = r.usage
        cached = getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0
        return Reply(r.choices[0].message.content or "",
                     dict(input=(u.prompt_tokens or 0) - cached, cached_in=cached, out=u.completion_tokens or 0),
                     model_version=getattr(r, "model", None))


PROVIDERS = {"fake": Fake, "ollama": Ollama, "gemini": Gemini, "anthropic": Anthropic, "openai": OpenAI,
             "openai_compat": OpenAI}


def make(cfg, fake=False):
    return Fake(cfg) if fake else PROVIDERS[cfg["provider"]](cfg)


def cost(usage, cfg):
    """USD for one call's usage at the model's configured prices (USD per 1M tokens)."""
    p_in = cfg.get("price_in", 0.0)
    return (usage.get("input", 0) * p_in
            + usage.get("cached_in", 0) * cfg.get("price_cached_in", p_in)
            + usage.get("cache_write", 0) * cfg.get("price_cache_write", p_in)
            + usage.get("out", 0) * cfg.get("price_out", 0.0)) / 1e6
