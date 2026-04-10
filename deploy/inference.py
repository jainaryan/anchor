"""
Model manager — loads all configured GGUFs at startup, one asyncio lock per
model so concurrent requests queue rather than crash.
"""

import asyncio
import os
import threading
from llama_cpp import Llama
from config import MODELS, SYSTEM_PROMPT, N_GPU_LAYERS, N_CTX, MAX_TOKENS, TEMPERATURE, INFERENCE_TIMEOUT


class ModelManager:
    def __init__(self):
        self._models: dict[str, Llama] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._load_all()

    def _load_all(self):
        for m in MODELS:
            print(f"[inference] Loading '{m['name']}' from {m['path']} ...", flush=True)
            self._models[m["id"]] = Llama(
                model_path=m["path"],
                n_gpu_layers=N_GPU_LAYERS,
                n_ctx=N_CTX,
                n_threads=os.cpu_count(),
                verbose=False,
                chat_format=m.get("chat_format", "llama-3"),
            )
            self._locks[m["id"]] = asyncio.Lock()
            print(f"[inference] '{m['name']}' ready.", flush=True)

    def model_ids(self) -> list[str]:
        return list(self._models.keys())

    def _trim_to_context(self, model: Llama, messages: list[dict]) -> list[dict]:
        """Trim oldest messages so history + system prompt + response fit in N_CTX."""
        budget = N_CTX - MAX_TOKENS - 64  # 64 tokens of formatting overhead

        # Subtract system prompt token count
        try:
            sys_tokens = len(model.tokenize(SYSTEM_PROMPT.encode(), add_bos=True, special=True))
        except Exception:
            sys_tokens = len(SYSTEM_PROMPT) // 4 + 10
        budget -= sys_tokens

        # Walk messages newest-first, keep what fits
        trimmed = []
        for msg in reversed(messages):
            try:
                n = len(model.tokenize(msg["content"].encode(), add_bos=False, special=False)) + 6
            except Exception:
                n = len(msg["content"]) // 4 + 6
            if budget - n < 0 and trimmed:
                # Budget exceeded but we already have at least one message — stop
                break
            budget -= n
            trimmed.insert(0, msg)

        return trimmed

    async def stream(self, model_id: str, messages: list[dict]):
        """
        Async generator that yields string tokens.
        Holds the per-model lock for the full duration — requests queue behind it.
        Raises TimeoutError if no token arrives within INFERENCE_TIMEOUT seconds.
        """
        if model_id not in self._models:
            raise ValueError(f"Unknown model: {model_id}")

        model = self._models[model_id]
        lock = self._locks[model_id]

        trimmed = self._trim_to_context(model, messages)
        full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + trimmed

        async with lock:
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue = asyncio.Queue()

            def _run():
                try:
                    for chunk in model.create_chat_completion(
                        messages=full_messages,
                        stream=True,
                        max_tokens=MAX_TOKENS,
                        temperature=TEMPERATURE,
                    ):
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        if delta:
                            asyncio.run_coroutine_threadsafe(queue.put(delta), loop)
                except Exception as e:
                    asyncio.run_coroutine_threadsafe(queue.put(RuntimeError(str(e))), loop)
                finally:
                    asyncio.run_coroutine_threadsafe(queue.put(None), loop)

            t = threading.Thread(target=_run, daemon=True)
            t.start()

            while True:
                try:
                    token = await asyncio.wait_for(queue.get(), timeout=INFERENCE_TIMEOUT)
                except asyncio.TimeoutError:
                    raise TimeoutError(
                        f"Inference stalled — no token received in {INFERENCE_TIMEOUT}s"
                    )
                if token is None:
                    break
                if isinstance(token, Exception):
                    raise token
                yield token
