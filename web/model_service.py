import threading
import queue
import json
from pathlib import Path
from mlx_lm import load, generate, stream_generate

class ModelService:
    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._queue = queue.Queue()
        self._stop_event = threading.Event()
        self._worker_thread = None
        self._lock = threading.Lock()

    def start(self, model_path, adapter_path):
        """Initializes the model and starts the worker thread."""
        print(f"[ModelService] Loading model from {model_path}...")
        self._model, self._tokenizer = load(model_path, adapter_path=adapter_path)
        print("[ModelService] Model loaded.")
        
        self._worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self._worker_thread.start()
        print("[ModelService] Worker thread started.")

    def get_tokenizer(self):
        """Returns the loaded tokenizer."""
        return self._tokenizer

    def stop(self):
        """Stops the worker thread."""
        self._stop_event.set()
        if self._worker_thread:
            self._worker_thread.join()

    def generate_chat(self, prompt, max_tokens=1024):
        """Blocking/Streaming call for chat generation."""
        result_queue = queue.Queue()
        # Message format: (type, prompt, max_tokens, result_queue)
        self._queue.put(("chat", prompt, max_tokens, result_queue))
        
        # Yield results as they come in
        while True:
            chunk = result_queue.get()
            if chunk is None: # Sentinel for end
                break
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk

    def submit_analysis(self, prompt, callback):
        """Non-blocking call for background analysis (low priority)."""
        # Message format: (type, prompt, max_tokens, callback)
        self._queue.put(("analysis", prompt, 512, callback))

    def _process_queue(self):
        """Worker loop that processes requests sequentially."""
        while not self._stop_event.is_set():
            try:
                # Wait for a job
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            msg_type, prompt, max_tokens, output_handler = item
            
            try:
                if msg_type == "chat":
                    # Streaming Generation
                    for response in stream_generate(
                        model=self._model,
                        tokenizer=self._tokenizer,
                        prompt=prompt,
                        max_tokens=max_tokens
                    ):
                        output_handler.put(response.text)
                    
                    # Signal end
                    output_handler.put(None)
                    
                elif msg_type == "analysis":
                    # Blocking Generation (for backward compatibility / simplicity for analysis)
                    response = generate(
                        model=self._model,
                        tokenizer=self._tokenizer,
                        prompt=prompt,
                        max_tokens=max_tokens,
                        verbose=False
                    )
                    text = response.text if hasattr(response, 'text') else response
                    # Cleanup prompt echo if needed (generate sometimes echoes)
                    if text.startswith(prompt):
                         text = text[len(prompt):].strip()
                         
                    try:
                        output_handler(text)
                    except Exception as e:
                        print(f"[ModelService] Analysis callback failed: {e}")

            except Exception as e:
                print(f"[ModelService] Error processing job: {e}")
                if msg_type == "chat":
                    output_handler.put(e)
                    output_handler.put(None)
            
            finally:
                self._queue.task_done()

# Global Singleton
service = ModelService()
