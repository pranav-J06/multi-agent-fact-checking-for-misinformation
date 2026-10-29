"""
Base Agent module for LLM interaction using Ollama.
Provides resilient chat completions, JSON schema enforcement, error handling, and latency tracking.
"""

import time
import json
import re
import logging
from typing import Dict, Any, List, Optional, Tuple
import ollama

from src.config import LLM_MODEL_NAME, LLM_TEMPERATURE, LLM_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)

class BaseAgent:
    def __init__(self, model_name: str = LLM_MODEL_NAME, temperature: float = LLM_TEMPERATURE):
        self.model_name = model_name
        self.temperature = temperature
        # Configure client with configured timeout
        self.client = ollama.Client(timeout=LLM_TIMEOUT_SECONDS)

    def _extract_json_from_text(self, text: str) -> Optional[Any]:
        """Extracts JSON object or list from messy LLM text or markdown code fences."""
        text = text.strip()
        # 1. Direct parse attempt
        try:
            return json.loads(text)
        except Exception:
            pass

        # 2. Markdown code fence extraction: ```json ... ``` or ``` ... ```
        fence_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text, re.IGNORECASE)
        if fence_match:
            try:
                return json.loads(fence_match.group(1).strip())
            except Exception:
                pass

        # 3. Bracket bounds extraction
        start_obj = text.find("{")
        end_obj = text.rfind("}")
        start_arr = text.find("[")
        end_arr = text.rfind("]")

        if start_obj != -1 and end_obj != -1 and (start_arr == -1 or start_obj < start_arr):
            try:
                return json.loads(text[start_obj:end_obj+1])
            except Exception:
                # Attempt to close truncated JSON array/object if cut off
                candidate = text[start_obj:end_obj+1]
                for closer in ["]}", "}", "\"]}", "}\n]"]:
                    try:
                        return json.loads(candidate + closer)
                    except Exception:
                        pass
        elif start_arr != -1 and end_arr != -1:
            try:
                return json.loads(text[start_arr:end_arr+1])
            except Exception:
                pass

        return None

    def generate_json(self, prompt: str, system_prompt: Optional[str] = None) -> Tuple[Dict[str, Any], float]:
        """
        Sends a prompt to the LLM requiring JSON output.
        Returns:
            parsed_json: Dict[str, Any]
            latency_seconds: float
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        t0 = time.perf_counter()
        options = {
            "temperature": self.temperature,
            "num_predict": 1024,
            "num_ctx": 4096
        }

        # Attempt 1: Native format='json'
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=messages,
                format="json",
                options=options
            )
            elapsed = time.perf_counter() - t0
            raw_text = response["message"]["content"]
            parsed = self._extract_json_from_text(raw_text)
            if parsed is not None:
                if isinstance(parsed, list):
                    return {"items": parsed}, elapsed
                return parsed, elapsed
            logger.warning(f"[BaseAgent] format='json' output failed parse. Raw: {raw_text[:120]}...")
        except Exception as e:
            logger.warning(f"[BaseAgent] Primary JSON call failed: {e}. Retrying without format='json' constraint...")

        # Attempt 2: Direct call without forced JSON format constraint (handles models that balk at format='json')
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=messages,
                options=options
            )
            elapsed = time.perf_counter() - t0
            raw_text = response["message"]["content"]
            parsed = self._extract_json_from_text(raw_text)
            if parsed is not None:
                if isinstance(parsed, list):
                    return {"items": parsed}, elapsed
                return parsed, elapsed
            logger.error(f"[BaseAgent] Could not extract valid JSON from LLM response: {raw_text[:200]}")
            return {"parse_error": True, "raw": raw_text}, elapsed
        except Exception as e2:
            elapsed = time.perf_counter() - t0
            logger.error(f"[BaseAgent] Both LLM generation attempts failed: {e2}")
            return {"error": str(e2), "parse_error": True}, elapsed
