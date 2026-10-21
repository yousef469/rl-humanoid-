import json
import logging
import asyncio
import httpx
from models import LLMResponse, LLMCommand, PerceptionData
from prompt_templates import build_prompt

logger = logging.getLogger(__name__)


class Reasoner:
    def __init__(self, ollama_url: str, model: str):
        self.ollama_url = ollama_url
        self.model = model
        self.client = httpx.Client(timeout=300.0)

    def close(self):
        self.client.close()

    async def reason(self, perception: PerceptionData, user_input: str) -> LLMResponse:
        prompt = build_prompt(perception, user_input)

        try:
            logger.info("Sending request to Ollama...")
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.post(self.ollama_url, json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 256,
                    }
                })
            )
            logger.info("Ollama responded")
            raw_text = response.json()["response"]
            raw_text = raw_text.strip()
            if raw_text.startswith("```"):
                raw_text = raw_text.split("\n", 1)[1] if "\n" in raw_text else raw_text
                raw_text = raw_text.rsplit("```", 1)[0].strip()
            raw = json.loads(raw_text)
            return self._parse_response(raw)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            return self._fallback()
        except httpx.TimeoutException:
            logger.error("Ollama request timed out")
            return self._fallback("I'm thinking about that. Give me a moment.")
        except Exception as e:
            logger.error(f"Ollama request failed: {e}")
            return self._fallback()

    def _parse_response(self, raw: dict) -> LLMResponse:
        commands = []
        for c in raw.get("commands", []):
            commands.append(LLMCommand(action=c["action"], params=c.get("params", {})))
        return LLMResponse(
            thought=raw.get("thought", ""),
            speech=raw.get("speech", "I'm not sure what to say."),
            commands=commands,
        )

    def _fallback(self, speech: str = "I'm still thinking. One moment please.") -> LLMResponse:
        return LLMResponse(
            thought="LLM request failed, using fallback.",
            speech=speech,
            commands=[LLMCommand(action="speak", params={"text": speech})],
        )
