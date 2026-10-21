"""Speech-only model — uses Gemma 1b via Ollama to generate natural language responses.
Runs async, doesn't block the robot's motion pipeline."""

import logging
import httpx

logger = logging.getLogger(__name__)

SPEECH_SYSTEM = """You are a friendly humanoid robot. Reply naturally to the user.
Keep responses 1-2 sentences. Don't mention your internal commands."""

SPEECH_EXAMPLES = """
User: what do you see?
You: I see a tree about 45 meters ahead of me.

User: walk forward
You: Moving forward now!

User: wave
You: *waves* Hello there!

User: look at the tree
You: Turning to look at the tree.

User: stop
You: OK, stopping now.
"""


class SpeechGenerator:
    def __init__(self, ollama_url: str, model: str = "gemma3:1b"):
        self.ollama_url = ollama_url
        self.model = model
        self.client = httpx.AsyncClient(timeout=60.0)

    async def close(self):
        await self.client.aclose()

    async def generate(self, scene_desc: str, user_input: str) -> str:
        prompt = f"""{SPEECH_SYSTEM}
{SPEECH_EXAMPLES}

Current scene: {scene_desc}
User: {user_input}
You:"""

        try:
            resp = await self.client.post(self.ollama_url, json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.5,
                    "num_predict": 80,
                }
            })
            data = resp.json()
            return data["response"].strip()
        except Exception as e:
            logger.warning(f"Speech generation failed: {e}")
            return ""
