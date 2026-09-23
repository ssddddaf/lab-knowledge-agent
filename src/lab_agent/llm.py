import base64
import json
from pathlib import Path
from typing import TypeVar

from openai import OpenAI
from pydantic import BaseModel

from .config import Settings

T = TypeVar("T", bound=BaseModel)


class LLM:
    def __init__(self, config: Settings):
        self.config = config
        self.enabled = bool(config.openai_api_key and config.chat_model)
        self.client = OpenAI(api_key=config.openai_api_key, base_url=config.openai_base_url, timeout=45, max_retries=2) if self.enabled else None

    def json(self, schema: type[T], system: str, user: str, images: list[Path] | None = None) -> T:
        if not self.client:
            raise RuntimeError("LAB_OPENAI_API_KEY and LAB_CHAT_MODEL are required for model generation")
        content: list[dict] = [{"type":"text","text":user}]
        for image in (images or [])[:4]:
            encoded = base64.b64encode(image.read_bytes()).decode("ascii")
            content.append({"type":"image_url","image_url":{"url":f"data:image/png;base64,{encoded}"}})
        response = self.client.chat.completions.create(
            model=self.config.chat_model,
            messages=[{"role":"system","content":system + "\nReturn only JSON matching: " + json.dumps(schema.model_json_schema(),ensure_ascii=False)},
                      {"role":"user","content":content}],
            response_format={"type":"json_object"}, temperature=0,
        )
        return schema.model_validate_json(response.choices[0].message.content or "{}")
