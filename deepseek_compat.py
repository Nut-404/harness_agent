from __future__ import annotations

import json
import os
from types import SimpleNamespace
from typing import Any

from openai import OpenAI


class DeepSeekMessagesClient:
    def __init__(self, openai_client: OpenAI):
        self._openai_client = openai_client
        self.messages = self

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 8000,
        **kwargs: Any,
    ) -> SimpleNamespace:
        request: dict[str, Any] = {
            "model": model,
            "messages": self._convert_messages(messages, system),
            "max_tokens": max_tokens,
        }
        if tools is not None:
            request["tools"] = [self._convert_tool(tool) for tool in tools]
        if "temperature" in kwargs:
            request["temperature"] = kwargs["temperature"]

        response = self._openai_client.chat.completions.create(**request)
        choice = response.choices[0]
        return SimpleNamespace(
            content=self._convert_response_content(choice.message),
            stop_reason=self._convert_stop_reason(choice),
            raw=response,
        )

    def _convert_messages(self, messages: list[dict[str, Any]], system: str | None) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        if system:
            converted.append({"role": "system", "content": system})

        for message in messages:
            role = message["role"]
            content = message.get("content")
            if role == "assistant" and isinstance(content, list):
                converted.append(self._convert_assistant_message(content))
            elif role == "user" and isinstance(content, list) and self._is_tool_result_list(content):
                converted.extend(self._convert_tool_result_messages(content))
            else:
                converted.append({"role": role, "content": self._convert_text_content(content)})
        return converted

    def _convert_assistant_message(self, content: list[Any]) -> dict[str, Any]:
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for block in content:
            block_type = self._get(block, "type")
            if block_type == "text":
                text_parts.append(self._get(block, "text") or "")
            elif block_type == "tool_use":
                tool_calls.append({
                    "id": self._get(block, "id"),
                    "type": "function",
                    "function": {
                        "name": self._get(block, "name"),
                        "arguments": json.dumps(self._get(block, "input") or {}, ensure_ascii=False),
                    },
                })
        message = {"role": "assistant", "content": "\n".join(text_parts) or None}
        if tool_calls:
            message["tool_calls"] = tool_calls
        return message

    def _convert_tool_result_messages(self, content: list[Any]) -> list[dict[str, Any]]:
        return [
            {
                "role": "tool",
                "tool_call_id": self._get(block, "tool_use_id"),
                "content": self._convert_text_content(self._get(block, "content")),
            }
            for block in content
        ]

    def _convert_stop_reason(self, choice: Any) -> str:
        if getattr(choice.message, "tool_calls", None):
            return "tool_use"
        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason == "length":
            return "max_tokens"
        return finish_reason or "end_turn"

    def _convert_response_content(self, message: Any) -> list[SimpleNamespace]:
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            return [
                SimpleNamespace(
                    type="tool_use",
                    id=tool_call.id,
                    name=tool_call.function.name,
                    input=json.loads(tool_call.function.arguments or "{}"),
                )
                for tool_call in tool_calls
            ]
        return [SimpleNamespace(type="text", text=getattr(message, "content", "") or "")]

    def _convert_tool(self, tool: dict[str, Any]) -> dict[str, Any]:
        if tool.get("type") == "function":
            return tool
        return {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
            },
        }

    def _is_tool_result_list(self, content: list[Any]) -> bool:
        return bool(content) and all(self._get(block, "type") == "tool_result" for block in content)

    def _convert_text_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                self._get(block, "text") or self._get(block, "content") or str(block)
                for block in content
            )
        return "" if content is None else str(content)

    def _get(self, value: Any, key: str) -> Any:
        if isinstance(value, dict):
            return value.get(key)
        return getattr(value, key, None)


def create_deepseek_client() -> DeepSeekMessagesClient:
    return DeepSeekMessagesClient(
        OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )
    )
