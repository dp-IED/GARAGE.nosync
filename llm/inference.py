"""
Shared HTTP client for LM Studio inference.

Provides a simple HTTP client that uses requests library to communicate
with LM Studio's OpenAI-compatible HTTP API, replacing the OpenAI library.
"""

import requests
from typing import List, Dict, Optional, Any
from dataclasses import dataclass


@dataclass
class LMStudioConfig:
    """Configuration for LM Studio HTTP client."""
    model_name: str
    base_url: str
    timeout: int = 60
    max_tokens: Optional[int] = None
    temperature: float = 0.0


class LMStudioClient:
    """
    HTTP-based client for LM Studio using requests library.

    Replaces OpenAI client with direct HTTP POST requests to LM Studio's
    OpenAI-compatible API endpoints.
    """

    def __init__(self, config: LMStudioConfig):
        self.config = config
        self.session = requests.Session()
        # Ensure base_url ends with /v1
        if not self.config.base_url.endswith("/v1"):
            self.config.base_url = self.config.base_url.rstrip("/") + "/v1"

    def check_connection(self) -> bool:
        """Verify connection to LM Studio by querying /v1/models endpoint."""
        try:
            response = self.session.get(
                f"{self.config.base_url}/models",
                timeout=5
            )
            response.raise_for_status()
            return True
        except Exception as e:
            raise RuntimeError(
                f"Failed to connect to LM Studio at {self.config.base_url}: {e}. "
                f"Please ensure LM Studio is running with the HTTP server enabled."
            )

    def chat_completions_create(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a chat completion with structured output support.

        Args:
            messages: List of message dicts with 'role' and 'content'
            response_format: Response format dict (e.g., for JSON schema)
            temperature: Sampling temperature (overrides config default)
            max_tokens: Max tokens in response (overrides config default)
            **kwargs: Additional parameters

        Returns:
            Response dict with 'choices' containing message content
        """
        url = f"{self.config.base_url}/chat/completions"

        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.config.temperature,
            **kwargs
        }

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        elif self.config.max_tokens is not None:
            payload["max_tokens"] = self.config.max_tokens

        if response_format is not None:
            payload["response_format"] = response_format

        response = self.session.post(
            url,
            json=payload,
            timeout=self.config.timeout
        )
        response.raise_for_status()
        return response.json()

    def close(self):
        """Close the requests session."""
        self.session.close()


def create_json_schema_response_format(
    schema_name: str,
    schema_dict: Dict[str, Any],
    strict: bool = True
) -> Dict[str, Any]:
    """
    Create a response format dict for JSON schema structured output.

    Args:
        schema_name: Name for schema (e.g., 'FaultDiagnosis')
        schema_dict: The JSON schema dict (e.g., from Pydantic model)
        strict: Whether to use strict mode

    Returns:
        Response format dict compatible with LM Studio's API
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "strict": strict,
            "schema": schema_dict
        }
    }


def create_client(
    model_name: str = "granite-4.0-h-micro",
    base_url: str = "http://localhost:1234/v1",
    timeout: int = 60,
    max_tokens: Optional[int] = None,
    temperature: float = 0.0,
    check_connection: bool = True
) -> LMStudioClient:
    """
    Factory function to create and optionally verify an LM Studio client.

    Args:
        model_name: Model name in LM Studio
        base_url: Base URL for LM Studio HTTP API
        timeout: Request timeout in seconds
        max_tokens: Default max tokens for responses
        temperature: Default sampling temperature
        check_connection: Whether to verify connection on creation

    Returns:
        Configured LMStudioClient instance
    """
    config = LMStudioConfig(
        model_name=model_name,
        base_url=base_url,
        timeout=timeout,
        max_tokens=max_tokens,
        temperature=temperature
    )

    client = LMStudioClient(config)

    if check_connection:
        client.check_connection()
        print(f"  Connected to LM Studio at {base_url} (model: {model_name})")

    return client
