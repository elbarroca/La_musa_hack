"""
LLM Client Module using LangChain with OpenAI
Provides a simple interface for interacting with OpenAI's API
"""

import os
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


class LLMClient:
    """
    A client for interacting with OpenAI via LangChain.
    Provides a unified interface for making LLM calls.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ):
        """
        Initialize the LLM client with LangChain's ChatOpenAI.

        Args:
            model: The OpenAI model to use (defaults to env var or gpt-4o-mini)
            temperature: Sampling temperature (defaults to env var or 0.5)
            max_tokens: Maximum tokens to generate (defaults to 1500)
        """
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables. "
                "Please set it in your .env file."
            )

        # Get configuration from environment or use defaults
        # Per Master.md guidance: temperature 0.5 for balanced reasoning
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.temperature = temperature if temperature is not None else float(os.getenv("OPENAI_TEMPERATURE", "0.5"))
        self.max_tokens = max_tokens or int(os.getenv("OPENAI_MAX_TOKENS", "4000"))

        # Initialize LangChain ChatOpenAI with correct parameter syntax
        # Using 2025 LangChain best practices - 'api_key' parameter is correct
        self.llm = ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=self.api_key,
            streaming=True  # Enable streaming by default for better UX
        )

        print(f"INFO: LLMClient initialized with model={self.model}, temperature={self.temperature}")

    def invoke(self, prompt: str, system_message: Optional[str] = None) -> str:
        """
        Send a prompt to the LLM and return the response.

        Args:
            prompt: The user prompt/query
            system_message: Optional system message to set context

        Returns:
            The LLM's response as a string
        """
        try:
            messages = []

            if system_message:
                messages.append(SystemMessage(content=system_message))

            messages.append(HumanMessage(content=prompt))

            # Invoke the LLM
            response = self.llm.invoke(messages)

            return response.content

        except Exception as e:
            error_message = f"ERROR: Failed to invoke OpenAI model: {e}"
            print(error_message)
            raise RuntimeError(
                "Could not connect to OpenAI API. "
                "Please check your API key and network connection."
            ) from e

    def get_llm(self) -> ChatOpenAI:
        """
        Get the underlying LangChain ChatOpenAI instance.
        Useful for building chains and more complex workflows.

        Returns:
            ChatOpenAI instance
        """
        return self.llm

    def update_temperature(self, temperature: float):
        """
        Update the temperature for subsequent calls.

        Args:
            temperature: New temperature value (0.0 to 2.0)
        """
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")

        self.temperature = temperature
        self.llm.temperature = temperature
        print(f"INFO: Temperature updated to {temperature}")

    def update_model(self, model: str):
        """
        Update the model for subsequent calls.

        Args:
            model: New model name (e.g., 'gpt-4o', 'gpt-4o-mini')
        """
        self.model = model
        self.llm = ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=self.api_key,
            streaming=True
        )
        print(f"INFO: Model updated to {model}")
