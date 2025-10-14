import os
from openai import AzureOpenAI

class LLMClient:
    """A client for interacting with the Azure OpenAI service."""
    def __init__(self):
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        if not all([self.client.api_key, self.client.base_url, self.deployment_name]):
            raise ValueError("Azure OpenAI environment variables are not fully configured.")

    def invoke(self, prompt: str) -> str:
        """Sends a prompt to the model and returns the response content."""
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[{"role": "system", "content": prompt}],
                temperature=0.7,
                max_tokens=800
            )
            return response.choices[0].message.content
        except Exception as e:
            error_message = f"ERROR: Failed to invoke Azure OpenAI model: {e}"
            print(error_message)
            # Propagate a user-friendly error message
            raise RuntimeError("Could not connect to the AI model. Please check your API credentials and Azure service status.") from e