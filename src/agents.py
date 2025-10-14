from typing import List
from src.knowledge_base import KnowledgeBase
from src.llm_client import LLMClient

class SymphonyAgent:
    """A context-aware agent that uses a knowledge base to inform its responses."""

    def __init__(self, agent_name: str, persona_prompt: str, llm_client: LLMClient, retriever: KnowledgeBase):
        self.name = agent_name
        self.persona_prompt = persona_prompt
        self.llm_client = llm_client
        self.retriever = retriever

    def _format_prompt_with_context(self, task: str, context: List[str], history: str) -> str:
        """Formats the final prompt including the persona, history, and retrieved context."""
        context_str = "\n".join(f"- {doc}" for doc in context)
        
        return f"""
        {self.persona_prompt}

        ### CONTEXT FROM COMPANY KNOWLEDGE BASE ###
        {context_str if context else "No specific company knowledge was found for this query."}
        
        ### ONGOING CONVERSATION ###
        {history}

        ### YOUR CURRENT TASK ###
        Analyze the following user-submitted plan and provide your expert opinion based on your persona and the context provided.
        PLAN: "{task}"
        """

    def execute(self, task: str, conversation_history: List[str]) -> str:
        """
        Executes the agent's turn: retrieves knowledge, formats prompt, and calls the LLM.
        """
        print(f"INFO: [{self.name}] Executing task...")
        context = self.retriever.retrieve(query=f"{self.name} analysis of {task}", top_k=2)
        
        history_str = "\n".join(conversation_history)
        full_prompt = self._format_prompt_with_context(task, context, history_str)
        
        response = self.llm_client.invoke(full_prompt)
        
        # We format the output nicely for the UI
        return f"### 🤵 {self.name}\n\n{response}"