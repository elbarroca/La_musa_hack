import yaml
from typing import List, Dict
from src.agents import SymphonyAgent
from src.knowledge_base import KnowledgeBase, MockKnowledgeBase
from src.llm_client import LLMClient

class Orchestrator:
    """Manages the sequence of agent interactions and state."""

    def __init__(self, agent_config_path: str, use_mock_kb: bool = True):
        with open(agent_config_path, 'r') as f:
            agent_configs = yaml.safe_load(f)['agents']

        self.llm_client = LLMClient()
        
        if use_mock_kb:
            self.knowledge_base: KnowledgeBase = MockKnowledgeBase()
        else:
            # Placeholder for real knowledge base implementation
            # self.knowledge_base = AzureAISearchKnowledgeBase(...)
            raise NotImplementedError("AzureAISearchKnowledgeBase is not yet configured.")

        self.agents = [
            SymphonyAgent(
                agent_name=cfg['name'],
                persona_prompt=cfg['prompt'],
                llm_client=self.llm_client,
                retriever=self.knowledge_base
            ) for cfg in agent_configs
        ]

    def run_analysis(self, initial_task: str) -> List[str]:
        """Runs the full multi-agent analysis, returning the complete conversation."""
        if not initial_task or not initial_task.strip():
            raise ValueError("Initial task cannot be empty.")
            
        conversation_history = [f"## Initial Plan for Analysis:\n\n> {initial_task}"]
        
        for agent in self.agents:
            response = agent.execute(task=initial_task, conversation_history=conversation_history)
            conversation_history.append(response)
        
        return conversation_history