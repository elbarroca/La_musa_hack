"""
Orchestrator Module with LangGraph StateGraph
Manages multi-agent workflow with proper state management
"""

import yaml
from typing import List, Dict, TypedDict, Annotated
from operator import add

from langgraph.graph import StateGraph, END

from src.agents import SymphonyAgent
from src.knowledge_base import KnowledgeBase, ChromaKnowledgeBase
from src.llm_client import LLMClient


# Define the state structure for the agent graph
class AgentState(TypedDict):
    """State that gets passed between agents in the graph."""
    task: str  # The original user task/plan
    conversation_history: Annotated[List[str], add]  # Accumulates all agent outputs
    current_agent_index: int  # Tracks which agent is currently active
    agent_outputs: Dict[str, str]  # Maps agent name to their output


class Orchestrator:
    """
    Manages the sequence of agent interactions using LangGraph.
    Implements proper state management and agent routing.
    """

    def __init__(self, agent_config_path: str, use_mock_kb: bool = False):
        """
        Initialize the orchestrator with agents and knowledge base.
        Mock mode removed - always uses real ChromaKnowledgeBase for production RAG.

        Args:
            agent_config_path: Path to the agents.yaml configuration file
            use_mock_kb: Deprecated parameter - always uses ChromaKnowledgeBase
        """
        # Load agent configurations
        with open(agent_config_path, 'r') as f:
            agent_configs = yaml.safe_load(f)['agents']

        # Initialize LLM client
        self.llm_client = LLMClient()

        # Initialize knowledge base - always use real RAG
        self.knowledge_base: KnowledgeBase = ChromaKnowledgeBase()
        print("INFO: Using ChromaKnowledgeBase with production RAG")

        # Create agents
        self.agents = [
            SymphonyAgent(
                agent_name=cfg['name'],
                persona_prompt=cfg['prompt'],
                llm_client=self.llm_client,
                retriever=self.knowledge_base
            ) for cfg in agent_configs
        ]

        self.agent_names = [agent.name for agent in self.agents]
        print(f"INFO: Initialized {len(self.agents)} agents: {', '.join(self.agent_names)}")

        # Build the LangGraph workflow
        self._build_graph()

    def _build_graph(self):
        """
        Build the LangGraph StateGraph for agent orchestration.
        Creates a sequential workflow: Shrek -> Sonic -> Hulk -> Trevor -> END
        """
        # Create the graph
        workflow = StateGraph(AgentState)

        # Add nodes for each agent
        for i, agent in enumerate(self.agents):
            workflow.add_node(agent.name, self._create_agent_node(agent, i))

        # Define the sequential flow
        workflow.set_entry_point(self.agents[0].name)

        for i in range(len(self.agents) - 1):
            workflow.add_edge(self.agents[i].name, self.agents[i + 1].name)

        # Last agent leads to END
        workflow.add_edge(self.agents[-1].name, END)

        # Compile the graph
        self.graph = workflow.compile()
        print("INFO: LangGraph workflow compiled successfully")

    def _create_agent_node(self, agent: SymphonyAgent, index: int):
        """
        Create a node function for an agent.

        Args:
            agent: The SymphonyAgent instance
            index: The index of this agent in the sequence

        Returns:
            A function that executes the agent and updates state
        """
        def agent_node(state: AgentState) -> AgentState:
            """Execute the agent and update the state."""
            print(f"\nINFO: Executing agent {index + 1}/{len(self.agents)}: {agent.name}")

            # Execute the agent
            response = agent.execute(
                task=state["task"],
                conversation_history=state["conversation_history"]
            )

            # Update state
            return {
                "task": state["task"],
                "conversation_history": [response],  # Will be accumulated via the add operator
                "current_agent_index": index + 1,
                "agent_outputs": {**state.get("agent_outputs", {}), agent.name: response}
            }

        return agent_node

    def run_analysis(self, initial_task: str) -> List[str]:
        """
        Run the full multi-agent analysis using the LangGraph workflow.

        Args:
            initial_task: The user's strategic plan or business idea

        Returns:
            List of all agent outputs (conversation history)
        """
        if not initial_task or not initial_task.strip():
            raise ValueError("Initial task cannot be empty.")

        print(f"\n{'='*80}")
        print("🎼 SYMPHONY ANALYSIS STARTING")
        print(f"{'='*80}\n")

        # Initialize state
        initial_state: AgentState = {
            "task": initial_task,
            "conversation_history": [f"## Initial Plan for Analysis:\n\n> {initial_task}"],
            "current_agent_index": 0,
            "agent_outputs": {}
        }

        try:
            # Execute the graph
            final_state = self.graph.invoke(initial_state)

            # Extract conversation history
            conversation_history = final_state["conversation_history"]

            print(f"\n{'='*80}")
            print("✅ SYMPHONY ANALYSIS COMPLETE")
            print(f"{'='*80}\n")

            return conversation_history

        except Exception as e:
            print(f"\n❌ ERROR: Symphony analysis failed: {e}")
            raise

    def run_analysis_streaming(self, initial_task: str):
        """
        Run the analysis with streaming output using latest LangGraph patterns.
        Implements 2025 best practices with proper state streaming.

        Args:
            initial_task: The user's strategic plan or business idea

        Yields:
            Tuple of (agent_name, chunk) for each piece of output
        """
        if not initial_task or not initial_task.strip():
            raise ValueError("Initial task cannot be empty.")

        print(f"\n{'='*80}")
        print("🎼 SYMPHONY ANALYSIS STARTING (STREAMING)")
        print(f"{'='*80}\n")

        # Yield the initial plan
        initial_entry = f"## 📋 Initial Plan for Analysis:\n\n> {initial_task}\n\n---\n"
        yield ("System", initial_entry)

        conversation_history = [initial_entry]

        try:
            # Execute agents sequentially with streaming
            # Following Master.md orchestration sequence: Shrek -> Sonic -> Hulk -> Trevor
            for i, agent in enumerate(self.agents, 1):
                print(f"\nINFO: [{i}/{len(self.agents)}] Streaming agent: {agent.name}")

                full_response = ""
                for chunk in agent.execute_streaming(
                    task=initial_task,
                    conversation_history=conversation_history
                ):
                    yield (agent.name, chunk)
                    full_response += chunk

                # Add separator between agents for better readability
                full_response += "\n\n---\n\n"
                conversation_history.append(full_response)

            print(f"\n{'='*80}")
            print("✅ SYMPHONY ANALYSIS COMPLETE")
            print(f"{'='*80}\n")

        except Exception as e:
            error_msg = str(e)
            print(f"\n❌ ERROR: Symphony analysis failed: {error_msg}")

            # Provide helpful error messages
            if "API key" in error_msg or "authentication" in error_msg.lower():
                yield ("Error", f"❌ **API Key Error**: Please check your OPENAI_API_KEY in .env file")
            elif "rate limit" in error_msg.lower():
                yield ("Error", f"❌ **Rate Limit**: OpenAI API rate limit exceeded. Please wait and try again.")
            else:
                yield ("Error", f"❌ **Error**: {error_msg}")

    def get_knowledge_base(self) -> KnowledgeBase:
        """
        Get the knowledge base instance.
        Useful for document management in the UI.

        Returns:
            The KnowledgeBase instance
        """
        return self.knowledge_base

    def add_documents_to_kb(self, file_paths: List[str]) -> int:
        """
        Add documents to the knowledge base.

        Args:
            file_paths: List of file paths to add

        Returns:
            Number of chunks added
        """
        return self.knowledge_base.add_documents(file_paths)

    def clear_knowledge_base(self):
        """Clear all documents from the knowledge base."""
        self.knowledge_base.clear()
