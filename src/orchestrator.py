"""
Orchestrator Module with LangGraph StateGraph and Agent-Evaluator Feedback Loops
Manages multi-agent workflow with quality validation and retry logic
"""

import yaml
from typing import List, Dict, TypedDict, Annotated, Literal
from operator import add

from langgraph.graph import StateGraph, END

from src.agents import SymphonyAgent
from src.knowledge_base import KnowledgeBase, ChromaKnowledgeBase
from src.llm_client import LLMClient
from src.output_parser import AgentOutputParser


# Define the state structure for the agent graph
class AgentState(TypedDict):
    """State that gets passed between agents in the graph."""
    task: str  # The original user task/plan
    conversation_history: Annotated[List[str], add]  # Accumulates all agent outputs
    current_agent: str  # Currently active agent
    agent_outputs: Dict[str, str]  # Maps agent name to their output
    agent_attempts: Dict[str, int]  # Track retry attempts per agent
    quality_scores: Dict[str, Dict]  # Quality scores from evaluator
    rag_context: Dict[str, List[str]]  # Track RAG context per agent
    max_retries: int  # Maximum retry attempts per agent


class Orchestrator:
    """
    Manages the sequence of agent interactions using LangGraph with quality control.
    Implements agent-evaluator feedback loops for quality assurance.
    """

    def __init__(self, agent_config_path: str, use_mock_kb: bool = False):
        """
        Initialize the orchestrator with agents and knowledge base.

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
        self.agents = []
        self.agent_by_name = {}
        
        for cfg in agent_configs:
            agent = SymphonyAgent(
                agent_name=cfg['name'],
                persona_prompt=cfg['prompt'],
                llm_client=self.llm_client,
                retriever=self.knowledge_base
            )
            self.agents.append(agent)
            self.agent_by_name[cfg['name']] = agent

        # Separate evaluator from main agents
        self.evaluator = self.agent_by_name.get("Evaluator")
        self.main_agents = [a for a in self.agents if a.name != "Evaluator"]
        
        self.agent_names = [agent.name for agent in self.agents]
        print(f"INFO: Initialized {len(self.agents)} agents: {', '.join(self.agent_names)}")
        print(f"INFO: Main agents: {[a.name for a in self.main_agents]}")

        # Build the LangGraph workflow with feedback loops
        self._build_graph()

    def _build_graph(self):
        """
        Build the LangGraph StateGraph with agent-evaluator feedback loops.
        Each agent is evaluated before proceeding to the next.
        """
        workflow = StateGraph(AgentState)

        # Add nodes for each main agent and their evaluation
        for agent in self.main_agents:
            # Agent execution node
            workflow.add_node(agent.name, self._create_agent_node(agent))
            # Evaluation node for this agent
            workflow.add_node(f"{agent.name}_eval", self._create_evaluator_node(agent.name))

        # Add final evaluator node for overall assessment
        if self.evaluator:
            workflow.add_node("final_evaluation", self._create_final_evaluator_node())

        # Set entry point to first agent
        workflow.set_entry_point(self.main_agents[0].name)

        # Create the feedback loop flow for each agent
        for i, agent in enumerate(self.main_agents):
            # Agent → Evaluator
            workflow.add_edge(agent.name, f"{agent.name}_eval")
            
            # Evaluator → Decision (retry or next agent)
            workflow.add_conditional_edges(
                f"{agent.name}_eval",
                self._should_retry_agent(agent.name),
                {
                    "retry": agent.name,  # Loop back to agent
                    "next": self.main_agents[i + 1].name if i + 1 < len(self.main_agents) else "final_evaluation",  # Next agent or final eval
                }
            )

        # Final evaluation → END
        if self.evaluator:
            workflow.add_edge("final_evaluation", END)
        else:
            # If no evaluator, last agent goes to END
            workflow.add_edge(self.main_agents[-1].name, END)

        # Compile the graph
        self.graph = workflow.compile()
        print("INFO: LangGraph workflow with feedback loops compiled successfully")

    def _create_agent_node(self, agent: SymphonyAgent):
        """
        Create a node function for an agent with RAG tracking.

        Args:
            agent: The SymphonyAgent instance

        Returns:
            A function that executes the agent and updates state
        """
        def agent_node(state: AgentState) -> AgentState:
            """Execute the agent and update the state."""
            attempt = state.get("agent_attempts", {}).get(agent.name, 0) + 1
            
            print(f"\n{'='*80}")
            print(f"🎯 {agent.name} - Attempt {attempt}/{state.get('max_retries', 3)}")
            print(f"{'='*80}")

            # Get RAG context (we'll track this separately)
            query = f"{agent.name} analysis: {state['task']}"
            rag_docs = self.knowledge_base.retrieve(query=query, top_k=5)
            
            print(f"📚 RAG Context Retrieved: {len(rag_docs)} documents")
            for i, doc in enumerate(rag_docs[:2], 1):  # Show first 2
                preview = doc[:100] + "..." if len(doc) > 100 else doc
                print(f"   {i}. {preview}")

            # Execute the agent
            response = agent.execute(
                task=state["task"],
                conversation_history=state["conversation_history"]
            )

            # Update state
            updated_rag = state.get("rag_context", {})
            updated_rag[agent.name] = rag_docs
            
            updated_attempts = state.get("agent_attempts", {})
            updated_attempts[agent.name] = attempt

            return {
                "task": state["task"],
                "conversation_history": [response],
                "current_agent": agent.name,
                "agent_outputs": {**state.get("agent_outputs", {}), agent.name: response},
                "agent_attempts": updated_attempts,
                "rag_context": updated_rag,
                "max_retries": state.get("max_retries", 3)
            }

        return agent_node

    def _create_evaluator_node(self, agent_name: str):
        """
        Create an evaluator node that assesses an agent's output.

        Args:
            agent_name: Name of the agent to evaluate

        Returns:
            A function that evaluates the agent's output
        """
        def evaluator_node(state: AgentState) -> AgentState:
            """Evaluate the agent's output quality."""
            if not self.evaluator:
                # No evaluator, skip validation
                return state

            agent_output = state.get("agent_outputs", {}).get(agent_name, "")
            
            print(f"\n🔍 Evaluator assessing {agent_name}'s output...")

            # Quick validation using output parser
            validation = AgentOutputParser.validate_output_format(agent_output, agent_name)
            
            # Calculate quality score (0-10)
            quality_score = 0.0
            if validation["has_header"]:
                quality_score += 3.0
            if validation["follows_format"]:
                quality_score += 4.0
            if validation["has_evidence"]:
                quality_score += 3.0
            
            print(f"   ✓ Header: {validation['has_header']}")
            print(f"   ✓ Format: {validation['follows_format']}")
            print(f"   ✓ Evidence: {validation['has_evidence']}")
            print(f"   → Quality Score: {quality_score}/10")

            # Store quality score
            updated_scores = state.get("quality_scores", {})
            updated_scores[agent_name] = {
                "overall": quality_score,
                "validation": validation
            }

            return {
                **state,
                "quality_scores": updated_scores
            }

        return evaluator_node

    def _should_retry_agent(self, agent_name: str):
        """
        Create a decision function for whether to retry an agent.

        Args:
            agent_name: Name of the agent

        Returns:
            A function that decides retry or proceed
        """
        def decide(state: AgentState) -> Literal["retry", "next"]:
            """Decide whether to retry the agent or proceed."""
            quality_scores = state.get("quality_scores", {})
            agent_score = quality_scores.get(agent_name, {})
            score = agent_score.get("overall", 0)
            
            attempts = state.get("agent_attempts", {}).get(agent_name, 0)
            max_retries = state.get("max_retries", 3)

            # Retry if score < 7 and under max attempts
            if score < 7.0 and attempts < max_retries:
                print(f"   ❌ Quality insufficient ({score}/10) - Retrying {agent_name}...")
                return "retry"
            elif score < 7.0:
                print(f"   ⚠️  Quality insufficient ({score}/10) but max retries reached - Proceeding...")
                return "next"
            else:
                print(f"   ✅ Quality acceptable ({score}/10) - Proceeding to next agent")
                return "next"

        return decide

    def _create_final_evaluator_node(self):
        """Create the final evaluator node for overall assessment."""
        def final_evaluator_node(state: AgentState) -> AgentState:
            """Run final evaluation on all agents."""
            if not self.evaluator:
                return state

            print(f"\n{'='*80}")
            print("🏆 FINAL QUALITY ASSESSMENT")
            print(f"{'='*80}\n")

            # Execute evaluator with full conversation history
            response = self.evaluator.execute(
                task=state["task"],
                conversation_history=state["conversation_history"]
            )

            # Parse evaluator output for detailed scores
            eval_data = AgentOutputParser.parse_evaluator_output(response)
            
            return {
                **state,
                "conversation_history": [response],
                "agent_outputs": {**state.get("agent_outputs", {}), "Evaluator": response},
                "quality_scores": {**state.get("quality_scores", {}), **eval_data.get("scores", {})}
            }

        return final_evaluator_node

    def run_analysis(self, initial_task: str, max_retries: int = 3) -> List[str]:
        """
        Run the full multi-agent analysis with quality control.

        Args:
            initial_task: The user's strategic plan or business idea
            max_retries: Maximum retry attempts per agent

        Returns:
            List of all agent outputs (conversation history)
        """
        if not initial_task or not initial_task.strip():
            raise ValueError("Initial task cannot be empty.")

        print(f"\n{'='*80}")
        print("🎼 SYMPHONY AGENTIC ANALYSIS STARTING")
        print(f"{'='*80}\n")

        # Initialize state
        initial_state: AgentState = {
            "task": initial_task,
            "conversation_history": [f"## 📋 Initial Plan for Analysis:\n\n> {initial_task}\n\n---\n"],
            "current_agent": "",
            "agent_outputs": {},
            "agent_attempts": {},
            "quality_scores": {},
            "rag_context": {},
            "max_retries": max_retries
        }

        try:
            # Execute the graph
            final_state = self.graph.invoke(initial_state)

            # Extract conversation history
            conversation_history = final_state["conversation_history"]

            print(f"\n{'='*80}")
            print("✅ SYMPHONY AGENTIC ANALYSIS COMPLETE")
            print(f"{'='*80}\n")

            # Print summary
            print("📊 EXECUTION SUMMARY:")
            for agent_name in [a.name for a in self.main_agents]:
                attempts = final_state.get("agent_attempts", {}).get(agent_name, 0)
                score = final_state.get("quality_scores", {}).get(agent_name, {}).get("overall", 0)
                print(f"   {agent_name}: {attempts} attempt(s), final score: {score}/10")

            return conversation_history

        except Exception as e:
            print(f"\n❌ ERROR: Symphony analysis failed: {e}")
            raise

    def run_analysis_streaming(self, initial_task: str, max_retries: int = 2):
        """
        Run the analysis with streaming output and quality control.

        Args:
            initial_task: The user's strategic plan or business idea
            max_retries: Maximum retry attempts per agent

        Yields:
            Tuple of (agent_name, chunk) for each piece of output
        """
        if not initial_task or not initial_task.strip():
            raise ValueError("Initial task cannot be empty.")

        print(f"\n{'='*80}")
        print("🎼 SYMPHONY AGENTIC ANALYSIS (STREAMING)")
        print(f"{'='*80}\n")

        # Yield the initial plan
        initial_entry = f"## 📋 Initial Plan for Analysis:\n\n> {initial_task}\n\n---\n"
        yield ("System", initial_entry)

        conversation_history = [initial_entry]
        agent_attempts = {}
        quality_scores = {}

        try:
            # Execute agents with feedback loops
            for agent in self.main_agents:
                attempts = 0
                quality_acceptable = False

                while attempts < max_retries and not quality_acceptable:
                    attempts += 1
                    
                    # Header
                    header = f"\n{'='*60}\n🎯 {agent.name} (Attempt {attempts}/{max_retries})\n{'='*60}\n"
                    yield (agent.name, header)

                    # Show RAG context with preview
                    query = f"{agent.name} analysis: {initial_task}"
                    rag_docs = self.knowledge_base.retrieve(query=query, top_k=5)
                    
                    rag_info = f"\n📚 **Retrieved {len(rag_docs)} knowledge chunks:**\n\n"
                    yield (agent.name, rag_info)
                    
                    # Show snippet of each chunk
                    for i, doc in enumerate(rag_docs[:3], 1):  # Show top 3
                        # Extract meaningful snippet (first 200 chars)
                        snippet = doc.split('\n')[0] if '\n' in doc else doc
                        snippet = snippet[:200] + "..." if len(snippet) > 200 else snippet
                        chunk_preview = f"   {i}. {snippet}\n\n"
                        yield (agent.name, chunk_preview)

                    # Stream agent output
                    full_response = ""
                    for chunk in agent.execute_streaming(
                        task=initial_task,
                        conversation_history=conversation_history
                    ):
                        yield (agent.name, chunk)
                        full_response += chunk

                    # Evaluate
                    validation = AgentOutputParser.validate_output_format(full_response, agent.name)
                    quality_score = 0.0
                    if validation["has_header"]:
                        quality_score += 3.0
                    if validation["follows_format"]:
                        quality_score += 4.0
                    if validation["has_evidence"]:
                        quality_score += 3.0

                    quality_scores[agent.name] = quality_score

                    # Show evaluation
                    eval_msg = f"\n\n🔍 **Evaluation**: Quality Score {quality_score}/10"
                    if quality_score >= 7.0:
                        eval_msg += " ✅ (Acceptable)\n\n---\n\n"
                        quality_acceptable = True
                    elif attempts < max_retries:
                        eval_msg += f" ❌ (Retrying...)\n\n---\n\n"
                    else:
                        eval_msg += f" ⚠️  (Max retries reached, proceeding)\n\n---\n\n"
                        quality_acceptable = True

                    yield (agent.name, eval_msg)

                    if quality_acceptable:
                        conversation_history.append(full_response + "\n\n---\n\n")

            # Final evaluation
            if self.evaluator:
                yield ("Evaluator", f"\n{'='*60}\n🏆 FINAL QUALITY ASSESSMENT\n{'='*60}\n\n")
                
                full_eval = ""
                for chunk in self.evaluator.execute_streaming(
                    task=initial_task,
                    conversation_history=conversation_history
                ):
                    yield ("Evaluator", chunk)
                    full_eval += chunk

                conversation_history.append(full_eval)

            print(f"\n{'='*80}")
            print("✅ SYMPHONY ANALYSIS COMPLETE")
            print(f"{'='*80}\n")

        except Exception as e:
            error_msg = str(e)
            print(f"\n❌ ERROR: {error_msg}")
            yield ("Error", f"❌ **Error**: {error_msg}")

    def get_knowledge_base(self) -> KnowledgeBase:
        """Get the knowledge base instance."""
        return self.knowledge_base

    def add_documents_to_kb(self, file_paths: List[str]) -> int:
        """Add documents to the knowledge base."""
        return self.knowledge_base.add_documents(file_paths)

    def clear_knowledge_base(self):
        """Clear all documents from the knowledge base."""
        self.knowledge_base.clear()

    def parse_conversation_history(self, conversation_history: List[str]) -> Dict:
        """Parse all agent outputs to extract structured data."""
        return AgentOutputParser.parse_all_outputs(conversation_history)
    
    def get_agent_count(self) -> int:
        """Get the total number of agents in the orchestrator."""
        return len(self.agents)
