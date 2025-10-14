# src/agents.py
"""
Symphony Agent System - LangGraph + LangChain Implementation
Agents use proper LangChain tool calling with LangGraph state management
"""

from typing import List, Dict, Any, Optional, Annotated, TypedDict
from operator import add

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from src.llm_client import LLMClient
from src.knowledge_base import KnowledgeBase
from src.output_parser import AgentOutputParser
import re
import json


# Define the agent state for LangGraph
class AgentReasoningState(TypedDict):
    """State for agent reasoning loop using LangGraph"""
    messages: Annotated[List[Any], add]  # Message history
    task: str  # The task to analyze
    conversation_history: List[str]  # Context from other agents
    agent_name: str  # Current agent name
    iteration: int  # Current iteration
    final_answer: Optional[str]  # Final structured answer


class SymphonyAgent:
    """
    A TRUE agentic Symphony analyst using LangChain tools + LangGraph orchestration.
    Uses proper tool calling, reasoning, and state management.
    """

    def __init__(
        self,
        agent_name: str,
        persona_prompt: str,
        llm_client: LLMClient,
        retriever: KnowledgeBase,
        max_iterations: int = 5
    ):
        """
        Initialize the LangChain-powered Symphony Agent.

        Args:
            agent_name: Name of the agent (e.g., "Shrek", "Sonic")
            persona_prompt: The agent's role and instructions
            llm_client: LLM client instance
            retriever: Knowledge base for RAG
            max_iterations: Maximum reasoning iterations
        """
        self.name = agent_name
        self.persona_prompt = persona_prompt
        self.llm_client = llm_client
        self.retriever = retriever
        self.max_iterations = max_iterations
        
        # Create tools for this agent
        self.tools = self._create_tools()
        
        # Create the agent with tool calling
        self.agent = self._create_agent()
        
        # Build the reasoning graph
        self.graph = self._build_reasoning_graph()

    def _create_tools(self):
        """Create LangChain tools for the agent"""
        
        # Capture self in closures
        retriever = self.retriever
        agent_name = self.name
        
        @tool
        def search_knowledge(query: str) -> str:
            """Search the knowledge base for relevant company documents and evidence.
            
            Args:
                query: The search query to find relevant information
                
            Returns:
                Relevant documents and evidence from the knowledge base
            """
            try:
                docs = retriever.retrieve(query, top_k=5)
                if not docs or (isinstance(docs, list) and len(docs) == 0):
                    return "No documents found. Use general expertise."
                
                if isinstance(docs, list) and len(docs) > 0 and "No documents" in str(docs[0]):
                    return "No documents found. Use general expertise."
                
                result = "📚 KNOWLEDGE FINDINGS:\n"
                for i, doc in enumerate(docs[:3], 1):
                    doc_preview = str(doc)[:250]
                    result += f"{i}. {doc_preview}...\n"
                return result
            except Exception as e:
                return f"KB unavailable. Using general expertise."

        @tool
        def analyze_patterns(text: str) -> str:
            """Analyze text to identify risk patterns and concerns.
            
            Args:
                text: The text to analyze for patterns
                
            Returns:
                Identified risk patterns
            """
            patterns = []
            if re.search(r'cost|expensive|budget', text, re.I):
                patterns.append("💰 Financial risk")
            if re.search(r'delay|slow|time', text, re.I):
                patterns.append("⏱️ Timeline risk")
            if re.search(r'complex|difficult', text, re.I):
                patterns.append("🔧 Complexity risk")
            if re.search(r'market|customer|user', text, re.I):
                patterns.append("👥 Market risk")
            
            return "🔍 PATTERNS: " + ", ".join(patterns) if patterns else "No critical patterns detected."

        @tool
        def evaluate_severity(description: str) -> str:
            """Evaluate the severity level of a risk or threat.
            
            Args:
                description: Description of the risk or threat
                
            Returns:
                Severity assessment
            """
            desc_lower = description.lower()
            
            if any(kw in desc_lower for kw in ['catastrophic', 'fatal', 'critical']):
                return "🎯 SEVERITY: Critical (Immediate action required)"
            elif any(kw in desc_lower for kw in ['major', 'significant', 'serious']):
                return "🎯 SEVERITY: High (Address urgently)"
            elif any(kw in desc_lower for kw in ['moderate', 'notable']):
                return "🎯 SEVERITY: Medium (Monitor closely)"
            return "🎯 SEVERITY: Low (Track)"

        @tool
        def synthesize_analysis(findings: str) -> str:
            """Synthesize findings into a structured analysis format.
            
            Args:
                findings: The raw findings to synthesize
                
            Returns:
                Guidance on structuring the analysis
            """
            return f"""
🎯 Synthesis Guide for {agent_name}:
1. Start with your role header (###)
2. Present key findings with evidence
3. Identify 2-3 specific threats with severity
4. Provide actionable recommendations
5. Support claims with data/patterns

Structure your response following your persona format exactly.
"""

        return [search_knowledge, analyze_patterns, evaluate_severity, synthesize_analysis]

    def _create_agent(self):
        """Create a LangChain agent with tool calling"""
        
        llm = self.llm_client.get_llm()
        
        # Create a prompt that emphasizes the persona and structured output
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""{self.persona_prompt}

CRITICAL INSTRUCTIONS:
1. Use your tools (search_knowledge, analyze_patterns, evaluate_severity) to gather evidence
2. After gathering evidence, use synthesize_analysis to get your output format guide
3. Provide your FINAL ANSWER in the EXACT format specified in your persona

Your output MUST include:
- Your role header (e.g., "### 🌟 SHREK'S OPPORTUNITY MAP")
- All required sections with emojis
- Evidence citations with document names
- Specific action items

Do NOT provide generic responses. Follow your persona format EXACTLY."""),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # Create the tool-calling agent
        agent = create_tool_calling_agent(llm, self.tools, prompt)
        
        return agent

    def _build_reasoning_graph(self):
        """Build LangGraph workflow for agent reasoning"""
        
        # Create tool node
        tool_node = ToolNode(self.tools)
        
        # Define the graph
        workflow = StateGraph(AgentReasoningState)
        
        # Add nodes
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", tool_node)
        
        # Set entry point
        workflow.add_edge(START, "agent")
        
        # Agent decides: use tool or finish
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "tools",
                "end": END
            }
        )
        
        # After tools, go back to agent
        workflow.add_edge("tools", "agent")
        
        return workflow.compile()

    def _agent_node(self, state: AgentReasoningState) -> AgentReasoningState:
        """Node that runs the agent"""
        
        # Build input for agent
        messages = state.get("messages", [])
        task = state.get("task", "")
        history = state.get("conversation_history", [])
        iteration = state.get("iteration", 0)
        
        # Create context
        history_str = "\n\n".join(history[-2:]) if history else "Start of conversation."
        
        input_text = f"""
TASK TO ANALYZE:
{task}

CONVERSATION CONTEXT (from other agents):
{history_str}

ITERATION: {iteration + 1}/{self.max_iterations}

Analyze this thoroughly. Use your tools to gather evidence, then provide your complete structured analysis following your persona format EXACTLY.
"""
        
        # If no messages yet, start fresh
        if not messages:
            messages = [HumanMessage(content=input_text)]
        
        # Run agent with executor
        agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            max_iterations=self.max_iterations,
            return_intermediate_steps=True,
            handle_parsing_errors=True,
            early_stopping_method="generate"  # Ensure we get a final answer
        )
        
        try:
            result = agent_executor.invoke({
                "input": input_text,
                "chat_history": messages[:-1] if len(messages) > 1 else []
            })
            
            output = result.get("output", "")
            
            # Add to messages
            new_messages = [AIMessage(content=output)]
            
            # Check if this is a final answer
            is_final = self._is_final_answer(output, iteration)
            
            return {
                "messages": new_messages,
                "task": task,
                "conversation_history": history,
                "agent_name": self.name,
                "iteration": iteration + 1,
                "final_answer": output if is_final else state.get("final_answer")
            }
            
        except Exception as e:
            print(f"ERROR in agent node: {e}")
            error_message = f"### 🎯 {self.name}\n\n❌ **Error during analysis**: {str(e)}"
            return {
                "messages": [AIMessage(content=error_message)],
                "task": task,
                "conversation_history": history,
                "agent_name": self.name,
                "iteration": iteration + 1,
                "final_answer": error_message
            }

    def _should_continue(self, state: AgentReasoningState) -> str:
        """Decide whether to continue or end"""
        
        messages = state.get("messages", [])
        iteration = state.get("iteration", 0)
        
        if not messages:
            return "continue"
        
        last_message = messages[-1]
        
        # Check if we have a final answer
        if state.get("final_answer"):
            return "end"
        
        # Check max iterations
        if iteration >= self.max_iterations:
            return "end"
        
        # Check if agent wants to use tools (has tool calls)
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "continue"
        
        # Check if response looks complete (has proper formatting)
        if isinstance(last_message.content, str):
            content = last_message.content
            
            # Check for agent-specific markers
            format_markers = {
                "Shrek": "SHREK'S OPPORTUNITY MAP",
                "Sonic": "SONIC'S LEAN EXECUTION AUDIT",
                "Hulk": "HULK SMASH ASSUMPTIONS",
                "Trevor": "TREVOR'S STRATEGIC SYNTHESIS",
                "Evaluator": "EVALUATOR'S QUALITY ASSESSMENT"
            }
            
            agent_marker = format_markers.get(self.name)
            if agent_marker and agent_marker in content:
                # Has proper header - check for content
                has_action_items = "ACTION ITEMS" in content
                has_evidence = "📄" in content or "Evidence" in content
                
                if has_action_items and has_evidence:
                    return "end"
        
        # Default: continue if under max iterations
        return "continue" if iteration < self.max_iterations else "end"

    def _is_final_answer(self, output: str, iteration: int) -> bool:
        """Check if output is a final answer"""
        
        # Check for agent-specific formatting
        format_markers = {
            "Shrek": "SHREK'S OPPORTUNITY MAP",
            "Sonic": "SONIC'S LEAN EXECUTION AUDIT",
            "Hulk": "HULK SMASH ASSUMPTIONS",
            "Trevor": "TREVOR'S STRATEGIC SYNTHESIS",
            "Evaluator": "EVALUATOR'S QUALITY ASSESSMENT"
        }
        
        agent_marker = format_markers.get(self.name)
        has_proper_header = agent_marker and agent_marker in output
        has_content = len(output) > 300
        
        # Check for required sections
        has_action_items = "ACTION ITEMS" in output or "CONSOLIDATED ACTION PLAN" in output
        has_evidence = "📄" in output or "Evidence from" in output
        
        # Consider it final if it has proper structure OR we're at max iterations
        is_complete = has_proper_header and has_content and has_action_items
        
        return is_complete or iteration >= self.max_iterations - 1

    def execute(self, task: str, conversation_history: List[str]) -> str:
        """
        Execute agent analysis using LangChain + LangGraph.
        
        Args:
            task: The user's strategic plan/task
            conversation_history: List of previous agent outputs

        Returns:
            The agent's structured analysis
        """
        try:
            print(f"\n{'='*70}")
            print(f"🤖 [{self.name}] Starting LangChain Agent with Tools")
            print(f"{'='*70}\n")
            
            # Initialize state
            initial_state: AgentReasoningState = {
                "messages": [],
                "task": task,
                "conversation_history": conversation_history,
                "agent_name": self.name,
                "iteration": 0,
                "final_answer": None
            }
            
            # Run the graph
            final_state = self.graph.invoke(initial_state)
            
            # Extract final answer
            response = final_state.get("final_answer")
            
            if not response:
                # Get from last message
                messages = final_state.get("messages", [])
                if messages:
                    last_msg = messages[-1]
                    response = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
                else:
                    response = f"### 🎯 {self.name}\n\n⚠️ No analysis generated"
            
            # Ensure proper formatting
            if not response.startswith("###"):
                response = f"### 🎯 {self.name}\n\n{response}"
            
            # Validate
            validation = AgentOutputParser.validate_output_format(response, self.name)
            print(f"\n✅ [{self.name}] Analysis complete (validation: {validation['is_valid']})")
            
            return response

        except Exception as e:
            error_msg = f"ERROR: [{self.name}] Failed: {e}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return f"### 🎯 {self.name}\n\n❌ **Error**: {e}"

    def execute_streaming(self, task: str, conversation_history: List[str]):
        """
        Execute agent analysis with streaming.
        
        Note: Full streaming of LangGraph is complex, so we run the graph
        and stream the final output.
        """
        print(f"INFO: [{self.name}] Starting analysis with streaming...")
        
        try:
            # Yield header
            yield f"### 🎯 {self.name}\n\n"
            yield f"*[Using LangChain tools for evidence gathering...]*\n\n"
            
            # Run the full graph
            initial_state: AgentReasoningState = {
                "messages": [],
                "task": task,
                "conversation_history": conversation_history,
                "agent_name": self.name,
                "iteration": 0,
                "final_answer": None
            }
            
            final_state = self.graph.invoke(initial_state)
            
            # Extract response
            response = final_state.get("final_answer")
            if not response:
                messages = final_state.get("messages", [])
                if messages:
                    last_msg = messages[-1]
                    response = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
                else:
                    response = "⚠️ No analysis generated"
            
            # Stream the response in chunks
            chunk_size = 50
            for i in range(0, len(response), chunk_size):
                chunk = response[i:i + chunk_size]
                yield chunk
            
            # Validate
            validation = AgentOutputParser.validate_output_format(response, self.name)
            print(f"INFO: [{self.name}] Streaming complete (validation: {validation['is_valid']})")

        except Exception as e:
            print(f"ERROR: [{self.name}] Streaming failed: {e}")
            import traceback
            traceback.print_exc()
            yield f"\n\n❌ **Error**: {e}"
    
    def validate_output(self, output: str) -> Dict[str, bool]:
        """Validate agent output format."""
        return AgentOutputParser.validate_output_format(output, self.name)
