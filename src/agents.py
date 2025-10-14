# src/agents.py
"""
Symphony Agent System - TRUE AGENTIC Implementation
Agents REASON iteratively, DECIDE what tools to use, OBSERVE results, and THINK again
"""

from typing import List, Dict, Any, Optional
from src.llm_client import LLMClient
from src.knowledge_base import KnowledgeBase
from src.output_parser import AgentOutputParser
import re
import json

class SymphonyAgent:
    """
    A TRUE agentic Symphony analyst that iteratively THINKS, DECIDES, ACTS, and OBSERVES.
    Uses iterative reasoning loop - not just a single LLM call.
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
        Initialize the TRUE agentic Symphony Agent.

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
        
        # Tool registry - these are the actions the agent can take
        self.tools = {
            "search_knowledge": self._search_kb_tool,
            "analyze_patterns": self._analyze_patterns_tool,
            "evaluate_severity": self._evaluate_risk_tool,
            "think_deeply": self._think_tool
        }
        
        # Track reasoning process
        self.reasoning_log = []

    def _search_kb_tool(self, query: str) -> str:
        """Tool: Search knowledge base for evidence"""
        try:
            docs = self.retriever.retrieve(query, top_k=5)
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

    def _analyze_patterns_tool(self, text: str) -> str:
        """Tool: Analyze patterns"""
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

    def _evaluate_risk_tool(self, description: str) -> str:
        """Tool: Evaluate severity"""
        desc_lower = description.lower()
        
        if any(kw in desc_lower for kw in ['catastrophic', 'fatal', 'critical']):
            return "🎯 SEVERITY: Critical (Immediate action required)"
        elif any(kw in desc_lower for kw in ['major', 'significant', 'serious']):
            return "🎯 SEVERITY: High (Address urgently)"
        elif any(kw in desc_lower for kw in ['moderate', 'notable']):
            return "🎯 SEVERITY: Medium (Monitor closely)"
        return "🎯 SEVERITY: Low (Track)"

    def _think_tool(self, thought: str) -> str:
        """Tool: Deep thinking without external tools"""
        return f"💭 REASONING: {thought}"

    def _run_agent_loop(self, task: str, conversation_history: List[str]) -> str:
        """
        Run the TRUE agentic reasoning loop.
        The agent iteratively thinks, decides actions, observes results.
        """
        print(f"\n{'='*70}")
        print(f"🤖 [{self.name}] TRUE AGENTIC LOOP STARTING")
        print(f"{'='*70}\n")
        
        # Initialize agent state
        history_str = "\n\n".join(conversation_history[-2:]) if conversation_history else "Start of conversation."
        observations = []
        iteration = 0
        
        # Agentic loop: Think → Decide → Act → Observe → Repeat
        while iteration < self.max_iterations:
            iteration += 1
            print(f"[{self.name}] 🔄 Iteration {iteration}/{self.max_iterations}")
            
            # Build the reasoning prompt for this iteration
            thinking_prompt = self._build_thinking_prompt(
                task=task,
                history=history_str,
                observations=observations,
                iteration=iteration
            )
            
            # Agent THINKS and DECIDES what to do next
            llm = self.llm_client.get_llm()
            decision = llm.invoke(thinking_prompt)
            decision_text = decision.content if hasattr(decision, 'content') else str(decision)
            
            print(f"[{self.name}] 💭 Decision: {decision_text[:150]}...")
            
            # Parse the decision
            action = self._parse_action(decision_text)
            
            if action["type"] == "FINAL_ANSWER":
                # Agent has decided it has enough information
                print(f"[{self.name}] ✅ Agent reached conclusion after {iteration} iterations")
                return action["content"]
            
            elif action["type"] == "USE_TOOL":
                # Agent wants to use a tool
                tool_name = action.get("tool")
                tool_input = action.get("input", "")
                
                if tool_name in self.tools:
                    print(f"[{self.name}] 🔧 Using tool: {tool_name}")
                    observation = self.tools[tool_name](tool_input)
                    observations.append({
                        "iteration": iteration,
                        "action": tool_name,
                        "input": tool_input[:100],
                        "result": observation[:200]
                    })
                    print(f"[{self.name}] 👁️ Observation: {observation[:100]}...")
                else:
                    observations.append({
                        "iteration": iteration,
                        "action": "ERROR",
                        "result": f"Unknown tool: {tool_name}"
                    })
            
            elif action["type"] == "THINK":
                # Agent is thinking without tools
                thought = action.get("content", "")
                observations.append({
                    "iteration": iteration,
                    "action": "THINK",
                    "result": thought[:200]
                })
                print(f"[{self.name}] 💭 Thinking: {thought[:100]}...")
        
        # Max iterations reached - generate final answer
        print(f"[{self.name}] ⚠️ Max iterations reached, generating final answer...")
        final_prompt = self._build_final_prompt(task, history_str, observations)
        final_response = llm.invoke(final_prompt)
        return final_response.content if hasattr(final_response, 'content') else str(final_response)

    def _build_thinking_prompt(self, task: str, history: str, observations: List[Dict], iteration: int) -> str:
        """Build the prompt for agent's thinking at each iteration"""
        
        obs_summary = "\n".join([
            f"Iteration {obs['iteration']}: {obs['action']} → {obs['result'][:150]}"
            for obs in observations[-3:]  # Last 3 observations
        ]) if observations else "No observations yet."
        
        tools_desc = """
Available Tools:
- search_knowledge: Search company documents for evidence
- analyze_patterns: Identify patterns and risks in text
- evaluate_severity: Assess risk severity level
- think_deeply: Reason through complex issues
"""
        
        return f"""{self.persona_prompt}

=== AGENTIC REASONING LOOP (Iteration {iteration}/{self.max_iterations}) ===

TASK TO ANALYZE:
{task[:500]}

CONVERSATION CONTEXT:
{history[:1000]}

OBSERVATIONS SO FAR:
{obs_summary}

{tools_desc}

=== YOUR DECISION ===

You are in iteration {iteration} of {self.max_iterations}. Analyze what you know and decide your next action.

You can respond in ONE of these formats:

1. If you need more information, use a tool:
ACTION: USE_TOOL
TOOL: [tool_name]
INPUT: [what to analyze/search]
REASONING: [why you need this]

2. If you need to think through something:
ACTION: THINK
CONTENT: [your reasoning]

3. If you have enough information to provide your final analysis:
ACTION: FINAL_ANSWER
CONTENT: [Your complete structured analysis following your persona format]

What is your decision?
"""

    def _build_final_prompt(self, task: str, history: str, observations: List[Dict]) -> str:
        """Build prompt for final answer generation"""
        
        obs_summary = "\n".join([
            f"{obs['action']}: {obs['result'][:200]}"
            for obs in observations
        ])
        
        return f"""{self.persona_prompt}

=== FINAL ANALYSIS GENERATION ===

TASK:
{task}

CONTEXT:
{history[:1000]}

YOUR REASONING PROCESS:
{obs_summary}

Now provide your COMPLETE structured analysis following the exact format specified in your persona.
Include all required sections with proper emoji markers.
"""

    def _parse_action(self, decision_text: str) -> Dict[str, Any]:
        """Parse the agent's decision into structured action"""
        
        # Check for FINAL_ANSWER
        if re.search(r'ACTION:\s*FINAL_ANSWER', decision_text, re.IGNORECASE):
            # Extract content after FINAL_ANSWER
            content_match = re.search(r'CONTENT:(.*)', decision_text, re.DOTALL | re.IGNORECASE)
            if content_match:
                return {
                    "type": "FINAL_ANSWER",
                    "content": content_match.group(1).strip()
                }
            return {"type": "FINAL_ANSWER", "content": decision_text}
        
        # Check for USE_TOOL
        if re.search(r'ACTION:\s*USE_TOOL', decision_text, re.IGNORECASE):
            tool_match = re.search(r'TOOL:\s*(\w+)', decision_text, re.IGNORECASE)
            input_match = re.search(r'INPUT:\s*(.+?)(?=REASONING:|$)', decision_text, re.DOTALL | re.IGNORECASE)
            
            return {
                "type": "USE_TOOL",
                "tool": tool_match.group(1) if tool_match else "think_deeply",
                "input": input_match.group(1).strip() if input_match else decision_text[:200]
            }
        
        # Check for THINK
        if re.search(r'ACTION:\s*THINK', decision_text, re.IGNORECASE):
            content_match = re.search(r'CONTENT:\s*(.+)', decision_text, re.DOTALL | re.IGNORECASE)
            return {
                "type": "THINK",
                "content": content_match.group(1).strip() if content_match else decision_text
            }
        
        # Default: treat as thinking
        return {"type": "THINK", "content": decision_text}

    def execute(self, task: str, conversation_history: List[str]) -> str:
        """
        Execute TRUE agentic analysis with iterative reasoning loop.
        
        Args:
            task: The user's strategic plan/task
            conversation_history: List of previous agent outputs

        Returns:
            The agent's structured analysis
        """
        try:
            # Run the true agentic loop
            response = self._run_agent_loop(task, conversation_history)
            
            # Ensure proper formatting
            if not response.startswith("###"):
                response = f"### 🎯 {self.name}\n\n{response}"
            
            # Validate
            validation = AgentOutputParser.validate_output_format(response, self.name)
            print(f"INFO: [{self.name}] Analysis complete (validation: {validation['is_valid']})")
            
            return response

        except Exception as e:
            error_msg = f"ERROR: [{self.name}] Failed: {e}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return f"### 🎯 {self.name}\n\n❌ **Error**: {e}"

    def execute_streaming(self, task: str, conversation_history: List[str]):
        """
        Execute TRUE agentic analysis with streaming.
        Shows the agent's reasoning process in real-time.
        """
        print(f"INFO: [{self.name}] Starting TRUE agentic analysis (streaming)...")
        
        try:
            # Yield header
            yield f"### 🎯 {self.name}\n\n"
            yield f"*[Agentic reasoning loop initiated...]*\n\n"
            
            # For streaming, we'll run the full loop and stream the final output
            # (True streaming of the reasoning loop would be complex)
            response = self._run_agent_loop(task, conversation_history)
            
            # Stream the response
            llm = self.llm_client.get_llm()
            full_text = ""
            
            for chunk in llm.stream(response):
                content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                full_text += content
                yield content
            
            # Validate
            validation = AgentOutputParser.validate_output_format(full_text, self.name)
            print(f"INFO: [{self.name}] Streaming complete (validation: {validation['is_valid']})")

        except Exception as e:
            print(f"ERROR: [{self.name}] Streaming failed: {e}")
            import traceback
            traceback.print_exc()
            yield f"❌ **Error**: {e}"
    
    def validate_output(self, output: str) -> Dict[str, bool]:
        """Validate agent output format."""
        return AgentOutputParser.validate_output_format(output, self.name)
