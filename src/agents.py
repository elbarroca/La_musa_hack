"""
Symphony Agent Module with LangChain LCEL Chains
Each agent uses a RAG chain: retriever -> prompt -> LLM -> parser
"""

from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

from src.knowledge_base import KnowledgeBase
from src.llm_client import LLMClient


class SymphonyAgent:
    """
    A context-aware agent that uses LangChain LCEL chains for RAG.
    Retrieves relevant knowledge, formats prompts, and generates responses.
    """

    def __init__(
        self,
        agent_name: str,
        persona_prompt: str,
        llm_client: LLMClient,
        retriever: KnowledgeBase
    ):
        """
        Initialize the Symphony Agent.

        Args:
            agent_name: Name of the agent (e.g., "Shrek", "Sonic")
            persona_prompt: The agent's role and instructions
            llm_client: LLM client instance
            retriever: Knowledge base for RAG
        """
        self.name = agent_name
        self.persona_prompt = persona_prompt
        self.llm_client = llm_client
        self.retriever = retriever

        # Build the RAG chain using LCEL
        self._build_chain()

    def _build_chain(self):
        """
        Build the LangChain LCEL chain for this agent.
        Uses RunnableParallel for better debugging and intermediate value access.
        Implements 2025 best practices for LCEL chains.

        Chain structure: context retrieval -> prompt formatting -> LLM -> output parsing
        """
        # Create the prompt template aligned with Master.md specifications
        template = """
{persona_prompt}

### CONTEXT FROM COMPANY KNOWLEDGE BASE ###
{context}

### ONGOING CONVERSATION ###
{history}

### USER INPUT / PLAN ###
{task}

Based on your role and the context provided, execute your task on the user input above.
Provide your analysis following the output format specified in your role description.
"""

        self.prompt = ChatPromptTemplate.from_template(template)

        # Build the LCEL chain with RunnableParallel for better debugging
        # This allows us to access intermediate values and improves traceability
        self.chain = (
            RunnableParallel(
                context=lambda x: self._retrieve_context(x["task"]),
                history=lambda x: x["history"],
                task=lambda x: x["task"],
                persona_prompt=lambda x: self.persona_prompt
            )
            | self.prompt
            | self.llm_client.get_llm()
            | StrOutputParser()
        )

    def _retrieve_context(self, task: str) -> str:
        """
        Retrieve relevant context from the knowledge base.

        Args:
            task: The task/query to retrieve context for

        Returns:
            Formatted context string
        """
        # Create a query that incorporates the agent's role
        query = f"{self.name} analysis: {task}"

        # Retrieve relevant documents
        docs = self.retriever.retrieve(query=query, top_k=2)

        if not docs or docs[0].startswith("No"):
            return "No specific company knowledge was found for this query. Use your general expertise."

        # Format the documents
        context_str = "\n\n".join(f"{i+1}. {doc}" for i, doc in enumerate(docs))
        return context_str

    def execute(self, task: str, conversation_history: List[str]) -> str:
        """
        Execute the agent's turn using the RAG chain.

        Args:
            task: The user's strategic plan/task
            conversation_history: List of previous agent outputs

        Returns:
            The agent's formatted response
        """
        print(f"INFO: [{self.name}] Executing task...")

        # Format conversation history
        history_str = "\n\n".join(conversation_history) if conversation_history else "This is the start of the conversation."

        try:
            # Invoke the chain
            response = self.chain.invoke({
                "task": task,
                "history": history_str
            })

            # Format the output for UI display
            formatted_response = f"### 🎯 {self.name}\n\n{response}"

            print(f"INFO: [{self.name}] Analysis complete")
            return formatted_response

        except Exception as e:
            error_msg = f"ERROR: [{self.name}] Failed to execute: {e}"
            print(error_msg)
            return f"### 🎯 {self.name}\n\n❌ **Error**: Failed to generate analysis. Please check your API configuration."

    def execute_streaming(self, task: str, conversation_history: List[str]):
        """
        Execute the agent's turn with streaming output.
        Yields chunks of text as they're generated.

        Args:
            task: The user's strategic plan/task
            conversation_history: List of previous agent outputs

        Yields:
            Chunks of the agent's response
        """
        print(f"INFO: [{self.name}] Executing task with streaming...")

        # Format conversation history
        history_str = "\n\n".join(conversation_history) if conversation_history else "This is the start of the conversation."

        try:
            # Yield the agent header first
            yield f"### 🎯 {self.name}\n\n"

            # Stream the response
            for chunk in self.chain.stream({
                "task": task,
                "history": history_str
            }):
                yield chunk

            print(f"INFO: [{self.name}] Streaming complete")

        except Exception as e:
            error_msg = f"ERROR: [{self.name}] Failed to execute streaming: {e}"
            print(error_msg)
            yield f"❌ **Error**: Failed to generate analysis. Please check your API configuration."
