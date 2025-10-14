from abc import ABC, abstractmethod
from typing import List

class KnowledgeBase(ABC):
    """Abstract base class for a knowledge base retriever."""
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        pass

class MockKnowledgeBase(KnowledgeBase):
    """
    A mock knowledge base with data tailored to the Azure Specialist personas.
    This ensures the demo is stable, impressive, and perfectly aligned with the prompts.
    """
    def __init__(self):
        # Data is now specific to Microsoft/Azure context
        self.documents = {
            "architect": "Azure Well-Architected Framework Review for 'Project Phoenix': The project failed to implement Azure Policy for cost management, leading to a 200% budget overrun in its AKS cluster. A lack of private endpoints also created a significant security vulnerability.",
            "customer": "Dynamics 365 Customer Advisory Board Feedback (Q4): Users reported 'workflow fatigue' from tools that don't seamlessly integrate with their M365 identity. A feature requiring more than 3 clicks to see value had an 80% drop-off in adoption.",
            "competitor": "CI Battle Card - AWS vs. Azure (Logistics Vertical): AWS is aggressively bundling Amazon Redshift and SageMaker for logistics customers. Their primary sales tactic is to offer deep discounts on data ingestion to lock customers into their ecosystem."
        }

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        print(f"INFO: MockKnowledgeBase received query: '{query}'")
        query = query.lower()
        results = []
        # Match queries to the new, more specific data
        if "architect" in query or "risk" in query:
            results.append(self.documents["architect"])
        if "customer" in query or "pm" in query:
            results.append(self.documents["customer"])
        if "competitor" in query or "market" in query:
            results.append(self.documents["competitor"])

        if not results:
            results.append("No specific documents found in the mock database for this query.")

        return results[:top_k]

# --- To use a REAL knowledge base, you would implement this class ---
# from azure.core.credentials import AzureKeyCredential
# from azure.search.documents import SearchClient

# class AzureAISearchKnowledgeBase(KnowledgeBase):
#     def __init__(self, endpoint: str, key: str, index_name: str):
#         self.client = SearchClient(endpoint=endpoint, index_name=index_name, credential=AzureKeyCredential(key))
#
#     def retrieve(self, query: str, top_k: int = 3) -> List[str]:
#         try:
#             results = self.client.search(search_text=query, top=top_k, include_total_count=True)
#             return [result['content'] for result in results]
#         except Exception as e:
#             print(f"ERROR: Failed to retrieve from Azure AI Search: {e}")
#             return []