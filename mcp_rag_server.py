import os
from fastmcp import FastMCP
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
import openai

load_dotenv()

mcp = FastMCP("rag-server")

class RAGSystem:
    def __init__(self):
        self.setup_openai()
    
    def setup_openai(self):
        """Setup OpenAI API key."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Please set OPENAI_API_KEY in your environment variables")
        openai.api_key = api_key
    
    def get_system_prompt(self):
        """Returns the system prompt that defines the AI persona."""
        return (
            "You are a friendly and helpful friend of a Charusat University student. "
            "You're here to help them understand their study materials and answer questions about the PDF content. "
            "Be supportive, encouraging, and explain things in a way that's easy to understand. "
            "Use a friendly tone and feel free to add relevant examples or analogies that a college student would relate to. "
            "Always base your answers on the content from the PDF, and make learning feel like a collaborative discussion with a friend."
        )
    
    def ask_pdf(self, pdf_path: str, question: str) -> str:
        """Load PDF and query it in one operation."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Setup LlamaIndex settings
        Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)
        Settings.embed_model = OpenAIEmbedding(embed_batch_size=50)
        Settings.text_splitter = SentenceSplitter(chunk_size=1024)
        
        # Load the PDF
        documents = SimpleDirectoryReader(input_files=[pdf_path]).load_data()
        
        # Create vector index
        index = VectorStoreIndex.from_documents(documents)
        
        # Create query engine with system prompt
        query_engine = index.as_query_engine(
            similarity_top_k=3,
            response_mode="compact",
            system_prompt=self.get_system_prompt()
        )
        
        # Query the PDF
        response = query_engine.query(question)
        
        # Extract source information
        sources_info = []
        for i, node in enumerate(response.source_nodes, 1):
            page = node.metadata.get("page_label", "Unknown")
            score = node.score
            text_preview = node.text[:150] + "..." if len(node.text) > 150 else node.text
            sources_info.append(f"Source {i} (Page {page}, Score: {score:.2f}): {text_preview}")
        
        result = f"Answer: {str(response)}\n\nSources:\n" + "\n".join(sources_info)
        return result

rag_system = RAGSystem()

@mcp.tool()
def ask_pdf(pdf_path: str, question: str) -> str:
    """Ask a question about a PDF document using RAG (Retrieval-Augmented Generation).
    This tool will automatically load and index the PDF, then answer your question with source citations.
    
    Args:
        pdf_path: Full path to the PDF file (e.g., 'c:\\path\\to\\file.pdf')
        question: Your question about the PDF content
    
    Returns:
        Answer with source citations from the PDF
    """
    try:
        return rag_system.ask_pdf(pdf_path, question)
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    mcp.run(transport="sse", port=8001)
