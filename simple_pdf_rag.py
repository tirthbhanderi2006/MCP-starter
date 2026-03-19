import os
import sys
from typing import List
import openai
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter

load_dotenv()

class SimplePDFAQ:
    def __init__(self, pdf_path: str):
        """
        Initialize the Simple PDF QA system.
        
        Args:
            pdf_path (str): Path to the PDF file
        """
        self.pdf_path = pdf_path
        self.setup_openai()
        self.setup_llama_index()
        self.load_and_index_pdf()
        # Initialize OpenAI client for non-RAG mode
        self.openai_client = openai.OpenAI()
    
    def setup_openai(self):
        """Setup OpenAI API key."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Please set OPENAI_API_KEY in your environment variables")
        openai.api_key = api_key
    
    def setup_llama_index(self):
        """Setup LlamaIndex settings."""
        # Use GPT-4o-mini for chat
        Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)
        
        # Use OpenAI embeddings
        Settings.embed_model = OpenAIEmbedding(embed_batch_size=50)
        
        # Setup text splitter
        Settings.text_splitter = SentenceSplitter(chunk_size=1024)
        
        print("✓ LlamaIndex settings configured")
    
    def get_system_prompt(self):
        """
        Returns the system prompt that defines the AI persona.
        """
        return (
            "You are a friendly and helpful friend of a Charusat University student. "
            "You're here to help them understand their study materials and answer questions about the PDF content. "
            "Be supportive, encouraging, and explain things in a way that's easy to understand. "
            "Use a friendly tone and feel free to add relevant examples or analogies that a college student would relate to. "
            "Always base your answers on the content from the PDF, and make learning feel like a collaborative discussion with a friend."
        )
    
    def load_and_index_pdf(self):
        """Load PDF and create vector index."""
        print(f"Loading PDF: {self.pdf_path}")
        
        # Load the PDF
        documents = SimpleDirectoryReader(
            input_files=[self.pdf_path]
        ).load_data()
        
        print(f"✓ Loaded {len(documents)} document(s)")
        
        # Create vector index
        self.index = VectorStoreIndex.from_documents(documents)
        
        # Create query engine with system prompt
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=3,
            response_mode="compact",
            system_prompt=self.get_system_prompt()
        )
        
        print("✓ PDF indexed successfully")
    
    def ask(self, question: str) -> dict:
        """
        Ask a question about the PDF content.
        
        Args:
            question (str): Your question
            
        Returns:
            dict: Contains answer and source information
        """
        response = self.query_engine.query(question)
        
        # Extract source information
        sources = []
        for node in response.source_nodes:
            sources.append({
                "page": node.metadata.get("page_label", "Unknown"),
                "text": node.text[:200] + "..." if len(node.text) > 200 else node.text,
                "score": node.score
            })
        
        return {
            "answer": str(response),
            "sources": sources
        }
    
    def ask_without_rag(self, question: str) -> dict:
        """
        Ask a question without using RAG (general knowledge only).
        
        Args:
            question (str): Your question
            
        Returns:
            dict: Contains answer without sources
        """
        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": question}
        ]
        
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0
        )
        
        return {
            "answer": response.choices[0].message.content,
            "sources": []  # No sources since this is not RAG
        }

def main():
    """Example usage of the Simple PDF QA system."""
    print("\n" + "="*50)
    print("PDF QA System")
    print("="*50)
    
    # Get mode selection first
    while True:
        print("\nSelect mode:")
        print("1. RAG Mode - Answers based on PDF content")
        print("2. General Mode - Answers from general knowledge")
        mode_input = input("\nSelect mode (1 for RAG, 2 for General): ").strip()
        
        if mode_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            return
        
        if mode_input in ['1', '2']:
            use_rag = mode_input == '1'
            mode_name = "RAG" if use_rag else "General"
            break
        
        print("Please enter 1 or 2")
    
    # If RAG mode, get PDF file
    if use_rag:
        print("\n[RAG Mode] Please provide your PDF file")
        pdf_path = input("Enter the path to your PDF file: ").strip()
        
        if not os.path.exists(pdf_path):
            print(f"Error: File not found at {pdf_path}")
            return
        
        try:
            # Initialize the QA system
            qa_system = SimplePDFAQ(pdf_path)
        except Exception as e:
            print(f"Error initializing PDF: {str(e)}")
            return
    else:
        print("\n[General Mode] No PDF needed - using general knowledge")
        # Initialize a minimal system for general mode
        try:
            qa_system = SimplePDFAQ.__new__(SimplePDFAQ)
            qa_system.setup_openai()
            qa_system.openai_client = openai.OpenAI()
            qa_system.get_system_prompt = SimplePDFAQ.get_system_prompt.__get__(qa_system, SimplePDFAQ)
        except Exception as e:
            print(f"Error initializing: {str(e)}")
            return
    
    print(f"\nPDF QA System Ready in {mode_name} Mode!")
    print("Type 'quit' to exit")
    print("="*50 + "\n")
    
    # Interactive Q&A loop
    while True:
        # Get question
        question = input(f"\n[{mode_name} Mode] Your question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not question:
            continue
        
        # Get answer
        try:
            if use_rag:
                result = qa_system.ask(question)
            else:
                result = qa_system.ask_without_rag(question)
            
            # Display answer
            print(f"\nAnswer [{mode_name} Mode]: {result['answer']}")
            
            # Display sources (only for RAG mode)
            if result['sources']:
                print("\nSources from PDF:")
                for i, source in enumerate(result['sources'], 1):
                    print(f"\n{i}. Page {source['page']} (Score: {source['score']:.2f})")
                    print(f"   {source['text']}")
            elif use_rag:
                print("\n[No sources found - this might indicate the answer is not in the PDF]")
            
            print("\n" + "-"*50)
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
