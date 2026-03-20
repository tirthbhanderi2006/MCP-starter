import os
import nest_asyncio
nest_asyncio.apply()

import sys
import tempfile
import asyncio
import logging

import streamlit as st
import groq
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, Settings
from llama_index.readers.file import PyMuPDFReader
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="PDF QA · MCP Starter Kit",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS — dark glassmorphism theme
# ──────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stChatMessage { border-radius: 12px; padding: 0.5rem; }
    .hero { text-align: center; padding: 2rem 0 1rem 0; }
    .hero h1 { font-size: 2.5rem; font-weight: 700; }
    .hero p  { color: #aaa; font-size: 1.1rem; }
    .badge { display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; }
    .badge-green { background: #1a3d2b; color: #4caf50; }
    .badge-yellow { background: #3d3010; color: #ffc107; }
    .badge-blue  { background: #102040; color: #4fc3f7; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def get_system_prompt() -> str:
    return (
        "You are a friendly and helpful friend of a Charusat University student. "
        "You're here to help them understand their study materials and answer questions about the PDF content. "
        "Be supportive, encouraging, and explain things in a way that's easy to understand. "
        "Use a friendly tone and feel free to add relevant examples or analogies that a college student would relate to. "
        "Always base your answers on the content from the PDF, and make learning feel like a collaborative discussion with a friend."
    )


@st.cache_resource(show_spinner="📄 Loading & indexing PDF…")
def build_index(pdf_bytes: bytes, filename: str):
    """Persist temp file, load via LlamaIndex, return query engine."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(pdf_bytes)
    tmp.flush()
    tmp.close()

    Settings.llm = Groq(model="llama-3.3-70b-versatile", temperature=0.5)
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.text_splitter = SentenceSplitter(chunk_size=1024)

    reader = PyMuPDFReader()
    documents = reader.load_data(file_path=tmp.name)
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine(
        similarity_top_k=3,
        response_mode="compact",
        system_prompt=get_system_prompt(),
    )
    return query_engine, len(documents)


async def ask_agent(question: str, api_key: str, query_engine) -> dict:
    """Run the unified LangChain agent connecting to MCP server and PDF."""
    mcp_server_url = os.getenv("MCP_SERVER_URL", "http://localhost:8000/sse")
    logger.info(f"🔗 Connecting to MCP server at: {mcp_server_url}")

    mcp_config = {
        "arithmetic": {
            "url": mcp_server_url,
            "transport": "sse",
        }
    }

    model = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=api_key,
        temperature=0.7,
    )

    system_msg = (
        "system",
        "You are a helpful mathematical and document assistant. "
        "You have access to a PDF document and arithmetic tools. "
        "Use them to answer the user's questions. "
        "Do not hallucinate tools. If you calculate something, simply state the result.",
    )

    try:
        logger.info("📡 Initializing MCP client...")
        client = MultiServerMCPClient(mcp_config)

        logger.info("🔍 Fetching MCP tools via session...")
        async with client.session("arithmetic") as session:
            mcp_tools = await load_mcp_tools(session)
            logger.info(f"✅ Loaded {len(mcp_tools)} MCP tools: {[t.name for t in mcp_tools]}")

            @tool
            def ask_pdf_document(query: str) -> str:
                """Use this tool to search the uploaded PDF document for information."""
                if query_engine is None:
                    return "No PDF has been uploaded yet."
                return str(query_engine.query(query))

            all_tools = mcp_tools + [ask_pdf_document]
            agent_executor = create_react_agent(model, all_tools)

            logger.info("🤖 Invoking agent...")
            response = await agent_executor.ainvoke(
                {"messages": [system_msg, ("user", question)]}
            )
            logger.info("✅ Agent response received")
            return {"answer": response["messages"][-1].content, "sources": []}

    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Agent Error: {error_msg}")

        if "Connection" in error_msg or "refused" in error_msg or "Connect" in error_msg:
            return {
                "answer": (
                    f"❌ Cannot connect to MCP Server at `{mcp_server_url}`.\n\n"
                    "**Check:**\n"
                    "- Run `docker compose ps` — is the backend container Up?\n"
                    "- Run `docker compose logs backend` — did the SSE server start?\n"
                    "- Run `docker compose exec frontend curl http://backend:8000/sse` — does it respond?"
                ),
                "sources": [],
            }
        elif "400" in error_msg or "Bad Request" in error_msg:
            return {
                "answer": (
                    f"❌ Backend returned 400 Bad Request at `{mcp_server_url}`.\n\n"
                    "The backend may still be initializing. Wait 10s and try again."
                ),
                "sources": [],
            }
        else:
            return {
                "answer": (
                    f"❌ Agent Error: {error_msg}\n\n"
                    f"**Make sure:**\n"
                    f"1. MCP server is running on `{mcp_server_url}`\n"
                    f"2. `GROQ_API_KEY` is set in `.env`\n"
                    f"3. Both backend and frontend containers are running"
                ),
                "sources": [],
            }


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.divider()

    api_key_set = bool(os.getenv("GROQ_API_KEY"))

    if api_key_set:
        st.markdown(
            '<span class="badge badge-green">🟢 System Ready (API Key Found)</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="badge badge-yellow">🟡 Missing GROQ_API_KEY in .env</span>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("## 📎 Upload PDF")
    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"],
        help="Upload the PDF you want to ask questions about.",
    )

    query_engine = None
    if uploaded_file:
        query_engine, n_docs = build_index(uploaded_file.getvalue(), uploaded_file.name)
        st.markdown(
            f'<span class="badge badge-blue">📄 {uploaded_file.name} — {n_docs} page(s) indexed</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="badge badge-yellow">⏳ Waiting for PDF</span>',
            unsafe_allow_html=True,
        )

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption("Built with ❤️ using LlamaIndex + Groq")


# ──────────────────────────────────────────────
# Hero header
# ──────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>📚 PDF QA System</h1>
    <p>Ask questions about your study materials — powered by RAG &amp; Groq</p>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Chat history
# ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍🎓" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 Sources from PDF", expanded=False):
                for src in msg["sources"]:
                    st.markdown(
                        f"""
**Page {src['page']}** · Score: {src['score']:.2f}

{src['text']}
""",
                        unsafe_allow_html=True,
                    )


# ──────────────────────────────────────────────
# Chat input
# ──────────────────────────────────────────────
prompt = st.chat_input("Ask a question…")

if prompt:
    if not api_key_set:
        st.error("⚠️ Please set your Groq API key in the `.env` file.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking…"):
            try:
                api_key = os.getenv("GROQ_API_KEY", "")
                loop = asyncio.get_event_loop()
                result = loop.run_until_complete(ask_agent(prompt, api_key, query_engine))

                st.markdown(result["answer"])

                if result.get("sources"):
                    with st.expander("📚 Sources from PDF", expanded=True):
                        for src in result["sources"]:
                            st.markdown(
                                f"""
**Page {src['page']}** · Score: {src['score']:.2f}

{src['text']}
""",
                                unsafe_allow_html=True,
                            )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result.get("sources", []),
                })

            except Exception as e:
                st.error(f"❌ Error: {e}")
