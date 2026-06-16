import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Zyro Dynamics HR Help Desk",
    page_icon="🤖",
    layout="centered",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fb; }
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 1.5px solid #4A90E2;
        padding: 10px;
    }
    .answer-box {
        background: #e8f4ea;
        border-left: 5px solid #2e7d32;
        border-radius: 8px;
        padding: 16px 20px;
        margin-top: 12px;
        font-size: 15px;
        line-height: 1.7;
    }
    .refusal-box {
        background: #fff3e0;
        border-left: 5px solid #e65100;
        border-radius: 8px;
        padding: 14px 20px;
        margin-top: 12px;
        font-size: 15px;
    }
    .source-box {
        background: #f0f4ff;
        border-radius: 8px;
        padding: 10px 16px;
        margin-top: 6px;
        font-size: 13px;
        color: #333;
    }
    .badge {
        background: #4A90E2;
        color: white;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 12px;
        font-weight: bold;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🤖 Zyro Dynamics HR Help Desk")
st.caption("Powered by RAG · Ask anything about Zyro Dynamics HR policies")
st.divider()

# ── API Key ────────────────────────────────────────────────────────────────────
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# ── Out-of-scope keyword pre-filter ────────────────────────────────────────────
OUT_OF_SCOPE_SIGNALS = [
    "ipl", "cricket", "football", "sports", "movie", "film", "stock",
    "weather", "recipe", "cook", "politics", "news", "bitcoin", "crypto",
    "who won", "president", "celebrity", "actor", "actress", "song", "music",
    "instagram", "twitter", "facebook", "tiktok", "game", "gaming",
    "restaurant", "hotel", "book recommendation", "science", "history",
]

REFUSAL_MESSAGE = (
    "I can only answer HR-related questions from Zyro Dynamics policy documents. "
    "Please ask about leave, compensation, onboarding, work-from-home, conduct, "
    "performance reviews, or other HR topics."
)

def is_out_of_scope_by_keyword(question: str) -> bool:
    q = question.lower()
    return any(signal in q for signal in OUT_OF_SCOPE_SIGNALS)

# ── Build RAG (cached so it runs once) ────────────────────────────────────────
@st.cache_resource(show_spinner="Loading HR policy documents…")
def build_rag():
    pdf_folder = "data"
    documents = []
    for file in sorted(os.listdir(pdf_folder)):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(pdf_folder, file))
            documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 8,
            "fetch_k": 40,
            "lambda_mult": 0.6
        }
    )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=600
    )

    prompt = ChatPromptTemplate.from_template("""
You are an expert HR Help Desk Assistant for Zyro Dynamics Pvt. Ltd.
Answer employee questions using ONLY the context provided below.

Rules:
1. Answer concisely and factually using ONLY information from the context.
2. Include specific details like numbers, durations, thresholds, and process steps when present.
3. If the question is not covered by the context, respond with EXACTLY:
   "I can only answer HR-related questions from Zyro Dynamics policy documents."
4. Do NOT make up information. Do NOT answer from general knowledge.

Context:
{context}

Question: {question}

Answer:""")

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return retriever, rag_chain


retriever, rag_chain = build_rag()

# ── Sidebar: quick topic guide ─────────────────────────────────────────────────
with st.sidebar:
    st.header("📋 Topics I Can Help With")
    topics = [
        "🏖️ Leave Policy (EL, SL, Maternity, Paternity)",
        "🏠 Work From Home / Hybrid",
        "💰 Compensation & Benefits",
        "📋 Performance Reviews & PIP",
        "🆕 Onboarding & Separation",
        "✈️ Travel & Expense Reimbursement",
        "🛡️ Code of Conduct & Ethics",
        "🔒 IT & Data Security",
        "⚖️ POSH / Prevention of Harassment",
        "🏢 Company Profile & Culture",
    ]
    for t in topics:
        st.markdown(f"- {t}")
    st.divider()
    st.caption("Zyro Dynamics Internal HR Bot · For employees only")

# ── Chat history ───────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── Input form ─────────────────────────────────────────────────────────────────
with st.form(key="qa_form", clear_on_submit=True):
    question = st.text_input(
        "💬 Ask your HR question",
        placeholder="e.g. How many earned leaves do I get per year?",
    )
    submitted = st.form_submit_button("Ask →", use_container_width=True)

# ── Handle submission ──────────────────────────────────────────────────────────
if submitted and question.strip():
    q = question.strip()

    with st.spinner("Searching HR policies…"):
        # 1. Fast keyword pre-filter
        if is_out_of_scope_by_keyword(q):
            answer = REFUSAL_MESSAGE
            sources = []
        else:
            # 2. Run RAG chain
            answer = rag_chain.invoke(q)

            # 3. Normalise refusals from the LLM
            if (
                "i can only answer hr" in answer.lower()
                or "information not found" in answer.lower()
                or len(answer.strip()) == 0
            ):
                answer = REFUSAL_MESSAGE
                sources = []
            else:
                docs = retriever.invoke(q)
                # Deduplicate sources by filename
                seen = set()
                sources = []
                for doc in docs:
                    src = os.path.basename(doc.metadata.get("source", "Unknown"))
                    page = doc.metadata.get("page", "?")
                    key = f"{src}::{page}"
                    if key not in seen:
                        seen.add(key)
                        sources.append((src, page, doc.page_content[:180].strip()))

    # Save to history
    st.session_state.history.insert(0, {
        "question": q,
        "answer": answer,
        "sources": sources,
    })

# ── Render history ─────────────────────────────────────────────────────────────
for entry in st.session_state.history:
    st.markdown(f"**🙋 {entry['question']}**")

    is_refusal = entry["answer"] == REFUSAL_MESSAGE or "i can only answer" in entry["answer"].lower()

    if is_refusal:
        st.markdown(
            f'<div class="refusal-box">⚠️ {entry["answer"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="answer-box">✅ {entry["answer"]}</div>',
            unsafe_allow_html=True,
        )
        if entry["sources"]:
            with st.expander(f"📄 Sources ({len(entry['sources'])} documents)"):
                for src, page, snippet in entry["sources"]:
                    st.markdown(
                        f'<div class="source-box">'
                        f'<span class="badge">p.{page}</span><strong>{src}</strong><br>'
                        f'<em style="color:#555">{snippet}…</em>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    st.divider()
