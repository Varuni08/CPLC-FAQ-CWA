import os
import re
import time
from tqdm import tqdm
from pypdf import PdfReader
from pinecone import Pinecone, ServerlessSpec
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from getpass import getpass

st.set_page_config(
    page_title="CPLC FAQ Bot",
    layout="centered",
)

INDEX_NAME  = "cplc-faq-bot"
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL   = "llama-3.1-8b-instant"

DEFAULT_SUGGESTIONS = [
    "What is CPLC?",
    "What courses does CPLC offer?",
    "How long are the courses?",
    "Does CPLC offer placement support?",
    "How do I enroll?",
]

GREETINGS   = {"hi","hello","hey","heya","wassup","yo","hiya","howdy","sup","what's up","whats up","namaste","vannakam"}
THANKS      = {"thanks","thank you","thank you so much","thx","ty","appreciate it"}
HOW_ARE_YOU = {"how are you","how are you doing","how's it going","hows it going","you good"}

TOPIC_LINKS = {
    "courses":           "https://www.cplc.in/courses",
    "enroll":            "https://www.cplc.in/enroll",
    "contact":           "https://www.cplc.in/contact",
    "placement":         "https://www.cplc.in/placement",
    "corporate":         "https://www.cplc.in/corporate-training",
    "about":             "https://www.cplc.in/about",
    "location":          "https://www.cplc.in/contact",
    "machine learning":  "https://www.cplc.in/courses/machine-learning",
    "deep learning":     "https://www.cplc.in/courses/deep-learning",
    "digital marketing": "https://www.cplc.in/courses/digital-marketing",
    "software testing":  "https://www.cplc.in/courses/software-testing",
    "nlp":               "https://www.cplc.in/courses/nlp",
    "computer vision":   "https://www.cplc.in/courses/computer-vision",
    "applied ai":        "https://www.cplc.in/courses/applied-ai",
}

@st.cache_resource(show_spinner="Loading models…")
def load_clients():
    pinecone_key = st.secrets["PINECONE_API_KEY"]
    groq_key     = st.secrets["GROQ_API_KEY"]
    pc       = Pinecone(api_key=pinecone_key)
    groq_cli = Groq(api_key=groq_key)
    embedder = SentenceTransformer(EMBED_MODEL)
    index    = pc.Index(INDEX_NAME)
    return groq_cli, embedder, index

# R
def retrieve_relevant_chunks(query, embedder, index, top_k=5):
    q_emb = embedder.encode([query], convert_to_numpy=True)[0].tolist()
    results = index.query(vector=q_emb, top_k=top_k, include_metadata=True)
    return [
        {
            "text":        m["metadata"]["text"],
            "score":       m["score"],
            "page_number": m["metadata"].get("page_number"),
        }
        for m in results.matches
    ]

def ask_cplc(question, groq_cli, embedder, index):
    chunks  = retrieve_relevant_chunks(question, embedder, index)
    context = "\n\n".join(c["text"] for c in chunks)

    system_prompt = (
        "You are a helpful FAQ assistant for CPLC (Codework Pro Learn Centre), "
        "a technology and AI training institute in Chennai. "
        "Answer questions clearly, concisely and in a warm, welcoming tone based only on the context provided. "
        "If the answer is not in the context, say: "
        "'I do not have the information you're asking for. You can visit https://www.cplc.in/ "
        "or contact admissions@codework.ai or call +91 72004 21678.' "
        "Do not make up information. "
        "If contact details, course names, duration, training type, or process are mentioned "
        "in the context, use them accurately. Keep answers clear, natural, and factual. "
        "At the end of your answer, always include a relevant link from the CPLC website "
        "based on what the user asked. For example, if they ask about courses, include "
        "https://www.cplc.in/courses. If about placement, include https://www.cplc.in/placement. "
        "If about contact, include https://www.cplc.in/contact. "
        "If no specific page applies, include https://www.cplc.in/. "
    )
    user_prompt = f"Context from CPLC knowledge base:\n{context}\n\nQuestion: {question}\n\nAnswer:"

    response = groq_cli.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        model=LLM_MODEL,
        temperature=0.1,
        max_tokens=800,
    )
    return response.choices[0].message.content

def get_suggested_questions(question, answer, groq_cli):
    prompt = (
        f"A user asked a question to a FAQ chatbot for CPLC (Codework Pro Learn Centre), "
        f"a tech and AI training institute in Chennai.\n\n"
        f"User's question: {question}\nBot's answer: {answer}\n\n"
        "Based on this, suggest exactly 3 short follow-up questions the user might want to ask next.\n"
        "These should be natural next questions related to CPLC's courses, fees, placement, enrollment, "
        "corporate training, or contact info.\n\n"
        "Return ONLY a numbered list like:\n1. <question>\n2. <question>\n3. <question>\n\nNo extra text."
    )
    resp = groq_cli.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=LLM_MODEL,
        temperature=0.1,
        max_tokens=250,
    )
    raw = resp.choices[0].message.content.strip()
    suggestions = []
    for line in raw.split("\n"):
        line = line.strip()
        if line and line[0].isdigit():
            suggestions.append(line)
    return suggestions[:3]

def handle_small_talk(user_input):
    text = user_input.lower().strip().rstrip("!?.").strip()
    if text in GREETINGS:
        return "Hello! Welcome to CPLC! I'm here to help with questions about our courses, enrollment, placement support, and more. What would you like to know?"
    if text in THANKS:
        return "You're most welcome! Feel free to ask if you have any more questions about CPLC."
    if text in HOW_ARE_YOU:
        return "I'm doing great, thank you for asking! Ready to help you with anything about CPLC. What would you like to know?"
    return None

# ui
st.title("CPLC FAQ Assistant")
st.caption("Ask me anything about Codework Pro Learn Centre – courses, placement, enrollment & more!")

groq_cli, embedder, index = load_clients()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "suggestions" not in st.session_state:
    st.session_state.suggestions = DEFAULT_SUGGESTIONS

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if st.session_state.suggestions:
    st.markdown("**You might want to ask:**")
    cols = st.columns(len(st.session_state.suggestions))
    for i, suggestion in enumerate(st.session_state.suggestions):
        label = re.sub(r"^\d+\.\s*", "", suggestion)
        if cols[i % len(cols)].button(label, key=f"sug_{i}_{label[:10]}"):
            st.session_state["pending_input"] = label

user_input = st.chat_input("Type your question here…")

if "pending_input" in st.session_state:
    user_input = st.session_state.pop("pending_input")

if user_input:
    user_input = user_input.strip()
    if not user_input:
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking !"):
            small_talk = handle_small_talk(user_input)
            if small_talk:
                answer = small_talk
                new_suggestions = DEFAULT_SUGGESTIONS
            else:
                answer = ask_cplc(user_input, groq_cli, embedder, index)
                new_suggestions = get_suggested_questions(user_input, answer, groq_cli)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.suggestions = new_suggestions
    st.rerun()
