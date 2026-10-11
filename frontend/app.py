import os
import time

import streamlit as st

from api import ApiError, SnapRagClient, ping

st.set_page_config(
    page_title="SnapRAG",
    page_icon=":material/neurology:",
    layout="wide",
)

DEFAULT_URL = os.getenv("SNAPRAG_API_URL", "http://localhost:8003")
SUGGESTIONS = [
    "Summarize this document in five bullet points",
    "What are the key concepts covered?",
    "List the important definitions",
    "Explain the main topic step by step",
]

for key, default in {
    "messages": [],
    "docs": None,
    "docs_error": None,
    "uploader_n": 0,
}.items():
    st.session_state.setdefault(key, default)


# ---------- helpers ----------

@st.cache_data(ttl=5, show_spinner=False)
def backend_up(url: str) -> bool:
    return ping(url)


def client() -> SnapRagClient:
    return SnapRagClient(st.session_state.api_url, st.session_state.token)


def reset_docs():
    st.session_state.docs = None
    st.session_state.docs_error = None


def refresh_docs():
    if not st.session_state.token.strip():
        st.session_state.docs, st.session_state.docs_error = None, None
        return
    try:
        st.session_state.docs, st.session_state.docs_error = client().documents(), None
    except ApiError as exc:
        st.session_state.docs, st.session_state.docs_error = None, str(exc)


def choose_document():
    try:
        client().select_document(st.session_state.doc_choice)
    except ApiError as exc:
        st.session_state.docs_error = str(exc)
        return
    refresh_docs()


def ask(question: str):
    st.session_state.pending = question


def pick_suggestion():
    if st.session_state.suggestion:
        ask(st.session_state.suggestion)
    st.session_state.suggestion = None


def describe_step(event: dict) -> str:
    detail = event.get("detail")
    extra = ", ".join(f"{k}: {v}" for k, v in detail.items()) if detail else ""
    return event["message"] + (f" ({extra})" if extra else "")


def confidence_badge(value):
    pct = value * 100 if value <= 1 else value
    color = "green" if pct >= 80 else "orange" if pct >= 55 else "red"
    st.badge(f"Confidence {pct:.0f}%", icon=":material/speed:", color=color)


# ---------- answering ----------

def answer_stream(question: str) -> dict:
    """Stream the answer live; progress steps go into a compact status above it."""

    steps, errors = [], []
    started = time.perf_counter()
    status = st.status(":shimmer[Searching the document]", type="compact")

    def tokens():
        try:
            for event in client().stream(question):
                if event["type"] == "token":
                    yield event["token"]
                elif event["type"] == "step":
                    steps.append(describe_step(event))
                    status.write(steps[-1])
                elif event["type"] == "error":
                    errors.append(event["message"])
        except ApiError as exc:
            errors.append(str(exc))

    text = st.write_stream(tokens())
    elapsed = time.perf_counter() - started
    status.update(
        label=f"Finished in {elapsed:.1f}s" if not errors else "Stopped with an error",
        state="error" if errors else "complete",
    )
    for message in errors:
        st.error(message, icon=":material/error:")

    return {"role": "assistant", "content": str(text), "steps": steps,
            "elapsed": elapsed, "error": "; ".join(errors) or None}


def answer_batch(question: str) -> dict:
    started = time.perf_counter()
    with st.spinner("Searching and writing an answer..."):
        try:
            result = client().query(question)
        except ApiError as exc:
            return {"role": "assistant", "content": "", "error": str(exc)}

    verification = result.get("verification")
    return {
        "role": "assistant",
        "content": result.get("answer", ""),
        "elapsed": time.perf_counter() - started,
        "confidence": result.get("confidence"),
        "verified": verification.get("supported") if isinstance(verification, dict) else None,
        "counts": {k: result.get(f"retrieved_{k}") for k in ("dense", "bm25", "hybrid")},
        "citations": result.get("citations") or [],
        "sources": result.get("sources") or [],
        "followups": result.get("followup_questions") or [],
    }


def render_assistant(index: int, message: dict):
    if message.get("steps"):
        with st.expander(f"Searched in {message['elapsed']:.1f}s", type="compact"):
            for step in message["steps"]:
                st.caption(step)

    if message.get("content"):
        st.markdown(message["content"])
    if message.get("error"):
        st.error(message["error"], icon=":material/error:")

    badges = st.container(horizontal=True)
    with badges:
        if message.get("confidence") is not None:
            confidence_badge(message["confidence"])
        if message.get("verified") is not None:
            st.badge("Verified" if message["verified"] else "Not verified",
                     icon=":material/verified:" if message["verified"] else ":material/gpp_maybe:",
                     color="green" if message["verified"] else "red")
        for name, value in (message.get("counts") or {}).items():
            if value is not None:
                st.badge(f"{name} {value}", color="gray")

    if message.get("citations") or message.get("sources"):
        with st.expander("Sources", icon=":material/menu_book:"):
            for c in message.get("citations", []):
                st.markdown(f"**[{c['source_id']}]** {c['document']}, page {c['page']}")
            for s in message.get("sources", []):
                st.caption(f"{s['document']}, page {s['page']}"
                           + (f" · rerank {s['rerank_score']:.2f}" if s.get("rerank_score") is not None else ""))
                st.text((s.get("text") or "")[:400])

    if message.get("followups"):
        with st.container(horizontal=True):
            for j, question in enumerate(message["followups"]):
                st.button(question, key=f"followup_{index}_{j}", icon=":material/subdirectory_arrow_right:",
                          type="tertiary", on_click=ask, args=(question,))


# ---------- sidebar ----------

with st.sidebar:
    st.title("SnapRAG", icon=":material/neurology:")
    st.caption("Ask questions about your PDFs")

    with st.expander("Connection", icon=":material/lan:", expanded=not os.getenv("SNAPRAG_TOKEN")):
        st.text_input("Backend URL", value=DEFAULT_URL, key="api_url", on_change=reset_docs)
        st.text_input("Access token", value=os.getenv("SNAPRAG_TOKEN", ""), type="password",
                      key="token", on_change=reset_docs,
                      help="Run `python dev_token.py` in rag-service to get one.")
    if backend_up(st.session_state.api_url):
        st.badge("Backend online", icon=":material/check_circle:", color="green")
    else:
        st.badge("Backend unreachable", icon=":material/cloud_off:", color="red")

    if st.session_state.docs is None and st.session_state.docs_error is None:
        refresh_docs()

    st.subheader("Documents")
    docs = st.session_state.docs
    if not st.session_state.token.strip():
        st.caption("Add your access token under Connection to load documents.")
    elif st.session_state.docs_error:
        st.error(st.session_state.docs_error, icon=":material/error:")
    elif docs and docs["documents"]:
        names = [d["name"] for d in docs["documents"]]
        st.session_state.doc_choice = docs["active_document"] if docs["active_document"] in names else None
        st.radio("Active document", names, index=None, key="doc_choice", on_change=choose_document,
                 captions=[f"{d['pages']} pages · {d['chunks']} chunks" for d in docs["documents"]],
                 label_visibility="collapsed", width="stretch")
    else:
        st.caption("No documents yet. Add a PDF below.")

    with st.expander("Add a PDF", icon=":material/upload_file:"):
        file = st.file_uploader("PDF file", type="pdf", key=f"pdf_{st.session_state.uploader_n}",
                                label_visibility="collapsed")
        if st.button("Index document", type="primary", icon=":material/database:",
                     disabled=file is None, width="stretch"):
            with st.spinner("Indexing... large PDFs can take a few minutes"):
                try:
                    result = client().upload(file.name, file.getvalue())
                except ApiError as exc:
                    st.error(str(exc), icon=":material/error:")
                else:
                    st.session_state.flash = (f"Indexed {result['filename']}: "
                                              f"{result['pages']} pages, {result['total_chunks']} chunks")
                    st.session_state.uploader_n += 1
                    refresh_docs()
                    st.rerun()

    st.toggle("Stream answers", value=True, key="stream",
              help="On: live answer with progress. Off: waits for the full answer and shows sources, confidence and follow-ups.")
    if st.button("Clear chat", icon=":material/delete_sweep:", width="stretch",
                 disabled=not st.session_state.messages):
        st.session_state.messages = []
        st.rerun()

if flash := st.session_state.pop("flash", None):
    st.toast(flash, icon=":material/check_circle:")


# ---------- main ----------

active = (st.session_state.docs or {}).get("active_document")

# Read the input first: the intro card is only drawn when nothing is being asked,
# otherwise it would stay on screen above the first answer until the next rerun.
prompt = st.chat_input("Ask about the active document", submit_mode="disable") \
    or st.session_state.pop("pending", None)

with st.container(horizontal=True, vertical_alignment="center"):
    st.header("Chat", icon=":material/forum:")
    if active:
        st.badge(active, icon=":material/description:", color="blue")
    else:
        st.caption("No active document: pick or upload one in the sidebar.")

if not st.session_state.messages and not prompt:
    with st.container(border=True):
        st.markdown("#### Ask anything about your document")
        st.caption("Answers come from the active PDF, with page citations when streaming is off.")
        st.pills("Try asking", SUGGESTIONS, key="suggestion", on_change=pick_suggestion,
                 label_visibility="collapsed")

for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            render_assistant(i, message)

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if st.session_state.stream:
            reply = answer_stream(prompt)
        else:
            reply = answer_batch(prompt)
            render_assistant(len(st.session_state.messages), reply)

    st.session_state.messages.append(reply)
