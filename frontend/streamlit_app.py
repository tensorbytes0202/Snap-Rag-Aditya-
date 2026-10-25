"""Streamlit UI for the SnapRAG backend: upload a PDF, chat about it, and watch the snapshot memory grow.

    ..\\venv\\Scripts\\python.exe -m streamlit run frontend/streamlit_app.py

The backend (rag-service/main.py) must be running, by default on http://localhost:8001
(port 8000 is often taken by another project):  python -m uvicorn main:app --port 8001
To use another address set the environment variable SNAPRAG_BACKEND before starting Streamlit.
"""

import os

import requests
import streamlit as st

st.set_page_config(page_title="SnapRAG", page_icon="📚", layout="wide")

TIMEOUT = 300
# The backend is fixed by the server's environment; visitors cannot point the app (and their token) at another address.
BACKEND = os.environ.get("SNAPRAG_BACKEND", "http://localhost:8001").rstrip("/")


def api(method: str, path: str, **kwargs):
    """One call to the backend with the saved token; returns (json, error_text)."""

    headers = {"Authorization": f"Bearer {st.session_state.token}"} if st.session_state.get("token") else {}

    try:
        reply = requests.request(method, BACKEND + path, headers=headers, timeout=TIMEOUT, allow_redirects=False, **kwargs)
    except requests.RequestException as exc:
        return None, f"Cannot reach the backend: {exc}"

    if reply.status_code >= 400:
        try:
            detail = reply.json().get("detail", "request failed")
        except ValueError:
            detail = "request failed"
        return None, f"{reply.status_code}: {detail}"

    return reply.json(), None


def start_session() -> None:

    st.session_state.setdefault("token", "")
    st.session_state.setdefault("chat_id", None)
    st.session_state.setdefault("messages", [])


def sidebar() -> None:

    with st.sidebar:
        st.title("📚 SnapRAG")
        st.caption(f"Backend: {BACKEND}")

        if not st.session_state.token:
            if st.button("Start a guest session", use_container_width=True):
                data, error = api("POST", "/auth/guest")
                if error:
                    st.error(error)
                else:
                    st.session_state.token = data["token"]
                    st.rerun()
            st.session_state.token = st.text_input("...or paste a token", type="password")
            return

        st.success("Signed in")

        upload = st.file_uploader("Upload a PDF", type="pdf")

        if upload and st.button("Index this PDF", use_container_width=True):
            with st.spinner("Extracting, embedding and indexing..."):
                _, error = api("POST", "/upload", files={"file": (upload.name, upload.getvalue(), "application/pdf")})
            st.error(error) if error else st.success("Indexed.")

        data, _ = api("GET", "/documents")
        documents = [d["name"] for d in (data or {}).get("documents", [])]

        if documents:
            active = (data or {}).get("active_document")
            choice = st.selectbox("Active document", documents, index=documents.index(active) if active in documents else 0)
            if choice != active:
                api("POST", f"/documents/active/{choice}")

        if st.button("New chat", use_container_width=True):
            data, error = api("POST", "/chats")
            if error:
                st.error(error)
            else:
                st.session_state.chat_id, st.session_state.messages = data["chat_id"], []

        if st.session_state.chat_id:
            st.caption(f"Chat: {st.session_state.chat_id[:8]}")

        if st.button("Sign out", use_container_width=True):
            st.session_state.token, st.session_state.chat_id, st.session_state.messages = "", None, []
            st.rerun()


def show_details(response: dict) -> None:
    """Confidence, citations and the memory used for one answer."""

    confidence = response.get("confidence")
    cols = st.columns(3)
    cols[0].metric("Confidence", f"{confidence:.2f}" if isinstance(confidence, (int, float)) else "-")
    cols[1].metric("Retrieval attempts", len(response.get("attempts") or []))
    verification = response.get("verification")
    cols[2].metric("Verified", str(verification.get("supported")) if isinstance(verification, dict) else str(verification))

    citations = response.get("citations") or []
    if citations:
        with st.expander(f"Citations ({len(citations)})"):
            st.json(citations)

    memory = response.get("retrieved_memory")
    if memory and memory.get("sentences"):
        with st.expander(f"Verified memory used (lambda = {memory.get('lambda')})"):
            for sentence in memory["sentences"]:
                st.write("- " + (sentence["text"] if isinstance(sentence, dict) else str(sentence)))


def chat_tab() -> None:

    if not st.session_state.chat_id:
        st.info("Press **New chat** in the sidebar, upload a PDF, and ask a question.")
        return

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("response"):
                show_details(message["response"])

    question = st.chat_input("Ask about the document")

    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving, answering and verifying..."):
            response, error = api("POST", "/query", json={"question": question, "chat_id": st.session_state.chat_id})

        if error:
            st.error(error)
            return

        st.markdown(response["answer"])
        show_details(response)

    st.session_state.messages.append({"role": "assistant", "content": response["answer"], "response": response})


def memory_tab() -> None:

    if not st.session_state.chat_id:
        st.info("Start a chat to see its memory.")
        return

    if st.button("Refresh memory"):
        st.rerun()

    data, error = api("GET", f"/chats/{st.session_state.chat_id}/memory")

    if error:
        st.error(error)
        return

    snapshots, cqm = data["snapshots"], data["cqm"]
    cols = st.columns(3)
    cols[0].metric("Snapshots", len(snapshots))
    cols[1].metric("Turns measured", len(cqm))
    cols[2].metric("State items", len(data["state"]))

    if cqm:
        st.subheader("Context quality (CIS) per turn")
        st.line_chart({"CIS": [r.get("cis") for r in cqm], "headroom": [r.get("headroom") for r in cqm]})

    st.subheader("Snapshots (hash-chained)")
    for snapshot in snapshots:
        meta = snapshot.get("meta", {})
        title = f"#{snapshot['seq']}  turns {snapshot['from_turn']}-{snapshot['to_turn']}  ·  {meta.get('compression', '?')}x compression  ·  {snapshot['state_hash'][:10]}"
        with st.expander(title):
            for facet, items in snapshot["facets"].items():
                if items:
                    st.markdown(f"**{facet}**")
                    for item in items:
                        st.write("- " + (item if isinstance(item, str) else str(item)))
            st.caption(f"previous hash: {(snapshot.get('prev_hash') or 'none')[:16]}")

    st.subheader("Conversation state (facts and corrections)")
    st.json(data["state"])


def main() -> None:

    start_session()
    sidebar()

    if not st.session_state.token:
        st.title("SnapRAG")
        st.write("Self-improving snapshot memory with confidence-gated hybrid retrieval for long-conversation document QA.")
        st.info("Start a guest session in the sidebar.")
        return

    chat, memory = st.tabs(["Chat", "Memory"])

    with chat:
        chat_tab()

    with memory:
        memory_tab()


main()
