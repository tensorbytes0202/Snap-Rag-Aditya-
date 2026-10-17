import os
import time

import streamlit as st

from api import ApiError, SnapRagClient, ping, start_guest

st.set_page_config(
    page_title="SnapRAG",
    page_icon=":material/neurology:",
    layout="centered",
)

# The backend URL comes from the environment (or Streamlit secrets in the
# cloud), never from the page, so reviewers only see the product.
API_URL = os.getenv("SNAPRAG_API_URL", "http://localhost:8003")
GITHUB_URL = "https://github.com/tensorbytes0202/Snap-Rag-Aditya-"

SUGGESTIONS = {
    ":material/summarize: Summarize it": "Summarize this document in five bullet points",
    ":material/key: Key concepts": "What are the key concepts covered?",
    ":material/menu_book: Definitions": "List the important definitions",
    ":material/stairs: Step by step": "Explain the main topic step by step",
}

FEATURES = [
    (":material/document_scanner:", "Reads every page",
     "Text, screenshots and diagrams are all indexed; image-heavy pages are read by a vision model."),
    (":material/manage_search:", "Hybrid search",
     "Dense and keyword search, fused and reranked, with page citations on every answer."),
    (":material/verified:", "Verified memory",
     "Long chats are compressed into snapshots; anything recalled is checked against the source first."),
]

for key, default in {
    "token": os.getenv("SNAPRAG_TOKEN", ""),
    "messages": [],
    "chat_id": None,
    "docs": None,
    "usage": None,
    "notice": None,
    "uploader_n": 0,
    "stream": True,
}.items():
    st.session_state.setdefault(key, default)


# ---------- helpers ----------

@st.cache_data(ttl=10, show_spinner=False)
def backend_up() -> bool:
    return ping(API_URL)


def client() -> SnapRagClient:
    return SnapRagClient(API_URL, st.session_state.token)


def sign_out(notice: str | None = None):
    st.session_state.token = ""
    st.session_state.messages = []
    st.session_state.chat_id = None
    st.session_state.docs = None
    st.session_state.usage = None
    st.session_state.notice = notice


def new_chat():
    st.session_state.messages = []
    st.session_state.chat_id = None


def resume_chat(chat_id: str):
    """Reload a past chat: its turns become the visible conversation and
    later questions continue it (its memory is recalled server-side)."""

    try:
        turns = client().messages(chat_id)
    except ApiError as exc:
        st.session_state.notice = handle(exc)
        return

    messages = []
    for turn in turns:
        messages.append({"role": "user", "content": turn["question"]})
        messages.append({
            "role": "assistant", "content": turn["answer"], "confidence": turn.get("confidence"),
            "verified": turn.get("verified"), "citations": turn.get("citations") or [],
        })

    st.session_state.messages = messages
    st.session_state.chat_id = chat_id


def handle(exc: ApiError) -> str:
    """React to an API error and return the message to show."""

    if exc.status == 401:
        sign_out("Your session has expired. Start a new one to continue.")
        st.rerun()

    if exc.status == 429:
        refresh()

    return str(exc)


def refresh():
    """Reload the caller's documents and remaining demo allowance."""

    try:
        st.session_state.docs = client().documents()
        st.session_state.usage = client().usage()
    except ApiError as exc:
        if exc.status == 401:
            sign_out("That access token is not valid. Start a demo or paste a valid token.")
        else:
            st.session_state.notice = str(exc)


def choose_document():
    try:
        client().select_document(st.session_state.doc_choice)
    except ApiError as exc:
        st.session_state.notice = handle(exc)
    refresh()


def current_chat() -> str | None:
    """The server-side chat for this conversation, created on the first
    question. None if that fails: the question is still answered, just
    without history or memory."""

    if st.session_state.chat_id is None:
        try:
            st.session_state.chat_id = client().create_chat()
        except ApiError:
            return None

    return st.session_state.chat_id


def ask(question: str):
    st.session_state.pending = question


def pick_suggestion():
    if st.session_state.suggestion:
        ask(SUGGESTIONS[st.session_state.suggestion])
    st.session_state.suggestion = None


def describe_step(event: dict) -> str:
    detail = event.get("detail")
    extra = ", ".join(f"{k}: {v}" for k, v in detail.items()) if detail else ""
    return event["message"] + (f" ({extra})" if extra else "")


def confidence_badge(value):
    pct = value * 100 if value <= 1 else value
    color = "green" if pct >= 80 else "orange" if pct >= 55 else "red"
    st.badge(f"Confidence {pct:.0f}%", icon=":material/speed:", color=color)


# ---------- welcome (no token yet) ----------

def welcome():
    st.space("medium")

    with st.container(horizontal_alignment="center"):
        st.title("SnapRAG", icon=":material/neurology:", text_alignment="center")
        st.markdown("#### Chat with your PDFs, with memory you can trust", text_alignment="center")
        st.caption(
            "Upload a document and ask questions in a conversation. Answers cite their pages, and "
            "long chats are remembered through verified snapshots instead of being forgotten.",
            text_alignment="center",
        )

    st.space("small")

    for column, (icon, title, body) in zip(st.columns(3), FEATURES):
        with column.container(border=True, height="stretch"):
            st.markdown(f"### {icon}")
            st.markdown(f"**{title}**")
            st.caption(body)

    st.space("small")

    with st.container(horizontal_alignment="center"):
        if st.button("Start free demo", type="primary", icon=":material/rocket_launch:"):
            try:
                session = start_guest(API_URL, st.context.ip_address)
            except ApiError as exc:
                st.error(str(exc), icon=":material/error:")
            else:
                st.session_state.token = session["token"]
                st.session_state.usage = session["limits"]
                st.rerun()
        st.caption("No sign-up needed. A demo session includes a small allowance of questions and uploads.",
                   text_alignment="center")

    with st.expander("I have an access token", icon=":material/key:"):
        with st.form("token_form", border=False):
            token = st.text_input("Access token", type="password", placeholder="Paste your token",
                                  label_visibility="collapsed")
            if st.form_submit_button("Connect", icon=":material/login:"):
                st.session_state.token = token.strip()
                st.rerun()

    if not backend_up():
        st.warning("The service is offline right now. Please try again in a few minutes.",
                   icon=":material/cloud_off:")

    st.caption(f"[Source code on GitHub]({GITHUB_URL})", text_alignment="center")


if notice := st.session_state.pop("notice", None):
    st.toast(notice, icon=":material/info:")

if not st.session_state.token:
    welcome()
    st.stop()

if st.session_state.docs is None:
    refresh()
    if not st.session_state.token:   # refresh signed us out (bad token)
        st.rerun()


# ---------- answering ----------

def answer_stream(question: str, chat_id: str | None) -> dict:
    """Stream the answer live; progress steps go into a compact status above it."""

    steps, errors, memory, check, revised, citations = [], [], {}, {}, {}, []
    started = time.perf_counter()
    status = st.status(":shimmer[Searching the document]", type="compact")
    box = st.empty()

    def tokens():
        try:
            for event in client().stream(question, chat_id):
                if event["type"] == "token":
                    yield event["token"]
                elif event["type"] == "step":
                    steps.append(describe_step(event))
                    status.write(steps[-1])
                    if event["message"] == "Memory recalled":
                        memory.update(event.get("detail") or {})
                elif event["type"] == "citations":
                    citations[:] = event["citations"]
                elif event["type"] == "verification":
                    check.update(event)
                elif event["type"] == "revised":
                    revised.update(event)
                elif event["type"] == "error":
                    errors.append(event["message"])
        except ApiError as exc:
            errors.append(handle(exc))

    with box.container():
        text = st.write_stream(tokens())

    content = str(text) if text else ""

    # A low-confidence answer is searched for again; the better one replaces it.
    if revised:
        content = revised["answer"]
        citations[:] = revised.get("citations") or citations
        box.markdown(content)
        check.update(supported=revised.get("supported"), confidence=revised.get("confidence"))

    elapsed = time.perf_counter() - started
    status.update(
        label=f"Finished in {elapsed:.1f}s" if not errors else "Stopped with an error",
        state="error" if errors else "complete",
    )
    for message in errors:
        st.error(message, icon=":material/error:")

    reply = {"role": "assistant", "content": content, "steps": steps, "elapsed": elapsed,
             "memory_count": memory.get("sentences", 0) + memory.get("state_items", 0),
             "confidence": check.get("confidence"), "verified": check.get("supported"),
             "revised": bool(revised), "citations": citations,
             "error": "; ".join(errors) or None}

    render_badges(reply)

    return reply


def answer_batch(question: str, chat_id: str | None) -> dict:
    started = time.perf_counter()
    with st.spinner("Searching and writing an answer..."):
        try:
            result = client().query(question, chat_id)
        except ApiError as exc:
            return {"role": "assistant", "content": "", "error": handle(exc)}

    verification = result.get("verification")
    memory = result.get("retrieved_memory") or {}
    return {
        "role": "assistant",
        "content": result.get("answer", ""),
        "elapsed": time.perf_counter() - started,
        "confidence": result.get("confidence"),
        "verified": verification.get("supported") if isinstance(verification, dict) else None,
        "citations": result.get("citations") or [],
        "sources": result.get("sources") or [],
        "followups": result.get("followup_questions") or [],
        "memory": memory.get("sentences") or [],
        "memory_count": len(memory.get("sentences") or []),
    }


def render_badges(message: dict):
    with st.container(horizontal=True):
        if message.get("confidence") is not None:
            confidence_badge(message["confidence"])
        if message.get("verified") is not None:
            st.badge("Verified" if message["verified"] else "Not verified",
                     icon=":material/verified:" if message["verified"] else ":material/gpp_maybe:",
                     color="green" if message["verified"] else "red")
        if message.get("revised"):
            st.badge("Revised after a second search", icon=":material/refresh:", color="orange")
        if message.get("memory_count"):
            st.badge(f"Used {message['memory_count']} verified memory facts", icon=":material/neurology:",
                     color="violet")


def render_assistant(index: int, message: dict):
    if message.get("steps"):
        with st.expander(f"Searched in {message['elapsed']:.1f}s", type="compact"):
            for step in message["steps"]:
                st.caption(step)

    if message.get("content"):
        st.markdown(message["content"])
    if message.get("error"):
        st.error(message["error"], icon=":material/error:")

    render_badges(message)

    if message.get("memory"):
        with st.expander("Verified memory used", icon=":material/neurology:"):
            for item in message["memory"]:
                st.markdown(f"- {item['text']}  :gray-badge[{item.get('source') or 'memory'}]")

    if message.get("citations") or message.get("sources"):
        with st.expander("Sources", icon=":material/menu_book:"):
            for c in message.get("citations", []):
                st.markdown(f"**[{c['source_id']}]** {c['document']}, page {c['page']}")
            for s in message.get("sources", []):
                st.caption(f"{s['document']}, page {s['page']}")
                st.text((s.get("text") or "")[:400])

    if message.get("followups"):
        with st.container(horizontal=True):
            for j, question in enumerate(message["followups"]):
                st.button(question, key=f"followup_{index}_{j}", icon=":material/subdirectory_arrow_right:",
                          type="tertiary", on_click=ask, args=(question,))


# ---------- sidebar ----------

docs = st.session_state.docs or {"documents": [], "active_document": None}
usage = st.session_state.usage or {"guest": False}
guest = usage.get("guest", False)
questions_left = usage["query_limit"] - usage["queries_used"] if guest else None
uploads_left = usage["upload_limit"] - usage["uploads_used"] if guest else None

with st.sidebar:
    st.title("SnapRAG", icon=":material/neurology:")

    if guest:
        st.badge("Demo session", icon=":material/science:", color="violet")
        st.progress(usage["queries_used"] / usage["query_limit"],
                    text=f"Questions: {usage['queries_used']} of {usage['query_limit']} used")
        st.progress(usage["uploads_used"] / usage["upload_limit"],
                    text=f"PDF uploads: {usage['uploads_used']} of {usage['upload_limit']} used")
    else:
        st.badge("Full access", icon=":material/verified_user:", color="green")

    st.subheader("Documents", icon=":material/folder:")
    names = [d["name"] for d in docs["documents"]]
    if names:
        st.session_state.doc_choice = docs["active_document"] if docs["active_document"] in names else None
        st.radio("Active document", names, index=None, key="doc_choice", on_change=choose_document,
                 captions=[f"{d['pages']} pages · {d['chunks']} chunks" for d in docs["documents"]],
                 label_visibility="collapsed", width="stretch")
    else:
        st.caption("No documents yet. Add a PDF below.")

    with st.expander("Add a PDF", icon=":material/upload_file:", expanded=not names):
        file = st.file_uploader("PDF file", type="pdf", key=f"pdf_{st.session_state.uploader_n}",
                                label_visibility="collapsed", disabled=uploads_left == 0)
        if guest:
            st.caption(f"Up to {usage['max_pages']} pages per PDF.")
        if st.button("Index document", type="primary", icon=":material/database:",
                     disabled=file is None or uploads_left == 0, width="stretch"):
            with st.spinner("Indexing... pages with images take a little longer"):
                try:
                    result = client().upload(file.name, file.getvalue())
                except ApiError as exc:
                    st.error(handle(exc), icon=":material/error:")
                else:
                    st.session_state.notice = (f"Indexed {result['filename']}: "
                                               f"{result['pages']} pages, {result['total_chunks']} chunks")
                    st.session_state.uploader_n += 1
                    refresh()
                    st.rerun()

    st.toggle("Stream answers", key="stream",
              help="On: the answer appears live. Off: wait for the full answer with sources, confidence and follow-ups.")

    with st.container(horizontal=True):
        st.button("New chat", icon=":material/add_comment:", on_click=new_chat,
                  disabled=not st.session_state.messages)
        st.button("Sign out", icon=":material/logout:", on_click=sign_out)

    if st.session_state.chat_id:
        with st.expander("Where we left off", icon=":material/bookmark:"):
            try:
                st.text(client().handoff(st.session_state.chat_id)["handoff"] or "Nothing yet.")
            except ApiError as exc:
                st.caption(str(exc))

    with st.expander("Past chats", icon=":material/history:"):
        try:
            past = [c for c in client().chats() if c["turns"] and c["chat_id"] != st.session_state.chat_id][:8]
        except ApiError:
            past = []
        if not past:
            st.caption("No earlier chats.")
        for chat in past:
            st.button(f"{chat['title'][:40]} ({chat['turns']})", key=f"resume_{chat['chat_id']}",
                      icon=":material/replay:", type="tertiary", on_click=resume_chat, args=(chat["chat_id"],))

    st.caption(f"[Source code on GitHub]({GITHUB_URL})")


# ---------- main ----------

active = docs["active_document"] if docs["active_document"] in names else None
out_of_questions = questions_left is not None and questions_left <= 0

# Read the input first: the intro card is only drawn when nothing is being asked,
# otherwise it would stay on screen above the first answer until the next rerun.
placeholder = ("Demo limit reached" if out_of_questions
               else "Ask about your document" if active else "Add a PDF to start asking")
prompt = st.chat_input(placeholder, submit_mode="disable", disabled=out_of_questions or not active) \
    or st.session_state.pop("pending", None)

with st.container(horizontal=True, vertical_alignment="center"):
    st.header("Chat", icon=":material/forum:")
    if active:
        st.badge(active, icon=":material/description:", color="blue")
    if guest and not out_of_questions:
        st.badge(f"{questions_left} questions left", icon=":material/hourglass_top:", color="gray")

if not names:
    with st.container(border=True):
        st.subheader("Start with a PDF", icon=":material/upload_file:")
        st.markdown(
            "1. Open **Add a PDF** in the sidebar and choose a file.\n"
            "2. Press **Index document**; pages with screenshots are read by a vision model.\n"
            "3. Ask questions here. Follow-ups work, and answers cite their pages."
        )
elif not st.session_state.messages and not prompt and not out_of_questions:
    with st.container(border=True):
        st.markdown("#### Ask anything about your document")
        st.caption("Try one of these, or type your own question below.")
        st.pills("Try asking", list(SUGGESTIONS), key="suggestion", on_change=pick_suggestion,
                 label_visibility="collapsed")

for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            render_assistant(i, message)

if out_of_questions:
    st.info(f"You've used all {usage['query_limit']} demo questions. Thanks for trying SnapRAG! "
            f"The full project is [on GitHub]({GITHUB_URL}).", icon=":material/celebration:")

if prompt and active and not out_of_questions:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        chat_id = current_chat()
        if st.session_state.stream:
            reply = answer_stream(prompt, chat_id)
        else:
            reply = answer_batch(prompt, chat_id)
            render_assistant(len(st.session_state.messages), reply)

    st.session_state.messages.append(reply)

    if guest:
        refresh()
        st.rerun()
