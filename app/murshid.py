"""Murshid — the runnable system.

This is the same architecture the notebook demonstrates, packaged to run
locally. The notebook remains the graded evidence; this module is the app.

    from app.murshid import build
    murshid, info = build()
    result = murshid.invoke({"student_id": "s2201", "question": "..."},
                            {"configurable": {"thread_id": "abc"}})
"""

from __future__ import annotations

import inspect
import os
import time
from pathlib import Path
from typing import Literal

import groq
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import entrypoint, task
from langgraph.store.memory import InMemoryStore
from langgraph.types import RetryPolicy, interrupt
from pydantic import BaseModel, Field, ValidationError

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CHROMA_DIR = ROOT / ".chroma"

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
PROGRAMME_TOTAL_CREDITS = 132
MAX_CONTEXT_CHARS_PER_SOURCE = 1800
NOT_FOUND_AR = ("لم أجد ذلك في وثائق جامعة النور المتاحة لديّ. "
                "يُرجى التواصل مع خدمات الطلاب مباشرة.")
NOT_FOUND_EN = ("I could not find that in the Al-Noor University documents I "
                "have access to. Please contact Student Services directly.")

# Smallest capable first: Groq's free tier allows 8,000 tokens per minute PER
# MODEL, and nothing here is hard reasoning — it is classification, extraction
# and summarising retrieved text.
PREFERRED_MODELS = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "qwen/qwen3.6-27b",
    "llama-3.3-70b-versatile",
]

# Module-level handles, populated by build().
llm = None
academic_retriever = None
campus_retriever = None
router = None
REGISTRY_LOG: list[dict] = []

# Pending/resolved course-withdrawal and appeal tickets, keyed by thread_id.
# Lets the UI show an advisor queue across every conversation the process has
# seen, not just the one the current browser session happens to be on.
APPROVALS: dict[str, dict] = {}


# ─────────────────────────────── setup ────────────────────────────────────
def pick_model() -> str:
    """Ask the account which models it can use rather than hardcoding one.

    Groq retired llama-3.3-70b-versatile (the course default) on 2026-08-16;
    hardcoding a replacement just moves the problem.
    """
    available = {m.id for m in groq.Groq().models.list().data}
    for m in PREFERRED_MODELS:
        if m in available:
            return m
    raise RuntimeError(
        "None of the preferred models are available to this key.\n"
        f"Available: {sorted(available)}"
    )


def build_stores(embeddings, persist: bool = True):
    """Load -> split -> embed -> store. Two SEPARATE collections."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=80,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""])

    retrievers, counts = {}, {}
    for name in ("academic", "campus"):
        folder = DATA_DIR / name
        if not folder.is_dir():
            raise FileNotFoundError(f"{folder} not found — run from the repo root.")
        docs = DirectoryLoader(str(folder), glob="**/*.md", loader_cls=TextLoader,
                               loader_kwargs={"encoding": "utf-8"}).load()
        for d in docs:
            d.metadata["source"] = Path(d.metadata.get("source", "?")).name
            d.metadata["collection"] = name
        chunks = splitter.split_documents(docs)
        kwargs = {"collection_name": name}
        if persist:
            CHROMA_DIR.mkdir(exist_ok=True)
            kwargs["persist_directory"] = str(CHROMA_DIR)
        store = Chroma.from_documents(chunks, embeddings, **kwargs)
        retrievers[name] = store.as_retriever(search_kwargs={"k": 3})
        counts[name] = (len(docs), len(chunks))
    return retrievers, counts


# ──────────────────────────────── tools ───────────────────────────────────
@tool
def compute_gpa(grades: list[float], credits: list[int]) -> str:
    """Compute a credit-weighted GPA from grade points and matching credit hours."""
    if len(grades) != len(credits):
        raise ValueError("grades and credits must be the same length")
    if not credits or sum(credits) == 0:
        raise ValueError("credits must contain at least one non-zero value")
    total_points = sum(g * c for g, c in zip(grades, credits))
    total_credits = sum(credits)
    gpa = total_points / total_credits
    standing = ("Dean's List range" if gpa >= 3.75 else
                "Good Standing" if gpa >= 2.00 else
                "below the 2.00 threshold — academic probation applies")
    return (f"GPA {gpa:.2f} across {total_credits} credit hours "
            f"({total_points:.1f} total grade points) — {standing}.")


@tool
def credits_to_graduate(completed_credits: int) -> str:
    """How many credit hours a student still needs, and roughly how many terms."""
    if completed_credits < 0:
        raise ValueError("completed_credits cannot be negative")
    remaining = max(PROGRAMME_TOTAL_CREDITS - completed_credits, 0)
    if remaining == 0:
        return (f"{completed_credits} of {PROGRAMME_TOTAL_CREDITS} credit hours "
                f"complete — the credit-hour requirement is already met.")
    terms = -(-remaining // 15)
    return (f"{remaining} credit hours remaining of {PROGRAMME_TOTAL_CREDITS} "
            f"— about {terms} more full-time term(s) at 15 hours each.")


@tool
def search_academic_regulations(query: str) -> str:
    """Search academic regulations: grading, GPA, appeals, attendance,
    withdrawal, probation, transcripts, graduation, examinations."""
    docs = academic_retriever.invoke(query)
    if not docs:
        return "No matching academic regulation found."
    return "\n\n".join(f"[{d.metadata['source']}] {d.page_content}" for d in docs)


@tool
def search_campus_services(query: str) -> str:
    """Search campus services: library, IT helpdesk, student portal, housing,
    dining, careers, wellbeing, clubs and societies."""
    docs = campus_retriever.invoke(query)
    if not docs:
        return "No matching campus service page found."
    return "\n\n".join(f"[{d.metadata['source']}] {d.page_content}" for d in docs)


# ─────────────────────────────── schemas ──────────────────────────────────
class MurshidRoute(BaseModel):
    """Where a student question should be handled."""

    destination: Literal["academic", "campus", "both", "action"] = Field(
        description=(
            "Which specialist should handle this question. "
            "Pick 'academic' for grades, GPA, exams, attendance, transcripts, "
            "probation, withdrawal rules and deadlines, graduation requirements. "
            "Pick 'campus' for the library, IT helpdesk, student portal login, "
            "housing, dining, careers, wellbeing, clubs. "
            "Pick 'both' when the question genuinely needs each source. "
            "Pick 'action' when the student is asking to FILE or SUBMIT something "
            "now: a withdrawal, a grade appeal, a formal request. "
            "Asking HOW to withdraw is 'academic'; asking TO withdraw is 'action'."))
    reason: str = Field(description="One short sentence justifying the destination")
    language: Literal["ar", "en"] = Field(
        description="Which language the student wrote in. Use 'ar' for Arabic, 'en' for English.")
    confidence: Literal["high", "low"] = Field(
        description=("How certain you are about the destination. Use 'high' when "
                     "the right source is obvious. Use 'low' when the question is "
                     "ambiguous or underspecified."))


class ActionRequest(BaseModel):
    """What the student is asking us to file."""
    action_type: Literal["withdrawal", "appeal"]
    course_code: str | None = Field(
        default=None, description="e.g. STAT301; null if the student did not say")
    reason: str = Field(description="The student's stated reason, in their words")


class BilingualAnswer(BaseModel):
    """The same answer, written naturally in each language — not a literal
    translation of one into the other."""
    answer_ar: str = Field(description="The answer in Arabic")
    answer_en: str = Field(description="The same answer in English")


class Query(BaseModel):
    """Rejects malformed input before it reaches an LLM call."""
    student_id: str = Field(min_length=3, max_length=20)
    question: str = Field(min_length=1, max_length=2000)
    thread_id: str | None = Field(
        default=None, description="Conversation thread, echoed into the advisor "
                                   "queue so a pending ticket can be resumed.")


# ───────────────────────── memory (two lifetimes) ─────────────────────────
checkpointer = InMemorySaver()   # short-term: the run, keyed by thread_id
store = InMemoryStore()          # long-term: durable facts, keyed by namespace


def remember(student_id: str, key: str, value):
    store.put(("students", student_id), key, {"value": value})


def recall(student_id: str, key: str, default=None):
    item = store.get(("students", student_id), key)
    return item.value["value"] if item else default


def load_student_profile(student_id: str) -> dict:
    return {"preferred_language": recall(student_id, "preferred_language"),
            "questions_asked": recall(student_id, "questions_asked", 0),
            "last_topic": recall(student_id, "last_topic"),
            "thread_history": recall(student_id, "thread_history", [])}


# ─────────────────────── retry policy (transient errors) ──────────────────
RETRY_KW = ("retry_policy" if "retry_policy" in inspect.signature(task).parameters
            else "retry")

# LangGraph's default retry_on does not cover provider SDK exceptions, so the
# transient types are named explicitly. The backoff must be able to outlast a
# 60-second rate-limit window: 5+10+20+40 does, 2+4+8+16 does not.
TRANSIENT = tuple(e for e in (
    getattr(groq, "RateLimitError", None),
    getattr(groq, "APIConnectionError", None),
    getattr(groq, "APITimeoutError", None),
    getattr(groq, "InternalServerError", None),
    ConnectionError,
) if e is not None)

RETRY = {RETRY_KW: RetryPolicy(max_attempts=6, initial_interval=5.0,
                               backoff_factor=2.0, max_interval=45.0,
                               retry_on=TRANSIENT)}

# ValidationError = malformed object. groq.BadRequestError = the provider
# rejected the tool call before Pydantic ever saw it. Both are the LLM's
# mistake; anything else is a genuine bug and is left to bubble up.
RECOVERABLE = (ValidationError, groq.BadRequestError)


# ──────────────────────────────── tasks ───────────────────────────────────
@task(**RETRY)
def classify(question: str) -> MurshidRoute:
    """Route the question. On invalid model output, re-prompt WITH the error."""
    feedback = ""
    for attempt in range(3):
        try:
            return router.invoke(f"Route this student question.{feedback}\n\n{question}")
        except RECOVERABLE as e:
            print(f"  [classify] attempt {attempt + 1} rejected ({type(e).__name__})")
            feedback = (f"\n\nYour previous answer was rejected: {e}\n"
                        f"Return an object with EXACTLY these four keys: "
                        f"destination, reason, language, confidence. "
                        f"Never use a value as a key name.")
    print("  [classify] exhausted retries -> searching both sources")
    return MurshidRoute(destination="both", language="en", confidence="low",
                        reason="classifier could not produce valid output")


@task(**RETRY)
def retrieve_one(question: str, source: str) -> dict:
    """Retrieve from ONE store. Returns both the joined context and the
    filenames actually matched, so the UI can show real sources, not just
    which store was searched."""
    retriever = academic_retriever if source == "academic" else campus_retriever
    docs = retriever.invoke(question)
    if not docs:
        return {"context": "", "sources": []}
    joined = "\n\n".join(f"[{d.metadata['source']}] {d.page_content}" for d in docs)
    names = [d.metadata["source"] for d in docs]
    return {"context": joined[:MAX_CONTEXT_CHARS_PER_SOURCE], "sources": names}


@task(**RETRY)
def synthesize(question: str, context: str) -> BilingualAnswer:
    """Write the student-facing answer from the retrieved context only —
    always in both languages, since the interface shows them side by side.

    The structured-output tool call is unreliable on longer prompts (observed
    on the merged "both stores" context): the model sometimes answers in
    plain prose instead of calling the tool, or calls it in a form the
    provider or the parser then rejects — inconsistent exception types
    across runs (groq.BadRequestError one time, a LangChain tool-parsing
    error another). Retry with feedback first — same pattern as classify()
    — then fall back to two plain-text calls, which don't depend on tool-
    calling at all. Caught broadly here (not narrowed to RECOVERABLE)
    precisely because that failure mode isn't one fixed exception type and
    a solid non-tool-calling fallback exists right below.
    """
    if not context.strip():
        return BilingualAnswer(answer_ar=NOT_FOUND_AR, answer_en=NOT_FOUND_EN)

    prompt = (
        "You are Murshid, an Al-Noor University student services assistant.\n"
        "Answer using ONLY the context below. If it does not contain the "
        "answer, say so plainly. Cite the source file in square brackets.\n"
        "Provide the answer in BOTH Arabic and English — each natural in its "
        "own language, not a literal translation of the other.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {question}"
    )
    for attempt in range(3):
        try:
            return llm.with_structured_output(BilingualAnswer).invoke(prompt)
        except Exception as e:
            print(f"  [synthesize] attempt {attempt + 1} rejected ({type(e).__name__})")
            prompt += (f"\n\nYour previous answer was rejected: {e}\n"
                      f"You MUST call the tool with both answer_ar and answer_en "
                      f"filled in — do not reply with plain text.")

    print("  [synthesize] exhausted structured-output retries -> plain-text fallback")
    ar = llm.invoke(f"Answer this question in Arabic only, using ONLY the context. Cite "
                    f"the source file in square brackets.\n\nCONTEXT:\n{context}\n\n"
                    f"QUESTION: {question}").content
    en = llm.invoke(f"Answer this question in English only, using ONLY the context. Cite "
                    f"the source file in square brackets.\n\nCONTEXT:\n{context}\n\n"
                    f"QUESTION: {question}").content
    return BilingualAnswer(answer_ar=ar, answer_en=en)


@task(**RETRY)
def normalize_for_retrieval(question: str, language: str) -> str:
    """English retrieval query for an Arabic question.

    The document stores are English-only and search runs on multilingual
    embeddings, which tend to match on surface/lexical overlap rather than
    intent — a literal, word-for-word rendering of the question can retrieve
    the wrong chunks even when the documents cover it. Restating the
    question's MEANING in plain English query terms fixes retrieval without
    touching the language the student is answered in.
    """
    if language != "ar":
        return question
    return llm.invoke(
        "Restate this student's question as a natural English question, using "
        "plain university-services terminology. Capture the MEANING and "
        "INTENT behind it — this is not a word-for-word translation. Return "
        "ONLY the restated question.\n\n"
        f"STUDENT'S QUESTION (Arabic): {question}"
    ).content.strip()


@task(**RETRY)
def rewrite_followup(question: str, history: list) -> str:
    """Turn a follow-up into a standalone question using this thread's history."""
    convo = "\n".join(f"Student: {h['q']}\nMurshid: {h['a'][:200]}" for h in history[-3:])
    rewritten = llm.invoke(
        "Rewrite the student's latest message as a standalone question, resolving "
        "any pronouns from the conversation. If it is already standalone, return "
        "it unchanged. Return ONLY the question text.\n\n"
        f"CONVERSATION SO FAR:\n{convo}\n\nLATEST MESSAGE: {question}"
    ).content.strip()
    return rewritten or question


@task(**RETRY)
def parse_action(question: str) -> ActionRequest:
    """Extract the formal request. Degrades into a human question on failure."""
    try:
        return llm.with_structured_output(ActionRequest).invoke(
            f"Extract the formal request from this student message. If no course "
            f"code is named, leave course_code null.\n\n{question}")
    except RECOVERABLE as e:
        print(f"  [parse_action] extraction failed ({type(e).__name__})")
        return ActionRequest(action_type="withdrawal", course_code=None, reason=question)


@task
def get_course_code(code: str | None) -> str:
    """Pause and ask if the student did not name a course."""
    if not code:
        code = interrupt({"message": "Which course code do you want to withdraw from?",
                          "field": "course_code"})
    return str(code).upper().strip()


@task
def request_approval(payload: dict) -> dict:
    """Pause until a student advisor approves, edits, or rejects."""
    return interrupt({"action": "A student advisor must approve this before it is filed",
                      **payload})


@task
def submit_withdrawal(student_id: str, course_code: str, reason: str) -> str:
    """File a course withdrawal. IRREVERSIBLE once the Registrar processes it."""
    ref = f"WD-{student_id}-{course_code}"
    REGISTRY_LOG.append({"ref": ref, "type": "withdrawal", "student": student_id,
                         "course": course_code, "reason": reason})
    return ref


def _policy_note(action_type: str) -> str:
    """A real, retrieved snippet of the governing policy for an approval
    ticket — not a fabricated deadline. Best-effort: falls back to a plain
    pointer if nothing is retrieved."""
    query = ("course withdrawal deadline" if action_type == "withdrawal"
             else "grade appeal deadline procedure")
    try:
        docs = academic_retriever.invoke(query)
    except Exception:
        docs = []
    if not docs:
        return "See the relevant academic regulation for the current deadline."
    snippet = docs[0].page_content.strip().replace("\n", " ")
    return (snippet[:240] + "…") if len(snippet) > 240 else snippet


# ────────────────────────────── the workflow ──────────────────────────────
@entrypoint(checkpointer=checkpointer, store=store)
def murshid(inputs: dict, *, previous: list | None = None):
    t0 = time.perf_counter()
    q = Query(**inputs)
    thread_key = q.thread_id or "default"
    history = previous or []                       # SHORT-TERM: this thread
    profile = load_student_profile(q.student_id)   # LONG-TERM: this student

    steps: list[dict] = []

    def stamp(name: str, detail: str) -> None:
        steps.append({"name": name, "detail": detail})

    stamp("Query()", f"Pydantic guardrail: text ok, student_id={q.student_id}")
    stamp("load_profile", f'Store read ("students","{q.student_id}") → '
                          f'preferred_language={profile["preferred_language"]}, '
                          f'questions_asked={profile["questions_asked"]}')

    question = q.question
    if history:
        question = rewrite_followup(q.question, history).result()
        stamp("rewrite_followup",
              f'resolved to: "{question}"' if question != q.question else "already standalone")

    route = classify(question).result()
    stamp("classify", f'MurshidRoute(destination="{route.destination}", '
                      f'confidence="{route.confidence}")')

    sources_meta: list[dict] = []

    if route.destination == "action":
        stamp("transfer_to_action_agent", "supervisor handoff — retrieval skipped, "
                                          "this is a request to file")
        req = parse_action(question).result()
        stamp("parse_action", f'action_type="{req.action_type}", '
                              f'course_code={req.course_code or "—"}')
        course = get_course_code(req.course_code).result()
        stamp("get_course_code", f"course={course}")

        APPROVALS[thread_key] = {
            "thread_id": thread_key,
            "student_id": q.student_id,
            "status": "pending",
            "action_type": req.action_type,
            "course_code": course,
            "student_stated_reason": req.reason,
            "policy_note": _policy_note(req.action_type),
            "filed_reason": "",
            "receipt": "",
        }
        stamp("interrupt()", "run paused in the checkpointer, awaiting advisor decision")

        decision = request_approval({
            "type": req.action_type, "student_id": q.student_id,
            "course_code": course, "student_stated_reason": req.reason,
        }).result()

        if not decision.get("approved"):
            note = decision.get("note", "no reason given")
            answer = "Your request was not filed. Advisor note: " + note
            outcome = "rejected_by_advisor"
            APPROVALS[thread_key]["status"] = "rejected"
            APPROVALS[thread_key]["filed_reason"] = answer
        else:
            edited_reason = decision.get("edited_reason", req.reason)
            ref = submit_withdrawal(
                q.student_id, decision.get("course_code", course), edited_reason).result()
            answer = f"Withdrawal {ref} filed for {course}. Recorded reason: {edited_reason}"
            outcome = "filed"
            APPROVALS[thread_key]["status"] = "approved"
            APPROVALS[thread_key]["filed_reason"] = edited_reason
            APPROVALS[thread_key]["receipt"] = ref
        searched = []

    elif route.destination == "both" or route.confidence == "low":
        search_query = normalize_for_retrieval(question, route.language).result()
        a_fut = retrieve_one(search_query, "academic")  # launched...
        c_fut = retrieve_one(search_query, "campus")    # ...concurrently
        stamp("search_academic ∥ search_campus",
              "both Chroma collections queried concurrently, k=3 each, contexts merged")
        a_res, c_res = a_fut.result(), c_fut.result()
        context = a_res["context"] + "\n\n" + c_res["context"]
        sources_meta = [{"source": s, "collection": "academic"} for s in a_res["sources"]]
        sources_meta += [{"source": s, "collection": "campus"} for s in c_res["sources"]]
        searched = ["academic", "campus"]
        bilingual = synthesize(question, context).result()
        stamp("synthesize_answer", "bilingual answer generated from merged context")
        outcome = "answered"

    else:
        search_query = normalize_for_retrieval(question, route.language).result()
        stamp(f"search_{route.destination}",
              f'Chroma collection "{route.destination}", k=3')
        res = retrieve_one(search_query, route.destination).result()
        context = res["context"]
        sources_meta = [{"source": s, "collection": route.destination} for s in res["sources"]]
        searched = [route.destination]
        if not context.strip():                        # retrieval validation
            other = "campus" if route.destination == "academic" else "academic"
            stamp("validate_retrieval", f"no chunks in {route.destination} — re-routed to {other}")
            res = retrieve_one(search_query, other).result()
            context = res["context"]
            sources_meta = [{"source": s, "collection": other} for s in res["sources"]]
            searched.append(other)
        else:
            stamp("validate_retrieval", "chunks relevant — no re-route needed")
        bilingual = synthesize(question, context).result()
        stamp("synthesize_answer", "bilingual answer generated from retrieved context")
        outcome = "answered"

    asked = profile["questions_asked"] + 1
    thread_history = (profile["thread_history"] +
                      [{"thread_id": thread_key, "question": q.question,
                        "route": route.destination}])[-8:]
    remember(q.student_id, "questions_asked", asked)
    remember(q.student_id, "preferred_language", route.language)
    remember(q.student_id, "last_topic", route.destination)
    remember(q.student_id, "thread_history", thread_history)
    stamp("save_profile", f'questions_asked {profile["questions_asked"]} → {asked}, '
                          f'last_topic={route.destination}')

    result = {"outcome": outcome, "routed_to": route.destination, "reason": route.reason,
              "confidence": route.confidence,
              "resolved_question": question, "sources_searched": searched,
              "sources": sources_meta[:3], "questions_asked_total": asked,
              "turns_on_this_thread": len(history) + 1, "steps": steps,
              "latency_seconds": round(time.perf_counter() - t0, 2)}

    if outcome == "answered":
        result["answer_ar"] = bilingual.answer_ar
        result["answer_en"] = bilingual.answer_en
        history_answer = bilingual.answer_en
    else:
        result["answer"] = answer
        history_answer = answer

    new_history = (history + [{"q": q.question, "a": history_answer}])[-3:]
    return entrypoint.final(value=result, save=new_history)


# ──────────────────────────────── wiring ──────────────────────────────────
def build(verbose: bool = True):
    """Initialise everything and return (workflow, info)."""
    global llm, academic_retriever, campus_retriever, router

    if not os.environ.get("GROQ_API_KEY"):
        try:
            from dotenv import load_dotenv
            load_dotenv(ROOT / ".env")
        except ImportError:
            pass
    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key, "
            "or export it in your shell. Free key: https://console.groq.com/keys")

    model = pick_model()
    if verbose:
        print(f"model      : {model}")
    llm = ChatGroq(model=model, temperature=0)
    router = llm.with_structured_output(MurshidRoute)

    if verbose:
        print(f"embeddings : {EMBEDDING_MODEL}  (downloads ~1 GB on first run)")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    retrievers, counts = build_stores(embeddings)
    academic_retriever = retrievers["academic"]
    campus_retriever = retrievers["campus"]
    if verbose:
        for name, (d, c) in counts.items():
            print(f"{name:11}: {d} documents -> {c} chunks")

    return murshid, {"model": model, "counts": counts}
