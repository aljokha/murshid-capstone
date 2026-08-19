# Architecture and design decisions

This document explains *why* Murshid is built the way it is. The **what** is in
[`../README.md`](../README.md); the rubric-by-rubric account is in
[`../WRITEUP.md`](../WRITEUP.md).

---

## The problem

A university's answers live in two administratively separate places:

- **Academic regulations** (Registrar) — grading, GPA, appeals, attendance,
  withdrawal, probation, transcripts, graduation, examinations.
- **Campus services** (Student Affairs) — library, IT, portal access, housing,
  dining, careers, wellbeing, clubs.

A student does not know which of those two worlds their question belongs to.
They also frequently ask in Arabic, and sometimes ask a question that spans
both. Occasionally they are not asking a question at all — they are asking the
university to *do* something.

Those four cases are exactly the four destinations the router chooses between.

---

## Decision 1 — routing is an LLM decision, not a keyword rule

The whole system hangs on `MurshidRoute`, a Pydantic model with a
`Literal["academic", "campus", "both", "action"]` destination returned through
`with_structured_output`.

`Literal` does two jobs. It **constrains the model** — it cannot emit a
destination the workflow has no branch for, which makes it the cheapest guardrail
in the system. And it **documents the branches** — the type *is* the list of
destinations, so the router and the control flow cannot drift apart.

The alternative, keyword matching, fails on real student input in at least four
distinguishable ways, all of which are in the notebook's routing test:

| Question | Why keywords fail |
|---|---|
| `لا أستطيع الدخول إلى بوابة الطالب` | No English keyword exists to match |
| "I was told my attendance is short but I have a medical note" | Contains no literal category term |
| "How do I appeal a grade, and where is the IT helpdesk?" | Matches **both** categories; `if/elif` picks arbitrarily |
| "How do I withdraw?" vs "I want to withdraw from STAT301" | Identical keywords, **different destinations** |

The last row is the interesting one, and it is why `action` is a separate
destination rather than a flag: the difference between asking *how* to withdraw
and asking *to* withdraw is intent, not vocabulary. Only a language model sees it.

---

## Decision 2 — two vector stores, not one

Two `Chroma` collections with different `collection_name` values and disjoint
document sets. The alternative — one store with a metadata filter — would work
functionally, but it makes the routing decision cosmetic: the same index is
searched either way.

Keeping them physically separate means the routing decision has a physical
consequence, which is both the point of the architecture and the thing the
isolation test in the notebook demonstrates.

The trade-off is real and worth naming: two stores cannot answer a spanning
question in one search. That is what the `both` destination and the
Parallelization branch exist to handle.

---

## Decision 3 — multilingual embeddings

The course lessons use `sentence-transformers/all-mpnet-base-v2`, which is
English-only. Murshid's students ask in Arabic against an English knowledge base
with Arabic summaries, so the project uses
`sentence-transformers/paraphrase-multilingual-mpnet-base-v2` instead.

Same family, same cost (free, local, no API key), but it embeds Arabic and
English into a shared space, so an Arabic query can retrieve an English
regulation. Without this swap the router would route Arabic questions correctly
and then retrieve nothing useful — a failure that is easy to miss, because the
routing evidence still looks perfect.

The knowledge base also carries a short Arabic summary at the foot of the six
most-asked documents, as a hedge: it gives cross-lingual retrieval an easier
target without translating the whole corpus.

---

## Decision 4 — Hybrid RAG

| | Why not |
|---|---|
| **2-Step** | Retrieval is not unconditional — the classifier chooses the store first, and `action` questions skip retrieval entirely |
| **Agentic** | The workers do not decide *when* to retrieve; the route is fixed before they run, which caps LLM calls per request and keeps latency predictable |
| **Hybrid** ✅ | Adds query classification *before* retrieval and retrieval validation *after* it |

The validation step is real, not decorative: if the routed store returns nothing,
the workflow re-routes to the other store once before concluding the answer is
not in its sources. Student questions are frequently ambiguous, and confidently
answering from the wrong domain is worse than one extra round-trip.

---

## Decision 5 — two architectural layers, both real

The notebook contains a supervisor **and** a Functional API entrypoint. This is
deliberate, and it is not duplication.

**Layer 1 — supervisor + workers (§10).** The declared Track A shape. A
dedicated supervisor delegates to two specialists that do not know about each
other, and the printed `transfer_to_*` tool calls are unambiguous evidence that
the *LLM* chose the worker.

**Layer 2 — the `@entrypoint` workflow (§12).** The supervisor shape cannot
express two things this project needs: a fourth `action` destination where the
student is asking to file something, and a human approval gate that pauses
before that action becomes irreversible. So the entrypoint wraps *the same tools
and the same retrievers* in explicit control flow that adds the four-way route,
retrieval validation, `RetryPolicy`, memory, and `interrupt()`.

Neither layer is dead code. Layer 1 is the routing evidence; Layer 2 is the
system.

---

## Decision 6 — what gets remembered, and where

| | Short-term (checkpointer) | Long-term (Store) |
|---|---|---|
| Object | `InMemorySaver` | `InMemoryStore` |
| Key | `thread_id` | `("students", student_id)` |
| Contents | the in-progress run, paused interrupts | `preferred_language`, `questions_asked`, `last_topic` |
| Lifetime | one conversation | across all conversations |
| Production swap | `SqliteSaver` / `PostgresSaver` | `PostgresStore` |

`preferred_language` is the one that earns its place. A student who asks in
Arabic once is answered in Arabic in every future conversation without ever
saying so — which is only possible because the fact is keyed by *student*, not by
*thread*. That is the concrete difference between a Store and a chat history, and
the cross-thread test in the notebook is what proves it.

---

## Decision 7 — where the human sits

Escalation is triggered by the model, through two fields on `MurshidRoute`:

- `destination == "action"` — the student is asking to file something.
- `confidence == "low"` — the classifier was unsure, so escalate rather than guess.

The second trigger doubles as the failure mode of the first. When
`classify` exhausts its re-prompt attempts, the safe default it returns carries
`confidence="low"` — so a broken classification becomes an escalation to a
person, not a confidently wrong answer.

What the advisor can do matters as much as that they are asked. They can
**approve**, **reject with a note**, or **approve with edits** — and an edited
reason is what reaches the registry, not the student's original wording. A gate
that can only say yes is not really a gate.

---

## Known limitations

- The action tools append to an in-process Python list, not a real Registrar
  system. The approval flow, the state handling, and the irreversibility
  semantics are real; the backend is a stub.
- `InMemoryStore` does not survive a kernel restart. `SqliteSaver` demonstrates
  durable short-term state; the Store's production equivalent is `PostgresStore`.
- Sixteen documents is small enough that `k=3` similarity search is sufficient.
  A real corpus of thousands would need reranking and metadata filtering.
- The two corpora legitimately cross-reference each other in places (a transcript
  can be blocked by an unreturned library item; library fines block transcript
  requests). Retrieval handles this correctly because routing sends the question
  to the store that *owns* the policy, but it is a reminder that "disjoint" is a
  property of ownership, not of vocabulary.
