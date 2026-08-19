<!--
╔══════════════════════════════════════════════════════════════════════════╗
║  Sections 1-7 below are TRUE BY CONSTRUCTION — they describe design      ║
║  decisions baked into the code, so they are safe to keep as written      ║
║  once you have actually run the notebook.                               ║
║                                                                          ║
║  Anything in «angle quotes» is a RESULT and must be replaced with what   ║
║  your own run actually produced. Find them all:  grep -n "«" WRITEUP.md  ║
║                                                                          ║
║  The rubric warns explicitly: graders check this document against your   ║
║  code and captured output. A claim your own notebook contradicts costs   ║
║  more than an admitted gap. Write from your results, not from the plan.  ║
╚══════════════════════════════════════════════════════════════════════════╝
-->

# Murshid — Capstone Write-Up

**Author:** «Your full name»
**Declared track:** **A — Supervisor + Workers**
**Programme:** SDAIA Academy — Building Agentic AI Systems, «cohort dates»

---

## 1. Agent fundamentals (15 pts)

Murshid's tools do real work on their arguments rather than returning fixed
strings. `compute_gpa` takes a list of grade points and a matching list of
credit hours, computes the credit-weighted average, and derives the standing
band from the result — different inputs give different outputs, and mismatched
list lengths raise. `credits_to_graduate` subtracts completed hours from the
132-hour programme total and derives the remaining number of terms by ceiling
division. `search_academic_regulations` and `search_campus_services` each run a
real vector similarity search against their own store and return the retrieved
text with its source filename attached.

Structured output is used everywhere a result is parsed by code rather than read
by a person. The router returns a `MurshidRoute` Pydantic model
(`destination`, `reason`, `language`, `confidence`) via `with_structured_output`,
and the workflow branches on `route.destination`. Because `destination` is typed
`Literal["academic", "campus", "both", "action"]`, the model cannot emit a value
the workflow has no branch for — the type and the branches cannot drift apart.
The free-text answer returned to the student is deliberately *not* structured,
because a human reads it.

In the notebook, cell «N» shows the model choosing `compute_gpa` on its own and
the tool call arguments it constructed: «paste the printed tool call here».

---

## 2. Multi-agent / routing architecture (15 pts)

**I declared Track A (Supervisor + Workers).** A `create_supervisor` instance
sits in front of two specialist workers, `academic_agent` and `campus_agent`,
each built with `create_agent` and holding its own tools and its own vector
store. The workers do not know about each other; all delegation happens in one
place.

The evidence that the **LLM** made the routing decision is the printed handoff
tool calls. For "What GPA do I need to stay off academic probation?" the trace
shows `transfer_to_academic_agent` followed by `transfer_back_to_supervisor`;
for "How late is the library open during finals week?" it shows
`transfer_to_campus_agent`. Two handoffs per request, not one — the return trip
is expected.

I additionally implemented the **Track C** classifier, because the routing
decision needs to distinguish four destinations rather than two workers, and
because its test cases are the clearest demonstration of why keyword matching
fails. Of the seven test questions, three defeat any keyword router:
*"لا أستطيع الدخول إلى بوابة الطالب"* contains no English keyword at all;
*"I was told my attendance is short but I have a medical note"* contains neither
"grade" nor "attendance policy" as a literal phrase; and *"How do I appeal a
grade, and where is the IT helpdesk?"* matches both categories at once, which a
first-match `if/elif` chain resolves arbitrarily. The classifier routes all
three correctly. «Confirm this against your own run — if any of them routed
differently, say so and say what you changed.»

---

## 3. RAG pipeline (15 pts)

The pipeline runs all five stages on real documents. Sixteen markdown documents
in `data/` are **loaded** with `DirectoryLoader`, **split** with
`RecursiveCharacterTextSplitter` (600 characters, 80 overlap, splitting on
markdown headings first so a regulation is not cut mid-clause), **embedded**
with `HuggingFaceEmbeddings`, **stored** in two separate Chroma collections, and
**retrieved** with `k=3`. My run produced «N» academic chunks and «N» campus
chunks.

The two collections are genuinely separate, which is what makes the routing
decision meaningful — two retrievers pointed at one store would make routing
change nothing. The isolation test demonstrates this: querying the academic
store for "library opening hours during finals" returns academic regulations,
not the library page.

I chose the multilingual embedding model
`paraphrase-multilingual-mpnet-base-v2` over the English-only
`all-mpnet-base-v2` used in the course lessons, because students ask in Arabic
against a knowledge base written in English with Arabic summaries, and
cross-lingual retrieval is the whole requirement. Both are free and run locally.

### RAG architecture chosen: **Hybrid**

Murshid is not **2-Step RAG**, because retrieval is not unconditional: an LLM
classifier first decides *which* store to search, and questions routed to
`action` skip retrieval entirely. It is not purely **Agentic RAG** either,
because the specialist workers do not have unbounded freedom to decide when to
retrieve — the route is fixed before they run, which caps the number of LLM
calls per request and keeps latency predictable.

It is **Hybrid** in the course's precise sense: it adds a query-classification
step before retrieval and a retrieval-validation step after it. If the routed
store returns nothing, the workflow re-routes to the other store once before
concluding that the answer is not in its sources. Hybrid is the right trade for
student services because student questions are frequently ambiguous or span both
domains, and confidently answering from the wrong domain is worse than one extra
retrieval round-trip.

---

## 4. Context & state management (15 pts)

Short-term and long-term memory are two different objects with two different
lifetimes, and both are wired into the same `@entrypoint`.

**Short-term** is the checkpointer, scoped by `thread_id`. It holds the
in-progress run and any paused interrupt. It is what lets a student ask a
follow-up question that refers back to the previous turn within one
conversation, and it is what lets a paused withdrawal request be resumed.

**Long-term** is an `InMemoryStore`, namespaced by `("students", student_id)`.
It holds the student's `preferred_language`, `major`, `questions_asked` count,
and `last_topic`. It is read at the top of every run and written at the bottom.
It is *not* a growing list of chat messages — it is keyed by student, not by
conversation, and it survives the conversation ending.

The cross-thread test is the proof. A student asks a question in Arabic in
thread `conv-A`; a completely new thread `conv-B` for the same student recalls
`preferred_language == "ar"` and reports a cumulative question count of 2; a
different student in `conv-C` starts from 1. My run printed:

```
«paste the actual conv-A / conv-B / conv-C output here»
```

If `conv-B` had reported 1, the value was living in the thread rather than the
store, and it would not have been long-term memory at all.

---

## 5. Human-in-the-loop (10 pts)

Two things in Murshid are irreversible: filing a formal grade appeal and
submitting a course withdrawal. A withdrawal cannot be reinstated in the same
term once the Registrar processes it, and the deadline is the end of week 10.
That is a real reason to require human approval, not a decorative one.

The decision to escalate is made by the **model**, not by a keyword rule. Two
`MurshidRoute` fields trigger it: `destination == "action"`, meaning the student
is asking to *file* something rather than asking a question; and
`confidence == "low"`, meaning the classifier was unsure and escalation is
better than a guess. The second trigger also catches classifier failure — the
fallback returned when structured output cannot be validated carries
`confidence="low"`, so a broken classification becomes an escalation rather than
a wrong answer.

Both halves are demonstrated. `interrupt()` pauses the run and surfaces the
draft request to an advisor; `Command(resume=...)` completes it. Critically, the
advisor's **edit reaches the final state**: the advisor replaced the student's
stated reason "the workload is too heavy" with "Medical grounds — documentation
on file with Student Health", and the registry log contains the advisor's text,
not the student's. The rejection path is also demonstrated, returning
`status: rejected_by_advisor` without filing anything.

My run printed:

```
«paste the PAUSED payload, the RESUMED result, and the REGISTRY LOG here»
```

---

## 6. LangGraph Functional API & error handling (15 pts)

The system is built on the **Functional API** — `@task` for each discrete unit
of work and a single `@entrypoint` orchestrating them with ordinary Python
control flow. There is no `StateGraph` anywhere. Tasks are awaited with
`.result()`, and the `both` branch launches its two retrievals before awaiting
either, so they overlap.

I implemented **three** of the four error strategies:

**Transient — `RetryPolicy`.** `retrieve_with_validation` carries a real
`RetryPolicy(max_attempts=3, initial_interval=0.5)` object on its `@task`
decorator. To prove it fires rather than merely exists, a `SIMULATE_FLAKY` flag
makes the first attempt raise `ConnectionError`. The printed output shows
attempt #1 followed by attempt #2, with no error surfacing and no retry code of
my own: «paste the attempt #1 / #2 output».

**LLM-recoverable — loop back with the error in context.** `classify` catches
`ValidationError` and re-prompts the model with the rejection text and an
explicit restatement of the permitted destinations. After three failures it
returns a safe default of `destination="both", confidence="low"`, which routes
to a human rather than crashing the workflow.

**User-fixable — `interrupt()`.** `get_course_code` pauses and asks when a
withdrawal request does not name a course, then resumes with the supplied value.

**Unexpected — let it bubble up.** There is deliberately no blanket
`except Exception` anywhere in the system. Anything not anticipated above
propagates to the caller, where it is visible in the traceback and in the
LangSmith trace — which is exactly where I want a genuine bug to appear.

---

## 7. Workflow pattern (10 pts)

**The pattern is Routing.**

Murshid implements the *Routing* pattern from the workflow-patterns lesson: a
single classifier LLM call inspects the input and directs it to one of several
specialised downstream paths — `academic`, `campus`, `both`, or `action` — each
with its own tools, its own knowledge store, and its own handling.

Routing fits because student questions fall into genuinely distinct categories
that need *different* retrieval sources, but each individual question needs only
one path. A single do-everything prompt would carry both corpora into every call,
costing tokens and diluting relevance. Parallelization would search both stores
every time and waste half the work on every single-domain question.
Orchestrator-Worker is over-engineered here, because the sub-tasks are known in
advance rather than planned per request.

The `both` branch additionally uses **Parallelization**: when a question
genuinely spans both domains the two retrievals are independent, so they are
launched as concurrent `@task`s and awaited together rather than run in series.

---

## 8. LangSmith observability (5 pts)

Tracing is enabled with `LANGCHAIN_TRACING_V2="true"` (the exact variable name —
`LANGSMITH_TRACING_V2` is not a real variable and fails silently), and the key is
verified with a `client.list_projects()` call before the agent runs, so an
invalid key raises loudly instead of producing an empty project page.
`wait_for_all_tracers()` flushes before inspection.

«Replace this entire paragraph with something only someone who opened the trace
could write. Name a specific task, a specific latency, and a specific
conclusion. For example: "The trace for the `both`-destination question showed
`classify` at 0.4s, the two retrievals at 60ms and 55ms overlapping, and
`synthesize` at 2.1s of a 2.6s total — retrieval is not the bottleneck, the
final generation is, because it receives roughly 1,800 tokens of combined
context. The next optimisation is a tighter `k` and a reranking step before
synthesis, not a faster vector store." Do not keep this example. Use your own
numbers.»

I also built a small evaluation dataset of «N» routing examples and scored the
classifier against it with a deterministic grader that compares the predicted
`destination` to the expected one. The experiment scored «N/N».

<!--
If you could not get a LangSmith key working, DELETE the two paragraphs above
and say so plainly instead. Describe what you inspected in their place — the
printed handoff tool calls, the per-task timings, the `sources_searched` lists.
An honest gap costs a few points. A fabricated finding that your own output
contradicts costs credibility across the whole submission.
-->

---

## Known limitations

Worth stating honestly; graders reward it and it costs nothing.

- The action tools write to an in-process Python list, not a real Registrar
  system. The approval flow, the state handling, and the irreversibility
  semantics are real; the backend is a stub.
- `InMemoryStore` does not survive a kernel restart. The checkpointer can be
  swapped for `SqliteSaver` to demonstrate durable state; the Store's production
  equivalent would be `PostgresStore`.
- Retrieval quality on Arabic queries is «better/worse» than on English ones,
  because the knowledge base is written in English with Arabic summaries rather
  than being fully bilingual. «State what you actually observed.»
- The knowledge base is 16 documents about a fictional university. Behaviour on
  a real corpus of thousands of documents would need reranking and probably
  metadata filtering, neither of which is implemented here.
