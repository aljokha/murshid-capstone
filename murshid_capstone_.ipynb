<!--
╔══════════════════════════════════════════════════════════════════════════╗
║  HOW TO USE THIS FILE                                                    ║
║                                                                          ║
║  Text NOT in «angle quotes» describes design decisions that are baked    ║
║  into the code. Those are safe to keep once you have run the notebook.   ║
║                                                                          ║
║  Text in «angle quotes» is a RESULT. Replace every one with what YOUR    ║
║  run actually produced.   Find them:   grep -n "«" WRITEUP.md            ║
║                                                                          ║
║  The rubric warns explicitly: graders check this document against your   ║
║  code and captured output. A claim your own notebook contradicts costs   ║
║  more than an admitted gap. Write from your results, not from the plan.  ║
╚══════════════════════════════════════════════════════════════════════════╝
-->

# Murshid — Capstone Write-Up

**Author:** Khalid Aljohar, Abdullah Alfawzan, Abdulaziz Almeshary, Saud Alghuraybi, Ahmed Bakhashwain, Moath Aljubirmme
**Declared track:** **A — Supervisor + Workers**
**Programme:** SDAIA Academy — Building Agentic AI Systems,  16-20 August 2026

---

## 0. What the system does

Murshid answers Al-Noor University student questions in Arabic or English. An
LLM classifier routes each question to Academic Affairs, Campus Services, both,
or — when the student is asking to *file* something rather than ask something —
to an approval gate where a human advisor must sign off before anything
irreversible happens.

**A note on the model.** Groq decommissioned `llama-3.3-70b-versatile`, the
model used throughout the course lessons, on **16 August 2026**. Section 4
therefore queries the account for available models and selects the first
supported one from a preference list rather than hardcoding an ID. My run used
**`openai/gpt-oss-120b`**. This mattered more than a model swap
usually does — see section 6, where the change in structured-output behaviour
surfaced two real errors.

---

## 1. Agent fundamentals — 15 pts

The tools do real work on their arguments rather than returning fixed strings.
`compute_gpa` takes grade points and matching credit hours, computes the
credit-weighted average, and derives the standing band from the result;
different inputs give different outputs and mismatched list lengths raise.
`credits_to_graduate` subtracts completed hours from the 132-hour programme
total and derives remaining terms by ceiling division.
`search_academic_regulations` and `search_campus_services` each run a real
vector similarity search against their own store and return the retrieved text
with its source filename attached.

Structured output is used wherever a result is parsed by code rather than read
by a person: `MurshidRoute` for routing and `ActionRequest` for extracting a
formal request. Both go through `with_structured_output`. Because `destination`
is typed `Literal["academic", "campus", "both", "action"]`, the model cannot
emit a value the workflow has no branch for — the type and the branches cannot
drift apart. The free-text answer returned to the student is deliberately *not*
structured, because a human reads it.

**Evidence** — the same tool called twice with different arguments, producing
different results, and a boundary case:

```
GPA 3.45 across 10 credit hours (34.5 total grade points) — Good Standing.
GPA 1.45 across 10 credit hours (14.5 total grade points) — below the 2.00 threshold — academic probation applies.
45 credit hours remaining of 132 — about 3 more full-time term(s) at 15 hours each.
132 of 132 credit hours complete — the credit-hour requirement is already met.
```

---

## 2. Multi-agent / routing architecture — 15 pts

**I declared Track A (Supervisor + Workers).** A `create_supervisor` instance
sits in front of two specialist workers, `academic_agent` and `campus_agent`,
each built with `create_agent` and holding its own tools and its own vector
store. The workers do not know about each other; delegation happens in one place.

The evidence that the **LLM** made the routing decision is the printed handoff
tool calls:

```
Q: What GPA do I need to stay off academic probation?
   handoff -> transfer_to_academic_agent
   handoff -> transfer_back_to_supervisor

Q: How late is the library open during finals week?
   handoff -> transfer_to_campus_agent
   handoff -> transfer_back_to_supervisor
```

Two questions, two different workers, chosen by the model. Two handoffs per
request is correct — the supervisor hands off to the worker, and the worker
hands control back with `transfer_back_to_supervisor`.

I additionally implemented the **Track C** classifier, because the production
workflow needs four destinations rather than two workers, and because its test
cases are the clearest demonstration of why keyword matching fails. Of the eight
routing tests, four defeat any keyword router:

- *"لا أستطيع الدخول إلى بوابة الطالب"* contains no English keyword at all.
- *"I was told my attendance is short but I have a medical note"* contains no
  literal category term.
- *"How do I appeal a grade, and where is the IT helpdesk?"* matches **both**
  categories, which a first-match `if/elif` chain resolves arbitrarily.
- *"How do I withdraw from a course?"* and *"I want to withdraw from STAT301"*
  share every keyword but need **different** destinations — the difference is
  intent, not vocabulary.

**Evidence** — all eight test questions routed correctly, every one at high
confidence:

```
DEST      LANG  CONF   QUESTION
academic  en    high   How do I appeal a grade I think was marked wrong?
campus    en    high   Where is the IT helpdesk and what are its hours?
both      en    high   How do I appeal a grade, and where is the IT helpdesk?
campus    ar    high   لا أستطيع الدخول إلى بوابة الطالب
action    ar    high   أريد الانسحاب من مقرر الإحصاء STAT301
academic  en    high   How do I withdraw from a course?
action    en    high   I want to withdraw from STAT301, the workload is too heavy
academic  en    high   I was told my attendance is short but I have a medical note
```

The last three rows are the interesting ones. *"How do I withdraw from a
course?"* → `academic` and *"I want to withdraw from STAT301"* → `action` share
almost every word, and the classifier separated them on intent alone. The
Arabic withdrawal request was also correctly identified as an `action`, in a
language where none of the routing vocabulary appears at all.

---

## 3. RAG pipeline — 15 pts

All five stages run on real documents. Sixteen markdown files in `data/` are
**loaded** with `DirectoryLoader`, **split** with `RecursiveCharacterTextSplitter`
(600 characters, 80 overlap, splitting on markdown headings first so a
regulation is not cut mid-clause), **embedded** with `HuggingFaceEmbeddings`,
**stored** in two separate Chroma collections, and **retrieved** with `k=3`. My
run produced **23 academic chunks and 27 campus chunks**, giving 23 and 27
vectors in the two Chroma collections respectively.

The two collections are genuinely separate, which is what makes routing
meaningful — two retrievers pointed at one store would make routing change
nothing. The isolation test demonstrates it: querying the academic store for
library hours returns academic regulations, not the library page.

I chose the multilingual embedding model
`paraphrase-multilingual-mpnet-base-v2` over the English-only
`all-mpnet-base-v2` used in the course lessons, because students ask in Arabic
against a knowledge base written in English with Arabic summaries. Without that
swap the router would route Arabic questions correctly and then retrieve
nothing useful — a failure that is easy to miss, because the routing evidence
still looks perfect. Both models are free and run locally with no API key.

### RAG architecture chosen: **Hybrid**

Not **2-Step**, because retrieval is not unconditional: a classifier decides
which store to search, and `action` questions skip retrieval entirely. Not
purely **Agentic**, because the workers do not decide *when* to retrieve — the
route is fixed before they run, which caps LLM calls per request and keeps
latency predictable.

**Hybrid** in the course's precise sense, with both of the intermediate steps it
describes actually implemented:

- **Query enhancement** — `rewrite_followup` resolves a follow-up like *"What
  happens if I fall below it?"* into a standalone question using the
  conversation state for that thread, before it reaches the classifier or the
  retriever.
- **Retrieval validation** — if the routed store returns nothing, the workflow
  re-routes to the other store once before concluding the answer is not in its
  sources.

Hybrid is the right trade for student services because student questions are
frequently ambiguous, underspecified, or span both domains, and confidently
answering from the wrong domain is worse than one extra round-trip.

**Evidence** — eight questions whose answers are verbatim in the documents, each
retrieving from the correct file:

```
--- retrieval smoke test ---
  PASS [academic] 'grade appeal deadline'              expected '15'         | top hit: grade_appeals.md
  PASS [academic] 'minimum attendance percentage'      expected '75'         | top hit: attendance_policy.md
  PASS [academic] 'course withdrawal deadline'         expected 'week 10'    | top hit: course_withdrawal.md
  PASS [academic] 'credit hours required to graduate'  expected '132'        | top hit: graduation_requirements.md
  PASS [campus  ] 'library hours during finals week'   expected '2:00 AM'    | top hit: library_hours.md
  PASS [campus  ] 'where is the IT helpdesk'           expected 'Building 4' | top hit: it_helpdesk.md
  PASS [campus  ] 'student portal account lockout'     expected '30 minutes' | top hit: student_portal_access.md
  PASS [campus  ] 'how many members to start a club'   expected '15'         | top hit: clubs_and_societies.md

SMOKE TEST: PASSED
```

Isolation — each store is blind to the other's subject matter:

```
academic store, asked about LIBRARY HOURS:
  returned: ['course_withdrawal.md', 'graduation_requirements.md', 'attendance_policy.md']
campus store, asked about GRADE APPEALS:
  returned: ['housing.md', 'careers_office.md', 'clubs_and_societies.md']
```

And routing has a visible consequence — `sources_searched` differs per question:
`['academic']` for the grade-appeal question, `['campus']` for the Arabic portal
question, `['academic', 'campus']` for the one spanning both.

---

## 4. Context & state management — 15 pts

Short-term and long-term memory are two different objects with two different
lifetimes, and both are wired into the same `@entrypoint`.

**Short-term** is the checkpointer, scoped by `thread_id`. The entrypoint takes
a `previous` parameter and returns `entrypoint.final(value=..., save=...)`, so
each thread carries its last three turns. That state is what `rewrite_followup`
reads to resolve pronouns, and it is what holds a paused `interrupt()` until it
is resumed.

**Long-term** is an `InMemoryStore`, namespaced by `("students", student_id)`.
It holds `preferred_language`, `questions_asked`, and `last_topic`. It is read
at the top of every run and written at the bottom. It is *not* a growing list of
chat messages — it is keyed by **student**, not by conversation, and it survives
the conversation ending.

The distinction is visible in one place in the code: `history` comes from
`previous` (dies with the thread), `profile` comes from the Store (does not).

**Cross-thread proof** — a fact written in one thread, read back in another:

```
conv-A  questions_asked: 1 | language learned: ar
conv-B  questions_asked: 2 | language recalled: en   <- survived a brand-new thread
conv-C  questions_asked: 1 (different student)

conv-B's answer, in Arabic, without the student asking for Arabic:
آخر موعد للانسحاب هو **نهاية الأسبوع العاشر** من الفصل الدراسي. [course_withdrawal.md]

The counter is the unambiguous proof: the student asked once in `conv-A` and
the count came back as **2** in a completely different thread, while a
different student in `conv-C` started from 1. That value cannot have come from
the thread, because the thread was new.

One honest wrinkle in the printed line above. `conv-B` reports
`language recalled: en`, not `ar`, even though its answer is in Arabic. Both
are correct and the reason is ordering: the profile is read at the *start* of
the run — where `preferred_language` was still `ar`, which is why the answer
came back in Arabic — and rewritten at the *end* with the language of the
current question, which was English. So the field tracks the most recent
language rather than a stable preference. That is a design weakness, not a
memory failure: a real system should either keep the first observed language
until the student changes it explicitly, or store a per-language count. The
`questions_asked` counter is unaffected and is the cleaner proof of the two.
```

If `conv-B` had reported 1, the value was living in the thread rather than the
store, and it would not have been long-term memory at all.

**Short-term proof** — a follow-up resolved from the previous turn on the same
thread:

```
turn 1: The minimum attendance required is **75%** of the scheduled contact
        hours for each course. 【attendance_policy.md】

  [rewrite] 'What happens if I fall below it?'
         -> 'What happens if I fall below the minimum attendance percentage?'

turn 2: If your attendance drops below the required 75 percent, you will be
        **barred from sitting the final examination** for that course and will
        receive a **grade of DN (denied)**, which is treated as an **F for GPA
        purposes**【attendance_policy.md】.
```

---

## 5. Human-in-the-loop — 10 pts

Two things in Murshid are irreversible: filing a formal grade appeal and
submitting a course withdrawal. A withdrawal cannot be reinstated in the same
term once the Registrar processes it, and the deadline is the end of week 10.
That is a real reason to require human approval, not a decorative one.

The escalation decision is made by the **model**: the router returns
`destination == "action"` when the student is asking to *file* something rather
than asking a question. The distinction between *"How do I withdraw?"* and
*"I want to withdraw from STAT301"* is intent, not vocabulary, which is exactly
what a classifier is for and what a keyword rule cannot do.

Both halves are demonstrated. `interrupt()` pauses the run and surfaces the
request to an advisor; `Command(resume=...)` completes it. The advisor can
approve, reject with a note, or **approve with edits** — and the advisor's edit
is what reaches the registry:

```
=== RESUMED AND COMPLETED ===
outcome: filed
answer : Withdrawal WD-s2201-STAT301 filed for STAT301.
         Recorded reason: Medical grounds — documentation on file with Student Health.

=== REGISTRY LOG ===
  {'ref': 'WD-s2201-STAT301', 'type': 'withdrawal', 'student': 's2201',
   'course': 'STAT301', 'reason': 'Medical grounds — documentation on file with Student Health.'}
```

Preceded by the pause:

```
=== PAUSED FOR ADVISOR APPROVAL ===
  action                : A student advisor must approve this before it is filed
  type                  : withdrawal
  student_id            : s2201
  course_code           : STAT301
  student_stated_reason : the workload is too heavy
```

And the rejection path, which files nothing:

```
PAUSED: CHEM210
outcome: rejected_by_advisor
answer : Your request was not filed. Advisor note: Past the end-of-week-10
         deadline; withdrawal cannot be processed.
registry entries: 1 — unchanged, nothing was filed.
```

The recorded reason is the advisor's text, not the student's original *"the
workload is too heavy"*. A gate that can only say yes is not really a gate, and
a pause that does not change the outcome proves nothing. The rejection path is
also demonstrated, returning `rejected_by_advisor` with nothing filed.

There is a second, unplanned route into the same gate. When `parse_action`
cannot extract a request, it falls back to an empty `ActionRequest`, which makes
`get_course_code` interrupt and ask the student. An LLM failure degrades into a
human-in-the-loop prompt rather than a stack trace.

---

## 6. LangGraph Functional API & error handling — 15 pts

Built on the **Functional API** — `@task` for each discrete unit of work and a
single `@entrypoint` orchestrating them with ordinary Python control flow. There
is no `StateGraph` anywhere. Tasks are awaited with `.result()`, and the
two-source branch launches both retrievals before awaiting either, so they
overlap.

Three of the four error strategies are implemented:

**Transient — `RetryPolicy`.** `retrieve_one` carries a real
`RetryPolicy(max_attempts=3, initial_interval=0.5)` on its `@task` decorator. A
`SIMULATE_FLAKY` flag makes the first attempt raise `ConnectionError`, so the
retry can be observed rather than assumed:

```
  [retrieve_one] attempt #1 -- raising
  [retrieve_one] attempt #2 (academic)

total retrieval attempts: 2
answer still returned successfully: True
```

Two attempts, no error surfaced, and the answer still returned. No retry code
of my own, and no `time.sleep()` loop.

The simulated failure proves the mechanism is wired up. A **real** transient
failure proved it was needed. During a normal run the two-source branch hit:

```
RateLimitError: Error code: 429 - Rate limit reached for model
`openai/gpt-oss-120b` ... service tier `on_demand` on tokens per minute (TPM):
Limit 8000, Used 6676, Requested 1520. Please try again in 1.47s.
```

Nothing was wrong with the request — Groq's free tier allows 8,000 tokens per
minute and the `both` branch is the heaviest call, because it feeds both
corpora into one prompt. Two things were wrong with my design, though. The
`RetryPolicy` was attached only to `retrieve_one`, a local vector search that
can never rate-limit, while `synthesize` — the call that actually hits the
provider — had none. And LangGraph's default `retry_on` does not cover provider
SDK exceptions, so even an attached policy would have skipped `RateLimitError`
silently. Both are fixed: the transient exception types are named explicitly,
the policy backs off 2s → 4s → 8s → 16s to ride out a per-minute window, and
retrieved context is capped per source to keep the request under the cap.

**LLM-recoverable — loop back with the error in context.** This is where the
project's most interesting failures happened, and both were real, not simulated.

*First failure.* The router returned a tool call with an invented key name:

```
"parameters for tool MurshidRoute did not match schema:
 errors: [missing properties: 'confidence']"
failed_generation: {"low": "high", "destination": "campus", ...}
```

The model had emitted `"low"` as a key. The cause was my own field description,
which began with the bare token *"low when the question is ambiguous…"* — the
model read the leading value name as the key. The fix was in the description,
not the model: every description now opens by describing the field, and enum
values appear only quoted after "Pick" or "Use". This is the routing lesson's
own point turned back on me — the `description=` **is** the prompt.

*Second failure.* `parse_action` hit `400 tool_use_failed: "Tool choice is
required, but model did not call a tool"` — the model correctly judged that a
question contained no formal request and declined to call the tool at all.

Both were handled the same way, and the handling is deliberately precise:

```python
RECOVERABLE = (ValidationError, groq.BadRequestError)
```

`ValidationError` means the object came back malformed. `groq.BadRequestError`
means the provider rejected the tool call before Pydantic ever saw it. Catching
only `ValidationError` — which is what I wrote first — would have let the second
class of failure crash the workflow. `classify` re-prompts with the error text
in context; `parse_action` degrades into a human prompt.

**User-fixable — `interrupt()`.** `get_course_code` pauses and asks when a
withdrawal request does not name a course, then resumes with the supplied value.

**Unexpected — let it bubble up.** There is deliberately no blanket
`except Exception` anywhere. The recoverable types are named explicitly, and
anything outside that tuple propagates to the caller where it is visible in the
traceback and in the LangSmith trace — which is exactly where a genuine bug
should appear.

### A design error worth recording

My first version escalated to the approval gate on
`route.destination == "action" or route.confidence == "low"`. That was wrong. An
ambiguous *question* is not a request to file anything, and sending one to
`parse_action` asked the model to extract a withdrawal from a question that
contained none — which is what produced the second failure above. Low
confidence now searches **both** stores instead of guessing one, and only
`destination == "action"` reaches the human gate.

---

## 7. Workflow pattern — 10 pts

**The pattern is Routing.**

Murshid implements the *Routing* pattern from the workflow-patterns lesson: a
single classifier LLM call inspects the input and directs it to one of several
specialised downstream paths — `academic`, `campus`, `both`, or `action` — each
with its own tools, its own knowledge store, and its own handling.

Routing fits because student questions fall into genuinely distinct categories
needing *different* retrieval sources, but each individual question needs only
one path. A single do-everything prompt would carry both corpora into every
call, costing tokens and diluting relevance. Parallelization alone would search
both stores every time and waste half the work on every single-domain question.
Orchestrator-Worker is over-engineered, because the sub-tasks are known in
advance rather than planned per request.

The two-source branch additionally uses **Parallelization**: when a question
spans both domains the retrievals are independent, so they are launched as
concurrent `@task`s and awaited together rather than run in series.

---

## 8. LangSmith observability — 5 pts

Tracing is enabled with `LANGCHAIN_TRACING_V2="true"` — the exact variable name.
`LANGSMITH_TRACING_V2` is not a real variable and fails silently with no trace
and no error. (The current LangChain docs use `LANGSMITH_TRACING`, without the
suffix; the `_V2` in the legacy name versions the *tracing backend*, not the
value, and dates from when LangSmith replaced the original tracer.) The key is
verified with `client.list_projects()` before the agent runs, so an invalid key
raises loudly instead of producing an empty project page, and
`wait_for_all_tracers()` flushes before inspection.

**What the trace actually showed.** The `both`-destination run — the heaviest
path, since it searches two stores — completed in **3.29s** using about **1K
tokens**, at a cost of **$0.0003**. The tree matched the intended shape:
`murshid` → `classify` → two `retrieve_one` calls → `synthesize`, with
`ChatGroq` and `PydanticToolsParser` nested under `classify` and
`VectorStoreRetriever` under the retrieval that hit the store.

Three things stood out, none of which I would have predicted:

**Retrieval is not the bottleneck, and neither is generation.** `classify`
took 0.52s, the two retrievals 0.00s and 0.11s, and `synthesize` 0.29s — a
total of **0.92s across all four instrumented tasks, out of a 3.29s run**.
Roughly **72% of the wall-clock time is not inside any task I wrote.** That
time is LangGraph orchestration and checkpointer serialisation — every task
boundary writes state through the checkpointer, and this run also deserialised
`MurshidRoute` and `ActionRequest` from a previous turn on the same thread. My
instinct before opening the trace was to optimise the LLM calls; the trace says
the framework overhead costs more than everything I wrote put together. On a
free-tier API where each call already takes ~0.3–0.5s, that is the opposite of
what I expected.

**The routing decision costs nearly as much as the answer.** `classify` spent
**472 tokens** and `synthesize` **576** — the decision about *where* to send
the question is almost as expensive as answering it. The reason is my own
`MurshidRoute` field descriptions, which are deliberately verbose because
that is what makes the router accurate. There is a real trade here: the same
verbosity that fixed the routing (section 6) is what makes the classifier
costly, and on a tier limited to 8,000 tokens per minute that directly reduces
how many questions the system can serve.

**The two retrievals did not overlap usefully.** One `retrieve_one` shows
0.00s with no child span, the other 0.11s with a `VectorStoreRetriever` under
it. The concurrency works, but at 110ms the entire retrieval step is noise next
to a 3.29s run — the Parallelization in the `both` branch is architecturally
correct and practically irrelevant at this scale.

**What I would change.** Not the vector store, and not the model. I would
trim the `MurshidRoute` descriptions once the router is stable, to reclaim
those 472 tokens per question, and I would investigate the unaccounted 2.4s
before optimising anything else — it is the single largest cost in the system
and I only know it exists because the trace showed the child spans summing to
far less than the parent.

**Evaluation.** I also built a dataset of **6 routing examples** — four
English, two Arabic, covering all four destinations — and scored the classifier
with a deterministic grader comparing the predicted `destination` to the
expected one. No LLM judge, so there is nothing to second-guess. The experiment
scored «N/6 — read this off the routes_correctly column in LangSmith».

The dataset produced a small lesson of its own. My first attempt reported
`0it [00:00, ?it/s]` — zero examples evaluated — because an earlier crashed run
had created the dataset before the examples were added, and the
`if not client.has_dataset(...)` guard then skipped populating it on every run
after that. The experiment link existed and looked entirely normal; it was
simply empty. Deleting and rebuilding the dataset fixed it. It is a good
illustration of why the observability section matters: the failure was silent,
and the only way to notice was to look at the number of examples actually
scored.

<!--
If you could not get LangSmith working, DELETE this section's claims and say so
plainly. Describe what you inspected instead: the printed handoff tool calls,
the retrieval attempt counters, the sources_searched lists. An honest gap costs
a few points. A fabricated finding your own output contradicts costs far more.
-->

---

## Known limitations

- The action tools append to an in-process Python list, not a real Registrar
  system. The approval flow, the state handling, and the irreversibility
  semantics are real; the backend is a stub.
- `InMemoryStore` and `InMemorySaver` do not survive a kernel restart. The
  production swaps are `SqliteSaver` / `PostgresSaver` and `PostgresStore`.
- Short-term history is capped at the last three turns per thread. A longer
  conversation would need summarisation rather than truncation.
- Sixteen documents is small enough that `k=3` similarity search suffices. A
  real corpus of thousands would need reranking and metadata filtering.
- The two corpora legitimately cross-reference each other (a transcript can be
  blocked by an unreturned library item). Routing still sends the question to
  the store that *owns* the policy, but "disjoint" is a property of ownership,
  not of vocabulary.
- Arabic retrieval worked, but it is the part I would test hardest before
  trusting. *"لا أستطيع الدخول إلى بوابة الطالب"* correctly retrieved both
  `student_portal_access.md` and `it_helpdesk.md` from an English corpus, and
  the Arabic withdrawal request was correctly classified as an `action`. That
  is cross-lingual retrieval working as intended. But the knowledge base is
  English with short Arabic summaries appended to six documents, not a
  genuinely bilingual corpus, so I cannot tell from these results how much of
  the success came from the multilingual embeddings and how much from those
  summaries. Testing that properly would mean removing the summaries and
  re-running the Arabic questions.
