# Murshid — Capstone Write-Up

**Authors:** Khalid ALjohar, Abdullah Alfawzan, Abdulaziz Almeshary, Saud Alghuraybi, Ahmed Bakhashwain, Moath Aljubir
**Declared track:** **A — Supervisor + Workers**
**Programme:** SDAIA Academy — Building Agentic AI Systems, 16–20 August 2026

---

## 0. What the system does

Murshid answers Al-Noor University student questions in Arabic or English. An
LLM classifier routes each question to Academic Affairs, Campus Services, both,
or — when the student is asking to *file* something rather than ask something —
to an approval gate where a human advisor must sign off before anything
irreversible happens.

**A note on the model.** Groq decommissioned `llama-3.3-70b-versatile`, the
model used throughout the course lessons, on **16 August 2026**. Rather than
hardcode a replacement that might also be retired, section 4 queries the account
for the models it can actually use and selects the first available from a
preference list. It printed:

```
chat models available to this key:
    allam-2-7b
    canopylabs/orpheus-arabic-saudi
    canopylabs/orpheus-v1-english
    groq/compound
    groq/compound-mini
    meta-llama/llama-prompt-guard-2-22m
    meta-llama/llama-prompt-guard-2-86m
    openai/gpt-oss-120b
    openai/gpt-oss-20b   <-- using this
    openai/gpt-oss-safeguard-20b
    qwen/qwen3.6-27b
```

We ran on **`openai/gpt-oss-20b`**. We started on `gpt-oss-120b` and moved down
deliberately: Groq's free tier allows 8,000 tokens per minute **per model**, a
full run of this notebook exceeds that on the larger model, and nothing here
needs a 120b model — every LLM call is classification, extraction, or
summarising retrieved text. Section 6 records what that ceiling taught us, and
section 2 records the one place where the smaller model is measurably weaker.

---

## 1. Agent fundamentals — 15 pts

The tools do real work on their arguments rather than returning fixed strings.
`compute_gpa` takes grade points and matching credit hours, computes the
credit-weighted average, and derives the standing band from the result;
mismatched list lengths raise. `credits_to_graduate` subtracts completed hours
from the 132-hour programme total and derives remaining terms by ceiling
division. `search_academic_regulations` and `search_campus_services` each run a
real vector similarity search against their own store and return the retrieved
text with its source filename attached.

The same tool called with different arguments produces different results:

```
GPA 3.45 across 10 credit hours (34.5 total grade points) — Good Standing.
GPA 1.45 across 10 credit hours (14.5 total grade points) — below the 2.00 threshold — academic probation applies.
45 credit hours remaining of 132 — about 3 more full-time term(s) at 15 hours each.
132 of 132 credit hours complete — the credit-hour requirement is already met.
```

Structured output is used wherever a result is parsed by code rather than read
by a person: `MurshidRoute` for routing and `ActionRequest` for extracting a
formal request, both via `with_structured_output`. Because `destination` is
typed `Literal["academic", "campus", "both", "action"]`, the model cannot emit a
value the workflow has no branch for — the type and the branches cannot drift
apart. The free-text answer returned to the student is deliberately *not*
structured, because a human reads it.

### The model choosing a tool for itself

The output above is us calling the tools. This is the **agent** deciding to:

```
model chose tool : search_academic_regulations
  arguments      : {'query': 'Al-Noor University academic regulations good standing GPA requirement'}
```

The query string was written by the model, not by us.

**Worth recording honestly:** we asked *"My grades this term were 4.0, 3.5 and
3.0, in courses worth 3, 3 and 4 credit hours. What is my GPA and am I in good
standing?"* expecting `compute_gpa` to be called. The model instead called
`search_academic_regulations` to look up the standing threshold and **computed
the GPA itself**, laying the arithmetic out in its answer and arriving at 3.45 —
the same figure `compute_gpa` returns. The tool call is genuine and
model-chosen, but it reveals something about tool design: a model will not call
a tool for work it believes it can already do. `compute_gpa` earns its place for
auditability and for larger inputs, not because the model needs it.

---

## 2. Multi-agent / routing architecture — 15 pts

**We declared Track A (Supervisor + Workers).** A `create_supervisor` instance
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
request is correct, not a bug — the supervisor hands off to the worker, and the
worker hands control back with `transfer_back_to_supervisor`.

We additionally implemented the **Track C** classifier, because the production
workflow needs four destinations rather than two workers, and because its test
cases are the clearest demonstration of why keyword matching fails:

```
DEST      LANG  CONF   QUESTION
academic  en    high   How do I appeal a grade I think was marked wrong?
campus    en    high   Where is the IT helpdesk and what are its hours?
academic  en    high   How do I appeal a grade, and where is the IT helpdesk?   <- see below
campus    ar    high   لا أستطيع الدخول إلى بوابة الطالب
action    ar    high   أريد الانسحاب من مقرر الإحصاء STAT301
academic  en    high   How do I withdraw from a course?
action    en    high   I want to withdraw from STAT301, the workload is too heavy
academic  en    high   I was told my attendance is short but I have a medical note
```

Four of these defeat any keyword router. *"لا أستطيع الدخول إلى بوابة الطالب"*
contains no English keyword at all. *"I was told my attendance is short but I
have a medical note"* contains no literal category term. And the pair *"How do I
withdraw from a course?"* → `academic` versus *"I want to withdraw from
STAT301"* → `action` share almost every word: the classifier separated them on
**intent**, which is precisely what a keyword rule cannot do. The Arabic
withdrawal request was also correctly identified as an `action`.

### One row we got wrong, and what it shows

Row 3 — *"How do I appeal a grade, and where is the IT helpdesk?"* — should be
`both`. The classifier returned `academic`, with the reason *"Appealing a grade
is an academic procedure."* It answered on the first clause and ignored the
second.

This is a real regression from moving to the smaller model. On
`openai/gpt-oss-120b` the same question routed to `both`. We are reporting it
rather than re-running until it looked tidy.

What makes it more interesting than a simple miss: **the same question routed
correctly inside the workflow.** Demo C, which reaches the classifier through
the `classify` task, produced:

```
routed to      : both
sources searched: ['academic', 'campus']
```

The difference is the prompt wrapper. The bare test calls
`router.invoke(question)`; `classify` calls
`router.invoke(f"Route this student question.\n\n{question}")`. That one framing
sentence is enough to change the outcome on a 20b model. The lesson is the
routing lesson's own point arriving from a different direction: the prompt
around a constrained-output call is not incidental, and a classifier that is
robust on a large model can become framing-sensitive on a small one. A
production system would test the classifier at the size it will actually run at,
not just at the size it was developed at.

---

## 3. RAG pipeline — 15 pts

All five stages run on real documents:

```
academic: loaded 8 documents -> 23 chunks
campus  : loaded 8 documents -> 27 chunks

academic collection: 23 vectors
campus   collection: 27 vectors
```

Sixteen markdown files in `data/` are **loaded** with `DirectoryLoader`,
**split** with `RecursiveCharacterTextSplitter` (600 characters, 80 overlap,
splitting on markdown headings first so a regulation is not cut mid-clause),
**embedded** with `HuggingFaceEmbeddings`, **stored** in two separate Chroma
collections, and **retrieved** with `k=3`.

Eight questions whose answers are verbatim in the documents, each retrieving
from the correct file:

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

The two collections are genuinely separate, which is what makes routing
meaningful — two retrievers pointed at one store would make routing change
nothing. Each store is blind to the other's subject matter:

```
academic store, asked about LIBRARY HOURS:
  returned: ['course_withdrawal.md', 'graduation_requirements.md', 'attendance_policy.md']
campus store, asked about GRADE APPEALS:
  returned: ['housing.md', 'careers_office.md', 'clubs_and_societies.md']
```

And routing has a visible consequence: `sources_searched` is `['academic']` for
the grade-appeal question, `['campus']` for the Arabic portal question, and
`['academic', 'campus']` for the spanning one.

We chose the multilingual embedding model
`paraphrase-multilingual-mpnet-base-v2` over the English-only
`all-mpnet-base-v2` used in the course lessons, because students ask in Arabic
against a knowledge base written in English with Arabic summaries. Without that
swap the router would route Arabic questions correctly and then retrieve nothing
useful — a failure that is easy to miss, because the routing evidence still
looks perfect. Both models are free and run locally with no API key.

### RAG architecture chosen: **Hybrid**

Not **2-Step**, because retrieval is not unconditional: a classifier decides
which store to search, and `action` questions skip retrieval entirely. Not
purely **Agentic**, because the workers do not decide *when* to retrieve — the
route is fixed before they run, which caps LLM calls per request and keeps
latency predictable.

**Hybrid** in the course's precise sense, with both intermediate steps actually
implemented:

- **Query enhancement** — `rewrite_followup` resolves a follow-up into a
  standalone question using the conversation state for that thread, before it
  reaches the classifier or the retriever.
- **Retrieval validation** — if the routed store returns nothing, the workflow
  re-routes to the other store once before concluding the answer is not in its
  sources.

Hybrid is the right trade for student services because student questions are
frequently ambiguous, underspecified, or span both domains, and confidently
answering from the wrong domain is worse than one extra round-trip.

---

## 4. Context & state management — 15 pts

Short-term and long-term memory are two different objects with two different
lifetimes, and both are wired into the same `@entrypoint`.

**Short-term** is the checkpointer, scoped by `thread_id`. The entrypoint takes
a `previous` parameter and returns `entrypoint.final(value=..., save=...)`, so
each thread carries its last three turns. That state is what `rewrite_followup`
reads, and it is what holds a paused `interrupt()` until it is resumed.

**Long-term** is an `InMemoryStore`, namespaced by `("students", student_id)`,
holding `preferred_language`, `questions_asked`, and `last_topic`. It is read at
the top of every run and written at the bottom. It is *not* a growing list of
chat messages — it is keyed by **student**, not by conversation.

The distinction is visible in one place in the code: `history` comes from
`previous` and dies with the thread; `profile` comes from the Store and does not.

### Cross-thread proof

```
conv-A  questions_asked: 1 | language learned: ar
conv-B  questions_asked: 2 | language recalled: en   <- survived a brand-new thread
conv-C  questions_asked: 1 (different student)

conv-B's answer, in Arabic, without the student asking for Arabic:
آخر موعد للانسحاب هو نهاية الأسبوع العاشر من الفصل الدراسي. [course_withdrawal.md]
```

The counter is the unambiguous proof: the student asked once in `conv-A` and the
count came back as **2** in a completely different thread, while a different
student in `conv-C` started from 1. That value cannot have come from the thread,
because the thread was new.

One honest wrinkle. `conv-B` reports `language recalled: en`, not `ar`, even
though its answer is in Arabic. Both are correct, and the reason is ordering:
the profile is read at the *start* of the run — where `preferred_language` was
still `ar`, which is why the answer came back in Arabic — and rewritten at the
*end* with the language of the current question, which was English. So the field
tracks the most recent language rather than a stable preference. That is a
design weakness, not a memory failure: a real system should keep the first
observed language until the student changes it explicitly, or store a
per-language count. The `questions_asked` counter is unaffected and is the
cleaner proof of the two.

### Short-term proof

```
turn 1: The minimum attendance required is **75 %** of the scheduled contact
        hours. [attendance_policy.md]

  [rewrite] 'What happens if I fall below it?'
         -> 'What happens if I fall below the minimum attendance percentage?'

turn 2: If your attendance in a course drops below the minimum 75 %, you will be
        barred from sitting the final examination for that course and will
        receive a grade of **DN** (Denied). This DN is treated as an **F** for
        GPA purposes. [attendance_policy.md]
```

The `[rewrite]` line is the evidence. "It" was resolved from the previous turn
on the same thread — a question meaningless in isolation was answered correctly
because the thread carried context.

---

## 5. Human-in-the-loop — 10 pts

Two things in Murshid are irreversible: filing a formal grade appeal and
submitting a course withdrawal. A withdrawal cannot be reinstated in the same
term once the Registrar processes it, and the deadline is the end of week 10.
That is a real reason to require human approval, not a decorative one.

The escalation decision is made by the **model**: the router returns
`destination == "action"` when the student is asking to *file* something rather
than asking a question. As section 2 shows, the distinction between *"How do I
withdraw?"* and *"I want to withdraw from STAT301"* is intent, not vocabulary.

Both halves are demonstrated:

```
=== PAUSED FOR ADVISOR APPROVAL ===
  action                : A student advisor must approve this before it is filed
  type                  : withdrawal
  student_id            : s2201
  course_code           : STAT301
  student_stated_reason : the workload is too heavy
```

```
=== RESUMED AND COMPLETED ===
outcome: filed
answer : Withdrawal WD-s2201-STAT301 filed for STAT301.
         Recorded reason: Medical grounds — documentation on file with Student Health.

=== REGISTRY LOG ===
  {'ref': 'WD-s2201-STAT301', 'type': 'withdrawal', 'student': 's2201',
   'course': 'STAT301', 'reason': 'Medical grounds — documentation on file with Student Health.'}
```

The advisor can approve, reject with a note, or **approve with edits** — and the
advisor's edit is what reaches the registry. The recorded reason is *"Medical
grounds"*, not the student's original *"the workload is too heavy"*. A gate that
can only say yes is not really a gate, and a pause that cannot change the
outcome proves nothing.

The rejection path files nothing:

```
PAUSED: CHEM210
outcome: rejected_by_advisor
answer : Your request was not filed. Advisor note: Past the end-of-week-10
         deadline; withdrawal cannot be processed.
registry entries: 1 — unchanged, nothing was filed.
```

There is a second, unplanned route into the same gate. When `parse_action`
cannot extract a request, it falls back to an empty `ActionRequest`, which makes
`get_course_code` interrupt and ask the student. An LLM failure degrades into a
human-in-the-loop prompt rather than a stack trace.

---

## 6. LangGraph Functional API & error handling — 15 pts

Built on the **Functional API** — `@task` for each discrete unit of work and a
single `@entrypoint` orchestrating them with ordinary Python control flow. There
is no `StateGraph` anywhere. Tasks are awaited with `.result()`, and the
two-source branch launches both retrievals before awaiting either.

Three of the four error strategies are implemented.

### Transient — `RetryPolicy`

Every task that calls the provider carries a real `RetryPolicy` object:
`classify`, `rewrite_followup`, `parse_action`, `synthesize`, and `retrieve_one`.
The three tasks that make no provider call — `submit_withdrawal`,
`get_course_code`, `request_approval` — deliberately do not.

To prove the policy fires rather than merely exists, a `SIMULATE_FLAKY` flag
makes the first attempt raise `ConnectionError`:

```
  [retrieve_one] attempt #1 -- raising
  [retrieve_one] attempt #2 (academic)

total retrieval attempts: 2
answer still returned successfully: True
```

Two attempts, no error surfaced, no retry code of our own, and no
`time.sleep()` loop.

**The simulated failure proves the mechanism. A real one proved it was needed —
and that we had it wrong three times.**

Groq's free tier allows 8,000 tokens per minute per model, and a full run
exceeds it:

```
RateLimitError: Error code: 429 - Rate limit reached for model `openai/gpt-oss-120b`
... on tokens per minute (TPM): Limit 8000, Used 7307, Requested 866.
Please try again in 1.297499999s.
```

Three separate mistakes surfaced through that one error:

1. **The policy was on the wrong task.** It was attached to `retrieve_one`, a
   local vector search that can never rate-limit, while `synthesize` — which
   actually calls the provider — had none. We fixed `synthesize`, and the next
   429 came from `classify`, which also had none. We were patching whichever
   task failed last instead of stating the rule. The rule is: *a provider call
   gets the policy.* Applying it uniformly ended the problem.
2. **`retry_on` did not cover the error.** LangGraph's default `retry_on` does
   not include provider SDK exceptions, so even an attached policy would have
   skipped `RateLimitError` silently. The transient types are now named
   explicitly in a `TRANSIENT` tuple.
3. **The backoff was too short to ever work.** Our first schedule was 2s → 4s →
   8s → 16s: 30 seconds total across five attempts. The TPM limit is a
   **60-second** window, so the policy gave up while still inside the window it
   was waiting on. It is now 5s → 10s → 20s → 40s → 45s across six attempts,
   which crosses a window boundary. A retry policy that cannot outlast the thing
   it retries against is decoration.

We also capped retrieved context at 1,800 characters per source, since the
two-source branch concatenates two of them.

### LLM-recoverable — loop back with the error in context

Two real failures, neither simulated.

**First**, the router returned a tool call with an invented key name:

```
"parameters for tool MurshidRoute did not match schema:
 errors: [missing properties: 'confidence']"
failed_generation: {"low": "high", "destination": "campus", ...}
```

The model emitted `"low"` as a *key*. The cause was our own field description,
which began with the bare token *"low when the question is ambiguous…"* — the
model read the leading value name as the key. The fix was in the description,
not the model: every description now opens by describing the field, and enum
values appear only quoted after "Pick" or "Use". This is the routing lesson's
own point turned back on us — the `description=` **is** the prompt.

**Second**, `parse_action` hit `400 tool_use_failed: "Tool choice is required,
but model did not call a tool"` — the model correctly judged that a question
contained no formal request and declined to call the tool at all.

Both are handled the same way, and the handling is deliberately precise:

```python
RECOVERABLE = (ValidationError, groq.BadRequestError)
```

`ValidationError` means the object came back malformed. `groq.BadRequestError`
means the provider rejected the tool call before Pydantic ever saw it. Catching
only `ValidationError` — which is what we wrote first — would have let the
second class of failure crash the workflow. `classify` re-prompts with the error
text in context; `parse_action` degrades into a human prompt.

### User-fixable — `interrupt()`

`get_course_code` pauses and asks when a withdrawal request does not name a
course, then resumes with the supplied value.

### Unexpected — let it bubble up

No workflow task catches a bare `Exception`. The recoverable types are named
explicitly in `RECOVERABLE`, and anything outside it propagates to the caller
where it is visible in the traceback and in the LangSmith trace — exactly where
a genuine bug should appear. The two broad `except Exception` clauses in the
notebook are both in setup code: one lets `load_secret` fall through from Colab
Secrets to environment variables to a `.env` file, and one turns an invalid
LangSmith key into a printed warning instead of aborting the run.

### A design error worth recording

Our first version escalated to the approval gate on
`route.destination == "action" or route.confidence == "low"`. That was wrong. An
ambiguous *question* is not a request to file anything, and sending one to
`parse_action` asked the model to extract a withdrawal from a question that
contained none — which is what produced the `tool_use_failed` error above. Low
confidence now searches **both** stores instead of guessing one, and only
`destination == "action"` reaches the human gate.

### An environment failure worth recording

The course lesson pins `chromadb==0.4.18` with `numpy<2` to work around a
conflict in older Colab images. Our runtime does not preinstall chromadb at all,
so the pin was obsolete — and actively harmful. Forcing numpy down to 1.26 broke
`sentence-transformers` with `ModuleNotFoundError: No module named
'numpy.strings'`, because Colab's scipy and sklearn are built against numpy 2.
Installing a current chromadb and leaving numpy alone fixed it. A pin that
solves yesterday's conflict can cause today's.

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
value, and dates from when LangSmith replaced the original tracer. Both work.)
The key is verified with `client.list_projects()` before the agent runs, so an
invalid key raises loudly instead of producing an empty project page, and
`wait_for_all_tracers()` flushes before inspection.

### What the trace showed

*Measured on the `openai/gpt-oss-120b` run, before we moved to the 20b model for
rate-limit headroom. The proportions are the point, not the absolute figures.*

The two-source run — the heaviest path, since it searches both stores —
completed in **3.29s** using about **1K tokens**, at a cost of **$0.0003**. The
tree matched the intended shape: `murshid` → `classify` → two `retrieve_one`
calls → `synthesize`, with `ChatGroq` and `PydanticToolsParser` nested under
`classify` and `VectorStoreRetriever` under the retrieval that hit the store.

**Neither retrieval nor generation is the bottleneck.** `classify` took 0.52s,
the two retrievals 0.00s and 0.11s, and `synthesize` 0.29s — **0.92s across all
four instrumented tasks, out of a 3.29s run.** Roughly **72% of the wall-clock
time is not inside any task we wrote.** That is LangGraph orchestration and
checkpointer serialisation; this run also deserialised `MurshidRoute` and
`ActionRequest` from a previous turn on the same thread. Our instinct before
opening the trace was to optimise the LLM calls. The trace says the framework
overhead costs more than everything we wrote put together.

**The routing decision costs nearly as much as the answer.** `classify` spent
**472 tokens** and `synthesize` **576** — deciding *where* to send the question
is almost as expensive as answering it, because our `MurshidRoute` field
descriptions are deliberately verbose. There is a real trade here: the same
verbosity that fixed the routing bug in section 6 is what makes the classifier
costly, and on a tier limited to 8,000 tokens per minute that directly caps
throughput.

**The parallel retrieval is architecturally right and practically irrelevant.**
At 110ms combined, the entire retrieval step is noise next to a 3.29s run.

**What we would change:** not the vector store, and not the model. Trim the
`MurshidRoute` descriptions once the router is stable to reclaim those 472
tokens per question, and investigate the unaccounted 2.4s before optimising
anything else — it is the single largest cost in the system, and we only know it
exists because the child spans summed to far less than the parent.

### Evaluation

A dataset of **6 routing examples** — four English, two Arabic, covering all
four destinations — scored with a deterministic grader comparing the predicted
`destination` to the expected one. No LLM judge, so there is nothing to
second-guess.

```
evaluated 6 examples
routes_correctly: 6/6 passed
```

Two earlier experiments on the 120b model also scored 1.00 across 6/6 runs, at
P50 latencies of 0.47s and 0.56s. Those figures cross-check the trace: the
evaluation calls the classifier *only*, and a P50 of ~0.5s matches the 0.52s
`classify` span in the full run — which is what tells us the 3.29s total is
genuinely not the model's doing.

### The failure this section caught twice

Our first evaluation run reported `0it [00:00, ?it/s]` — zero examples scored —
because an earlier crashed run had created the dataset before the examples were
added, and the `if not client.has_dataset(...)` guard then skipped populating it
on every run afterwards. The experiment link existed and looked entirely normal.
It was simply empty.

Rebuilding the dataset fixed the scoring, but the notebook's **saved output
still said `0it`** even after a successful re-run. The cause is that tqdm's
progress bar is a live display widget and Colab persists only its first frame:
the browser showed `6/?` while the file on disk showed `0`. We added an explicit
`print` of the pass count so the saved artefact reflects what actually happened —
which is why the output above reads `routes_correctly: 6/6 passed` on a line of
its own, immediately after the stale `0it` the widget left behind.

Both failures were silent, and both were only visible by reading the artefact
rather than trusting the display. That is a fair summary of why this section
exists.

---

## Known limitations

- The action tools append to an in-process Python list, not a real Registrar
  system. The approval flow, the state handling, and the irreversibility
  semantics are real; the backend is a stub.
- `InMemoryStore` and `InMemorySaver` do not survive a kernel restart. The
  production swaps are `SqliteSaver` / `PostgresSaver` and `PostgresStore`.
- `preferred_language` tracks the most recent language rather than a stable
  preference, as described in section 4.
- The classifier is framing-sensitive on `gpt-oss-20b`: the same question routes
  differently through `router.invoke(q)` and through `classify`, which wraps it
  in one instruction sentence. Section 2 documents the case.
- Short-term history is capped at the last three turns per thread. A longer
  conversation would need summarisation rather than truncation.
- Sixteen documents is small enough that `k=3` similarity search suffices. A
  real corpus of thousands would need reranking and metadata filtering.
- Arabic retrieval worked — *"لا أستطيع الدخول إلى بوابة الطالب"* correctly
  retrieved `student_portal_access.md` and `it_helpdesk.md` from an English
  corpus — but the knowledge base is English with short Arabic summaries on six
  documents, not a genuinely bilingual corpus. We cannot tell from these results
  how much of the success came from the multilingual embeddings and how much
  from those summaries. Testing that properly would mean removing the summaries
  and re-running the Arabic questions.
- The two corpora legitimately cross-reference each other (a transcript can be
  blocked by an unreturned library item). Routing still sends the question to
  the store that *owns* the policy, but "disjoint" is a property of ownership,
  not of vocabulary.
