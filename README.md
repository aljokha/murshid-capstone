

# Murshid (مُرشد) — A University Student Services Agent




**Authors:** Khalid ALjohar, Abdullah Alfawzan, Abdulaziz Almeshary, Saud Alghuraybi, Ahmed Bakhashwain, Moath Aljubir
**Training programme:** SDAIA Academy — Building Agentic AI Systems
**Cohort dates:** 16–20 August 2026
**Declared capstone track:** **A — Supervisor + Workers**
(Track C multi-source routing and Track B human escalation are also implemented — see [Architecture](#architecture).)
**SDAIA Academy GitHub:** https://github.com/SDAIAAcademy

---

## What it does

A university's answers live in two very different places. Academic regulations —
grading, attendance, appeals, withdrawal deadlines, graduation requirements —
belong to the Registrar. Campus services — library, IT helpdesk, housing,
dining, careers, wellbeing — belong to Student Affairs. A student asking a
question does not know or care which of those two worlds their question falls
into. They just ask.

**Murshid** takes one free-text question, in Arabic or English, and:

1. **Routes it** with an LLM classifier to the right specialist — Academic
   Affairs, Campus Services, both, or "this is a request to *file* something".
2. **Answers it** from that specialist's own private document store, so a
   question about library hours never retrieves a grading regulation.
3. **Remembers the student** across separate conversations — their preferred
   language, their major, how many times they have asked — so a student who
   asked in Arabic last week is answered in Arabic this week without saying so.
4. **Stops and asks a human** before doing anything irreversible. A course
   withdrawal is deadline-bound and cannot be undone, so the workflow pauses,
   a student advisor approves or edits the request, and only then is it filed.

### Worked examples

| Student asks | Routed to | What happens |
|---|---|---|
| "What GPA do I need to stay off probation?" | `academic` | Searches the academic store only, answers from the probation regulation |
| "لا أستطيع الدخول إلى بوابة الطالب" | `campus` | No English keyword appears anywhere in that sentence — the LLM classifier still routes it correctly, and the answer comes back in Arabic |
| "How do I appeal a grade, and where is the IT helpdesk?" | `both` | Searches both stores concurrently, merges the context, answers both halves. (On `gpt-oss-20b` the *bare* classifier returns `academic` for this one — see WRITEUP §2.) |
| "I want to withdraw from STAT301" | `action` | Pauses. An advisor reviews, edits the stated reason, approves. Only then is it filed |

---

## Architecture

```
                        ┌──────────────────────────────┐
   Student question ───▶│  Pydantic guardrail (Query)  │
                        └──────────────┬───────────────┘
                                       ▼
                        ┌──────────────────────────────┐
                        │  load_profile  (Store read)  │  long-term memory
                        │  ("students", student_id)    │
                        └──────────────┬───────────────┘
                                       ▼
                        ┌──────────────────────────────┐
                        │      SUPERVISOR / ROUTER     │  Track A + Track C
                        │  with_structured_output(     │
                        │        MurshidRoute)         │
                        │  Literal[academic|campus|    │
                        │          both|action]        │
                        └───┬─────────┬─────────┬──────┘
                            │         │         │
              ┌─────────────┘         │         └──────────────┐
              ▼                       ▼                        ▼
   ┌────────────────────┐  ┌────────────────────┐  ┌──────────────────────┐
   │  academic_agent    │  │   campus_agent     │  │   action_agent       │
   │  tools:            │  │   tools:           │  │   tools:             │
   │   search_academic  │  │    search_campus   │  │    submit_withdrawal │
   │   compute_gpa      │  │                    │  │    file_appeal       │
   │   credits_to_grad  │  │                    │  └──────────┬───────────┘
   ├────────────────────┤  ├────────────────────┤             │
   │ Chroma:"academic"  │  │ Chroma:"campus"    │             ▼
   │  8 regulations     │  │  8 service pages   │  ┌──────────────────────┐
   └─────────┬──────────┘  └─────────┬──────────┘  │  interrupt()         │
             │                       │             │  advisor approves /  │
             └───────────┬───────────┘             │  edits / rejects     │
                         ▼                         └──────────┬───────────┘
              ┌──────────────────────┐                        │
              │  synthesize_answer   │◀───────────────────────┘
              │  (style from profile)│
              └──────────┬───────────┘
                         ▼
              ┌──────────────────────┐
              │ save_profile (Store) │  long-term write
              └──────────┬───────────┘
                         ▼
                    Final answer
```

Everything above runs inside a single LangGraph **Functional API** `@entrypoint`,
with a checkpointer (short-term state) and a Store (long-term facts). Every box
is a `@task`.

### Why these choices

- **Track A (Supervisor + workers)** is the declared track because it produces
  the clearest evidence that the *LLM* made the routing decision: the printed
  `transfer_to_academic_agent` / `transfer_to_campus_agent` handoff tool calls.
- **Track C routing** is also implemented because two genuinely separate vector
  stores are what make routing meaningful — if both retrievers query the same
  store, routing changes nothing.
- **Track B escalation** is also implemented because a course withdrawal is
  irreversible, which makes a human approval gate a real requirement rather than
  a decorative one.
- **Model.** Groq decommissioned `llama-3.3-70b-versatile` (the course default)
  on 16 August 2026, so the notebook queries the account for available models
  and picks the first supported one. This run used `openai/gpt-oss-20b`, chosen
  over `gpt-oss-120b` because Groq's free tier allows 8,000 tokens per minute
  per model and a full run exceeds that on the larger one.
- **Multilingual embeddings.** The knowledge base is English with Arabic
  summaries, and students ask in both languages, so the embedding model is
  `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` rather than the
  English-only `all-mpnet-base-v2`. Both are free and run locally with no API
  key. Cross-lingual retrieval is the reason for the swap.

---

## Run it locally

A full local app with a web interface. Ask whatever you like — there are no
canned questions, and it answers in Arabic or English.

```bash
git clone https://github.com/aljokha/murshid-capstone.git
cd murshid-capstone

python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env          # then open .env and paste your GROQ_API_KEY
python run.py
```

It opens `http://127.0.0.1:7860` in your browser. Add `--share` for a temporary
public link.

> First start downloads the embedding model (~1 GB) and takes a few minutes.
> After that it is cached and startup is quick. A free Groq key is enough —
> get one at [console.groq.com/keys](https://console.groq.com/keys).

### What the interface exposes

It is not a wrapper around a chat model. Every message goes through
`murshid.invoke()` — the same workflow the notebook demonstrates — so what you
see is the real system:

| In the interface | What it is showing |
|---|---|
| **How the answer was produced** | Which office the question was routed to, why, and which document stores were searched |
| **Understood as…** | When you ask a follow-up, the standalone question it was rewritten into |
| **Student ID** | Change it and the system treats you as a different person, with different memory |
| **Start a new conversation** | Resets what was *said*, keeps what is known about the *student* |
| **Advisor panel** | Appears when you ask to withdraw from a course. Approve, edit the reason, or reject |
| **Registrar log** | What has actually been filed — showing the advisor's wording, not the student's |

### Things worth trying

- Ask in Arabic: *"لا أستطيع الدخول إلى بوابة الطالب"*
- Ask about attendance, then just ask *"what happens if I fall below it?"*
- Ask something that spans both offices in one sentence
- Ask to withdraw from a course, then edit the reason before approving

### Project layout

```
app/murshid.py    the system — documents, routing, memory, approval gate
app/ui.py         the web interface
run.py            start it
notebooks/        the graded notebook, with every result saved
```

---

## How to run

### Option A — Google Colab (recommended, no local setup)

No GitHub account required to run it.

1. Go to [colab.research.google.com](https://colab.research.google.com) and
   **Upload** `notebooks/murshid_capstone.ipynb`.
2. Open the **Files** panel (folder icon, left sidebar) and upload
   `murshid-capstone.zip`. The notebook unzips it automatically.
3. Add your `GROQ_API_KEY` to the Colab **Secrets** panel (the key icon in the
   left sidebar) and enable notebook access. Do not paste keys into a cell.
4. Optionally add `LANGSMITH_API_KEY` the same way, for section 8.
5. **Runtime → Restart session and run all.**

Once the project *is* on GitHub, you can skip the zip upload by setting
`REPO_URL` in section 2 and letting the notebook clone it instead.

### Option B — Local

```bash
git clone https://github.com/aljokha/murshid-capstone.git
cd murshid-capstone
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env      # then edit .env and add your keys

python -m src.ingest      # builds and smoke-tests both vector stores
jupyter notebook notebooks/murshid_capstone.ipynb
```

The embedding model downloads once (~1 GB) on first run and is cached
afterwards. It needs no API key.

---

## Repository layout

```
murshid-capstone/
├── README.md                     this file
├── WRITEUP.md                    one paragraph per rubric section
├── requirements.txt
├── .env.example                  variable NAMES only, no values
├── .gitignore                    excludes .env, *.db, chroma/, caches
├── notebooks/
│   └── murshid_capstone.ipynb    the graded artefact — run top to bottom
├── src/
│   └── ingest.py                 standalone load → split → embed → store
├── data/
│   ├── academic/                 8 academic regulation documents
│   └── campus/                   8 campus service documents
└── docs/
    ├── architecture.md           design decisions in detail
    └── evidence/                 screenshots (LangSmith trace, handoffs)
```

The notebook is the single source of truth for the graded logic. `src/ingest.py`
is the one piece deliberately duplicated outside it, because building the vector
stores is useful as a standalone step. Nothing else is mirrored, so there is no
risk of the notebook and a module drifting apart.

---

## Rubric map

| # | Rubric section | Pts | Where the code is | Where the evidence is |
|---|---|---|---|---|
| 1 | Agent fundamentals | 15 | Notebook §7 (tools), §8 (`MurshidRoute`) | Printed tool call with arguments |
| 2 | Multi-agent / routing | 15 | Notebook §8 (classifier), §9 (supervisor) | Routing table + printed `transfer_to_*` calls |
| 3 | RAG pipeline | 15 | Notebook §5–6, `src/ingest.py` | Smoke test + cross-store isolation test + Hybrid justification in WRITEUP |
| 4 | Context & state | 15 | Notebook §10 | Cross-thread test: conv-A → 1, conv-B → 2, conv-C → 1 |
| 5 | Human-in-the-loop | 10 | Notebook §11 (`request_approval`) | Interrupt payload + `Command(resume=...)` + advisor's edit in the registry |
| 6 | Functional API & errors | 15 | Notebook §11–12 | `@task`/`@entrypoint` throughout; printed retry attempt #1 → #2 |
| 7 | Workflow pattern | 10 | Notebook §8 routing branch | Named **Routing** explicitly in WRITEUP |
| 8 | LangSmith | 5 | Notebook §3, §18 | Trace observation in WRITEUP + evaluation experiment |

Full write-up: [WRITEUP.md](WRITEUP.md). Design detail: [docs/architecture.md](docs/architecture.md).

---

## Data

The 16 documents in `data/` describe a **fictional** institution, "Al-Noor
University". They were written for this project. No real university policy, no
personal data, and nothing confidential is included. The structure is designed
so real documents can replace them file-for-file without changing any code.

---

## Acknowledgements

Built for the SDAIA Academy programme **Building Agentic AI Systems**, following
the course material at
[mohammadyusif.github.io/agentic-ai-systems](https://mohammadyusif.github.io/agentic-ai-systems/)
(Days 1–4 by Hassan Algoz; Day 5 and the capstone preparation material by
Mohammad Yusif). 
