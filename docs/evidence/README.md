# Evidence

Screenshots and captured output that back the claims in
[`../../WRITEUP.md`](../../WRITEUP.md).

Put the following here as you produce them. Every one of these corresponds to a
specific thing a grader looks for.

| File | What it shows | Rubric section |
|---|---|---|
| `01_routing_table.png` | The 7-question routing table, including the two Arabic questions and the spanning question | §2 |
| `02_supervisor_handoffs.png` | Printed `transfer_to_academic_agent` and `transfer_to_campus_agent` | §2 |
| `03_retrieval_smoke_test.png` | Verbatim text retrieved from your own documents | §3 |
| `04_cross_thread_memory.png` | conv-A → 1, conv-B → 2, conv-C → 1 | §4 |
| `05_interrupt_and_resume.png` | The interrupt payload, the resume, and the advisor's edit in the registry log | §5 |
| `06_retry_fired.png` | `[retrieve] attempt #1` → `attempt #2` | §6 |
| `07_langsmith_trace.png` | The run tree with per-step latency visible | §8 |
| `08_langsmith_experiment.png` | The routing evaluation experiment results | §8 |

The notebook already contains all of this as saved cell output. These
screenshots are for the README, the write-up, and your presentation — a grader
should not have to open the notebook to see that the system works.
