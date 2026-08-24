# ADTC 2026 -- Technical Report

## Problem
This tool addresses **documentation quality and handoff time**, not record
storage. Most facilities already have a working storage mechanism -- patient
booklets and paper records -- and this project does not attempt to replace
that; a booklet needs no power, no software, and travels with the patient.

What paper does not solve well: a rushed clinician's handwritten note is
often incomplete, illegible to a colleague, and missing fields a referral or
discharge letter requires (exact posology, the structured sections a
specialist expects). Dictating instead of hand-writing is faster and works
while examining a patient. A typed, structured, complete document is usable
by a third party -- another clinician, a specialist receiving a referral, an
insurer, the patient's family -- in a way a rushed paper scrawl often isn't.
The bilingual output (French for the professional record, Swahili for the
patient) also solves a handoff problem paper does not: nobody is translating
a handwritten booklet on the spot.

Target user: a clinician on a $150-500 laptop with no reliable internet,
producing the printable letter that then goes *into* the existing paper
record -- faster and more complete than writing it by hand. Cloud LLM
assistance is unavailable where clinics lack stable connectivity.

## Design Decisions

### Model: Ministral-3-3B-Instruct-2512, GGUF Q2_K
Selected after evaluating several smaller alternatives that all failed for
different reasons on this exact task (French clinical document generation
from a detailed but informally-worded dictation):

- **SmolLM2-1.7B-Instruct**: weak instruction-following in French, drifted
  off-topic, echoed system-prompt phrasing back instead of executing the task.
- **Qwen3.5-0.8B**: a hybrid "thinking" model. Its chain-of-thought reasoning
  consumed the entire token budget before reaching an actual answer, even
  with `enable_thinking: false` requested via the chat template.
- **Luth-1.7B-Instruct** (French fine-tune on Qwen3 base): correctly parsed
  system vs. user roles (confirmed via raw llama-server testing with an
  empty system prompt and no RAG context), but consistently declined to
  generate the requested document, citing "insufficient information" even
  when the request contained complete clinical detail. This behaviour
  persisted across multiple system-prompt rewordings and was reproduced
  with anonymised ("Patient X") and named patients alike, ruling out a
  privacy-trigger explanation. Root cause not conclusively identified;
  suspected to be an alignment/refusal bias from the base model's
  instruction-tuning data around generating "official"-looking documents.
- **Llama-3.2-1B-Instruct**: functional but hallucinated clinical facts
  from a RAG-provided *format example* into the actual patient's letter
  (confused the illustrative example in `modele_compte_rendu.txt`'s
  "EXEMPLE DE TRANSFORMATION" section with real facts about the current
  patient). At this parameter count, the model doesn't reliably keep
  "facts to use" and "format reference" separated under distribution shift.

Ministral-3B was the smallest model tested that reliably (a) followed the
system-prompt distinction between RAG-provided *format rules* and
user-provided *facts*, (b) did not hallucinate details absent from the
request, and (c) did not refuse the task category. Even at Q2_K -- the most
aggressive quantization tested, not the recommended default -- output
quality held up; Q4_K_M was not yet benchmarked head-to-head at submission
time (see Benchmarks section, flagged as an open optimization item).

### Serving architecture: llama-server (HTTP), not llama-cli per-turn
Initial implementation spawned `llama-cli` fresh per conversation turn,
passing the system prompt via `-p` and the question via stdin, then
regex-extracting the reply out of echoed CLI output. This proved unreliable:
whether `-p` is applied as a true templated SYSTEM role or just prepended as
raw completion text depends on the exact llama-cli build and flag
combination, and extraction-regex mismatches silently corrupted what was
treated as "the model's answer." Confirmed via side-by-side comparison: the
same model, given the same prompt, succeeded under Ollama (which sends
role-separated JSON messages to its own internal API) while failing under
the raw-CLI pipeline -- isolating the problem to *how* the prompt was
communicated, not model capability.

Fixed by switching to `llama-server`'s OpenAI-compatible
`/v1/chat/completions` endpoint: real `{role: system/user, content: ...}`
messages, server-side Jinja template application, JSON response with no
banner/echo to parse. The server is started once and reused across turns
(state tracked in `.llama_server_state.json`), avoiding per-turn model
reload cost.

### Quantization
GGUF Q2_K tested and functional for Ministral-3-3B; Q4_K_M (the generally
recommended speed/quality sweet spot per ADTC's own resource list) not yet
benchmarked head-to-head on this hardware at submission time -- see
Benchmarks.

### Grounding: local RAG over report-format templates, not fine-tuning
RAG corpus (`rag/my_docs/`) contains structural rules (expected sections,
formal tone, "never approximate a dosage") and one worked example -- not
patient case data. This is deliberate: the system prompt explicitly
instructs the model that RAG-retrieved content is format guidance only,
and that clinical facts come exclusively from the user's own request.
Chunking splits on markdown `##` headers rather than a fixed character
window, keeping each structural rule intact rather than split mid-sentence
across chunk boundaries.

### African-language handling: deterministic templating, not a second model
Considered and rejected: (a) asking Ministral itself to free-generate the
Swahili section, and (b) loading a second small model (e.g. InkubaLM-0.4B)
for translation. Both were rejected for the same reason -- neither author
can independently verify the correctness of free-generated Swahili medical
text, and Ministral's Swahili training data is thin relative to its French/
English coverage, given it is a European-focused multilingual model. Since
every other design decision in this project has been driven by eliminating
unverifiable hallucination risk (see Model section above), introducing a
new one in the language we are least equipped to fact-check ourselves would
be inconsistent with that standard.

Instead: a human-verified French-Swahili medical term dictionary
(`rag/my_docs/glossaire_medical.txt`) plus fixed Swahili sentence templates,
filled from the same structured facts Ministral already extracts for the
French document. No model call is made for the Swahili side -- pure
substitution against a checked dictionary. Trade-off: covers a fixed set of
common facts/phrases rather than arbitrary open-ended translation of
clinician dictation; judged an acceptable trade for zero hallucination risk
in an unverifiable language, given the timeline. Qualifies for the African
Alpha Bonus (+15%), which rewards meaningful functional coverage in an
African language, not a specific generation technique.

## Constraints
- 7GB RAM hard ceiling, no discrete GPU, offline-only during evaluation.
- French-language clinical documentation follows the structure in
  `format_lettre_sortie.txt` / `modele_compte_rendu.txt` -- currently
  illustrative placeholders (explicitly flagged as such in the corpus
  generation script), not yet validated against a real facility's actual
  templates. **Open item before final submission**: replace with a
  credible, sourced approximation of real hospital document structure.
- The embedder (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`)
  is stored as a persistent, project-local copy rather than relying on the
  system-wide HF cache, so it survives interruptions/power loss and never
  makes a network call at runtime unless explicitly asked to update.

## Benchmarks (development machine)

Measured with `adtc-profiler run --submission . --mode participant --output submission.json`
on an Intel Core i5-6200U (2C/4T, 2.3GHz, no discrete GPU) -- notably *below* the ADTC
Standard Laptop reference spec (10th-12th gen i5 / Ryzen 5 3000-5000). Absolute TPS is
expected to be higher on reference hardware; relative comparisons between quantizations
on this machine remain valid.

| Config | TPS (gen) | Peak RSS | Peak core temp | Throttled | Notes |
|---|---|---|---|---|---|
| Ministral-3-3B-Instruct-2512-Q2_K | 6.61 | 1.76 GB | 75.0°C | No | Baseline, re-confirmed on a second run (CPU p99 90.5%, still 75.0°C peak, not throttled). Well under 85°C thermal cap and 7GB RAM ceiling even on a weak CPU. |
| Ministral-3-3B-Instruct-2512-Q4_K_M | *TBD* | *TBD* | *TBD* | *TBD* | An earlier informal test on a different (RAM-constrained, heavily-swapping) machine appeared to throttle; not yet re-verified with `sensors` on hardware where thermal readings are trustworthy. Re-test before deciding against it -- Q2_K's headroom (75°C peak, 1.76GB peak RSS) suggests there may be room for a higher-quality quant. |

Efficiency: Seff = 100 x ((7GB - 1.76GB) / 7GB) ~= 75/100 at Q2_K.

Note: the profiler's built-in `arc_easy` accuracy score (English, generic knowledge,
lm-eval-harness) is a pipeline sanity check, not the competition's Sacc score. Sacc is
determined by the judge panel evaluating this submission's actual French clinical
prompts (tp_001, tp_002, plus 2 hidden prompts), not by this generic benchmark.
