# Design notes

Decisions that are not obvious from the code, with the evidence that forced them. Newest
first. The README carries the two that teach the most; the rest live here.

## Provider extras on tool calls must be echoed back (2026-09-09)

Gemini's OpenAI-compatible endpoint attaches `extra_content.google.thought_signature` to
every tool call it emits and rejects the next turn with HTTP 400 ("Function call is missing
a thought_signature in functionCall parts") unless the field comes back verbatim. The client
had been rebuilding the assistant message from id, name, and arguments, which is all the
OpenAI schema promises, so every Gemini-answered tool question died on turn two. Locally
this never showed because Groq answered first; it surfaced in CI only when Groq's daily cap
sent the chain to Gemini, and it took two knowledge-base rows and both tool rows with it.
Fix: `_echo_tool_call` in `llm/client.py` starts from the SDK's full dump of the tool call
and overlays the standard fields. Lesson: "OpenAI-compatible" covers the request shape, not
the conversation protocol; anything a provider returns on a tool call is part of its state
and must round-trip.

Second half of the same lesson, found in the very next CI run: a conversation whose first
turn was answered by Groq and whose second turn failed over to Gemini carries a Groq tool
call with no signature to echo, and Gemini rejects it the same way. Google documents a
placeholder signature, `skip_thought_signature_validator`, for calls the model did not
produce; `_messages_for` in `llm/client.py` adds it to any unsigned tool call before a
request goes to Gemini, so a chain can fail over in the middle of a tool conversation.

## The gate enforces only when the answer model matches the baseline (2026-09-09)

The smoke baseline was answered by `groq:openai/gpt-oss-20b`. When Groq is capped, CI
answers come from whichever provider is next, and a faithfulness gap between two different
models is not evidence about the commit under test. `gate.py` records the dominant answer
model of both reports (`dataset.answer_models`, reflection entries excluded) and, with
`--enforce-only-on-baseline-model`, downgrades a failing comparison to exit 3 when the models
differ. CI turns exit 3 into a workflow warning that names both models. The faithfulness
margin stays at 0.05; nothing about the threshold moved. The same separation applies one
step earlier: `make preflight` fails on a rejected key, and an eval that exits 2 because
every provider is capped is also a warning, not a failure.

## Citation markers are normalised before enforcement (2026-09-08)

gpt-oss writes `【1】`; the guardrail matched `[1]`, so correct answers were stripped to
nothing and reported as refusals. `normalize_markers` rewrites full-width, grouped, and
spaced variants before reflection and enforcement. A deterministic guardrail is only as good
as its tolerance for the output conventions of every model that can reach it.

## Context precision is scored one passage per call

RAGAS's stock `context_precision` issues five sequential few-shot prompts per row; that
never fit an 8k tokens-per-minute window. Each passage is judged in its own lean call and
combined with average precision, which is what the metric computes anyway.

## Judges and answers are chains, and every score records its judge

Free tiers cap per minute and per day. Each role (answer, reflect, judge-per-metric) is an
ordered chain of `provider:model` entries; the client moves on when a daily cap or a long
wait is reported and stays moved. Reports keep `answer_models` and `judges` per metric so a
failover onto a weaker model is visible in the numbers rather than hidden in them.

## The deployed image uses ONNX embeddings and SQLite

The free host allows 512 MB. Torch alone breaks that, so the image embeds with
`onnxruntime` and `tokenizers` (cosine parity 1.000 with the sentence-transformers path,
checked in tests) and serves the tools from SQLite seeded at build time. Local development
keeps Postgres so the MCP tools run against the same repository interface they would in
production.
