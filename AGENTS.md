# AGENTS.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with
project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial
tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:

- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes,
simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:

- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:

- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:

```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it
work") require constant clarification.

## 5. Prose Style

**No semicolons in prose.**

This covers docstrings, comments, and pydantic `Field(description=...)` text.
Use two sentences, or a comma with a conjunction.

- Bad: `"Recycling iterations; more = higher accuracy, slower."`
- Good: `"Recycling iterations. More gives higher accuracy and runs slower."`

Semicolons required by a format are fine, such as an HTTP `Content-Disposition`
header value.

Field descriptions are published API text. Pydantic promotes them into the
OpenAPI schema, so they are read by API consumers, not just developers.

## 6. Communication Style

**Plain language. No jargon. Say less.**

This governs how you talk to the user in replies and explanations, not just
code.

- Write for a smart person who is not deep in this codebase. Choose everyday
  words over technical ones.
- Avoid jargon. If a precise term is truly needed, define it once in a plain
  phrase, then use it.
- Lead with the answer, then the reason. Do not bury the point under setup.
- Be concise and to the point. Cut filler, preamble, and restating the question.
  If a sentence adds nothing, remove it.
- Prefer short sentences and concrete words over long, abstract ones.
- Explain a tradeoff in terms the user can act on, not in internal terminology.
- When the user asks for an explanation, match the depth they asked for. Do not
  pad a simple question with a wall of text.

- Bad: "We should leverage a discriminated union to disambiguate the polymorphic
  payload at the deserialization boundary."
- Good: "Label each form with its type, so the framework reads the label and
  checks it against the right form."

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer
rewrites due to overcomplication, clarifying questions come before
implementation rather than after mistakes, and explanations read plainly and
stay short.
