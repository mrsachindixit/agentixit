# Code sample style guide

Every sample under `code/` follows these rules. They are deliberate, and several of them
are the opposite of what you would do in production code — the goal here is that a
student can read one file top to bottom and see the whole idea.

## Teaching philosophy
- Keep examples concept-first and runnable.
- Prefer inline procedural code over heavy class hierarchies.
- Keep files succinct and practical.
- Avoid over-engineering and avoid academic/geeky framing.
- Show baseline vs improved patterns when teaching optimization or evaluation.
- Use real model invocation where feasible for demonstrations.

## The core rule: inline, hands-on, no prose
Every sample is something a student can open, **uncomment a line or tweak a value, and immediately run** to see the focus concept in action. The code itself is the lesson — the explanation lives in the book, never in the file.
- **No textual explanation inside code files.** No essay-style module docstrings, no banner comment blocks narrating "what this file is about", no paragraphs of prose. If you feel the urge to explain at length, that text belongs in the book, not the `.py` file.
- **Inline over elegant.** We deliberately skip "good engineering" niceties — shared utility modules for common work, base classes, dataclasses, wrapper abstractions — because they hide the very mechanics we are teaching. Duplication a student can read top-to-bottom beats a clever abstraction they have to jump around to follow.
- **Short and focused.** One file delivers exactly one concept, as directly as possible. Prefer a 40-line script over a 120-line "well-structured" one.
- **Comments are for pointing, not explaining.** A short inline comment to label a step or flag a line worth uncommenting is fine. Multi-line teaching prose is not.

These rules apply **retroactively**: when revisiting or editing any existing file, refactor it toward this style (strip prose headers, inline gratuitous classes/dataclasses) rather than preserving the old shape.

## Audience: capable engineers, but not Python experts
Many students come from other languages (Java, C#, Go, JS, ...) and use Python only because it is the lingua franca of AI. **Every line of Python cleverness they have to decode is attention stolen from the agentic-AI concept we are actually teaching.** So minimize the Python-language burden and spend the student's focus on the AI idea.
- **Prefer plain, explicit constructs over Pythonic shortcuts.** A simple `[x for x in items if cond]` list comprehension is fine — most people from other languages read those easily. The problem is *compound* cleverness: an explicit variable beats a walrus `:=`; a clear `if x is None` beats a terse `x or default` when the intent isn't obvious.
- **Avoid idioms that only Python natives parse at a glance:** nested or multi-`for` comprehensions, a comprehension with a `next(..., -1)` sentinel or another comprehension inside it, chained ternaries, `*unpacking` tricks, dunder access, metaclasses, decorators-as-cleverness, `contextvars`, comprehension side effects. A *single* simple comprehension is fine; the moment it nests or hides control flow, unroll it into a loop. If a Java/C# developer would have to stop and look it up, rewrite it plainly.
- **Spell out the steps.** Build a list with an obvious loop rather than a dense one-liner, even if it costs a few extra lines. The extra lines read top-to-bottom; the dense line does not.
- **Names over syntax.** Give intermediate values clear names so the code documents itself, instead of relying on the reader knowing what a slice, unpack, or comprehension yields.
- **The one exception:** if the Python construct *is* the lesson (e.g. async/await in a streaming chapter), use it — but introduce it plainly and comment the one line that matters.

## Flat script over functions + loops when teaching one concept
If a file demonstrates a single idea, write it as a linear script that reads like a recipe: set up the inputs, do the one thing, print the result. Do not introduce helper functions, a `run_case`-style wrapper, or a loop over multiple cases just to look "structured". A student should read the file top to bottom and see the whole story without jumping between function definitions and call sites.
- **One concept = one straight-line script** (ideally ~40 lines), not a mini-framework.
- **Don't define a function that is only called once** — inline its body.
- **Don't loop over a list of cases when a single concrete case makes the point.** Use one real example (e.g. "check Berlin's details, compare to our answer, print the verdict") instead of a generic case-runner.
- **Add a second case or a function ONLY when repetition/iteration is itself the lesson** (e.g. a regression suite, a batch eval). Otherwise keep it singular and concrete.
- **End with the actual result or verdict printed inline,** not returned from a function.
- **Code the happy path, not every hypothetical.** Read the one field or response shape the concept actually uses. Don't write multi-branch parsers or try/except fallback ladders to handle inputs that won't occur in the demo — defensive ceremony hides the lesson as much as a wrapper class does.
- The test: if you can delete a `def` and a `for` and the lesson gets *clearer*, delete them.

## Code style
- Keep scripts short and readable.
- Use clear variable names.
- Minimize abstraction unless it clearly improves learning.
- Avoid unnecessary comments; code should read naturally.

## Code samples — MANDATORY style rules
These are non-negotiable for every `.py` file added under this repo:
1. **Inline, direct, learning-first.** Top-level functions only. Do NOT introduce classes, dataclasses, pydantic models, or utility modules unless the chapter being taught is specifically about that abstraction.
2. **No text introduction at the file header.** No multi-line module docstring, no banner comment block describing what the file is "about". The file should start with imports, then code. The book explains the concept; the code shows it.
3. **No abstraction for its own sake.** Prefer a 40-line script over a 200-line one with helpers. If you find yourself extracting a helper, ask whether the learner gains more by seeing the duplication inline.
4. **Tiny dependency surface.** Stick to `utils/ollama_client.py`, the Python standard library, and (where the chapter requires it) LangChain/LangGraph. Do not pull in pydantic, dataclasses, attrs, or similar unless the topic is schema/validation itself.
5. **Print, don't return.** Sample files run end-to-end and print observable output. Avoid building libraries.

## Curriculum style
- One file should teach one clear idea.
- Preserve progressive learning order across modules.
- Favor deterministic scoring snippets around LLM calls so outcomes are inspectable.

## Book and code stay in their own lanes
- Explanation lives in `book/`. Code lives in `code/`. A sample file does not
  re-explain what its chapter already covers.
- When the two need to point at each other, link by path and anchor
  (`book/03-execution-patterns.md`, `code/module03_langchain/3.5_lg_basic.py`).
  Do not copy prose from the book into code comments, or code out of the samples
  into the book beyond the short illustrative snippets already there.
- Keep the numbering aligned. If a sample is renumbered or renamed, update the
  chapter that references it and its test in `code/tests/` in the same change.
