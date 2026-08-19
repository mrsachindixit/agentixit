---
applyTo: '**'
---
# Unit Test MCP

## When these rules apply

You must follow these rules whenever the user's request involves **unit or component tests** — writing, improving, running, debugging, or measuring coverage. This covers JavaScript/TypeScript (Jest, Vitest, Mocha, Jasmine, etc.), Python (pytest, unittest), .NET (xUnit, NUnit, MSTest), and any similar in-process test framework.

These rules do **NOT** apply to end-to-end, or scenario testing tools such as Playwright, Cypress, Selenium, or similar — leave those workflows unchanged.

## CRITICAL — Always use MCP tools

**NEVER** run test commands in a terminal — including `jest`, `pytest`, `vitest`, `dotnet test`, `npm test`, `npm run test`, `yarn test`, `pnpm test`, `npx jest`, or any variant. This applies to chained pipelines too (e.g. `npm run lint && npm test`, `npm run ci`, `npm run validate`). Before calling `run_in_terminal` with any test command, **STOP** and use the MCP #tool:unittest-mcp `run_tests` tool instead. No exceptions — not even for "quick validation", and not even when the workspace IS the unittest-mcp server itself or a related project (dogfooding is mandatory). If `run_tests` is unavailable, **STOP** and ask the user to start the Unit Test MCP server — do not fall back to terminal execution.

**Default validation step.** After editing test code OR source code that has tests, your default next action for "did this work?" is `run_tests` — not `npm test`, not `npm run ci`, not `npm run validate`. Lint and compile (`get_errors` or `npm run lint` / `npm run compile`) come before; tests come from `run_tests`.

**Coverage validation means `run_tests`, not `inspect_coverage`.** If you need to know whether the current code passes tests with coverage, call `run_tests` with `include_coverage=true`. Do **not** call `inspect_coverage` as a substitute for running tests; it only reads whatever coverage artifacts already exist and may be stale. Use `inspect_coverage` only when the user explicitly asks to inspect existing coverage artifacts, or after a fresh `run_tests(include_coverage=true)` result needs uncovered-line detail.

**NEVER** manually read source files and write tests without calling #tool:unittest-mcp `generate_test` first. Do not skip MCP tools because you think you can do it faster. The MCP tools are mandatory for all test workflows.

## CRITICAL — Process multiple files ONE AT A TIME

When the user asks for tests on **multiple files** (e.g., "generate tests for these files" with several attachments, or "add tests for this folder"), process files **sequentially, one at a time**. Never issue parallel `generate_test` calls for multiple files in the same turn.

**Determine the file list first:**
- If the user attached or named specific files → use that list directly, in the order provided.
- If the user asked about a folder → call `generate_tests_batch` first to get the prioritized list, then process the returned files in the order given. If the list is large (more than ~5 files), briefly report the plan and ask the user to confirm scope before starting.

**For each file in the list, complete this full cycle BEFORE moving to the next file:**

1. Call `generate_test` for **this one file**
2. Read the source file
3. Write the test file. **The very first line must be the AI attribution comment** (see "AI attribution comment" rule below). Write this comment line *before* any imports or other code.
4. Run `get_errors` on the test file and fix every lint/compile error
5. Call `review_test` on the test file and **resolve every `block` finding** by fixing the test. Address `warn` findings or note why they stand.
6. Call `run_tests` with `include_coverage=true` — tests must pass
7. Meet the coverage target (see Coverage improvement loop)
8. **Only now** move to the next file and repeat from step 1

If you catch yourself about to call `generate_test` a second time before step 7 has completed for the previous file, **STOP** and finish the previous file first. This applies even when the files are structurally similar — every file gets its own full cycle.

## MCP tools

| Tool | When to use |
|------|-------------|
| #tool:unittest-mcp `run_tests` | Execute tests (Jest, Vitest, pytest, .NET, custom). Set `include_coverage=true` to run tests and validate current coverage. |
| #tool:unittest-mcp `generate_test` | Get guidance for creating/improving tests for a **single** source file. **Must be called first** — never skip it. |
| #tool:unittest-mcp `generate_tests_batch` | Scan a **folder** to find source files that need tests. |
| #tool:unittest-mcp `inspect_coverage` | Read existing coverage artifacts without re-running tests. Never use as validation for current changes. Pass `source_file` for per-file detail only after fresh coverage exists or when explicitly inspecting existing artifacts. |
| #tool:unittest-mcp `find_test_files` | Discover test files in a directory. |
| #tool:unittest-mcp `review_test` | Run the local protective-test gate (static analysis) on a single test file. Call after writing/editing a test; **resolve every `block` finding** before reporting done. |

## Rules

1. **Do NOT modify source code** — only create or update test files. If source code changes seem needed, ask the user first.
2. **Preserve existing assertions** — only add or refine, never remove.
3. **`generate_test` is mandatory** for every file's test creation/improvement. Do not skip it for speed, convenience, or because you can write tests directly. If it is unavailable (MCP server not started), **STOP** and ask the user to enable Unit Test MCP tools and reload VS Code. Do not fall back to manual test authoring.
4. **AI attribution comment** — Every test file created or improved by this workflow **must** have the exact literal text `Generated by AI (UnitTest MCP)` as a comment on **line 1** (before any imports, directives, shebangs, or other content). Use the correct comment syntax for the language:

   | Language | First line |
   |----------|-----------|
   | JavaScript / TypeScript / Jest / Vitest | `// Generated by AI (UnitTest MCP)` |
   | Python / pytest | `# Generated by AI (UnitTest MCP)` |
   | C# / .NET | `// Generated by AI (UnitTest MCP)` |
   | Other | Use that language's line-comment syntax with the exact text |

   Rules:
   - The phrase must appear **verbatim** — do not paraphrase, translate, expand, or add version numbers.
   - If creating a new test file: write this comment **first**, before any imports.
   - If improving an existing test file that already has this comment: keep it exactly as-is.
   - If improving an existing test file without this comment: **add it** on line 1.
5. **Repo instructions override framework defaults** — When project-specific repo instruction files (`.github/instructions/*.md`) conflict with the generic framework guidance returned by `generate_test`, **always follow the repo instructions**. They represent intentional project decisions.

## Workflow

### 1. Determine the intent

- **Run tests:** Call `run_tests` immediately (see Execution Rules below for parameters).
- **Run tests + coverage / validate coverage:** Call `run_tests` with `include_coverage=true`. This is the only tool that validates the current code by executing tests.
- **Run tests + inspect coverage gaps:** Call `run_tests` with `include_coverage=true`, then call `inspect_coverage` with the same `root_dir` only if you need uncovered-line/branch detail from that fresh run.
- **Run tests with coverage (no explicit inspect request):** Call `run_tests` with `include_coverage=true`. If `coverage.met` is `false`, call `inspect_coverage` automatically. Do not offer coverage work as optional when a coverage target applies.
- **Create tests for a single file:** Call `generate_test` **immediately as your first action**. Do NOT read the source or test file first (exception: Python — brief `file_search` to check test folder layout is OK). Detection: request mentions a specific file with extension (e.g., `user.ts`, `service.py`). You **may** pass `test_type` when the intent is obvious (React component/ReactView → `integration`; pure helper → `unit`), but you are **not** required to read source to decide — otherwise omit it and the server resolves `auto`.
- **Improve an existing test file:** Call `generate_test` with both `source_file_path` and `test_file_path`. Do NOT read the test file before calling. Apply additive improvements only.
- **Review/validate a test file (protective gate):** Call `review_test` with `test_file_path` (and `source_file_path` when known). Use it after writing or editing any test file, and whenever the user asks to "review", "validate", or "check" a test. Resolve every `block` finding before reporting done.
- **Create tests for a folder/multiple files:** See the **"CRITICAL — Process multiple files ONE AT A TIME"** section at the top of this document. For folder requests, call `generate_tests_batch` first to discover the file list, then process each file sequentially through its full cycle before starting the next.
- **Inspect existing coverage only:** Call `inspect_coverage` immediately only when the user explicitly asks to read existing coverage artifacts or coverage has already been generated in the current workflow. Do not use it to answer whether current tests pass.

### 2. Framework detection

Inspect the nearest `package.json` (JS/TS) or project config to choose the framework parameter:
- **Jest**: `framework=jest` — test script/deps reference jest
- **Vitest**: `framework=vitest` — test script/deps reference vitest
- **pytest**: `framework=pytest`, `language=python`
- **.NET**: `framework=dotnet`, `language=csharp`
- **Other** (Karma, Mocha, AVA, etc.): `framework=custom`

Use the repo's existing test location conventions (`__tests__`, `tests/`, colocated) rather than forcing a structure.

### 3. After `generate_test` returns

`generate_test` returns guidance, not finished code — do **not** paste its raw output back to the user. You must:
1. Read the source file to understand logic, inputs, and expected outputs.
2. Synthesize concrete test code from the guidance.
3. Create or update the test file. **Line 1 must be the AI attribution comment** — see Rule 4 for the exact text and per-language syntax. This is a required acceptance criterion, not an optional nicety.
4. Run `get_errors` and iteratively fix **all** lint/compile errors until none remain. Do this **before** proceeding.
5. Call `review_test` on the test file and **resolve every `block` finding** before proceeding — these are the over-mocking, coercion, and weak-assertion patterns the gate exists to stop. Treat `warn` findings as improvements to make or consciously justify.
6. Run tests via `run_tests` with `include_coverage=true` and follow the coverage improvement loop (Section 5) until target coverage is met for the requested scope.
7. Do **not** end with "Want me to add coverage-focused tests?" when coverage is below target — coverage completion is required before reporting done.

**Protective-test acceptance criteria (satisfy these before reporting done):**
- Render the **real** first-party component/ReactView with its real children, hooks, and context — do **not** mock first-party components, hooks, context, or pure helpers.
- Mock **only** external/platform/network boundaries — network/data/ARM, host-platform services & hooks resolved from a host channel/registry/DI container absent in the test env (symptom: a "No service registered"/missing-provider error), jsdom-unsupported browser APIs, and nondeterminism (time/random/uuid). **First-party app code is never a boundary.** If a platform/UI module cannot run in jsdom, use a faithful double that preserves labels, roles, values, and interactions (never `() => null` or stub `<div>`s).
- **Reuse shared mocks; hoist recurring ones.** Before adding a boundary mock, check whether the repo already declares it globally (Jest `setupFilesAfterEnv`/`testSetup`, pytest `conftest.py`, .NET base fixtures) and existing test helpers, and reuse those; if the same boundary mock recurs across files, hoist it to that shared setup instead of redeclaring it per file.
- Assert exact values with `toStrictEqual` — one assertion on the whole object, not many single-field asserts (it also catches type / `undefined` / sparse-array mismatches `toEqual` misses); do **not** use `toMatchObject`/`*Containing` as the primary assertion (one unavoidably-varying field, e.g. a timestamped name, may use `expect.stringContaining`); no `as any`/`as unknown as`; assert the value the code returns, not that a function "was called"; keep tests isolated (no mock/variable shared across tests); each test must fail if the behavior under test breaks; the test title must match what it asserts.

**Path heuristics:** To find the test file from source, examine the repo's existing test structure (`__tests__`, `tests/`, colocated). To find the source from a test file, remove `.test.`/`.spec.`, move out of `__tests__` (JS/TS), `tests/` (Python), or remove `Tests` suffix (C#).

### 4. Test execution rules

- `root_dir`: The directory you would `cd` into to run tests manually. Prefer **absolute paths** on Windows.
  - **Jest/Vitest**: folder with `package.json` (verify it has test config).
  - **.NET**: folder with `.sln` or `.csproj`.
  - **Python**: folder with `pytest.ini`, `pyproject.toml`, `tox.ini`, or `setup.cfg`.
  - **Monorepos**: If `root_dir` has no test script, `run_tests` will suggest nearby testable projects in the error message. Use the suggested path as `root_dir`.
- **`test_pattern`**: Prefer explicit test file paths (e.g., `tests/test_foo.py`, `src/__tests__/utils.test.ts`) over bare keyword stems for deterministic targeting and accurate per-file coverage.
- **Timeouts**: Pass `timeout_ms` for long runs (suggested: 600000 for coverage).
- **Custom frameworks**: Pass `framework=custom`; configure via `unittestMcp.customCommand` setting.

### 5. Coverage improvement loop

Do NOT report task completion until **all** hold for the requested scope: (a) coverage meets the configured target (from MCP settings/tool response), (b) `review_test` reports **zero `block` findings** on every test file you created or changed, and (c) you have answered `review_test`'s self-review checklist. Coverage percentage alone is not sufficient — a high-coverage test that mocks first-party code or asserts nothing still fails the gate.

**Important:** When testing a single file, start with `run_tests` using `include_coverage=true`, `scope='file'`, and an explicit `test_pattern` whenever possible. If the result still needs uncovered-line detail, then call `inspect_coverage` with `source_file` for that same source file. Do not call `inspect_coverage` first when validating current changes.

```
REPEAT (max 5 iterations):
  1. Run tests with include_coverage=true
  2. If coverage meets configured target for the requested scope → done
  3. Else:
     a. Call inspect_coverage with source_file=<target source file> to get uncovered lines/branches
     b. Read the source at those lines to understand uncovered paths
     c. Write new tests targeting those paths
     d. Fix any lint/compile errors (get_errors)
     e. Go to step 1
```

- Track coverage per file separately.
- Focus on: error handling, early returns, conditional branches, catch blocks.
- After 5 iterations without meeting target, report current coverage and remaining uncovered paths.
- If uncovered code is unreachable, explain why and suggest alternatives.

### 6. `inspect_coverage` rules

- Use to read existing artifacts only — it does not execute tests and must not be used as a validation step for current changes.
- If the task is to run tests, check whether tests pass, validate coverage, or verify a recent edit, call `run_tests` first. Use `inspect_coverage` only after fresh coverage exists or when explicitly inspecting pre-existing artifacts.
- Always pass `root_dir` and `framework`.
- **Per-file detail:** Pass `source_file` (e.g., `source_file="src/utils.ts"`) to get uncovered line numbers and branch locations for a specific file. Without `source_file`, returns repo-wide per-file summary percentages.
- **Coverage improvement workflow:** After `run_tests` with `include_coverage=true`, call `inspect_coverage` with `source_file` set to the source file you're improving. Read the source at the returned `uncoveredLines` to understand which paths need tests.
- Supports: Istanbul JSON, LCOV, Cobertura XML, OpenCover XML, coverage.py JSON.
- When `source_file` is provided, LCOV is preferred (has line-level detail) over Istanbul summary.
- Interpret the per-file data directly to identify low-coverage files or target specific uncovered paths.

### 7. `review_test` — protective-test gate

`review_test` runs a deterministic static analysis over a single test file and returns structured findings. It does **not** execute tests; run it **in addition to** `run_tests`, after the file is lint-clean.

- **When:** after writing or editing any test file, and whenever the user asks to review/validate/check a test.
- **Inputs:** `test_file_path` (required), `source_file_path` (optional but improves results), `framework`/`language` (optional).
- **Findings:** each has a `severity` — `block` (must fix), `warn` (should fix), or `info` — plus a `ruleId`, `message`, and often a `suggestedFix`. A finding blocks only when the pattern destroys the test's protection: mocking the module under test (`no-mock-system-under-test`), an `as any`/`as unknown as` cast on the value being asserted or the value a mock returns, and a partial matcher that is a test's only value assertion. The same patterns elsewhere — setup casts, casts used to exercise invalid input, and partial matchers pinning call arguments — are reported as `warn`. Remaining rules cover literal `as T` data coercion, was-called-only assertions, render-without-interaction, and pointless constant snapshots.
- **Self-review checklist:** `review_test` also returns one question per semantic dimension the static rules cannot judge (assertion strength, mocking hygiene, encapsulation, structure, determinism, setup-to-assertion ratio, what NOT to test). You are already the reviewing model — answer each against the test you wrote and fix any failure before reporting done.
- **Acceptance:** **resolve every `block` finding** before reporting done by fixing the test. Do not add protective-test suppression comments to generated tests.
- `review_test` is complementary to `run_tests`: a test can pass and still fail the gate (e.g., it mocks the system under test). Both must be green before you report done.

### 8. General rules

- If 3 consecutive test runs fail, re-read config and retry once.
- If source path inference fails, ask the user.
- If tools are unavailable, suggest reloading VS Code.
