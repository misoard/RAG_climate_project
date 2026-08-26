---
description: Review uncommitted work against project standards, then commit and push if clean — or report issues to fix if not
argument-hint: [optional commit message]
allowed-tools: Bash(git status:*), Bash(git diff:*), Bash(git log:*), Bash(git add:*), Bash(git commit:*), Bash(git push:*), Bash(git branch:*), Bash(git rev-parse:*), Bash(uv:*), Bash(pytest:*), Read, Grep, Glob
---

# /ship — review, then commit & push or report back

## Live context
- Branch: !`git rev-parse --abbrev-ref HEAD`
- Status: !`git status --short`
- Uncommitted diff: !`git diff HEAD`
- Recent commits: !`git log --oneline -5`

## What to do

You are shipping the current uncommitted work on a **solo** project. This is a review-and-ship gate, not a teaching moment — be concise. First read `CLAUDE.md` (esp. §3 the agentic-core boundary, §6 the contracts, §7 conventions) and `MILESTONES.md` to know which milestone is active and its **Done when** criteria. Then run the review below and take exactly one of two paths.

### 1. Review the uncommitted changes against this checklist

**Blocking (any one of these means DO NOT commit or push):**
- **Secrets.** No `.env`, API keys, tokens, or credentials are staged or present in the diff. `.env` must appear only as `.env.example`. If you see a real secret, this is blocking even if the user passed a commit message.
- **Ignored artifacts.** Nothing under `data/raw/`, `data/processed/`, `.venv/`, or any `.gitignore`d path is being committed.
- **Broken code.** No broken imports, syntax errors, or references to files that don't exist.
- **Failing tests.** If a test suite exists, run it offline (`uv run pytest -q`) and require green. Tests must not need a network/API key (they use `FakeRouter`). If there are no tests yet (early milestones), note that and continue — not blocking.
- **Scope jump.** The changes belong to the **active milestone**. Work pulled forward from a later milestone is blocking — flag it rather than shipping it.
- **Contract drift.** If `src/contracts/` changed, confirm it was intentional and consistent with `CLAUDE.md §6`. An accidental change to the boundary types is blocking.

**Non-blocking (report, but may proceed):**
- Debug prints, stray `TODO`/`FIXME` in non-critical paths, dead code, hardcoded values that should be config, thin or missing docstrings on non-trivial functions, model names or keys that should come from the `Deployment` registry / `.env` rather than being inline.

### 2. Decide

- **If any blocking issue exists:** do **not** stage, commit, or push anything. Output a numbered, prioritized list (most severe first). For each: the file (and line if you can), why it blocks, and a concrete fix. End by stating clearly that nothing was shipped and what to run `/ship` again after fixing. Stop here.
- **If clean (only non-blocking notes, or none):** proceed to step 3. List any non-blocking notes briefly first so they're not lost.

### 3. Commit & push (only when not blocked)

- Stage intentionally — add the specific files that make up this change. Do **not** `git add -A` blindly if it would sweep in unrelated or ignored files.
- Commit message: if `$ARGUMENTS` is non-empty, use it verbatim. Otherwise write a concise message derived from the diff and the active milestone, prefixed with the milestone tag — e.g. `M0: define boundary contracts and project scaffold`. Keep the subject under ~72 chars; add a short body only if the change needs it.
- Push to the current branch's upstream. If no upstream is set, set it: `git push -u origin <branch>`.
- If the push fails (auth, non-fast-forward, no remote), report the exact error and stop — **never** `--force`, and never create or rewrite remote history to force it through.

### 4. Report

Finish with a short summary: the commit hash and message, the files shipped, the branch/remote pushed to, and any non-blocking notes to address next time.

## Hard rules
- Never commit secrets or `.gitignore`d artifacts, even if explicitly told to in `$ARGUMENTS`.
- Never `git push --force` and never rewrite published history.
- Never push when the review is blocked.
