# Documentation rules

What earns a place in the documentation layer, repo-wide.

## The test

Delete it: would a reader plausibly make a **wrong decision** the code alone wouldn't prevent? If no, it does not belong. Rules and conventions stay; narration of what the code already says goes.

## Prose docs

1. **One home per fact.** State it in exactly one doc; everywhere else links. Two docs describing the same ownership will drift.
2. **Pointers carry a trigger.** `[import-rules.md](...) when adding imports or new modules` — the condition under which to read it, never a summary of it. No limit on how deep pointers nest.
3. **ADRs are a dated record.** Amend one when a rule elaborates the decision it already owns; never trim one for length.
4. **`GLOSSARY.md` defines words, not behaviour.** One or two sentences per term plus `_Avoid_`; what it IS, not what it does. Behaviour belongs in an ADR or in code.

## Docstrings

**Every function and class gets a one-line docstring** (or the language's doc comment) saying what it does, so a reader can scan a module without reading bodies. Route handlers, data models, exception classes, accessors and private helpers all included — role is not a reason to skip it in either direction, and privacy least of all.

**Say more only when the extra line carries something the name and signature cannot:**

- what the function **mutates**
- an **invariant or precondition** a caller must respect
- an **ownership rule** — who may call this, who may not
- **why** it exists when an obvious alternative doesn't work

**Module docstrings are mandatory** on every non-trivial module — one line naming what the module owns.

Style: imperative one-liner.

<!-- One "### <Language>" note per language the repo uses. Seeds:
### Python
No `Args` or `Returns` — the signature is typed. `Raises` only where the exception is part of the caller's contract.
### TypeScript
No JSDoc restating a TS type.
-->

## Comments

Explain **why**, never **what**. A comment narrating the line beneath it goes. Section dividers are fine.

Exempt: constraints imposed from outside the file that would otherwise read as a mistake.

## Tests

**Same rule: a one-line docstring on every test**, naming what it pins down. It does not excuse a vague name — the test name is still the specification, so rename a test that needs its docstring to be intelligible. Where the code can't reveal the origin of a regression (an issue number, a past bug), the docstring is where that goes.
