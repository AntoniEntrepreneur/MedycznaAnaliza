# Domain docs

How to consume this repo's glossary and ADRs.

## Before exploring

- [GLOSSARY.md](../../GLOSSARY.md) — glossary. Use these terms in names and issue titles; check each entry's `_Avoid_` line.
- [docs/adr/](../adr/) — ADRs that touch the area you're about to work in.

Either may not exist yet: proceed without it. `/domain-modeling` creates them when a term or decision actually gets resolved.

If a concept isn't in the glossary yet, either the project doesn't use that language (reconsider) or there's a gap for `/domain-modeling`.

## ADR conflicts

If your output contradicts an existing ADR, surface it rather than silently overriding:

> *Contradicts ADR-0007 (event-sourced orders) — but worth reopening because…*
