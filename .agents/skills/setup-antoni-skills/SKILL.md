---
name: setup-antoni-skills
description: "Set up a repo's agent docs (issue tracker, labels, domain docs, documentation rules, project map, skills) and the AGENTS.md pointers to them. Run once per repo, or again to converge an older setup."
disable-model-invocation: true
---

# Setup Antoni's skills

Scaffold the `docs/agents/` layer the engineering skills read, in the shape the seed templates in this folder carry, and point `AGENTS.md` / `CLAUDE.md` at each doc. Issues live on GitHub.

The run **converges**: a repo with nothing gets every doc; a repo set up earlier (by this skill or by `setup-matt-pocock-skills`) keeps what it has and gains what's missing. Explore, show a draft, let the user edit it, then write.

## 1. Explore

Read what exists; done when every item below has an answer.

- `git remote -v`: the GitHub repo. A remote that isn't GitHub ends the run: tell the user this skill only writes the GitHub tracker.
- `CLAUDE.md` and `AGENTS.md`: which exist, which one holds the content (a `CLAUDE.md` that is only `@AGENTS.md` means `AGENTS.md`), and whether it has a pointer list or Matt's `## Agent skills` block.
- `docs/agents/`: each existing doc, and `triage-labels.md` from Matt's setup.
- `GLOSSARY.md` and `docs/adr/` at the root.
- The languages in the repo (file extensions outside vendored and generated folders), for [documentation.md](./documentation.md).
- The top-level folders and what each holds, for [project-map.md](./project-map.md).
- `skills-lock.json` and `.claude/skills/`: skills installed from `AntoniEntrepreneur/skills`, and local ones with no lock entry, for [skills.md](./skills.md).
- `.sandcastle/`: whether the repo runs Sandcastle, for [labels.md](./labels.md).
- `gh label list --limit 100`: the labels the tracker already has.

## 2. Draft

Build each doc from its seed template:

| Doc | Template | Fill in |
|---|---|---|
| `docs/agents/issue-tracker.md` | [issue-tracker.md](./issue-tracker.md) | nothing |
| `docs/agents/labels.md` | [labels.md](./labels.md) | without `.sandcastle/`, drop the Sandcastle section and its family from the opening paragraph |
| `docs/agents/domain.md` | [domain.md](./domain.md) | nothing |
| `docs/agents/documentation.md` | [documentation.md](./documentation.md) | one note per language found in step 1 |
| `docs/agents/project-map.md` | [project-map.md](./project-map.md) | one row per top-level folder |
| `docs/agents/skills.md` | [skills.md](./skills.md) | the local skills with no lock entry |

**Converging an existing doc**: keep every line the repo already has, including its own sections and wording, and add only the template's sections it lacks. A `triage-labels.md` moves into `labels.md`: its rows become the Triage labels table, and the file is deleted.

Then draft the pointer lines for the content file from step 1. One bold-topic line per doc, each ending on the trigger that sends an agent there:

```markdown
- **Layout** — [docs/agents/project-map.md](docs/agents/project-map.md) when locating packages or test roots.
- **Domain language** — [GLOSSARY.md](GLOSSARY.md) before naming concepts; [docs/agents/domain.md](docs/agents/domain.md) for glossary and ADR consumption.
- **Documentation** — [docs/agents/documentation.md](docs/agents/documentation.md) before writing a doc, docstring, or comment.
- **Skills** — [docs/agents/skills.md](docs/agents/skills.md) before editing a skill under `.claude/skills/`.
- **Issues** — [docs/agents/issue-tracker.md](docs/agents/issue-tracker.md) when creating, fetching, labelling, or closing GitHub issues; [docs/agents/labels.md](docs/agents/labels.md) when choosing a label to apply, or reading what one already applied commits you to.
```

Without a `GLOSSARY.md`, the Domain language line points at `domain.md` alone. Lines the file already has for the same doc stay as they are. Matt's `## Agent skills` block is replaced by these lines. With neither file present, ask the user which to create, and draft a `# <project name>` heading and a one-line description from the README above the list.

## 3. Confirm

Show the user every drafted doc and the pointer lines, with each doc marked new or converged, plus the labels step 4 will create. Done when the user approves the draft, with their edits applied.

## 4. Write

Write the docs and the pointer lines. Then create each label from `labels.md` the tracker lacks, with these colours so every repo reads the same:

| Labels | Colour |
|---|---|
| `ready-for-agent` | `0E8A16` |
| `wontfix` | `ffffff` |
| `bug` / `documentation` / `enhancement` | `d73a4a` / `0075ca` / `a2eeef` |
| `spec` / `research` / `grilling` / `future idea` | `a66199` / `78b5c7` / `654321` / `0b505e` |
| every `wayfinder:*` | `5319E7` |
| `sandcastle` | `F9A825` |

```bash
gh label create "<name>" --color <colour> --description "<Meaning column from labels.md>"
```

## 5. Done

Report each doc written (new or converged), the labels created, and the skills that now read them: `to-spec`, `to-tickets`, `implement-spec`, `code-review`, `wayfinder` and `triage` read the tracker and labels; every skill that names domain concepts reads `domain.md`. The docs are the repo's own from here: edit them directly, and rerun this skill after changing a template here.
