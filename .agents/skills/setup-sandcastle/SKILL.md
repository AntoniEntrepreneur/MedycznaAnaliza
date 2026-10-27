---
name: setup-sandcastle
description: Install the Sandcastle loop from AntoniEntrepreneur/sandcastle-loop into this project, or update it to a newer version.
disable-model-invocation: true
---

The pack, [AntoniEntrepreneur/sandcastle-loop](https://github.com/AntoniEntrepreneur/sandcastle-loop), has two kinds of file:

- **Pack-owned files.** `manifest.json` lists them. This skill copies them in, and an update overwrites them.
- **Project-owned files.** `sandcastle.config.mts`, `project-notes.md`, `Dockerfile` and `CODING_STANDARDS.md` are drafted once, from `examples/iletobedzie/`, and never touched again.

`.sandcastle/pack.json` in the project records which version is installed and the hash of every pack-owned file. Its presence is what makes a run an **update** rather than an **install**.

Work on a branch, and finish with a PR.

## 1. Fetch the version

The version is the one the user named, otherwise the newest tag: `gh api repos/AntoniEntrepreneur/sandcastle-loop/tags --jq '.[0].name'`. No tags means no export has been merged yet; say so and stop.

Clone it into a scratch directory: `git clone --quiet --depth 1 --branch <tag> https://github.com/AntoniEntrepreneur/sandcastle-loop <scratch>/pack`.

A project with a `.sandcastle/lib/exportPlan.mts` but no `.sandcastle/pack.json` is the loop's **source** (Iletobędzie). It exports to the pack and never installs from it: stop.

**Done when** `<scratch>/pack/manifest.json` exists and you know whether this is an install or an update.

## 2. Guard the project's edits (update only)

For each path in `pack.json`'s `files`, compare `shasum -a 256 <path>` with the recorded hash. A mismatch means someone edited a pack-owned file in this project, and an update would erase that edit. Show the diff between the project's file and `<scratch>/pack/<path>`, and ask how to proceed:

- move the edit into the source repo, so it comes back through an export;
- move it into a project-owned file;
- drop it.

**Done when** every recorded file either matches its hash or has the user's decision.

## 3. Copy the pack-owned files

Copy every path in the new `manifest.json`'s `files` from `<scratch>/pack/` to the same path in the project. On an update, delete the paths that the old `pack.json` lists and the new manifest doesn't, so a file removed upstream goes away here too. This includes an engine patch for an older engine version.

**Done when** the project's pack-owned files are byte-identical to the pack's.

## 4. Draft the project-owned files (install, or a file that's missing)

Each of the four has a worked example in `<scratch>/pack/examples/iletobedzie/`. Draft this project's version by reading the project itself, never by copying Iletobędzie's values:

| File | Read the project for |
|---|---|
| `sandcastle.config.mts` | The setup command (dependency installs that leave the worktree clean), the full check (Makefile, `package.json` scripts, CI workflow), the formatter, lockfiles, the files the image bakes in, the download caches, a glossary with `_Avoid_:` lines |
| `Dockerfile` | The toolchains the check needs. Keep the example's agent user, Claude Code, `gh` and Chromium; swap in this project's languages and dependency layers |
| `project-notes.md` | How to run one test file, which CLIs the agent may `--help`, how to start the app for a screenshot, and the glossary section if there is a glossary |
| `CODING_STANDARDS.md` | Pointers into the project's own standards docs, each with the trigger for reading it |

Show the user each draft before writing it.

**Done when** all four exist and the user has seen each draft.

## 5. Wire up `package.json`

- `dependencies`: `@ai-hero/sandcastle` at exactly the manifest's `engineVersion`, with no caret, because the patch fits one version only.
- `devDependencies`: `tsx`, `patch-package`.
- `scripts`: `"sandcastle": "npx tsx .sandcastle/main.mts"`, `"postinstall": "patch-package"`.

Then run `npm install`; `patch-package` reports applying the Sandcastle patch.

## 6. Install the skills the prompts use

The pack's prompts call these skills by name:

| Skill | Used by |
|---|---|
| `pr` | `pr-prompt.md`, the PR writer |
| `tdd` | `implement-prompt.md`, the implementer (Claude or `agy`) |

For each one, if `.claude/skills/<name>/SKILL.md` is missing, run `npx skills add AntoniEntrepreneur/skills -a claude-code antigravity-cli -s <name>`. It puts the skill in `.agents/skills/<name>/`, where the `agy` implementer reads it, and symlinks `.claude/skills/<name>` to it for Claude. Commit both, because the sandbox works on a checkout of the branch, so a skill that is only on disk never reaches it.

When a pack prompt starts naming a new skill, add it to this table.

**Done when** every skill in the table has a committed `.claude/skills/<name>/SKILL.md`.

## 7. Record the version

Write `.sandcastle/pack.json`:

```json
{ "version": "<tag>", "engineVersion": "<manifest engineVersion>", "files": { "<path>": "<sha256>" } }
```

Include every path in the manifest's `files`.

## 8. Things only the user can do (install only)

- `.sandcastle/.env` (gitignored by the pack's `.sandcastle/.gitignore`) needs `CLAUDE_CODE_OAUTH_TOKEN` (from `claude setup-token`) and `GH_TOKEN`. The user pastes them in; never read the file back.
- The `sandcastle` label admits an issue: `gh label create sandcastle` if `gh label list --search sandcastle` shows none.

## 9. Verify

1. `node --test .sandcastle/lib/` passes.
2. `SANDCASTLE_DRY_RUN=1 npm run sandcastle` plans without error. It reports the sandbox image as missing until the first real run builds it.
3. Commit on the branch and open a PR named after the version.

**Done when** both checks pass and the PR is open.
