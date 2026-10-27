# Skills

The skills in `.claude/skills/` are installed from the pack [AntoniEntrepreneur/skills](https://github.com/AntoniEntrepreneur/skills), a fork of `mattpocock/skills`. `skills-lock.json` records each one's source and hash. <Local skills with no lock entry, e.g. "`notion-sync` is the exception: local to this repo, with no lock entry." Drop the sentence when there are none.>

An installed skill is a copy: `npx skills update` overwrites it. So a skill change lands in the pack first, as a PR to the fork (clone at `~/Code/skills`), and reaches this repo through a PR that runs:

```sh
npx skills update -p -y
```

A skill local to this repo is edited here directly.
