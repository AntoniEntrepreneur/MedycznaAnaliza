# Labels

Every label in this repo's issue tracker, and what applying it commits you to. Three families, and an issue may carry one from each: **triage** says what state it is in, **descriptive** says what it is, **wayfinder** marks a map's children. Operations — creating, fetching, applying labels, closing — are [issue-tracker.md](issue-tracker.md).

## Triage labels

The skills speak in terms of canonical triage roles; this table maps those roles to the label strings used here.

| Label in mattpocock/skills | Label in our tracker | Meaning                                 |
| -------------------------- | -------------------- | --------------------------------------- |
| `ready-for-agent`          | `ready-for-agent`    | Fully specified, ready for an AFK agent |
| `wontfix`                  | `wontfix`            | Will not be actioned                    |

When a skill mentions a role (e.g. "apply the AFK-ready triage label"), use the corresponding label string from this table.

`ready-for-agent` is a claim that an agent can finish the ticket alone. Withhold it whenever the ticket still contains a decision — an unpicked option, a threshold nobody has chosen, a trade-off with no stated winner — and apply `grilling` instead, so the ticket is routed to the conversation that settles it rather than to an agent who would settle it by guessing. A ticket carries one or the other, never both. `grilling` comes off once the decisions are written into the body, and `ready-for-agent` goes on, or `spec` alone when the body became a spec.

## Descriptive labels

Orthogonal to triage — they say what an issue *is*, not what state it is in. Apply as many as fit, or none.

| Label         | Meaning                                                             |
| ------------- | ------------------------------------------------------------------- |
| `bug`         | Something shipped behaves wrongly                                   |
| `enhancement` | A change to behaviour that already works                            |
| `documentation` | Docs-only work                                                    |
| `spec`        | A specification issue, typically the parent of a `to-tickets` batch. Carries `spec` alone unless the user names another label |
| `research`    | Answered by reading sources, not by changing code                   |
| `grilling`    | Answered by a conversation that settles a decision (use the `grill-with-docs` skill) |
| `future idea` | Worth doing, nobody is doing it, no commitment to when              |

## Wayfinder labels

`wayfinder:map` marks a map; `wayfinder:research` / `wayfinder:prototype` / `wayfinder:grilling` / `wayfinder:task` mark its child tickets by type. See [wayfinding operations](issue-tracker.md#wayfinding-operations).

These are meaningful **only inside a map**. An issue detached from its map — ruled out of scope, or orphaned when a map closes — swaps its `wayfinder:<type>` label for the plain descriptive one of the same name (`wayfinder:research` → `research`), because outside a map the type no longer names a ticket a session can claim.
