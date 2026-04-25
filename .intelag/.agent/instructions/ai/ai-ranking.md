# AI Agent Ranking for Planning and Coding

This file is the routing rubric for:

- `.intelag/.agent/instructions/plan/ai_planner.md`
- `.intelag/.agent/instructions/plan/ai-planner-splitter.md`

Its job is to help the planner pick the right agent family and reasoning effort quickly and consistently.

## Core Rule

Do not optimize for token price alone.

Optimize for reliable task completion using this weighted decision function:

```text
final_score = (2.5 * performance_score + 2.0 * speed_score + 1.5 * cost_efficiency_score) / 6.0 - uncertainty_penalty
```

Meaning:

- `performance_score` matters most
- `speed_score` matters next
- `cost_efficiency_score` still matters, but it is not allowed to dominate quality

Higher `final_score` is better.

## Canonical Agent Families

Use only these canonical names in plans:

- `GPT-5.4`
- `GPT-5.4-mini`
- `GPT-5.3-codex`
- `GPT-5.3-codex-spark`
- `Opus-4.6`
- `Sonnet-4.6`

Use one of these reasoning efforts:

- `low`
- `medium`
- `high`
- `xhigh`

## Quick Pick Table

Use this first. It is the fastest route to a good default choice.

| Task profile | Default pick | Escalate when | Avoid |
| --- | --- | --- | --- |
| Master planning, architecture, migrations, auth, security, framework rewrites, high-blast-radius work | `GPT-5.4/xhigh` | `Opus-4.6/xhigh` for very large-context review work; `GPT-5.3-codex/xhigh` when terminal or tool iteration dominates | `GPT-5.4-mini`, `GPT-5.3-codex-spark` |
| Tool-heavy multi-file debugging, log chasing, long command loops, repo-wide coding agents | `GPT-5.3-codex/high` | `GPT-5.4/xhigh` if ambiguity or architecture risk rises | `GPT-5.3-codex-spark` for large-context work |
| Routine multi-file implementation, standard refactors, ordinary tests | `GPT-5.4-mini/high` | `GPT-5.4/high` if blast radius or ambiguity becomes high | `Opus-4.6` unless long context is the actual bottleneck |
| Tight interactive edit loop, fast command-run-fix cycle, small local bugfixes | `GPT-5.3-codex-spark/high` | `GPT-5.3-codex/high` if deeper reasoning or larger context is needed | `Opus-4.6`, `GPT-5.4/xhigh` |
| Low-risk mechanical transforms, copy-adjust work, straightforward test fixes, cleanup | `GPT-5.4-mini/low` or `GPT-5.4-mini/medium` | `GPT-5.4-mini/high` if the edits stop being mechanical | `smart` tier agents by default |
| Huge repo, giant plan, very large working set, long-document reasoning | `GPT-5.4/xhigh` | `Opus-4.6/xhigh` if review quality matters more than speed and cost | `GPT-5.3-codex-spark`, `GPT-5.4-mini` |
| Claude-family required but `Opus-4.6` is unavailable or too expensive | `Sonnet-4.6/high` | `Sonnet-4.6/xhigh` only if the task is hard and no better-supported option is available | Using `Sonnet-4.6` as the default when evidence quality matters |

## At-a-Glance Model Card

| Agent family | Best use | Speed band | Cost band | Context signal | Evidence confidence |
| --- | --- | --- | --- | --- | --- |
| `GPT-5.4` | hard planning, high-risk coding, large-context work | medium | medium | 1.05M context, 128K output | high |
| `GPT-5.4-mini` | default balanced choice for routine work | fast | low | 400K context, 128K output | high |
| `GPT-5.3-codex` | tool-heavy long-horizon coding | medium | unknown | public context not confirmed in this file | medium |
| `GPT-5.3-codex-spark` | fastest interactive coding loop | very fast | unknown | 128K context | medium |
| `Opus-4.6` | slower, high-quality long-context review and planning | slow | high | 1M context, 128K output | medium |
| `Sonnet-4.6` | fallback Claude-family balanced option | medium | unknown | not confirmed in this file | low |

Interpretation:

- `high` confidence means this file includes enough published benchmark or price data to make the model a default choice.
- `medium` confidence means some key data is missing, but there is still enough signal to use the model when it fits the task well.
- `low` confidence means do not make it the default choice unless an external constraint forces it.

## Hard Guardrails

Apply these before scoring:

1. If the task has high blast radius, do not default to `GPT-5.4-mini` or `GPT-5.3-codex-spark`.
2. If the task is security, auth, migration, architecture, or framework-rewrite work, start in the `smart` tier.
3. If the working set is clearly above `400K`, restrict the candidate set to `GPT-5.4` and `Opus-4.6`.
4. If the task is a tight interactive loop and the working set is small, prefer `GPT-5.3-codex-spark`.
5. If benchmark or pricing data is materially missing, do not make that model the default unless another requirement forces it.

## Step 1: Classify the Task

Score the task on these axes before picking a model:

- `difficulty`: `mechanical`, `routine`, `hard`, `critical`
- `tool_intensity`: `low`, `medium`, `high`
- `blast_radius`: `low`, `medium`, `high`
- `context_pressure`: `small`, `medium`, `large`
- `speed_sensitivity`: `low`, `high`

Use these interpretations:

- `mechanical`: exact transforms, cleanup, repetitive file edits, little judgment
- `routine`: standard feature work, ordinary bugfixes, regular tests
- `hard`: multi-file refactors, ambiguous bugs, infra, concurrency, larger design choices
- `critical`: migrations, security, auth, major architecture, expensive-to-rework changes
- `tool_intensity high`: many commands, logs, grep loops, CI failures, or agentic repo exploration
- `blast_radius high`: a bad answer can break production behavior, many files, or future maintainability
- `context_pressure large`: the working set is too large for small-context models to hold comfortably

## Step 2: Compute the Performance Score

Do not average all benchmarks blindly.
Use the benchmark mix that matches the task.

### Routine coding

```text
performance_score = 0.45 * SWE_bench_Pro
                  + 0.35 * Terminal_Bench_2
                  + 0.20 * MCP_Atlas
```

### Tool-heavy coding

```text
performance_score = 0.30 * SWE_bench_Pro
                  + 0.40 * Terminal_Bench_2
                  + 0.20 * MCP_Atlas
                  + 0.10 * BrowseComp
```

### Planning, architecture, security, migrations

```text
performance_score = 0.50 * SWE_bench_Pro
                  + 0.20 * Terminal_Bench_2
                  + 0.15 * MCP_Atlas
                  + 0.15 * BrowseComp
```

If one required metric is missing:

- renormalize over the published metrics
- apply `uncertainty_penalty = 5`

If more than one required metric is missing:

- renormalize over the published metrics
- apply `uncertainty_penalty = 10`

## Step 3: Compute the Speed Score

Use these speed bands:

| Speed band | Score | Agent families |
| --- | ---: | --- |
| very fast | 100 | `GPT-5.3-codex-spark` |
| fast | 85 | `GPT-5.4-mini` |
| medium | 70 | `GPT-5.4`, `GPT-5.3-codex`, `Sonnet-4.6` |
| slow | 55 | `Opus-4.6` |

Notes:

- These are operational routing bands, not one shared published latency benchmark.
- Inside the same family, `xhigh` is slower than `high`, which is slower than `medium`, which is slower than `low`.
- Use the lowest effort that still safely clears the task.

## Step 4: Compute the Cost-Efficiency Score

Use these cost-efficiency bands:

| Cost band | Score | Agent families |
| --- | ---: | --- |
| low cost | 90 | `GPT-5.4-mini` |
| medium cost | 65 | `GPT-5.4` |
| high cost | 35 | `Opus-4.6` |
| unknown cost | 50 | `GPT-5.3-codex`, `GPT-5.3-codex-spark`, `Sonnet-4.6` |

Interpretation:

- `cost_efficiency_score` is higher when the model is cheaper to use for the same job.
- `unknown cost` is not a bonus. It is a neutral placeholder and should not beat a known cheaper option unless performance or speed justifies it.
- If output volume will be very large, prefer models with lower output pricing unless the task is clearly `hard` or `critical`.

## Step 5: Apply the Uncertainty Penalty

Use:

- `0` when the key benchmark and pricing picture is clear
- `5` when one major dimension is missing
- `10` when more than one major dimension is missing

Examples:

- `Sonnet-4.6` should usually carry an uncertainty penalty because this file does not include verified benchmark rows for it.
- `GPT-5.3-codex` and `GPT-5.3-codex-spark` should usually carry at least a small penalty on cost because public pricing is not confirmed in this file.

## Step 6: Choose the Reasoning Effort

Use this table after choosing the family:

| Situation | Effort |
| --- | --- |
| exact transforms, low-risk cleanup, deterministic edits | `low` |
| routine implementation, ordinary tests, straightforward bugfixes | `medium` |
| multi-file feature work, non-trivial refactors, ambiguous bugs | `high` |
| planning, migrations, auth, security, architecture, giant context, high blast radius | `xhigh` |

## Default Routing Policy

Use this when you do not need a full score calculation:

1. Start with `GPT-5.4-mini/high` for routine coding.
2. Move down to `GPT-5.4-mini/low` or `GPT-5.4-mini/medium` only for clearly mechanical work.
3. Move up to `GPT-5.4/high` or `GPT-5.4/xhigh` when the blast radius, ambiguity, or context size increases.
4. Prefer `GPT-5.3-codex/high` for tool-heavy loops and agentic coding work.
5. Prefer `GPT-5.3-codex-spark/high` for the fastest interactive loop when the task is local and small enough.
6. Prefer `Opus-4.6/high` or `Opus-4.6/xhigh` when long-context review quality matters more than speed and price.
7. Use `Sonnet-4.6` only as a fallback choice, not as the default recommendation.

## Planner-Facing Evidence Table

Blank cells mean the metric was not published in the checked source set already curated for this repo.

| Agent family | SWE-bench Pro | Terminal-Bench 2.0 | MCP Atlas | BrowseComp | Context | Price per 1M input / output | Notes |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| `GPT-5.4` | 57.7 | 75.1 | 67.2 | 89.3 | 1.05M / 128K | $2.50 / $15.00 | strongest OpenAI general-purpose planning and coding choice in this file |
| `GPT-5.4-mini` | 54.4 | 60.0 | 57.7 | — | 400K / 128K | $0.75 / $4.50 | best default price/performance choice |
| `GPT-5.3-codex` | 56.8 | 77.3 | — | — | not confirmed here | not confirmed here | strongest published Terminal-Bench row among the supported coding agents in this file |
| `GPT-5.3-codex-spark` | — | — | — | — | 128K / not confirmed here | not confirmed here | published `>1000 tokens/s`; use for speed-sensitive tight loops |
| `Opus-4.6` | 53.4 | 65.4 | 75.8 | 83.7 | 1M / 128K | $5.00 / $25.00 | slower, expensive, but strong on tool use and long-context review work |
| `Sonnet-4.6` | — | — | — | — | not confirmed here | not confirmed here | keep as fallback only because this file lacks verified benchmark support |

## Source Notes

The figures below were already curated into the previous version of this file. This rewrite keeps the same checked-source basis and reorganizes it into a planning rubric.

Sources:

- [OpenAI: GPT-5.4 mini and nano](https://openai.com/index/introducing-gpt-5-4-mini-and-nano/)
- [OpenAI: GPT-5.4 model page](https://developers.openai.com/api/docs/models/gpt-5.4)
- [OpenAI: GPT-5-Codex model page](https://developers.openai.com/api/docs/models/gpt-5-codex)
- [OpenAI: GPT-5.2-Codex model page](https://developers.openai.com/api/docs/models/gpt-5.2-codex)
- [OpenAI: GPT-5.3-Codex](https://openai.com/index/introducing-gpt-5-3-codex/)
- [OpenAI: GPT-5.3-Codex-Spark](https://openai.com/index/introducing-gpt-5-3-codex-spark/)
- [Anthropic: Claude Opus 4.6](https://www.anthropic.com/news/claude-opus-4-6)
- [Anthropic: Project Glasswing](https://www.anthropic.com/project/glasswing)

## Short Version

If you need the shortest usable answer:

- `GPT-5.4-mini/high` is the default for routine work.
- `GPT-5.4/xhigh` is the default for planning and high-risk work.
- `GPT-5.3-codex/high` is the default for tool-heavy coding.
- `GPT-5.3-codex-spark/high` is the default for the fastest edit loop.
- `Opus-4.6/xhigh` is a deliberate expensive choice for long-context review quality.
- `Sonnet-4.6` is fallback only.
