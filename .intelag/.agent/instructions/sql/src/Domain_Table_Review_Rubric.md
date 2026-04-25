# Domain Table Review Rubric (Improved)

**Purpose:** Structured pass/fail checklist for reviewing **all tables in one bounded context** (one PostgreSQL schema / one domain package) before merge or release. Enables consistent, rapid reviews with clear escalation paths.

**How to use**
- Work **domain by domain** (e.g. `org`, `hr`, `scheduling`); load the ERD or model list for that context only.
- Start at **Tier A**; only proceed to higher tiers when the domain is green at the previous tier (or explicitly defer with a written risk acceptance).
- Record outcomes in the sign-off block: **Pass** / **Fail** / **N/A** / **Deferred** (with owner + date + risk assessment).
- **Approval rule:** Tier A must be 100% Pass. Tier B has no Fail on B1–B3 (integrity checks). Higher tiers may skip if justified.

**Related doc:** [Golden Rules for SQL Database Design](./Golden%20Rules%20for%20SQL%20Database%20Design.md)

---

## Tier A — Basic (every reviewer)

*Goal: Tables are identifiable, keyed, and auditable. No silent foot-guns.*

| # | Check | Pass criteria | Failure mode |
|---|--------|---------------|----|
| A1 | **Naming** | Table and column names are `snake_case`; singular *or* plural—**consistent with rest of product**. Tables with abbreviations (e.g., `org_dept_role`) have glossary entry. | Silent confusion; inconsistency breeds bugs in joins and ORM codegen. |
| A2 | **Primary key** | Every table has exactly one surrogate PK column; follows project standard (e.g. `<entity>_id`). No multi-column PKs unless documented as legacy. | Ambiguity in inserts/updates; ORM struggles with identity. |
| A3 | **Timestamps** | Every table has non-nullable `created_at` (`TIMESTAMPTZ`); `updated_at` exists and is maintained on every write (or trigger/ORM handles it). | Audit trail gaps; can't debug data history. |
| A4 | **NULL policy** | Columns are `NOT NULL` unless NULL has documented business meaning (noted in review comment or docstring). Default to `NOT NULL`. | Silent NULLs creep in; queries crash on unexpected nulls. |
| A5 | **Money & time** | No `FLOAT` for money; use `NUMERIC(precision, scale)` or `BIGINT` (cents). Timestamps are `TIMESTAMPTZ` (not `TIMESTAMP`), UTC assumed. | Rounding errors in financial data; timezone bugs in prod. |
| A6 | **Schema ownership** | Domain tables live in the correct PostgreSQL **schema**; no obvious stray tables in wrong schema. Clear schema-per-domain or schema-per-team organization. | Namespace collisions; hard to understand ownership. |
| A7 | **Documentation** | Each new/changed table has a one-line purpose (comment in migration, model docstring, ADR, or ticket link); reviewer can answer "what is this for?" | Orphaned tables; nobody knows why they exist; hard to deprecate. |

---

## Tier B — Intermediate (reviewer familiar with SQL + ORMs)

*Goal: Referential integrity and domain rules are visible in the schema.*

| # | Check | Pass criteria | Failure mode |
|---|--------|---------------|----|
| B1 | **Foreign keys** | Every logical FK column has a declared `REFERENCES` / ORM `ForeignKey` to the correct `schema.table.column`. No soft FK strings. | Silent referential corruption; cascades fail; app must police referential integrity (slow, error-prone). |
| B2 | **ON DELETE / ON UPDATE** | Each FK has deliberate actions (`RESTRICT` / `CASCADE` / `SET NULL`); defaults are not accidental. Actions match domain semantics (e.g., org FK = `CASCADE`, not `RESTRICT`). Document rationale if non-obvious. | Accidental cascades wipe data; `RESTRICT` blocks valid ops; hard to recover. |
| B3 | **FK indexes** | Every FK column (or composite leading prefix) is covered by an index suitable for joins and parent deletes/updates. At minimum: `INDEX (fk_column)`. | Child table scans during parent delete (O(n)); slow lookups. |
| B4 | **Uniqueness** | Business-natural uniqueness (e.g. per-org name, email) has a `UNIQUE` constraint or unique index, not only app checks. Document why if missing (e.g., soft-delete conflict). | App-side race conditions; duplicate keys slip past in concurrent writes. |
| B5 | **CHECK constraints** | Status/enum columns have `CHECK (status IN (...))` or documented exception (e.g., app validation only). Paired dates/times have `CHECK (start_date <= end_date)` where applicable. | Invalid state creeps in during bulk updates or direct SQL. |
| B6 | **Cross-domain FKs** | References to other domains use **stable, qualified targets** (e.g., `org.users.id` not a hardcoded UUID string). If using a registry or federation, documented and centralized (not duplicate literals). | Refactoring one domain breaks another; hard to migrate or deprecate. |
| B7 | **Junction tables** | Many-to-many relationships use a **bridge table with PKs on both FKs**, not arrays/JSON/CSV in a column. Composite PK is `(fk1_id, fk2_id)` or separate surrogate PK. | Array/JSON queries degrade; can't efficiently filter or join; RLS impossible. |

---

## Tier C — Advanced (reviewer + DBA or senior backend)

*Goal: Performance, consistency, and operability under load.*

| # | Check | Pass criteria | Failure mode |
|---|--------|---------------|----|
| C1 | **Query patterns** | Top 3–5 critical queries per table have been identified (e.g., search by org + name, list active items, fetch by FK). `WHERE`/`JOIN`/`ORDER BY` columns are indexed or sequential scan is explicitly accepted (with cardinality estimate). | N+1 queries; unindexed large scans during peak load; query latency spikes. |
| C2 | **Composite indexes** | Composite index column order matches **equality filters first**, then range/sort; avoids redundant indexes that duplicate earlier prefixes without justification. Example: `(org_id, status, created_at)` for `WHERE org_id = ? AND status = ? ORDER BY created_at`. | Index bloat; queries still trigger full table scans (leading column not used). |
| C3 | **Partial / filtered indexes** | Low-cardinality flags (`is_active`, status) use partial indexes (e.g., `WHERE is_active = true`) or are leading columns in a composite; no standalone btree on a boolean. | Index wastes space; queries scan deleted/inactive rows unnecessarily. |
| C4 | **Covering / INCLUDE** | Hot read paths (e.g., count by org, summary stats) use `INCLUDE` or narrow indexes to avoid heap fetches; documented if non-obvious. | Extra disk I/O for every read; cache misses under load. |
| C5 | **JSON / JSONB** | JSON columns have documented shape (schema, example, constraints); optional `CHECK` or application validation strategy is explicit. Avoid unbounded nesting. | Silent invalid JSON; app crashes on unexpected shapes; can't index into it safely. |
| C6 | **Soft delete** | If used: `deleted_at` / `is_active` strategy is consistent; FK behavior still correct; all queries that must hide deleted rows have appropriate indexes (e.g., partial index on `deleted_at IS NULL`). | Deleted data leaks; FK constraints confuse soft-delete state; scan all deleted rows in every query. |
| C7 | **Write amplification** | Table has ≤ ~5–7 indexes on very write-heavy paths, or overhead is justified and documented. Account for triggers, generated columns, full-text indexes. | Write-heavy tables slow down; index maintenance becomes bottleneck; contention on heavily-written rows. |
| C8 | **Migration safety** | Alembic (or equivalent) migration plan avoids long exclusive locks on large tables. If adding index: use `CONCURRENTLY`. If backfilling: batch in smaller transactions or concurrent data migration pattern. Documented lock time / impact. | Live table locks; production downtime; replication lag spikes. |

---

## Tier D — Very advanced (staff/principal engineer + optional SRE)

*Goal: Scale, security, tenancy, and long-term evolution.*

| # | Check | Pass criteria | Failure mode |
|---|--------|---------------|----|
| D1 | **Multi-tenancy** | Tenant isolation column(s) present where required; RLS policy or equivalent documented; no implicit cross-tenant FK holes. Verify test coverage of isolation. | Data leakage across orgs; compliance violation. |
| D2 | **PII & secrets** | Sensitive columns (email, SSN, password hash, API key) identified and documented. Encryption/masking strategy for non-prod clear; no secrets stored in plaintext columns. | Accidental log/dump of sensitive data; compliance audit failure. |
| D3 | **Privilege model** | DB roles follow least privilege for this domain's tables (e.g., service role can INSERT/UPDATE but not DELETE). Blast radius of a compromised app credential understood and documented. | Compromised service account can drop or corrupt tables; hard to audit who did what. |
| D4 | **Partitioning / archival** | If table is or will be very large (>1GB or >10M rows expected): partition key, retention, or archival strategy is defined (even if "implemented in Q3"). Time-series or sharded data strategies documented. | Table bloats; queries slow on large rows; cost scales linearly with data. |
| D5 | **FK depth & cycles** | Dependency graph sketched; no unintended deep cascade chains (>4 levels). Cycles (if any) are intentional and have documented safe handling (e.g., deferred FK checks, explicit cleanup order). | Cascading deletes wipe entire org by accident; cycles cause deadlocks. |
| D6 | **ORM metadata correctness** | Cross-schema FKs resolve in ORM metadata; `ForeignKey(to='schema.Table')` or `db.ForeignKey('schema.table')` is correct. ORM migrations don't falsely report "unknown FK target." | ORM codegen fails; alembic migration incorrect; app doesn't boot. |
| D7 | **Round-trip integrity** | If YAML/model generation is used (e.g., ER diagram → migrations → Python models): round-trip test that FKs, constraints, and types survive export/import. Gaps listed as accepted technical debt. | Generated migrations omit constraints; models out of sync with DB; silent data corruption. |
| D8 | **Failure modes & idempotency** | For critical writes (outbox pattern, payments, state changes): timeout handling, deadlock retry strategy, and idempotency key documented. Example: `INSERT ... ON CONFLICT (idempotency_key) DO UPDATE` for idempotent APIs. | Duplicate transactions; payment double-charge; message loss on restart. |

---

## Scoring & Approval Guide

### Quick decision matrix

| Tier A | Tier B | Tier C | Tier D | Decision |
|--------|--------|--------|--------|----------|
| ✅ Pass | ✅ Pass | ✅ Pass | ✅ Pass | **Approve** |
| ✅ Pass | ✅ Pass | ✅ Pass | ❓ N/A | **Approve** (D not required for early-stage) |
| ✅ Pass | ✅ Pass | ⏸ Deferred | — | **Conditional approval** (open ticket for C; proceed if low-risk) |
| ✅ Pass | ❌ Fail (B1–B3) | — | — | **Block** (integrity issue) |
| ✅ Pass | ⚠ Partial (B4–B7) | — | — | **Conditional** (tech debt ticket + owner) |
| ❌ Fail (A1–A7) | — | — | — | **Block** (redo review) |

### Weighted scoring (optional, for rollup reports)

| Tier | Weight | Pass = | Notes |
|------|--------|--------|-------|
| A | 40% | Must be 100% unless risk accepted by tech lead (formal doc). | Foundation. |
| B | 35% | No Fail on B1–B3 (integrity). B4–B7 can be Deferred if ticketed. | Data safety. |
| C | 20% | Can defer if low cardinality / non-hot-path table; document. | Perf / ops. |
| D | 5% | Often N/A for MVP tables. Skip unless scale/security concerns. | Maturity. |

**Domain Score = (Tier A Pass% × 0.40) + (Tier B Pass% × 0.35) + (Tier C Pass% × 0.20) + (Tier D Pass% × 0.05)**

Suggested threshold: **Domain ≥ 85% = ship; <85% = rework or defer with ticket.**

---

## Sign-off block (copy per domain review)

| Field | Value |
|-------|-------|
| **Domain / schema** | e.g. `org`, `hr`, `scheduling` |
| **Reviewer(s)** | Names; domain expertise (SQL, DBA, Tier C, etc.) |
| **Date** | YYYY-MM-DD |
| **Tier A** | Pass / Fail / Partial |
| **Tier B** | Pass / Fail / Partial / N/A |
| **Tier C** | Pass / Fail / Partial / N/A |
| **Tier D** | Pass / Fail / Partial / N/A |
| **Domain Score** | (if using weighted approach) |
| **Open findings** | (Ticket links + owners + due dates) |
| **Blockers** | Any Fail items that prevent merge? |
| **Approval to merge** | Yes / No / Conditional (state condition) |
| **Risk acceptance** | (If deferring: tech lead name + date + reason) |

**Example:**
```
Domain: org
Reviewer: Alice (SQL), Bob (backend)
Date: 2026-03-15
Tier A: Pass
Tier B: Pass (B4 has unique index on email pending; ticketed as #4521)
Tier C: Partial (C1 analysis on user_org join pending; deferred to Q2 perf review)
Tier D: N/A (org domain not sensitive; no tenancy yet)
Score: 92%
Blockers: None
Approval: Yes, conditional on #4521 closed by EOW
```

---

## Tips for reviewers

### For Tier A (every reviewer)
- Use a linter (e.g., [sqlfluff](https://github.com/sqlfluff/sqlfluff)) to catch naming, NULL defaults automatically.
- Check the migration script: does it have a docstring explaining **why** the table exists?
- Skim the ORM model: does it match the schema?

### For Tier B (SQL-familiar)
- Sketch the FK graph on paper or in a tool; identify all edges.
- Validate ON DELETE / ON UPDATE semantics **in writing** (e.g., "deleting an org cascades to all child orgs because org is tree-like").
- Use `EXPLAIN` to check if FK columns are indexed: look for "Seq Scan on child_table" in the delete plan.

### For Tier C (DBA or senior)
- Run `pg_stat_statements` or slow-query log on staging to spot unindexed WHERE/JOIN columns.
- Ask: "What are the top 5 queries on this table?" If unsure, request them from the author.
- For new indexes: always compare `pg_indexes_size()` before/after and weigh against write cost.

### For Tier D (principal / SRE)
- Trace FKs three levels deep; are cascades intended?
- For sensitive tables: review RLS policies, masking, and non-prod anonymization.
- Partition planning: estimate growth; when will sharding/archival be needed?

---

## Common mistakes to avoid

| Mistake | Why it matters | Fix |
|---------|----------------|-----|
| "FK indexes are app-side; we'll handle it" | Parent deletes full-table-scan the child. | Add `INDEX (fk_col)` in migration. |
| "UNIQUE constraint isn't needed; app enforces it" | Race condition under concurrent writes. | Add `UNIQUE INDEX` to schema. |
| "We'll partition later when it's big" | Growth happens suddenly; retrofitting locks prod. | Define strategy upfront (even if deferred). |
| "Soft delete is just a flag" | Queries bloat; uniqueness breaks; RLS impossible. | Add partial indexes; explicit soft-delete pattern. |
| "Let's document this in Slack / Notion" | Doc rots; reviewer can't find it; next person guesses. | Put docstrings in migration files and ORM models. |
| "B1–B3 can be deferred; we'll fix it in a hotfix" | Referential corruption in prod; hard to unwind. | **Never defer B1–B3; block merge if Fail.** |

---

## Related reading

- [Golden Rules for SQL Database Design](./Golden%20Rules%20for%20SQL%20Database%20Design.md) — detailed design patterns
- PostgreSQL docs: [Foreign Keys](https://www.postgresql.org/docs/current/sql-createtable.html#SQL-CREATETABLE-CONSTRAINTS), [Indexes](https://www.postgresql.org/docs/current/indexes.html), [Partial Indexes](https://www.postgresql.org/docs/current/indexes-partial.html), [ON CONFLICT](https://www.postgresql.org/docs/current/sql-insert.html#SQL-ON-CONFLICT)
- [Aqua Data Studio — FK visualization](https://www.aquafold.com/) or similar ER tools

---

## Document history

| Version | Date | Summary |
|---------|------|---------|
| 1.0 | 2026-03 | Initial rubric (Tiers A–D + sign-off + scoring guide). |
| 2.0 | 2026-03-26 | **Improved:** Failure modes per check; decision matrix; approval rules clarified; common mistakes; tips for each tier; weighted scoring guide; risk-acceptance template. |