# Golden Rules for SQL Database Design

**A Senior Engineer's Rubric** · Version 1.1 · 2026

*Schema Design · Normalization · Indexing · Relationships · Performance · Multi-Schema ORM · Services*

---

## How to use this document

These rules are **product-agnostic** defaults. Where the Intelag SQL Suite uses a specific convention (PostgreSQL schemas per bounded context, SQLAlchemy `SchemaBase`, YAML/model generators), callouts mark **Intelag alignment** so agents and humans can stay consistent without forking the rubric.

---

## 1. Table & Schema Design

Well-structured tables form the foundation of a performant, maintainable database. These rules ensure clarity, consistency, and long-term scalability.

| # | Rule | Rationale | Severity |
|---|------|-----------|----------|
| 1.1 | Use lowercase `snake_case` table names. Choose **either** singular **or** plural per product and apply it everywhere (e.g. `customer_order` vs `customer_orders`). | Prevents quoting issues across engines; mixed singular/plural breaks mental models and automation. | **High** |
| 1.2 | Every table must have a single-column surrogate primary key, preferably `<entity>_id` (or `id` if globally consistent). | Guarantees row identity; simplifies joins and ORM mapping. | **Critical** |
| 1.3 | Prefer UUID (v4/v7) or `BIGINT IDENTITY/SERIAL` for primary keys over natural keys. | Natural keys change, compound keys complicate joins, and sequential IDs leak volume; UUIDv7 is time-sortable. | **High** |
| 1.4 | Every table must include `created_at` (`TIMESTAMPTZ`, non-nullable, default now) and `updated_at` (`TIMESTAMPTZ`, maintained on change). | Enables auditing, debugging, and incremental ETL. | **High** |
| 1.5 | Default all columns to `NOT NULL` unless NULL carries explicit business meaning. | Eliminates three-value logic bugs and enables index efficiency. | **Critical** |
| 1.6 | Use the smallest appropriate data type: `INT` vs `BIGINT`, `VARCHAR(n)` vs `TEXT`, `DATE` vs `TIMESTAMP`. | Reduces page size, improves cache hit ratio, and shrinks indexes. | **Medium** |
| 1.7 | Store monetary values as `DECIMAL/NUMERIC` with explicit precision (e.g. `DECIMAL(19,4)`), never `FLOAT`. | Floating-point rounding causes financial discrepancies. | **Critical** |
| 1.8 | Store all timestamps in UTC (`TIMESTAMPTZ`). Convert to local time only at the application layer. | Avoids DST ambiguity and cross-timezone comparison bugs. | **High** |
| 1.9 | Add `CHECK` constraints for domain validation (e.g. status enums, non-negative amounts, `start <= end` for paired date/time columns). | Catches invalid data at the engine level before it corrupts downstream. | **Medium** |
| 1.10 | Avoid storing derived/calculated data unless materialized for proven performance reasons. | Keeps a single source of truth; eliminates stale calculation risk. | **Medium** |
| 1.11 | **Intelag alignment:** In multi-schema PostgreSQL, set the table's schema in DDL and in the ORM (`__table_args__` / equivalent) so metadata, migrations, and `search_path` assumptions stay aligned. | Omitting schema while using schema-qualified FK strings breaks mapper configuration and validation tooling. | **High** |

> **💡 Pro Tip:** If you find yourself adding a column named `type` or `kind` that changes the meaning of other columns, consider splitting into separate tables (table-per-type) or using PostgreSQL's table inheritance.

---

## 2. Normalization & Denormalization

Normalize first for correctness, then selectively denormalize with measurement-backed justification.

| # | Rule | Rationale | Severity |
|---|------|-----------|----------|
| 2.1 | Achieve at least Third Normal Form (3NF) for all OLTP tables before considering denormalization. | Eliminates update/insert/delete anomalies. | **Critical** |
| 2.2 | Never store comma-separated values or JSON arrays when a junction table is appropriate. | Destroys referential integrity, prevents indexing, and blocks joins. | **Critical** |
| 2.3 | Denormalize only after profiling shows a measurable bottleneck; document every denormalization with a dated comment. | Premature denormalization creates technical debt with no measured payoff. | **High** |
| 2.4 | When denormalizing, use materialized views or summary tables—not duplicated columns—where the engine supports them. | Materialized views can be refreshed atomically; columns can drift. | **Medium** |
| 2.5 | If you store JSON/JSONB, define a JSON Schema or `CHECK` constraint to enforce structure. | Unvalidated JSON becomes a shadow schema no one understands. | **Medium** |
| 2.6 | Eliminate transitive dependencies: a non-key column should never depend on another non-key column. | Prevents update anomalies (e.g., updating a zip code without its city). | **High** |

---

## 3. Relationships & Referential Integrity

Foreign keys are contracts between tables. Treat them with the same rigor as application-level validation.

| # | Rule | Rationale | Severity |
|---|------|-----------|----------|
| 3.1 | Every foreign key must have an explicit `REFERENCES` target with deliberate `ON DELETE` and `ON UPDATE` actions (or database defaults you have explicitly accepted). | Missing cascade/restrict rules leave orphan records or silent failures. | **Critical** |
| 3.2 | Name foreign key columns as `<referenced_entity>_id` (e.g. `organization_id`, `employee_id`) when referencing surrogate keys. | Self-documenting; enables automated ORM discovery. | **High** |
| 3.3 | Use junction/bridge tables for many-to-many relationships; never embed arrays of IDs. | Preserves referential integrity and enables efficient querying. | **Critical** |
| 3.4 | Choose `ON DELETE` actions deliberately: `RESTRICT` for strong entities, `CASCADE` for owned children, `SET NULL` for optional associations. | Wrong action either blocks operations or silently destroys data. | **Critical** |
| 3.5 | Always index foreign key columns (most engines except MySQL/InnoDB do not auto-index them). | Unindexed FKs cause full table scans on `DELETE`/`UPDATE` of the parent row. | **High** |
| 3.6 | Use composite foreign keys only when the referenced table has a composite PK; prefer surrogate keys otherwise. | Reduces key width, join complexity, and migration difficulty. | **Medium** |
| 3.7 | For soft-delete patterns, add a `deleted_at TIMESTAMPTZ` (nullable) and/or an `is_active` flag with indexing strategy; maintain FK integrity. | Allows auditability and undo; avoids orphan cascades. | **Medium** |
| 3.8 | Self-referencing FKs (e.g. `employee.manager_id → employee.id`) must allow `NULL` for root nodes. | Prevents a chicken-and-egg insertion problem. | **Medium** |
| 3.9 | **PostgreSQL multi-schema:** Reference the full qualified target (`schema.table.column`) in FK definitions when tables live in different schemas, and keep `search_path` or ORM metadata consistent. | Ambiguous or path-dependent resolution causes migration drift and broken tooling. | **High** |
| 3.10 | **Intelag alignment:** Maintain **one module** (or equivalent) that defines schema names, domain-to-table ownership, and **schema-qualified FK string constants** used across models. Never scatter duplicate `"schema.table.column"` literals. | Centralizes renames, audits, ERD styling, and YAML/model round-trips. | **High** |

---

## 4. Indexing Strategy

Indexes are the single largest lever for read performance. Every index also carries a write penalty—balance accordingly.

| # | Rule | Rationale | Severity |
|---|------|-----------|----------|
| 4.1 | Index every column used in `WHERE`, `JOIN ON`, and `ORDER BY` clauses of frequent queries. | Without an index, the engine falls back to sequential scans. | **Critical** |
| 4.2 | Create composite indexes with the highest-selectivity column first, followed by range/sort columns. | Column order determines whether the index can satisfy the query plan. | **Critical** |
| 4.3 | Never index columns with very low cardinality (e.g. boolean flags) in isolation—combine with a selective column or use a **partial** index. | A standalone low-cardinality index is rarely chosen by the optimizer. | **High** |
| 4.4 | Use `INCLUDE` (covering index) columns to enable index-only scans for read-heavy queries. | Avoids expensive heap lookups by serving data directly from the index. | **Medium** |
| 4.5 | Use partial/filtered indexes for queries that target a subset (e.g. `WHERE status = 'active'` or `WHERE is_active = 'Y'`). | Dramatically smaller index; faster scans and lower maintenance. | **Medium** |
| 4.6 | Add `UNIQUE` indexes/constraints for business-level uniqueness (email, SKU, composite natural keys) beyond the PK. | Prevents duplicate data at the engine level, not just the app level. | **Critical** |
| 4.7 | Limit total indexes per table to roughly 5–7 on write-heavy tables; each index adds `INSERT`/`UPDATE` overhead. | Write amplification degrades throughput and increases WAL volume. | **High** |
| 4.8 | Audit unused indexes monthly (`pg_stat_user_indexes`, `sys.dm_db_index_usage_stats`) and drop them. | Dead indexes consume storage, slow writes, and bloat backups. | **Medium** |
| 4.9 | Use expression/functional indexes for computed predicates (e.g. `INDEX ON LOWER(email)`). | Allows the optimizer to use the index when the query applies the same function. | **Medium** |
| 4.10 | Use GIN/GiST indexes for full-text search, JSONB, and array containment queries (PostgreSQL). | B-tree cannot satisfy containment or text-search operators. | **Medium** |

> **💡 Index Audit Query (PostgreSQL):**
> ```sql
> SELECT relname, indexrelname, idx_scan
> FROM pg_stat_user_indexes
> WHERE idx_scan = 0 AND schemaname = 'public'
> ORDER BY pg_relation_size(indexrelid) DESC;
> ```

---

## 5. Performance & Scaling Patterns

Design for the query patterns you know today while keeping the door open for the scale you may need tomorrow.

| # | Rule | Rationale | Severity |
|---|------|-----------|----------|
| 5.1 | Design tables for read patterns: if a query always needs columns A, B, C together, ensure they share an index. | Aligning physical storage with access patterns minimizes I/O. | **High** |
| 5.2 | Partition large tables (100M+ rows) by range, list, or hash—based on actual query filters. | Partition pruning eliminates scanning irrelevant data. | **High** |
| 5.3 | Use connection pooling (PgBouncer, ProxySQL) and avoid long-held transactions in OLTP systems. | Idle connections consume memory and can cause lock contention. | **High** |
| 5.4 | Never use `SELECT *`; always specify required columns. | Reduces I/O, prevents index-only scan invalidation, and avoids schema-change surprises. | **High** |
| 5.5 | Avoid correlated subqueries in `SELECT`/`WHERE`; rewrite as JOINs or lateral joins. | Correlated subqueries execute per-row; joins execute set-based. | **Medium** |
| 5.6 | Use `EXPLAIN ANALYZE` on every new query before deploying to production. | Reveals sequential scans, nested loops, and sort spills before users feel them. | **Critical** |
| 5.7 | Implement read replicas for reporting/analytics queries; never run analytics on the primary. | Offloads read traffic and prevents long-running queries from blocking writes. | **High** |
| 5.8 | Set `statement_timeout` / `lock_timeout` at the session or role level to prevent runaway queries. | A single bad query can exhaust all connections and cascade into downtime. | **Medium** |

---

## 6. Naming Conventions & Documentation

A schema that explains itself reduces onboarding time and prevents misinterpretation.

| # | Rule | Rationale | Severity |
|---|------|-----------|----------|
| 6.1 | Use consistent naming: `snake_case` for tables/columns, `PascalCase` only if the project-wide standard demands it. | Mixing conventions causes quoting errors and slows development. | **High** |
| 6.2 | Prefix boolean columns with `is_`, `has_`, or `can_` (e.g. `is_active`, `has_verified_email`). If the engine stores `CHAR(1)` / `VARCHAR(1)` flags (`Y`/`N`), enforce with `CHECK`. | Self-documenting; avoids ambiguous truthy strings. | **Medium** |
| 6.3 | Name indexes as `idx_<table>_<columns>` (or `ix_<table>_<columns>` if consistent) and constraints with explicit names: `chk_`, `uq_`, `fk_`. | Allows instant identification in error messages and monitoring. | **Medium** |
| 6.4 | **Multi-schema:** Auto-generated `ix_*` names may include schema/table qualifiers; treat **column sets** as the semantic identity when comparing models or migrations—human-chosen `idx_*` names should still be stable and explicit. | Prevents false drift between environments that differ only by `search_path` or metadata naming. | **Medium** |
| 6.5 | Add `COMMENT ON TABLE` / `COMMENT ON COLUMN` for every object in the schema. | Surfaces documentation in IDE tooltips and data catalog tools. | **Medium** |
| 6.6 | Maintain versioned migration files (Flyway, Liquibase, Alembic)—never apply ad-hoc DDL in production. | Ensures reproducibility, rollback capability, and audit history. | **Critical** |
| 6.7 | Keep an up-to-date ERD in the repository, regenerated from models or the live schema. | Visual references accelerate understanding for new team members. | **Medium** |

---

## 7. Security & Access Control

The database is the last line of defense. Assume the application layer will eventually have a bug.

| # | Rule | Rationale | Severity |
|---|------|-----------|----------|
| 7.1 | Apply the principle of least privilege: application roles should have only `SELECT`/`INSERT`/`UPDATE` on needed tables. | Limits blast radius if credentials are compromised. | **Critical** |
| 7.2 | Never store plaintext passwords or API keys in any column; use hashed values with bcrypt/argon2. | Plaintext credentials in a breach expose all users. | **Critical** |
| 7.3 | Encrypt PII columns at rest (TDE or column-level encryption) and mask them in non-production environments. | Regulatory compliance (GDPR, HIPAA) and breach risk reduction. | **High** |
| 7.4 | Use row-level security (RLS) for multi-tenant databases instead of filtering only in the application. | Prevents data leakage even if the app has a logic bug. | **High** |
| 7.5 | Audit all DDL changes and privileged DML with `pg_audit` or equivalent engine feature. | Creates a forensic trail for security incidents and change management. | **Medium** |

---

## 8. Multi-schema PostgreSQL & bounded contexts (DDD)

When a single database hosts multiple **bounded contexts**, use PostgreSQL **schemas** as physical boundaries—not separate meanings crammed into one flat namespace.

| # | Rule | Rationale | Severity |
|---|------|-----------|----------|
| 8.1 | Map one bounded context → one PostgreSQL `schema` name (short, stable, lowercase). | Clarifies ownership, migration ordering, and permission boundaries. | **High** |
| 8.2 | Keep **shared kernel** tables (e.g. organization, location) in a dedicated schema; depend on them via explicit FKs rather than copying IDs without constraints. | Prevents silent divergence and orphan references across contexts. | **Critical** |
| 8.3 | Document which tables belong to which context (single source of truth); derive reverse lookups (table → context) from that map—do not duplicate. | Tools (ERD, validators, `SchemaManager`) stay consistent. | **High** |
| 8.4 | Prefer **many small, named migrations** per context over monolithic dumps when contexts evolve at different speeds. | Reduces merge conflict and blast radius. | **Medium** |
| 8.5 | For SQLite or tests, use `schema_translate_map` (or equivalent) so the same models target a single logical schema without forking class definitions. | Keeps test and prod metadata aligned. | **Medium** |

---

## 9. ORM alignment (SQLAlchemy-style declarative models)

These rules apply when the database is authored and migrated through an ORM layer.

| # | Rule | Rationale | Severity |
|---|------|-----------|----------|
| 9.1 | Declare **every** `ForeignKey` at the column (or constraint) level with the same qualification you use in DDL (`schema.table.column` when using multiple schemas). | Enables relationship join inference, Alembic autogenerate, and static validation. | **Critical** |
| 9.2 | When validating FK targets against `MetaData.tables`, use the table's **`key`** (schema-qualified internal key), not the bare `Table.name` only. | Otherwise every cross-schema FK appears to target a “missing” table. | **Critical** |
| 9.3 | Name all non-PK constraints (`ForeignKeyConstraint`, `UniqueConstraint`, `CheckConstraint`, indexes) explicitly. | Stable names in migrations and clearer error messages. | **High** |
| 9.4 | Keep `relationship()` definitions consistent with FK columns (`back_populates`, `uselist`); resolve string class names by ensuring the declarative registry or imports load related modules before `configure_mappers()`. | Avoids `InvalidRequestError` / `NoForeignKeysError` at runtime. | **High** |
| 9.5 | For model ↔ YAML ↔ generated-model round-trips: ensure FK targets are **serializable** (literal strings or constants resolvable to strings). Registry indirection (`ForeignKey(ORG_ID)`) must be expanded when exporting to YAML. | Prevents `foreign_key: null` in specs and empty FK metadata in generated code. | **High** |
| 9.6 | Type conventions: `Uuid`/`UUID` for `*_id` columns, `DateTime(timezone=True)` for timestamps, `Numeric` for money—match validator rules if the project runs structural checks. | Catches drift before production DDL. | **Medium** |

---

## 10. Application services & persistence boundary

Database design succeeds only if the layer above it respects transactions and invariants.

| # | Rule | Rationale | Severity |
|---|------|-----------|----------|
| 10.1 | **Validate at the boundary** (DTO/schema) then persist; do not rely on the database alone for app-level rules, but do rely on the database for integrity it enforces best (FK, uniqueness, checks). | Defense in depth; clearer error handling. | **High** |
| 10.2 | Use a **unit of work** / session per request or use-case; commit once at the end of a successful business operation. | Avoids partial updates and inconsistent read-your-writes behavior. | **Critical** |
| 10.3 | Map domain exceptions to HTTP/API errors explicitly; map integrity errors (`UniqueViolation`, `ForeignKeyViolation`) to conflict or client-error responses with safe messages. | Prevents leaking internals and improves client behavior. | **High** |
| 10.4 | Services should accept filters and payloads that mirror **indexed** columns for list endpoints (`organization_id`, `status`, date ranges). | Prevents accidental full scans from generic “search everything” APIs. | **Medium** |
| 10.5 | Idempotent writes where the business requires them (use natural or external keys plus `ON CONFLICT` or explicit upsert patterns). | Safer retries and message-driven workflows. | **Medium** |

---

## Quick-Reference Checklist

### ✅ Do This (Every Time)

- [ ] Surrogate PK (`UUID` or `BIGINT`/identity) on every table
- [ ] `NOT NULL` as the default; NULL only with documented business reason
- [ ] `created_at` and `updated_at` on every table (`TIMESTAMPTZ`)
- [ ] Foreign keys with explicit targets and deliberate `ON DELETE` / `ON UPDATE`
- [ ] Index on every FK column (or covering composite that includes it)
- [ ] `UNIQUE` constraint on natural business identifiers where applicable
- [ ] `CHECK` constraints on status/value domains and sensible date/time pairs
- [ ] `EXPLAIN ANALYZE` on every new or modified heavy query
- [ ] Migration file committed and peer-reviewed
- [ ] `COMMENT ON TABLE` / `COMMENT ON COLUMN` where the project expects catalog docs
- [ ] **Multi-schema:** schema on table metadata matches FK qualification
- [ ] **ORM:** FK registry / constants centralized; YAML exports resolve symbols to strings
- [ ] **Validation/tooling:** FK target checks use metadata `Table.key`, not name alone

### ❌ Never Do This

- [ ] Store comma-separated values or arrays of IDs in a column when a junction table fits
- [ ] Use `FLOAT`/`DOUBLE` for monetary values
- [ ] Apply ad-hoc DDL directly in production
- [ ] Use `SELECT *` in application queries
- [ ] Index a boolean (or `Y`/`N` flag) alone without partial predicate or leading selective column
- [ ] Omit `ON DELETE` / `ON UPDATE` choice without documenting the DB default
- [ ] Store plaintext passwords or secrets
- [ ] Denormalize without a profiled performance justification
- [ ] Leave unused indexes accumulating
- [ ] Mix naming conventions within the same schema
- [ ] Scatter duplicate schema-qualified FK strings across the codebase

---

## Severity Guide

| Severity | Meaning | Action Required |
|----------|---------|-----------------|
| **Critical** | Violation causes data loss, corruption, or security breach. | Must fix before merge; blocks deployment. |
| **High** | Causes significant performance degradation or maintenance burden. | Must fix within the current sprint. |
| **Medium** | Best practice; improves clarity, maintainability, or minor performance. | Should fix; acceptable to defer with documented reason. |

---

## Document history

| Version | Date | Summary |
|---------|------|---------|
| 1.0 | 2026 | Initial rubric (tables through security). |
| 1.1 | 2026 | Multi-schema DDD, ORM/metadata rules, registry FK strings, service boundary, checklist expansion, plural/singular table naming nuance, Intelag alignment callouts. |
