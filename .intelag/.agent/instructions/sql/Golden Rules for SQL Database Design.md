# Golden Rules for SQL Database Design

**A Senior Engineer's Rubric** · Version 1.0 · 2026

*Schema Design · Normalization · Indexing · Relationships · Performance*

---

## 1. Table & Schema Design

Well-structured tables form the foundation of a performant, maintainable database. These rules ensure clarity, consistency, and long-term scalability.

| # | Rule | Rationale | Severity |
|---|------|-----------|----------|
| 1.1 | Use singular, lowercase, `snake_case` table names (e.g., `customer_order`, not `CustomerOrders`). | Prevents quoting issues across engines; matches SQL convention. | **High** |
| 1.2 | Every table must have a single-column surrogate primary key, preferably named `id` or `<table>_id`. | Guarantees row identity; simplifies joins and ORM mapping. | **Critical** |
| 1.3 | Prefer UUID v7 or `BIGINT IDENTITY/SERIAL` for primary keys over natural keys. | Natural keys change, compound keys complicate joins, and GUIDs are non-sequential. UUIDv7 is time-sortable. | **High** |
| 1.4 | Every table must include `created_at` (`TIMESTAMPTZ, DEFAULT NOW()`) and `updated_at` columns. | Enables auditing, debugging, and incremental ETL. | **High** |
| 1.5 | Default all columns to `NOT NULL` unless NULL carries explicit business meaning. | Eliminates three-value logic bugs and enables index efficiency. | **Critical** |
| 1.6 | Use the smallest appropriate data type: `INT` vs `BIGINT`, `VARCHAR(n)` vs `TEXT`, `DATE` vs `TIMESTAMP`. | Reduces page size, improves cache hit ratio, and shrinks indexes. | **Medium** |
| 1.7 | Store monetary values as `DECIMAL/NUMERIC` with explicit precision (e.g., `DECIMAL(19,4)`), never `FLOAT`. | Floating-point rounding causes financial discrepancies. | **Critical** |
| 1.8 | Store all timestamps in UTC (`TIMESTAMPTZ`). Convert to local time only at the application layer. | Avoids DST ambiguity and cross-timezone comparison bugs. | **High** |
| 1.9 | Add `CHECK` constraints for domain validation (e.g., `CHECK (status IN ('active','inactive'))`, `CHECK (price >= 0)`). | Catches invalid data at the engine level before it corrupts downstream. | **Medium** |
| 1.10 | Avoid storing derived/calculated data unless materialized for proven performance reasons. | Keeps a single source of truth; eliminates stale calculation risk. | **Medium** |

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
| 3.1 | Every foreign key must have an explicit `REFERENCES` clause with `ON DELETE` and `ON UPDATE` actions. | Missing cascade/restrict rules leave orphan records or silent failures. | **Critical** |
| 3.2 | Name foreign key columns as `<referenced_table>_id` (e.g., `customer_id` references `customer.id`). | Self-documenting; enables automated ORM discovery. | **High** |
| 3.3 | Use junction/bridge tables for many-to-many relationships; never embed arrays of IDs. | Preserves referential integrity and enables efficient querying. | **Critical** |
| 3.4 | Choose `ON DELETE` actions deliberately: `RESTRICT` for strong entities, `CASCADE` for owned children, `SET NULL` for optional associations. | Wrong action either blocks operations or silently destroys data. | **Critical** |
| 3.5 | Always index foreign key columns (most engines except MySQL/InnoDB do not auto-index them). | Unindexed FKs cause full table scans on `DELETE`/`UPDATE` of the parent row. | **High** |
| 3.6 | Use composite foreign keys only when the referenced table has a composite PK; prefer surrogate keys otherwise. | Reduces key width, join complexity, and migration difficulty. | **Medium** |
| 3.7 | For soft-delete patterns, add a `deleted_at TIMESTAMPTZ` column instead of removing rows; maintain FK integrity. | Allows auditability and undo; avoids orphan cascades. | **Medium** |
| 3.8 | Self-referencing FKs (e.g., `employee.manager_id → employee.id`) must allow `NULL` for root nodes. | Prevents a chicken-and-egg insertion problem. | **Medium** |

---

## 4. Indexing Strategy

Indexes are the single largest lever for read performance. Every index also carries a write penalty—balance accordingly.

| # | Rule | Rationale | Severity |
|---|------|-----------|----------|
| 4.1 | Index every column used in `WHERE`, `JOIN ON`, and `ORDER BY` clauses of frequent queries. | Without an index, the engine falls back to sequential scans. | **Critical** |
| 4.2 | Create composite indexes with the highest-selectivity column first, followed by range/sort columns. | Column order determines whether the index can satisfy the query plan. | **Critical** |
| 4.3 | Never index columns with very low cardinality (e.g., boolean flags) in isolation—combine with a selective column. | A standalone low-cardinality index is rarely chosen by the optimizer. | **High** |
| 4.4 | Use `INCLUDE` (covering index) columns to enable index-only scans for read-heavy queries. | Avoids expensive heap lookups by serving data directly from the index. | **Medium** |
| 4.5 | Use partial/filtered indexes for queries that target a subset (e.g., `WHERE status = 'active'`). | Dramatically smaller index; faster scans and lower maintenance. | **Medium** |
| 4.6 | Add `UNIQUE` indexes/constraints for business-level uniqueness (email, SKU, etc.) beyond the PK. | Prevents duplicate data at the engine level, not just the app level. | **Critical** |
| 4.7 | Limit total indexes per table to roughly 5–7 on write-heavy tables; each index adds `INSERT`/`UPDATE` overhead. | Write amplification degrades throughput and increases WAL volume. | **High** |
| 4.8 | Audit unused indexes monthly (`pg_stat_user_indexes`, `sys.dm_db_index_usage_stats`) and drop them. | Dead indexes consume storage, slow writes, and bloat backups. | **Medium** |
| 4.9 | Use expression/functional indexes for computed predicates (e.g., `INDEX ON LOWER(email)`). | Allows the optimizer to use the index when the query applies the same function. | **Medium** |
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
| 6.2 | Prefix boolean columns with `is_`, `has_`, or `can_` (e.g., `is_active`, `has_verified_email`). | Self-documenting; removes ambiguity about the column's intent. | **Medium** |
| 6.3 | Name indexes as `idx_<table>_<columns>` and constraints as `chk_`, `uq_`, `fk_` prefixes. | Allows instant identification in error messages and monitoring. | **Medium** |
| 6.4 | Add `COMMENT ON TABLE` / `COMMENT ON COLUMN` for every object in the schema. | Surfaces documentation in IDE tooltips and data catalog tools. | **Medium** |
| 6.5 | Maintain versioned migration files (Flyway, Liquibase, Alembic)—never apply ad-hoc DDL in production. | Ensures reproducibility, rollback capability, and audit history. | **Critical** |
| 6.6 | Keep an up-to-date ERD (entity-relationship diagram) in the repository, regenerated from the live schema. | Visual references accelerate understanding for new team members. | **Medium** |

---

## 7. Security & Access Control

The database is the last line of defense. Assume the application layer will eventually have a bug.

| # | Rule | Rationale | Severity |
|---|------|-----------|----------|
| 7.1 | Apply the principle of least privilege: application roles should have only `SELECT`/`INSERT`/`UPDATE` on needed tables. | Limits blast radius if credentials are compromised. | **Critical** |
| 7.2 | Never store plaintext passwords or API keys in any column; use hashed values with bcrypt/argon2. | Plaintext credentials in a breach expose all users. | **Critical** |
| 7.3 | Encrypt PII columns at rest (TDE or column-level encryption) and mask them in non-production environments. | Regulatory compliance (GDPR, HIPAA) and breach risk reduction. | **High** |
| 7.4 | Use row-level security (RLS) for multi-tenant databases instead of filtering in the application. | Prevents data leakage even if the app has a logic bug. | **High** |
| 7.5 | Audit all DDL changes and privileged DML with `pg_audit` or equivalent engine feature. | Creates a forensic trail for security incidents and change management. | **Medium** |

---

## Quick-Reference Checklist

### ✅ Do This (Every Time)

- [ ] Surrogate PK (`BIGINT` or UUIDv7) on every table
- [ ] `NOT NULL` as the default; NULL only with documented business reason
- [ ] `created_at` and `updated_at` on every table
- [ ] Foreign keys with explicit `ON DELETE` / `ON UPDATE`
- [ ] Index on every FK column
- [ ] `UNIQUE` constraint on natural business identifiers
- [ ] `EXPLAIN ANALYZE` on every new or modified query
- [ ] Migration file committed and peer-reviewed
- [ ] `COMMENT ON TABLE` and `COMMENT ON COLUMN` documented
- [ ] Timestamps stored in UTC (`TIMESTAMPTZ`)

### ❌ Never Do This

- [ ] Store comma-separated values or arrays of IDs in a column
- [ ] Use `FLOAT`/`DOUBLE` for monetary values
- [ ] Apply ad-hoc DDL directly in production
- [ ] Use `SELECT *` in application queries
- [ ] Index a boolean column on its own
- [ ] Skip `ON DELETE` action on foreign keys
- [ ] Store plaintext passwords or secrets
- [ ] Denormalize without a profiled performance justification
- [ ] Leave unused indexes accumulating
- [ ] Mix naming conventions within the same schema

---

## Severity Guide

| Severity | Meaning | Action Required |
|----------|---------|-----------------|
| **Critical** | Violation causes data loss, corruption, or security breach. | Must fix before merge; blocks deployment. |
| **High** | Causes significant performance degradation or maintenance burden. | Must fix within the current sprint. |
| **Medium** | Best practice; improves clarity, maintainability, or minor performance. | Should fix; acceptable to defer with documented reason. |