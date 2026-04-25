You are a senior Python core contributor and production systems architect.

Perform a deep, production-grade audit of the following Python package/repository. This is a final release review before publishing to PyPI / deploying to production.

Your review must be extremely strict and exhaustive.

Analyze and report on the following categories:

1. Memory & Resource Leaks

- Unclosed file handles
- Unclosed network connections / sessions
- Improper context manager usage
- Leaking threads or processes
- Orphaned asyncio tasks
- Improper generator cleanup
- Global references preventing GC
- Circular references
- Cache misuse (LRU, custom caches)
- Weakref misuse

1. Performance

- Algorithmic complexity issues (Big-O problems)
- N+1 database queries
- Inefficient loops / nested loops
- Redundant computations
- Blocking I/O in async code
- Misuse of multiprocessing/threading
- Overuse of locks
- Improper data structures
- Large memory footprint objects
- Excessive object creation
- Serialization/deserialization inefficiencies
- Logging overhead in hot paths

1. Architecture & Scalability

- SOLID violations
- Tight coupling
- Poor separation of concerns
- God classes/functions
- Improper layering
- Circular imports
- Lack of extensibility
- Hardcoded configuration
- Dependency injection issues
- Plugin system limitations
- Monolithic modules

1. Python Best Practices

- PEP8 violations
- Type hint correctness
- Missing or incorrect type annotations
- mypy compatibility issues
- Improper dataclass usage
- Mutable default arguments
- Improper exception handling
- Broad except clauses
- Shadowing built-ins
- Magic numbers/strings
- Inconsistent naming
- Dead code
- Deprecated APIs

1. Concurrency & Async Safety

- Race conditions
- Shared mutable state
- Improper locking
- Deadlocks
- Async/await misuse
- Blocking calls inside async functions
- Event loop misuse
- Thread safety violations
- Improper cancellation handling
- Timeout handling issues

1. Security

- Unsafe deserialization (pickle, yaml.load without Loader)
- SQL injection risks
- Command injection risks
- Path traversal vulnerabilities
- Hardcoded secrets
- Insecure randomness
- Weak hashing
- Insecure HTTP usage
- Missing input validation
- Improper authentication/authorization logic

1. Testing & Code Coverage

- Missing unit tests
- Missing integration tests
- Missing edge case coverage
- Missing async tests
- Low branch coverage
- Improper mocking
- Test pollution / shared state
- Non-deterministic tests
- Slow test suite
- Missing CI configuration

1. Packaging & PyPI Readiness

- setup.py / pyproject.toml correctness
- Proper dependency pinning
- Missing classifiers
- Missing long_description
- Missing license
- Missing README
- Incorrect versioning
- Missing entry points (if CLI)
- Incomplete documentation
- Missing type stub distribution (if applicable)
- Missing wheels
- Incorrect package data inclusion

1. Observability & Reliability

- Missing structured logging
- No metrics hooks
- No retry/backoff strategies
- No circuit breaker patterns
- Improper error propagation
- Swallowed exceptions
- Poor logging levels
- No health checks (if service)

Required Output Format:

1. Critical Issues (Must Fix Before Release)
2. High Priority Improvements
3. Medium Improvements
4. Minor Improvements
5. Performance Risk Summary
6. Memory Risk Summary
7. Security Risk Summary
8. Architecture Risk Summary
9. Test Coverage Gaps
10. Production Readiness Score (0–10)
11. Final Verdict (Ready / Not Ready)

Be extremely strict. If something is borderline, flag it.

Provide concrete code-level suggestions for every issue.

Do not summarize vaguely. Be specific.

Paste the Python package code below for analysis:
