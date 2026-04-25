Intelag Repository and Package Standard

HARD RULES

* Full PEP8 compliance.
* Must pass MyPy strict mode.
* All functions fully type annotated.
* Avoid Any and type ignore unless absolutely unavoidable.
* Replace magic numbers and strings with constants or enums.
* Prefix unused variables with underscore.
* Separate imports into four groups: standard library, third party, intelag packages, internal modules.
* Use absolute imports only, no relative imports.
* Do not use if TYPE_CHECKING.
* Every file starts with a module level docstring.
* All public functions have exactly one line docstrings.
* Never use print, use logger with lazy percent formatting.
* Log errors before raising exceptions.
* Error handlers return safe defaults.
* Avoid getattr, setattr, delattr, hasattr.
* File size under 800 lines.
* Each file defines example_usage and calls it under main.
* Review code twice and remove duplication.
* Prefer minimal simple code.

Repository Structure

* Root contains intelag_main_package for orchestration.
* intelag_main_package includes cli, services, models, config, utils.
* intelag_packages for internal pip installable packages.
* intelag_submodules for shared git submodules.
* scripts for non package utilities.
* pyproject.toml for build system and CLI entry points.
* README.md architecture first.
* Lint and mypy configs at root.

Naming Conventions

* Packages and directories use snake_case with intelag_ prefix.
* Classes use PascalCase.
* Functions and attributes use snake_case.
* CLI subcommand functions use cmd_ prefix.
* Constants use UPPER_SNAKE_CASE.

CLI Design Pattern

* Use argparse with subcommands.
* cli module acts as facade and exports public CLI components via all.
* Complex subcommands use class pattern with configure_parser(parser) and run(args).

Coding Standards

* Each file begins with module docstring.
* Imports grouped as standard, third party, intelag shared, internal project.
* Use absolute imports only.
* No TYPE_CHECKING.
* Public functions have exactly one line docstrings.
* All parameters and return values fully typed.

Configuration Pattern

* Use YAML for all configurations.
* Load configs via intelag_config_model.
* Store defaults in config directory.
* Allow user override with --config flag.
* Use importlib.resources for packaged defaults.

Execution Block Pattern

* Each backend or utility file defines example_usage.
* example_usage is called under if main for manual testing.

---
Performance Optimization Rules:

You are an expert Python performance engineer. Follow these rules when optimizing Python code:

1) Measure first. Always profile before optimizing. Use cProfile, line_profiler, or py-spy to identify actual bottlenecks. Never guess.

2) Kill Python loops. Python loop overhead is expensive. Replace manual iteration with vectorized operations, list comprehensions, or built-ins like sum(), map(), and filter(). Use NumPy for numerical work.

3) Pick the correct concurrency model. The GIL blocks CPU-bound threads. Use threading or asyncio for I/O-bound work. Use multiprocessing or ProcessPoolExecutor for CPU-bound work.

4) Consider faster runtimes. For pure Python code without C extensions, PyPy can provide significant speedups with no code changes. It does not work well with NumPy, pandas, or other C-extension-heavy libraries.

5) Compile hotspots. Move performance-critical code to native execution using Numba (@njit decorator), Cython, or writing C extensions. This is especially effective for tight numerical loops.

6) Reduce allocations. Object creation and garbage collection have costs. Use generators instead of building large intermediate lists. Reuse objects where possible. Consider __slots__ for memory-heavy classes.

7) Upgrade data structures. Use sets for membership testing instead of lists. Use dicts for lookups. Use collections.deque for queue operations. Use heapq for priority queues. Wrong data structure choice turns O(1) operations into O(n).

8) Use faster serialization. Replace stdlib json with orjson or ujson for JSON-heavy workloads. Use pickle protocol 5 or msgpack for internal serialization.

9) Optimize database and I/O. Batch operations to reduce round trips. Use executemany instead of execute in loops. Wrap bulk operations in transactions. Use connection pooling.

10) Practice logging discipline. Avoid f-strings or .format() in logging calls on hot paths. Use lazy formatting with percent syntax: logger.info("value=%s", x). This avoids string construction when the log level is disabled.

11) Cache expensive computations. Use functools.lru_cache for pure functions with repeated inputs. Use external caching like Redis for distributed workloads.

12) Optimize string operations. Use str.join() instead of repeated concatenation with +=. Use io.StringIO for building large strings incrementally.

13) Cache attribute lookups in tight loops. Assign frequently accessed attributes or global functions to local variables before entering a loop. Local variable access is faster than attribute or global lookups.

Apply these rules only after profiling confirms where time is actually spent. Premature optimization is counterproductive.
