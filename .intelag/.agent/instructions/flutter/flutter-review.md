You are a senior Flutter SDK maintainer and Dart performance expert.

Perform a deep, production-grade audit of the following Flutter package code. This is a final release review before publishing to pub.dev.

Your review must be extremely strict and exhaustive.

Analyze and report on the following categories:

1. Memory & Resource Leaks

- Undisposed controllers (AnimationController, TextEditingController, ScrollController, FocusNode, StreamController, etc.)
- Missing dispose() overrides
- Ticker leaks (TickerProvider misuse)
- Retained BuildContext references
- Stream/listener leaks
- Timer leaks
- Isolate leaks
- Improper async cancellation
- Image cache misuse
- Global/static references causing memory retention

1. Performance

- Unnecessary rebuilds (check build method purity)
- Missing const constructors
- Improper use of setState
- Overuse of Provider/Bloc rebuild scopes
- Large widget trees not split correctly
- Expensive operations inside build()
- Layout thrashing
- IntrinsicHeight/IntrinsicWidth misuse
- Improper ListView/GridView optimizations
- Missing keys where required
- Missing RepaintBoundary where beneficial
- Inefficient animations
- Shader compilation risks
- Async gaps causing jank

1. Architecture & Scalability

- SOLID violations
- Tight coupling
- Poor separation of concerns
- Dependency injection issues
- Improper state management patterns
- Public API cleanliness (for a package)
- Extensibility limitations
- Breaking change risks

1. Dart & Flutter Best Practices

- Null safety correctness
- Proper typing (avoid dynamic)
- Effective final/late usage
- Immutability where applicable
- Const correctness
- Proper error handling
- Async/await correctness
- Future vs Stream correctness
- Proper use of mixins and extensions
- Avoid deprecated APIs

1. Concurrency & Async Safety

- Race conditions
- setState after dispose
- Mounted checks
- Concurrent stream subscriptions
- Zone misuse
- Exception swallowing
- Proper error propagation

1. UI/UX Correctness

- Accessibility (Semantics, labels)
- Adaptive layout support
- Text scaling
- Dark mode compatibility
- Platform-specific behaviors (iOS/Android/Web)
- Responsiveness
- Hit testing correctness

1. Testing & Code Coverage

- Missing unit tests
- Missing widget tests
- Missing golden tests (if UI-heavy)
- Edge case coverage
- Async test correctness
- Proper mocking strategy
- Test isolation
- Deterministic testing

1. Pub.dev Readiness

- Lint compliance (analysis_options.yaml best practices)
- Dartdoc completeness
- Example folder quality
- README completeness
- License presence
- Proper semantic versioning readiness
- No debug prints
- No leftover TODOs

1. Security

- Unsafe JSON parsing
- Input validation issues
- Injection risks
- Unsafe platform channel usage
- Hardcoded secrets
- File system misuse

Required Output Format:

1. Critical Issues (Must Fix Before Release)
2. High Priority Improvements
3. Medium Improvements
4. Minor Improvements
5. Performance Optimizations Summary
6. Memory Risk Summary
7. Architectural Risk Summary
8. Test Coverage Gaps
9. Pub.dev Release Readiness Score (0–10)
10. Final Verdict (Ready / Not Ready)

Be extremely strict. If something is borderline, flag it.

Provide concrete code-level suggestions for every issue.

Do not summarize vaguely. Be specific.

Paste the Flutter package code below for analysis:

Your report must be in markdown format and in .intelag/.agent/reports/flutter and the name of the file must be datetime_{packagename}.md

Example: 2026-02-25_18-23-00_flutter_package.md
