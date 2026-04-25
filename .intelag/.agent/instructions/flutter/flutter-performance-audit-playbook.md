# Flutter Performance-First Audit & Code Review Playbook

## AI Agent Operating Protocol

You are a performance-focused Flutter code reviewer. When given code alongside this playbook, follow this exact workflow:

### Step 1: Scope & Inventory

1. List all files in the audit scope.
2. Classify each file: widget, screen, controller/notifier, model, service, test, config, generated.
3. Exclude generated files (`*.g.dart`, `*.freezed.dart`, `*.mocks.dart`, `*.gr.dart`, `build/**`, `.dart_tool/**`) unless explicitly included.
4. Identify hot paths: scrollable lists, drag surfaces, animated widgets, tab bars, high-frequency input handlers.

### Step 2: Anti-Pattern Scan

1. Search for every anti-pattern listed in the **Anti-Pattern Detection Reference** section below.
2. For each match, record: file, line, rule violated, severity (P0/P1/P2).

### Step 3: Fix or Report

- **If instructed to fix**: Apply the prescribed fix from the anti-pattern reference. Verify the fix compiles and does not change behavior.
- **If instructed to audit only**: Emit the full audit report (see **Mandatory AI Audit Report Format**).
- **Default behavior**: Fix P0 issues in-place, report P1/P2 issues in the audit output.

### Step 4: Validate

1. Confirm `flutter analyze` passes after changes.
2. Confirm no new lint warnings from `prefer_const_constructors` or `prefer_const_literals_to_create_immutables`.
3. Run existing tests if available.

### Step 5: Emit Report

Output the structured audit report using the template at the end of this playbook. Every finding must have a status. Never omit the machine-readable JSON summary.

### Decision Rules

- **Fix autonomously**: Missing `const`, missing `dispose()`, `shrinkWrap: true` removal, `Opacity` widget → `color.withValues(alpha:)`, helper methods → widgets, `IntrinsicWidth`/`IntrinsicHeight` removal, missing `cacheWidth`/`cacheHeight`, missing `child` parameter on builders.
- **Report but ask before fixing**: State management restructuring, widget extraction/splitting, controller decomposition, theme refactoring.
- **Report only**: Architecture-level changes, package splits, navigation restructuring.

## Summary

This playbook enforces a strict priority hierarchy: **Performance (P0) → Modularity (M) → Documentation (D)**. A component must pass performance gates before structural refactoring begins. All performance claims require measured evidence in profile mode — debug-mode benchmarks are invalid.

## Project-Agnostic Interpretation Rules

This playbook is intentionally project-agnostic and must be interpreted broadly enough to apply to any Flutter codebase, whether the audited target is a widget, screen, route, feature module, design-system primitive, shell surface, or full application flow.

1. The word `component` means the audit target under review. It may be a single widget, a subtree, a screen, a package, or a complete user flow.
2. Terms such as tabs, panels, dashboards, grids, and drag layers are examples of high-risk UI patterns, not required project structures.
3. References to `README.md` mean the canonical engineering document for the target. This may be a README, ADR, feature doc, package doc, or equivalent markdown guidance.
4. If a project already defines stricter budgets, coding standards, or reporting formats, those stricter local rules override the defaults in this playbook.
5. If a project lacks automation or instrumentation, the audit must still be completed using manual evidence, but every missing measurement must be marked explicitly as `Not Measured` rather than inferred.
6. The auditor must never assume that absence of evidence implies a pass. Unverified items remain `UNKNOWN` or `NOT MEASURED` until proven.
7. Scope must explicitly declare included and excluded files. Generated code and tool outputs are excluded by default unless they are the direct audit target.
8. Default exclusions: `build/**`, `.dart_tool/**`, `.fvm/**`, `android/.gradle/**`, `ios/Pods/**`, `**/*.g.dart`, `**/*.freezed.dart`, `**/*.mocks.dart`, `**/*.gr.dart`, `**/generated/**`.
9. Any override to an exclusion rule must be documented in the report with a rationale.

## Definition of Done

On 120Hz displays, the per-frame budget is **8.33ms** (16.6ms at 60Hz). Exceeding this budget causes dropped frames (jank). A component is "done" only when all gates below are satisfied.

### Completion Criteria

| Metric                          | Measurement Methodology                                   | Target Threshold                                                                                    |
| :------------------------------ | :-------------------------------------------------------- | :-------------------------------------------------------------------------------------------------- |
| **Frame Timing Budgets**        | Automated `watchPerformance` integration tests            | Average/p90/p99 frame build and raster times < 8.33ms (120Hz) or < 16.6ms (60Hz)                    |
| **Rebuild Hotspot Elimination** | DevTools Rebuild Stats (v2.36+) during user interaction   | Static subtrees must exhibit a build count of exactly 1 during scroll or drag events                |
| **Visual State Resolution**     | Golden file assertions (`matchesGoldenFile`)              | Interactive styling (hover, selection, etc.) resolved via `WidgetState` without hardcoded overrides |
| **Architectural Documentation** | Manual code review against canonical target documentation | Documentation accurately details performance budgets, theming rules, and state listening protocols  |

### Default Budget Thresholds (when local policy is absent)

Use these defaults only when the target project has not defined stricter thresholds:

| Metric                         | Default Threshold                                | Notes                                                                  |
| :----------------------------- | :----------------------------------------------- | :--------------------------------------------------------------------- |
| Avg/P90 build time             | `<= frame_budget_ms`                             | `frame_budget_ms` = 8.33 for 120Hz, 16.6 for 60Hz                      |
| P99 build time                 | `<= frame_budget_ms * 1.25`                      | Tolerates limited tail latency                                         |
| Worst build time               | `<= frame_budget_ms * 1.50`                      | Worst-frame guardrail                                                  |
| Avg/P90 raster time            | `<= frame_budget_ms`                             | Same budget as build                                                   |
| P99 raster time                | `<= frame_budget_ms * 1.25`                      | Tail latency cap                                                       |
| Worst raster time              | `<= frame_budget_ms * 1.50`                      | Worst-frame cap                                                        |
| Janky frame ratio              | `<= 1.0%`                                        | Hard fail at `> 2.0%` unless waived                                    |
| Missed build/raster budget     | `<= 0` (critical paths), `<= 1` (non-critical)   | Critical paths include scroll, drag, and tab-switch interactions       |
| Rebuild hotspot count (static) | `0`                                              | Static containers should not rebuild during high-frequency interaction |
| Memory stability               | Net heap growth `<= +5%` after 3 repeated cycles | Compare post-GC snapshots at equivalent states                         |

## Profiling Workflows and Measurable Baselines

**All profiling must use profile mode** (`flutter run --profile`). Debug mode adds assertion/tracing overhead that invalidates timing data.

### Establishing Mandatory Baseline Scenarios

Define a reproducible baseline scenario before optimizing. The scenario must cover:

1. **Initialization:** Loading the component with a high volume of data or children.
2. **High-Frequency Interaction:** Simulating rapid interactions (e.g., tab switching, scrolling through deeply populated lists, or drag-and-drop actions).
3. **State Transitions:** Triggering state-dependent animations, such as hover elevations, press ripples, or dynamic resizing.

### Reproducibility and Sampling Protocol

Performance status is only valid if runs are reproducible.

1. Execute 2 warm-up runs and discard them.
2. Execute 5 measured runs per scenario (minimum 3 only if constrained, with explicit justification).
3. Report median plus percentile metrics for each measured run set.
4. Use stable data seeds/fixtures and fixed interaction scripts.
5. Record device model, OS version, Flutter version, renderer, and thermal/power conditions.
6. If run-to-run variance for a key metric exceeds 10%, mark scenario status as `UNSTABLE` and do not claim `PASS` without a waiver.
7. For CI gating, fail when median exceeds threshold or when at least 3 of 5 runs exceed threshold.

### Profiling Toolchain

| Diagnostic Tool                   | Primary Function                    | Extracted Metrics and Observations                                                                                                  |
| :-------------------------------- | :---------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------- |
| **DevTools Performance View**     | Frame timing and jank detection     | UI thread vs. Raster thread execution times; identification of red jank bars.                                                       |
| **DevTools Rebuild Stats**        | Widget build frequency analysis     | Aggregate build counts per widget across frames.                                                                                    |
| **DevTools Frame Analysis**       | Frame-specific bottleneck isolation | Deep inspection of individual slow frames to identify expensive operations.                                                         |
| **Inspector: Highlight Repaints** | Visual boundary debugging           | Identifies excessive layer invalidations where static UI elements are repainted unnecessarily.                                      |
| **Inspector: Oversized Images**   | Asset memory profiling              | Detects images decoded at resolutions exceeding their logical display size. Also triggered via `debugInvertOversizedImages = true`. |
| **DevTools Memory View**          | Heap and allocation tracking        | Monitors Dart heap and native memory footprints; utilizes diff snapshots and trace instances to find allocation churn and leaks.    |

### Debug Diagnostic Flags (use sparingly, debug-mode only)

These flags add overhead and are for pattern identification, not final timing:

- `debugPrintRebuildDirtyWidgets = true` — Logs every dirty widget rebuilt each frame to the console. Use short sessions only.
- `debugProfileBuildsEnabled = true` — Adds timeline events for each widget build, visible in DevTools timeline.
- `debugProfileLayoutsEnabled = true` — Adds timeline events for layout passes.
- `debugProfilePaintsEnabled = true` — Adds timeline events for paint passes.
- `debugInvertOversizedImages = true` — Visually inverts images decoded at resolutions exceeding their display size, making oversized decodes immediately obvious.

### ImageCache Configuration

Flutter's `ImageCache` is an LRU cache with defaults of up to 1000 images / 100MB. For components with heavy image usage, tune via `PaintingBinding.instance.imageCache.maximumSize` and `maximumSizeBytes`. Over-caching risks memory pressure; under-caching causes repeated decode jank.

## Automated Performance Measurement in CI

Flutter's `integration_test` package captures metrics in CI pipelines via two APIs:

- **`watchPerformance()`** — Watches `FrameTiming` during an action; reports avg, p90, p99, worst for build/raster. Primary budget enforcement tool.
- **`traceAction()`** — Records full timeline for deeper event-level analysis. Use when `watchPerformance` finds a regression but root cause is unclear.

```dart
import 'package:integration_test/integration_test.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter/material.dart';
// Import the component to be tested
// import 'package:your_package/src/components/your_component.dart';

void main() {
  final binding = IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('Component performance budget validation', (WidgetTester tester) async {
    // await tester.pumpWidget(const MaterialApp(home: YourComponent()));

    // Wait for initial rendering to settle
    await tester.pumpAndSettle();

    await binding.watchPerformance(() async {
      // Execute standard component interactions
      // final listFinder = find.byType(Scrollable);

      // Simulate interactions (e.g., fling, drag, tap)
      // await tester.fling(listFinder, const Offset(0, -800), 2000);
      await tester.pumpAndSettle();

    }, reportKey: 'component_perf_metrics');
  });
}
```

The resulting JSON artifact will output percentile data (e.g., `average_frame_build_time_millis`, `worst_frame_build_time_millis`, `missed_frame_build_budget_count`). These outputs function as the empirical gate for the P0 phase, ensuring that regression thresholds automatically fail the CI pipeline if performance degrades.

## Anti-Pattern Detection Reference

Use this table as a quick-lookup during code review. Search for the **Detection Pattern** (grep/regex) and apply the corresponding fix. This section is the primary action driver — scan the codebase for these patterns first, then consult the detailed Phase 1 explanations below for context when needed.

### Rendering Anti-Patterns

| ID   | Anti-Pattern                                          | Detection Pattern (grep/regex)                                              | Severity | Fix                                                                                      |
| :--- | :---------------------------------------------------- | :-------------------------------------------------------------------------- | :------- | :--------------------------------------------------------------------------------------- |
| R1   | `shrinkWrap: true` on scrollable                      | `shrinkWrap:\s*true`                                                        | P0       | Remove `shrinkWrap`, wrap in `Expanded`/`Flexible` or provide explicit height constraint |
| R2   | `Opacity` widget wrapping subtree                     | `\bOpacity\(` (verify not `color.withValues(alpha:`)                        | P0       | Apply opacity directly to color: `color.withValues(alpha: 0.8)`                          |
| R3   | `IntrinsicHeight` or `IntrinsicWidth`                 | `\bIntrinsicHeight\b` or `\bIntrinsicWidth\b`                               | P0       | Replace with `SizedBox`, `ConstrainedBox`, `Align`, or `Flex`                            |
| R4   | `Clip.antiAliasWithSaveLayer`                         | `antiAliasWithSaveLayer`                                                    | P0       | Use `Clip.antiAlias` or `Clip.hardEdge`                                                  |
| R5   | Missing `const` on constructors                       | Lint: `prefer_const_constructors`                                           | P1       | Add `const` keyword                                                                      |
| R6   | Missing `itemExtent`/`prototypeItem` on uniform lists | `ListView` or `ListView.builder` without `itemExtent`                       | P1       | Add `itemExtent` or `prototypeItem` for uniform children                                 |
| R7   | Image without `cacheWidth`/`cacheHeight`              | `Image.asset(`, `Image.network(`, `Image.file(` without `cacheWidth`        | P0       | Add `cacheWidth`/`cacheHeight` matching display size × `devicePixelRatio`                |
| R8   | Missing `RepaintBoundary` on animated subtree         | Manual: animated widget without `RepaintBoundary` ancestor                  | P1       | Wrap in `RepaintBoundary`, verify via Inspector "Highlight Repaints"                     |
| R9   | Missing `child` param on builder widgets              | `AnimatedBuilder(`, `ValueListenableBuilder(`, `Consumer(` without `child:` | P1       | Extract stable subtree into `child` parameter                                            |
| R10  | Nested `ListView` in unbounded `Column`               | `Column(` containing `ListView` without `Expanded`/`Flexible` wrapper       | P0       | Wrap inner `ListView` in `Expanded` or `Flexible`                                        |
| R11  | Oversized image decode                                | `debugInvertOversizedImages` (verify at runtime)                            | P1       | Add `cacheWidth`/`cacheHeight` to `Image` provider                                       |

### State Management Anti-Patterns

| ID   | Anti-Pattern                                       | Detection Pattern (grep/regex)                                                                          | Severity | Fix                                                                            |
| :--- | :------------------------------------------------- | :------------------------------------------------------------------------------------------------------ | :------- | :----------------------------------------------------------------------------- |
| S1   | `setState` in root/scaffold/screen widget          | `setState(` in `*_screen.dart`, `*_page.dart`, `*_shell.dart`                                           | P0       | Extract to `ValueNotifier` + `ValueListenableBuilder` or dedicated controller  |
| S2   | Broad `context.watch<T>()` in structural widget    | `context.watch<` in scaffold/shell/page widgets                                                         | P0       | Replace with `context.select<T, R>()` for specific fields                      |
| S3   | Broad `Consumer<T>` wrapping large subtree         | `Consumer<` wrapping `Column`, `Row`, `Scaffold`, `CustomScrollView`                                    | P0       | Narrow `Consumer` to leaf widget or use `Selector`                             |
| S4   | `Future`/`Stream` created in `build()`             | `Future.` or `Stream.` inside `Widget build(` method body                                               | P0       | Move creation to `initState`, `didChangeDependencies`, or controller           |
| S5   | `Provider.of<T>(context)` defaulting to listen     | `Provider.of<` without `listen: false`                                                                  | P1       | Add `listen: false` or use `context.read<T>()`                                 |
| S6   | `notifyListeners()` in a loop                      | `notifyListeners()` inside `for`/`while`/`forEach` block                                                | P0       | Batch mutations, call `notifyListeners()` once after loop                      |
| S7   | Missing `dispose()` for controllers                | `AnimationController`, `ScrollController`, `TabController`, `TextEditingController` without `dispose()` | P0       | Add `dispose()` override calling `.dispose()` on every controller              |
| S8   | In-place collection mutation + notify              | `.add(`, `.remove(`, `.clear(` followed by `notifyListeners`                                            | P1       | Create new collection: `state = [...state, newItem]` for immutable transitions |
| S9   | `ref.watch(provider)` returning large state object | `ref.watch(` returning entire state class (not `.select(...)`)                                          | P1       | Use `ref.watch(provider.select((s) => s.field))`                               |
| S10  | `BlocBuilder` without `buildWhen` on large subtree | `BlocBuilder<` without `buildWhen:`                                                                     | P1       | Add `buildWhen:` or use `BlocSelector` for narrow rebuilds                     |

### Architecture Anti-Patterns

| ID   | Anti-Pattern                            | Detection Pattern (grep/regex)                            | Severity | Fix                                                                            |
| :--- | :-------------------------------------- | :-------------------------------------------------------- | :------- | :----------------------------------------------------------------------------- |
| A1   | UI helper methods instead of widgets    | `Widget _build` prefix pattern in widget files            | P1       | Convert to dedicated `StatelessWidget` class                                   |
| A2   | Hardcoded colors in feature code        | `Color(0x` or `Colors.` outside theme/design-system files | P1       | Use `Theme.of(context).colorScheme` or design tokens                           |
| A3   | Hardcoded `TextStyle` in feature code   | `TextStyle(` outside theme files                          | P1       | Use `Theme.of(context).textTheme` or design tokens                             |
| A4   | Mega-controller (>300 lines)            | Controller/notifier file >300 lines                       | P1       | Split by domain responsibility (e.g., `TabsState`, `DragState`, `LayoutState`) |
| A5   | Business logic inside widget `build()`  | Network calls, DB queries, complex algorithms in widget   | P0       | Move to controller/service/use-case layer                                      |
| A6   | Deep inheritance instead of composition | `extends` chain >2 levels deep on widget classes          | P2       | Compose behavior with mixins or delegate objects                               |

### Before/After Fix Examples

**R1: Remove shrinkWrap**

```dart
// BAD — forces layout of ALL children upfront, destroys laziness
ListView.builder(
  shrinkWrap: true,
  itemCount: items.length,
  itemBuilder: (context, index) => ItemTile(items[index]),
)

// GOOD — lazy rendering with proper constraints
Expanded(
  child: ListView.builder(
    itemCount: items.length,
    itemBuilder: (context, index) => ItemTile(items[index]),
  ),
)
```

**R2: Opacity widget → color opacity**

```dart
// BAD — allocates offscreen buffer via saveLayer
Opacity(
  opacity: 0.5,
  child: Container(color: Colors.blue, child: content),
)

// GOOD — no saveLayer, opacity applied to color directly
Container(
  color: Colors.blue.withValues(alpha: 0.5),
  child: content,
)
```

**S1: setState in screen root → ValueNotifier at leaf**

```dart
// BAD — rebuilds entire screen on every hover change
class _ScreenState extends State<MyScreen> {
  int _hoveredIndex = -1;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Column(children: [
        Header(), // rebuilds unnecessarily
        ListView.builder(
          itemBuilder: (ctx, i) => MouseRegion(
            onEnter: (_) => setState(() => _hoveredIndex = i),
            child: ListTile(selected: _hoveredIndex == i),
          ),
        ),
      ]),
    );
  }
}

// GOOD — only the hovered tile rebuilds
class _ScreenState extends State<MyScreen> {
  final _hoveredIndex = ValueNotifier<int>(-1);

  @override
  void dispose() {
    _hoveredIndex.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Column(children: [
        const Header(), // never rebuilds
        Expanded(
          child: ListView.builder(
            itemBuilder: (ctx, i) => ValueListenableBuilder<int>(
              valueListenable: _hoveredIndex,
              builder: (ctx, hovered, child) => MouseRegion(
                onEnter: (_) => _hoveredIndex.value = i,
                child: ListTile(selected: hovered == i),
              ),
            ),
          ),
        ),
      ]),
    );
  }
}
```

**A1: Helper method → Widget class**

```dart
// BAD — no element reuse, no const, no own BuildContext
class MyScreen extends StatelessWidget {
  Widget _buildHeader() {
    return Padding(
      padding: EdgeInsets.all(16),
      child: Text('Header'),
    );
  }

  @override
  Widget build(BuildContext context) => Column(children: [_buildHeader()]);
}

// GOOD — const-capable, own element, reusable
class _Header extends StatelessWidget {
  const _Header();

  @override
  Widget build(BuildContext context) {
    return const Padding(
      padding: EdgeInsets.all(16),
      child: Text('Header'),
    );
  }
}
```

**S4: Future/Stream created in build → lifecycle method**

```dart
// BAD — new Future created on every rebuild, causing repeated fetches
@override
Widget build(BuildContext context) {
  return FutureBuilder<List<Item>>(
    future: fetchItems(), // called every build!
    builder: (ctx, snapshot) => /* ... */,
  );
}

// GOOD — Future created once in initState
late final Future<List<Item>> _itemsFuture;

@override
void initState() {
  super.initState();
  _itemsFuture = fetchItems();
}

@override
Widget build(BuildContext context) {
  return FutureBuilder<List<Item>>(
    future: _itemsFuture, // stable reference
    builder: (ctx, snapshot) => /* ... */,
  );
}
```

## Phase 1: Performance-First Checklist (P0 Gates)

Resolution of P0 items is mandatory before structural refactoring begins. Each item must pass or receive a documented waiver.

### P0.1: Constricting Rebuild Scope and Frequency

`build()` is called frequently. When parent widgets use broad state listeners (`setState`, unscoped `watch`), the entire descendant subtree is recursively checked.

- **Detection:** Use DevTools Rebuild Stats to identify excessively high build counts on root widgets or primary structural containers.
- **Remediation:** Separate static layouts from dynamic data and push state listeners as far down the widget tree as possible using `context.select`, `Consumer`, or `ValueListenableBuilder`.

### P0.2: Enforcing Strict Immutability and Subtree Caching

`const` widgets short-circuit Flutter's diff algorithm — the framework skips diffing and rebuilding the entire subtree.

- **Detection:** Static analysis (lints) will flag missing `const` modifiers.
- **Remediation:** Use `const` constructors aggressively. For subtrees that cannot be `const` but are structurally stable, cache the widget instance in the state of a `StatefulWidget`.

```dart
class _ComponentShellState extends State<ComponentShell> {
  // Cache expensive, stable UI elements to avoid rebuilding
  late final Widget _cachedChrome = const StaticChrome();

  @override
  Widget build(BuildContext context) {
    return _cachedChrome;
  }
}
```

### P0.3: Eradicating Expensive Computations from the Build Phase

Synchronous heavy work (sorting, filtering, JSON decoding, string manipulation) inside `build()` directly causes jank.

- **Detection:** DevTools CPU profiler will highlight Dart execution spikes within the build phase.
- **Remediation:** Move computations to controllers/ViewModels, cache derived state, or offload massive datasets to a background isolate using the `compute()` function.

### P0.4: Optimizing High-Frequency Interactions

Pointer movements and drag events fire at the screen's refresh rate. `setState` during these causes rebuild storms.

- **Detection:** Severe frame drops during gestures; red bars in DevTools frame chart.
- **Remediation:** Isolate ephemeral state (like cursor coordinates) into a `ValueNotifier` and consume it via `ValueListenableBuilder` only where needed. Use `childWhenDragging` for `Draggable` widgets to avoid layout recalculations.

### P0.5: List Rendering Efficiency and Scroll Physics

`shrinkWrap: true` forces layout of every child simultaneously, destroying lazy rendering.

- **Detection:** High memory consumption and massive layout times in DevTools timeline.
- **Remediation:** Eradicate `shrinkWrap`. Use `Expanded` or `Flexible` for constraints. Mandate the use of `itemExtent` or `prototypeItem` for uniform child sizes to allow mathematical layout calculation.

### P0.6: Eliminating Intrinsic Sizing and Speculative Layout Passes

`IntrinsicWidth`/`IntrinsicHeight` perform speculative layout on all children → O(N²) in deep trees.

- **Detection:** Static searches for `Intrinsic` widgets; repeated "Layout" segments in DevTools timeline.
- **Remediation:** Replace with constraint-based alternatives like `Align`, `SizedBox`, `Flex`, or `ConstrainedBox`.

### P0.7: Reducing Paint Costs and Optimizing Render Targets

`saveLayer()` (triggered by `Opacity`, `Clip.antiAliasWithSaveLayer`, heavy shadows) allocates offscreen buffers and stalls rendering.

- **Detection:** GPU thread bottlenecks in DevTools.
- **Remediation:** Apply opacity directly to colors (`color.withValues(alpha: 0.8)`) instead of using `Opacity` widgets. Use `RepaintBoundary` strictly to isolate heavily animated components from static subtrees.

### P0.8: Image Memory Discipline and Decoding Strategies

Images decode uncompressed. Without `cacheWidth`/`cacheHeight`, large images consume massive RAM.

- **Detection:** Heap spikes in DevTools Memory view; "Highlight oversized images" in Inspector.
- **Remediation:** Use `cacheWidth` and `cacheHeight` on `Image` providers to decode assets at display resolution. Use `precacheImage` to avoid decoding jank during transitions.

### P0.9: Dynamic Interaction Styling via WidgetState

Use `WidgetState` (formerly `MaterialState`) for interactive styling — never hardcode active/hover/focus colors.

- **Detection:** Visual inconsistencies (e.g., tabs not changing color); hardcoded `TextStyle(color:...)`.
- **Remediation:** Use `WidgetStateProperty` or theme-based inheritance (e.g., `labelColor` in `TabBarTheme`).

### P0.10: Memory Stability and Controller Disposal

Every controller must be `dispose()`d. Missing disposal = memory leaks.

- **Detection:** DevTools Memory view revealing orphaned controller objects.
- **Remediation:** Strictly implement `dispose()`. Use `RestorationMixin` and `RestorableProperty` classes to preserve user context (scroll offsets, active indices) across app restarts/reclamations.

### P0.11: Accessibility Semantics and Render Overhead

Accessibility must not compromise frame budgets.

- **Detection:** Use `showSemanticsDebugger: true` to audit the semantic map.
- **Remediation:** Use the `SemanticsRole` API. Use `excludeSemantics: true` for purely decorative elements to keep the accessibility tree lean.

## State Management Decision Rules and Fine Rebuild Policy

State management **is** performance in Flutter — it determines rebuild frequency and radius. Rule: choose the lightest primitive, keep ownership close to the consumer, make rebuild triggers explicit and narrow.

### State Primitive Selection Rules

| State Type                        | Correct Primitive                                                             | Use When                                                                                      | Avoid When                                                                                                |
| :-------------------------------- | :---------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------- |
| Ephemeral widget-only UI state    | `setState` in a small `StatefulWidget`                                        | Toggle, hover, local expansion, transient field focus, local tab index                        | The state is needed outside the widget subtree or updates at a very high frequency across a large subtree |
| High-frequency local values       | `ValueNotifier<T>` + `ValueListenableBuilder`                                 | Drag positions, hover targets, split ratios, selection index, panel focus, transient counters | Complex cross-feature orchestration or multi-field transactional state                                    |
| Screen/feature presentation state | `ChangeNotifier`, `Notifier`, `Cubit`, or equivalent                          | A feature owns multiple related view models and consumers need selectors                      | The entire screen is forced to listen to one large mutable object                                         |
| Async request state               | `AsyncValue`, sealed view state, `StreamBuilder`, `FutureBuilder` at the edge | Loading/error/data boundaries close to the consuming subtree                                  | The future/stream is recreated in `build()` or wrapped around the entire screen                           |
| App-wide session/config state     | Top-level provider/notifier                                                   | Auth session, locale, theme, feature flags, app configuration                                 | Feature-specific state is being hoisted globally for convenience                                          |

### Fine-Grained Rebuild Rules

1. A widget may only subscribe to the exact field, derived scalar, or small immutable view model that it renders.
2. Parent containers such as `Scaffold`, `CustomScrollView`, `PageView`, `Navigator` hosts, feature shells, and multi-pane layout roots must remain as stateless as possible during ordinary interaction.
3. Selector outputs must be stable and cheap to compare. Returning a new `List`, `Map`, or wrapper object on every read defeats selector-based optimization unless equality is intentionally implemented.
4. Expensive derived collections must be computed before the widget layer or cached behind invalidation rules. A selector that allocates on every rebuild is still a hot path.
5. High-frequency state must terminate at a leaf boundary. Pointer movement, drag progress, scroll-linked decoration, and resize feedback must not rebuild layout chrome, headers, or unrelated siblings.
6. `child` parameters on `AnimatedBuilder`, `ValueListenableBuilder`, `Consumer`, `Selector`, and similar builder APIs are mandatory whenever part of the subtree is stable.
7. `setState` is acceptable only when the rebuild radius is intentionally tiny and verified. It is not a substitute for a missing boundary widget.
8. `Key` usage does not reduce rebuild cost. Keys preserve identity; they do not make a subtree cheap.
9. Builder callbacks must be pure. No network calls, controller mutation, provider writes, or post-frame scheduling loops may originate from `build()` or builder closures unless narrowly justified.
10. Rebuild optimization must be verified with tooling, not assumed. A refactor is incomplete until Rebuild Stats or debug rebuild logs confirm the intended boundary.

### Common State Management Bad Practices

- Using `setState` on a screen root, scaffold shell, or feature root for hover, drag, filter chips, or tab selection.
- Watching an entire view model when the widget needs one field, one boolean, or one count.
- Returning freshly allocated collections from `select`/`selector` logic on every read.
- Recreating `Future`, `Stream`, controller, notifier, or provider instances inside `build()`.
- Wrapping a large subtree with one `Consumer`, `BlocBuilder`, or `AnimatedBuilder` instead of extracting narrow leaf widgets.
- Calling `notifyListeners()` repeatedly inside loops or multi-step mutations instead of batching updates.
- Mutating a list/map in place and then expecting selector equality to protect rebuilds.
- Storing ephemeral interaction state globally when it should remain local to a leaf widget.
- Using `GlobalKey` or imperative state reach-in as a substitute for explicit data flow.
- Triggering side effects from `FutureBuilder` / `StreamBuilder` builders or from `ref.watch(...)`-driven build paths.

### Framework-Specific Listening Rules

- **Provider:** Prefer `context.select`, `Selector`, and narrow `Consumer` placement. Avoid broad `context.watch<T>()` at structural widget boundaries.
- **Riverpod:** Prefer `ref.watch(provider.select(...))` or smaller providers per concern. Avoid large umbrella providers that expose a whole screen model to many widgets.
- **Bloc/Cubit:** Prefer `BlocSelector` or `buildWhen` for selective rebuilds. Avoid a single `BlocBuilder` rebuilding an entire page for one field change.
- **Listenable/ChangeNotifier:** Prefer leaf `ListenableBuilder` / `ValueListenableBuilder` usage with stable `child` subtrees. Avoid bridging every listener back into `setState`.
- **FutureBuilder/StreamBuilder:** Instantiate futures and streams outside `build()` and place builders at the smallest async boundary that actually depends on the result.

## Phase 2: Modularity and Architectural Readability (M Gates)

Modularity is implemented **after** P0 performance gates pass.

### M.1: Responsibility Separation via Widget Boundaries

Split monolithic `build()` methods along _change boundaries_. Extract static UI into `const`-capable `StatelessWidget` classes.

### M.2: Eradicate "Mega-Controllers"

Decompose monolithic state managers by domain (e.g., separate `LayoutState`, `DataState`, `InteractionState`).

### M.3: Replace UI Helper Methods with Widgets

UI helper functions (`Widget _buildHeader()`) bypass the widget lifecycle and prevent `const` caching. Convert to `StatelessWidget` classes.

### M.4: Enforce Strict Static Analysis Guardrails

Enforce `flutter_lints` with `prefer_const_constructors` and `prefer_const_literals_to_create_immutables` enabled.

## Phase 3: Documentation and Alignment (D Gates)

### D.1: Performance Strategy Documentation

Detail the target frame rate constraints, `watchPerformance` test execution, and mandatory profiling protocols in the target's canonical engineering documentation.

### D.2: Theming and Styling Protocols

Codify the theming paradigm. Explicitly forbid hardcoding raw colors; mandate the use of `ThemeData` inheritance and `WidgetState` mechanisms.

### D.3: Visual Regression Testing

Generate pixel-perfect golden tests (`matchesGoldenFile`) for various component states (empty, populated, interactive permutations) to prevent regressive visual changes.

## Audit Execution Matrix Template

Apply this table **per source file** in the target directory or audited scope.

### Audit Scope Inclusion/Exclusion Rules

- Include by default: `lib/**`, `test/**`, `integration_test/**`, feature/package docs, and configuration used by runtime behavior.
- Exclude by default: generated artifacts and dependency/vendor trees listed in the project-agnostic defaults above.
- If generated code is audited, mark it `INCLUDED_BY_EXCEPTION` with rationale.
- The report must include a short `Scope Exclusions` list containing excluded globs.

| File              | Role         | Hot Path? | Rebuild Hotspots | State Listening Issues | Build-Time Heavy Work | Layout Risks | Paint Risks | Image/Asset Risks | Theming Risks | Priority | Effort | Proposed Fix | Tests to Add | Status |
| :---------------- | :----------- | :-------- | :--------------- | :--------------------- | :-------------------- | :----------- | :---------- | :---------------- | :------------ | :------- | :----- | :----------- | :----------- | :----- |
| `example.dart`    | Root widget  |           |                  |                        |                       |              |             |                   |               |          |        |              |              |        |
| `header.dart`     | Tab header   |           |                  |                        |                       |              |             |                   |               |          |        |              |              |        |
| `drag_layer.dart` | Drag overlay |           |                  |                        |                       |              |             |                   |               |          |        |              |              |        |
| ...               | ...          |           |                  |                        |                       |              |             |                   |               |          |        |              |              |        |

### Scoring Matrix Rules

- **P0:** Causes jank/rebuild storms, breaks interactive styling state, heavy build/layout/paint costs, oversized images, memory leaks. Immediate resolution.
- **P1:** Maintainability issues that increase rebuild scope risk.
- **P2:** Cleanup, naming, docs, minor refactors.

### Deterministic Scoring Formula (0-10)

1. Start with `10.0`.
2. Apply deductions:

- `P0 FAIL`: `-1.00` each
- `P0 PARTIAL`: `-0.50` each
- `P1 FAIL`: `-0.35` each
- `P1 PARTIAL`: `-0.15` each
- `P2 FAIL`: `-0.10` each
- `UNKNOWN` or `NOT MEASURED` on required checklist items: `-0.05` each (cap `-1.00` total)

1. Apply hard caps:

- Any unwaived `P0 FAIL` caps final score at `6.9`.
- Any `UNSTABLE` performance scenario (without waiver) caps final score at `7.4`.

1. Clamp final score to `[0.0, 10.0]`.
2. Readiness band:

- `9.0-10.0`: Ready
- `7.5-8.9`: Conditionally Ready
- `0.0-7.4`: Not Ready

**Effort scale:** XS (< half-day), S (0.5–1 day), M (1–3 days), L (3–5 days), XL (1–2 weeks).

## Mandatory AI Audit Report Format

Any AI using this playbook must emit a deterministic audit report. The output is not complete unless it includes all required sections below in the specified order. This requirement exists so reports can be reviewed by humans, diffed over time, and partially parsed by automation.

### Output Rules

1. Report findings before narrative summary.
2. Every rule, measurement, and checklist item must be assigned one status: `PASS`, `FAIL`, `PARTIAL`, `UNKNOWN`, `NOT MEASURED`, or `NOT APPLICABLE`.
3. If a metric was not collected, the AI must say `NOT MEASURED` and explain why. It must not invent numbers.
4. Every `FAIL`, `PARTIAL`, or `UNKNOWN` item must include evidence, impact, and a concrete remediation.
5. Findings must be ordered by severity: `P0`, then `P1`, then `P2`.
6. File references must be precise whenever the evidence comes from source code or configuration.
7. The report must separate observed facts from assumptions or inferred risks.
8. The report must end with a machine-readable summary block in JSON.
9. Every waiver must include: `id`, `rule`, `severity`, `owner`, `reason`, `approver`, `issued_on`, `expires_on`.
10. The report must include a numeric final score and readiness band derived from the deterministic scoring formula.
11. The report must include the artifact output path and file names used for persistence.

### Required Report Sections

1. `Audit Target`
2. `Environment`
3. `Measured Budgets`
4. `Findings`
5. `Checklist Summary`
6. `Per-File Audit Matrix`
7. `Remediation Plan`
8. `Evidence Gaps and Assumptions`
9. `Scorecard`
10. `Report Artifact`
11. `Machine-Readable Summary`

### Required Markdown Template

````md
## Audit Target

- Target: <widget/screen/feature/package>
- Scope boundary: <what was audited>
- Commit/branch: <if known>
- Audit mode: <manual | automated | hybrid>

## Environment

- Flutter version: <version or unknown>
- Device/profile target: <device/emulator/web/desktop>
- Build mode: <profile/debug/release>
- Refresh-rate assumption: <60Hz | 90Hz | 120Hz | mixed | unknown>
- Frame budget ms: <8.33|11.11|16.6|custom>
- Tooling used: <DevTools, watchPerformance, traceAction, analyze, tests>
- Scope exclusions: <list of excluded globs>

## Measured Budgets

| Metric                     | Status                         | Value   | Threshold                      | Evidence | Notes  |
| :------------------------- | :----------------------------- | :------ | :----------------------------- | :------- | :----- |
| Avg build time             | PASS/FAIL/NOT MEASURED         | <value> | <= frame_budget_ms             | <source> | <note> |
| P90 build time             | PASS/FAIL/NOT MEASURED         | <value> | <= frame_budget_ms             | <source> | <note> |
| P99 build time             | PASS/FAIL/NOT MEASURED         | <value> | <= frame_budget_ms \* 1.25     | <source> | <note> |
| Worst build time           | PASS/FAIL/NOT MEASURED         | <value> | <= frame_budget_ms \* 1.50     | <source> | <note> |
| Avg raster time            | PASS/FAIL/NOT MEASURED         | <value> | <= frame_budget_ms             | <source> | <note> |
| P90 raster time            | PASS/FAIL/NOT MEASURED         | <value> | <= frame_budget_ms             | <source> | <note> |
| P99 raster time            | PASS/FAIL/NOT MEASURED         | <value> | <= frame_budget_ms \* 1.25     | <source> | <note> |
| Worst raster time          | PASS/FAIL/NOT MEASURED         | <value> | <= frame_budget_ms \* 1.50     | <source> | <note> |
| Janky frame ratio          | PASS/FAIL/NOT MEASURED         | <value> | <= 1.0%                        | <source> | <note> |
| Missed build budget count  | PASS/FAIL/NOT MEASURED         | <value> | <= 0 (critical) / <= 1 (other) | <source> | <note> |
| Missed raster budget count | PASS/FAIL/NOT MEASURED         | <value> | <= 0 (critical) / <= 1 (other) | <source> | <note> |
| Rebuild hotspot count      | PASS/FAIL/NOT MEASURED         | <value> | 0 for static containers        | <source> | <note> |
| Memory stability           | PASS/FAIL/PARTIAL/NOT MEASURED | <value> | <= +5% net heap after 3 cycles | <source> | <note> |

## Findings

| Severity | Status | Rule   | Location                   | Evidence | Impact                       | Required Fix           |
| :------- | :----- | :----- | :------------------------- | :------- | :--------------------------- | :--------------------- |
| P0       | FAIL   | <rule> | <file/section/tool result> | <fact>   | <user-visible/system impact> | <concrete remediation> |

## Checklist Summary

| Section         | Pass | Fail | Partial | Unknown | Not Measured | Not Applicable |
| :-------------- | ---: | ---: | ------: | ------: | -----------: | -------------: |
| P0 Rendering    |    0 |    0 |       0 |       0 |            0 |              0 |
| P0 State        |    0 |    0 |       0 |       0 |            0 |              0 |
| M Architecture  |    0 |    0 |       0 |       0 |            0 |              0 |
| Q Quality       |    0 |    0 |       0 |       0 |            0 |              0 |
| T Testing       |    0 |    0 |       0 |       0 |            0 |              0 |
| D Documentation |    0 |    0 |       0 |       0 |            0 |              0 |

## Per-File Audit Matrix

| File   | Hot Path | Primary Risks                       | Priority | Recommended Action | Status   |
| :----- | :------- | :---------------------------------- | :------- | :----------------- | :------- |
| <path> | Yes/No   | <rebuild/layout/paint/state/memory> | P0/P1/P2 | <action>           | <status> |

## Remediation Plan

1. <highest-priority fix>
2. <next fix>
3. <validation/test addition>

## Evidence Gaps and Assumptions

- Missing evidence: <what was not measured>
- Assumptions: <what was assumed and why>
- Waivers: <accepted exceptions, if any>

## Scorecard

- Final score (0-10): <value>
- Readiness band: <Ready | Conditionally Ready | Not Ready>
- Hard caps applied: <none | P0_FAIL_CAP | UNSTABLE_CAP>
- Deduction breakdown: <P0 fail, P0 partial, P1 fail, P1 partial, P2 fail, unknown/not_measured>

## Report Artifact

- Markdown path: `.intelag/.agent/reports/flutter/<YYYY-MM-DD_HH-mm-ss>_<target-slug>_audit.md`
- JSON path: `.intelag/.agent/reports/flutter/<YYYY-MM-DD_HH-mm-ss>_<target-slug>_audit.json`
- Persisted: <yes|no>

## Machine-Readable Summary

```json
{
  "target": "<target>",
  "audit_mode": "manual|automated|hybrid",
  "build_mode": "profile|debug|release|unknown",
  "refresh_rate": "60Hz|90Hz|120Hz|mixed|unknown",
  "frame_budget_ms": null,
  "overall_status": "pass|fail|partial|unknown",
  "final_score": 0.0,
  "readiness_band": "ready|conditionally_ready|not_ready",
  "hard_caps": [],
  "severity_counts": { "P0": 0, "P1": 0, "P2": 0 },
  "checklist_counts": {
    "pass": 0,
    "fail": 0,
    "partial": 0,
    "unknown": 0,
    "not_measured": 0,
    "not_applicable": 0
  },
  "measured_budgets": {
    "avg_build_ms": null,
    "p90_build_ms": null,
    "p99_build_ms": null,
    "worst_build_ms": null,
    "avg_raster_ms": null,
    "p90_raster_ms": null,
    "p99_raster_ms": null,
    "worst_raster_ms": null,
    "janky_frame_ratio_pct": null,
    "missed_frame_build_budget_count": null,
    "missed_frame_raster_budget_count": null,
    "memory_stable": null,
    "run_count": null,
    "run_variance_pct": null
  },
  "top_findings": [
    {
      "severity": "P0",
      "rule": "<rule>",
      "location": "<location>",
      "status": "FAIL",
      "impact": "<impact>",
      "fix": "<fix>"
    }
  ],
  "scope_exclusions": ["<glob>"],
  "report_artifacts": {
    "markdown": "<path>",
    "json": "<path>",
    "persisted": false
  },
  "evidence_gaps": ["<gap>"],
  "waivers": [
    {
      "id": "<waiver-id>",
      "rule": "<rule>",
      "severity": "P0|P1|P2",
      "owner": "<name-or-team>",
      "reason": "<business/technical rationale>",
      "approver": "<name>",
      "issued_on": "YYYY-MM-DD",
      "expires_on": "YYYY-MM-DD"
    }
  ]
}
```
````

### AI Auditor Discipline Rules

- Do not output a prose-only review.
- Do not claim performance compliance without measured evidence or an explicit waiver.
- Do not collapse multiple distinct findings into one vague paragraph.
- Do not hide uncertainty. Unknowns must remain visible.
- Do not recommend broad refactors before the P0 issues are identified and prioritized.

## CI Gates and Automated Checks

### Static Gates (must pass on every PR)

- `flutter analyze` with `flutter_lints` enabled.
- Enforced rules: `prefer_const_constructors`, `prefer_const_literals_to_create_immutables`.

### Widget + Golden Gates

- Run widget tests including goldens; update goldens only via explicit workflow (`flutter test --update-goldens`).
- Golden tests use `matchesGoldenFile` with tightly controlled tolerance.

### Performance Budget Gates

- Integration perf test(s) using `watchPerformance()` and/or `traceAction()` around component interactions.
- Budgets validate avg / p90 / p99 / worst for build/raster durations.
- CI generates JSON perf artifacts attached to build logs.
- Reproducibility requirement: 2 warm-up runs (discarded) + 5 measured runs per scenario.
- Failure rule: fail if median exceeds threshold or if at least 3 of 5 runs exceed threshold.
- Variance rule: if key metric variance exceeds 10%, mark `UNSTABLE` and block unconditional `PASS` unless waived.
- Note: perf variability on shared CI runners is real; use baselines, stable runners, pinned references, and fixed fixtures.

### Memory Regression Gates (optional but valuable)

- Not typically hard-fail in CI, but enforce:
  - No obvious leaks by repeating interactions and ensuring disposal correctness.
  - Manual review checklist using DevTools memory diff snapshots / trace instances.

## Refactor Workflow Template

1. **Baseline Capture (S)** — Record DevTools Performance + rebuild stats; add initial `watchPerformance()` test.
2. **Fix P0 Rebuild Scope (M–L)** — Convert broad watches to selects; push listeners down; split root widget.
3. **Fix P0 Build-Time Work (M)** — Move list transforms/sorts/derivations out of `build()`; cache derived values.
4. **Fix P0 Layout/Paint Hotspots (M–L)** — Remove `shrinkWrap`; remove intrinsic sizing; reduce `saveLayer` triggers; apply `RepaintBoundary` only where justified.
5. **Fix Interactive Styling Bug (S)** — Remove hardcoded colors; use theme-based or `TabController`-driven styling.
6. **Add/Expand Goldens + Docs (S–M)** — Goldens for component states; canonical documentation updates.

## Conclusion

Performance gates (P0) must pass before modularity (M) or documentation (D) work begins. All claims require measured evidence in profile mode. Use the Anti-Pattern Detection Reference for systematic code review and the Audit Report Format for structured output.

## Common False Positives — Do NOT Flag These

Not every pattern match is a true finding. The AI must apply judgment for these cases:

| Pattern                                | When It Is Acceptable                                                                                                                                                         |
| :------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `setState` in a small `StatefulWidget` | Acceptable when the widget is a leaf with a tiny rebuild radius (e.g., a toggle button, local expansion tile). Only flag when used in screens/scaffolds/shells.               |
| `Opacity` widget                       | Acceptable for animating opacity via `AnimatedOpacity` or `FadeTransition` — these cannot be replaced with color opacity. Only flag static `Opacity` wrapping large subtrees. |
| `shrinkWrap: true`                     | Acceptable in `ListView` with a known small, fixed item count (e.g., ≤5 items in a dialog/bottom sheet). Flag in scrollable lists with dynamic/large data.                    |
| Missing `const`                        | Do not flag in generated code (`*.g.dart`, `*.freezed.dart`). Only flag in hand-written source.                                                                               |
| `Provider.of` without `listen: false`  | Acceptable if the widget genuinely needs to rebuild when the value changes AND is a leaf widget.                                                                              |
| UI helper methods                      | Acceptable for private methods called exactly once in short widgets (<50 lines). Flag when methods are called from multiple places or in large widgets.                       |
| `IntrinsicHeight`/`IntrinsicWidth`     | Acceptable in cold paths (settings screens, dialogs shown once). Flag only on hot paths (scrollable items, frequently rebuilt UI).                                            |

---

## Performance Audit Checklist

Use this checklist during every component audit. It merges the P0/M/D gates from this playbook with the Golden Rules and Hard Rules from the optimization guidelines, and incorporates all verification patterns from the deep-research audit reports. Items are grouped by concern.

### P0 — Rendering Performance

- [ ] **Profiling Environment:** Audit conducted exclusively in **Profile Mode** (`flutter run --profile`) on a physical device. Debug-mode results are invalid.
- [ ] **Frame Budget:** `buildDuration` and `rasterDuration` percentile metrics (avg, p90, p99, worst) are < 8.33ms (120Hz) or < 16.6ms (60Hz).
- [ ] **Reproducibility:** Each perf scenario includes 2 warm-up runs and 5 measured runs; median and variance are reported.
- [ ] **Janky Frame Count:** Count of janky frames and their causes (UI vs raster) are recorded and trending downward.
- [ ] **Rebuild Scope (Golden Rule 1, 2, 14):** Each widget watches only the smallest possible state slice. No high-level parent listens to broad state unless absolutely necessary. Rebuild boundaries are optimized before micro-optimizing code.
- [ ] **Granular Selectors (Golden Rule 6):** `context.select` (Provider), `ref.watch(provider.select(...))` (Riverpod), `Consumer` with narrow scope, or `ValueListenableBuilder` is used everywhere instead of broad `watch`/`listen`.
- [ ] **Rebuild Budget Verification:** DevTools Rebuild Stats / widget build counts confirm key container widgets have low build counts during interaction. `debugPrintRebuildDirtyWidgets` used for short diagnostic sessions.
- [ ] **Provider Scoping (Golden Rule 7):** Providers live as close as possible to the widget subtree that actually needs them—not globally unless required.
- [ ] **Immutability & const (Golden Rule 8, 15):** All possible widgets are marked `const`. Immutable models are used for state. (`flutter analyze` must pass with `prefer_const_constructors`.) Stable subtrees that cannot be `const` are cached in `State` fields.
- [ ] **Build Phase Purity (Golden Rule 4, 16):** `build()` is free of `.sort()`, `.map().toList()`, JSON decoding, filtering, fetching, or any heavy computation. No unnecessary object allocations inside `build()`. Derived state is computed/cached in controllers and read via selectors.
- [ ] **Animation Efficiency:** `AnimatedBuilder` uses the `child` parameter to exclude stable subtrees from rebuilds. No broad `setState` inside animation callbacks.
- [ ] **High-Frequency Events:** Drag/scroll/pointer events are isolated via `ValueNotifier` and consumed by `ValueListenableBuilder` with a `child` parameter for stable subtrees. `childWhenDragging` is set on `Draggable` widgets.
- [ ] **Lazy List Rendering (Golden Rule 17):** `shrinkWrap: true` is removed from all `ListView`/`GridView`. No nested `ListView` inside `Column` with unbounded constraints. Lists, grids, and composite scrolling layouts use `Expanded`/`Flexible` for constraints.
- [ ] **Layout Predictability:** `itemExtent` or `prototypeItem` is provided for lists with uniform child sizes.
- [ ] **Intrinsic Elimination:** All `IntrinsicHeight`/`IntrinsicWidth` widgets are replaced with `Align`, `SizedBox`, `Flex`, or `ConstrainedBox`.
- [ ] **Paint Optimization:** `Opacity` widget wrapping large subtrees is replaced with `color.withValues(alpha: ...)` applied locally. `Clip.antiAlias` is preferred over `Clip.antiAliasWithSaveLayer`. Implicit `saveLayer` triggers from heavy shadows/blur are minimized.
- [ ] **RepaintBoundary:** Heavily animated or frequently updating subcomponents are wrapped in `RepaintBoundary` to prevent repainting static neighbors. Verified via Inspector "Highlight Repaints".
- [ ] **Image Memory:** All `Image` providers use `cacheWidth`/`cacheHeight` matching display resolution × `devicePixelRatio`. `precacheImage` is used in `didChangeDependencies()` for transition-critical assets. `ImageCache` is tuned via `maximumSize`/`maximumSizeBytes` if needed.
- [ ] **Oversized Image Detection:** `debugInvertOversizedImages = true` and Inspector "Highlight oversized images" have been used to verify no oversized decodes remain.
- [ ] **Interactive Styling:** All interactive states (hover, focus, active, disabled) use `WidgetStateProperty` or `ThemeData`-based inheritance—no hardcoded `TextStyle(color:...)`. Tab styling uses `TabBar.labelColor`/`unselectedLabelColor` or `TabController`-driven `AnimatedBuilder`.
- [ ] **Resource Lifecycle:** Every `AnimationController`, `ScrollController`, `TabController`, `StreamSubscription`, and `TextEditingController` is explicitly `dispose()`d.
- [ ] **Memory Stability:** Repeated interactions (tab switching, open/close panels) do not cause monotonically increasing heap or native memory. Verified via DevTools Memory view diff snapshots.
- [ ] **State Restoration:** `RestorationMixin` and `RestorableProperty` are used to preserve scroll offsets, active indices, and user context across app reclamation.
- [ ] **Accessibility:** `SemanticsRole` is used for custom widgets. `excludeSemantics: true` is applied to purely decorative elements. Semantics tree validated via `showSemanticsDebugger`.
- [ ] **Automated CI Gates:** `watchPerformance()` integration tests are implemented and enforced. `traceAction()` is available for deeper regression analysis. CI generates JSON perf artifacts attached to build logs.
- [ ] **Scope Exclusions Declared:** Generated/vendor/build artifacts are excluded by default and any exception is explicitly justified.
- [ ] **Shader Compilation Jank:** No visible shader compilation stutter on first route transition. For Skia: `ShaderWarmUp` subclass covers critical shaders. For Impeller: confirm Impeller is enabled (`--enable-impeller`) which eliminates runtime shader compilation by design.
- [ ] **Deferred Loading:** Large features not needed at startup use deferred imports (`import 'package:x/feature.dart' deferred as feature`) to reduce initial load time and memory footprint.
- [ ] **Route/Navigation Rebuild Avoidance:** Navigation events do not trigger rebuilds of screens remaining in the navigation stack. Tab-based navigation uses `AutomaticKeepAliveClientMixin` or equivalent for state preservation where needed.
- [ ] **Isolate Offloading:** CPU-intensive work (JSON parsing of large payloads, image processing, complex calculations >16ms) uses `Isolate.run()` or `compute()` to avoid blocking the UI thread.

### P0 — State Management & Providers

- [ ] **Smallest Possible Watch (Golden Rule 1):** If a widget needs only `user.name`, it does NOT rebuild on `user.email`, `user.role`, or `user.permissions`.
- [ ] **No Broad Parent Listeners (Golden Rule 2):** The higher the listener in the tree, the more expensive the blast radius. Parent widgets do not `watch` state that only a child needs.
- [ ] **No Broad Static Patterns:** Codebase is free of `context.watch<T>()` in high-level widgets, `Consumer<T>` wrapping huge subtrees, `Provider.of<T>(context)` with default listening, and `ref.watch(provider)` returning big state objects in large widgets.
- [ ] **Selector Stability:** Selectors do not allocate fresh `List`, `Map`, `Set`, or ad-hoc wrapper objects on every read unless stable equality semantics are implemented and intentional.
- [ ] **Local State Placement:** Ephemeral UI state such as hover, pressed, expanded, highlighted row, transient selection, and drag-preview state is kept local to the smallest practical widget instead of being hoisted into screen-level state.
- [ ] **High-Frequency Isolation:** Pointer move, drag, resize, scroll-linked, or animation-driven values terminate in leaf builders and do not rebuild shells, headers, scaffold-level layouts, or unrelated siblings.
- [ ] **No Futures/Streams in Build:** `Future`, `Stream`, and subscription-producing code is created in lifecycle methods or controllers, not inside `build()`.
- [ ] **Pure Builders:** `build()`, `builder`, `itemBuilder`, `Consumer`, `BlocBuilder`, `FutureBuilder`, and `StreamBuilder` closures have no side effects, no provider writes, and no repeated post-frame scheduling.
- [ ] **ChangeNotifier Discipline:** `addListener` callbacks do not trigger broad `setState` without boundary widgets. Prefer `ValueListenableBuilder` or extracted leaf widgets.
- [ ] **Batch Notifications:** Controllers/notifiers batch related state changes and avoid repeated `notifyListeners()` / emit cycles for one user action.
- [ ] **State Split by Responsibility (Golden Rule 3, 19):** No mega-controller manages unrelated concerns (screen state + filters + pagination + selection + loading + editing). Each notifier/controller has a single clear purpose.
- [ ] **Derived Over Duplicated State (Golden Rule 9):** No two values are manually kept in sync if one can be computed from the other. Expensive derived values are cached in controllers with dirty-flag invalidation.
- [ ] **Async State Isolation (Golden Rule 18):** Loading/error/data state does not unnecessarily rebuild unrelated UI. Async operations are scoped to their consuming subtree.
- [ ] **Framework-Specific Selectivity:** Provider uses `select`/`Selector`, Riverpod uses `.select(...)` or narrower providers, Bloc/Cubit uses `BlocSelector` or `buildWhen`, and Listenable-based flows use leaf `ListenableBuilder`/`ValueListenableBuilder`.
- [ ] **No Global Mutable State:** Scoped state is used instead of global mutable singletons unless global scope is genuinely required.
- [ ] **Predictable State Flow:** State flows in a predictable, unidirectional way. No implicit shared mutation between widgets.
- [ ] **Immutable State Transitions (Golden Rule 8):** State updates use immutable models with predictable transitions—no in-place mutation of state objects.
- [ ] **No Key-as-Optimization Myth:** Keys are not being used as a performance fix for rebuild problems that should instead be solved by state and widget boundaries.

### M — Architecture & Modularity

- [ ] **Widget Boundaries (Golden Rule 10, M.1):** Massive monolithic `build()` methods are split along _change boundaries_. Each widget has one clear UI responsibility.
- [ ] **No Helper Methods (M.3):** UI helper functions (`Widget _buildHeader()`, `_buildTab()`, `_buildDockItem()`) are converted to dedicated `StatelessWidget` classes. Helper methods lack their own `BuildContext` and bypass element reuse.
- [ ] **No Mega-Controllers (M.2):** Monolithic state managers are decomposed by domain (e.g., `TabsState`, `DragState`, `LayoutState`).
- [ ] **Separation of Concerns:** Presentation, state, domain logic, and data access are in clearly separated layers. Business logic is NOT inside UI widgets (except trivial view-only logic).
- [ ] **Feature-First Structure (Golden Rule 11):** Code is organized by feature/module, not dumped into generic `widgets/`, `utils/`, `helpers/` folders.
- [ ] **Loosely Coupled Modules:** Modules have well-defined boundaries and are easy to test, move, and maintain independently.
- [ ] **Composition Over Inheritance (Golden Rule 13):** Behavior is composed from small pieces, not through deep inheritance trees—unless inheritance is clearly justified.
- [ ] **Earned Abstractions (Golden Rule 12, 20):** Shared widgets solve a real repeated pattern (2+ occurrences). Abstractions are useful and practical—no over-engineering.
- [ ] **Minimal Boilerplate (Golden Rule 5):** Repeated widget structures, state mappings, or interaction patterns appearing 2+ times are extracted into reusable components, utilities, or builders.

### Q — Code Quality & Static Analysis

- [ ] **Static Analysis (M.4):** `flutter analyze` returns zero errors. `flutter_lints` is configured with `prefer_const_constructors` and `prefer_const_literals_to_create_immutables` enforced.
- [ ] **No Ad-hoc Styling:** No `ButtonStyle`/`styleFrom`, no inline `TextStyle`, no inline hardcoded colors/radius/spacing in feature code. All styling flows through theme tokens or UI kit components.
- [ ] **No Duplicate Layouts:** Repeated layouts or repeated state-handling code is extracted—not copy-pasted.
- [ ] **Strong Naming:** Classes, files, and variables use clear, descriptive names. Files are organized logically.
- [ ] **No Rebuild-Expanding Patterns:** No pattern or abstraction is introduced that increases rebuild scope compared to the previous state.
- [ ] **Clean Widget Trees:** Widget trees are lightweight—no unnecessary nesting, wrapper widgets, or structural dead weight.
- [ ] **Unused Code Removal:** Dead imports, unused variables, unreachable branches, and commented-out code are removed.
- [ ] **Error Handling at Boundaries:** Network, file I/O, and platform channel calls have proper error handling. Errors do not crash the widget tree — use `ErrorWidget.builder` or `FlutterError.onError` for graceful degradation.

### T — Test Coverage Requirements

- [ ] **Widget Tests:** Each extracted widget has its own widget test verifying behavior.
- [ ] **Controller/Notifier Unit Tests:** State transition logic (tab changes, drag enter/leave, layout updates, derived value invalidation) has unit tests.
- [ ] **Golden Tests:** Pixel-perfect goldens exist for selected/unselected states, empty vs. populated states, and at least one light/dark theme variant.
- [ ] **Integration Perf Tests:** `watchPerformance()` covers scroll, tab-switch, and drag scenarios with defined budget thresholds.
- [ ] **Memory Regression Check:** Repeated interaction scenario manually verified via DevTools Memory view diff snapshots to confirm heap stabilization.
- [ ] **Drag/Drop Correctness:** Integration test simulates drag path and asserts no crashes and correct `DragTarget` acceptance.
- [ ] **Edge Case Tests:** Empty states, error states, rapid state transitions, and boundary conditions (0 items, max items) are covered.

### D — Documentation & Visual Regression

- [ ] **Performance Strategy Documented (D.1):** Canonical target documentation details frame-rate constraints, `watchPerformance` test execution instructions, `traceAction()` for deeper analysis, and the mandate for profile-mode-only profiling.
- [ ] **Profiling How-To Documented:** Canonical target documentation explains: (1) run profile build, (2) open DevTools Performance view, (3) enable "Track widget builds" for rebuild hotspots, (4) use Inspector → Highlight repaints / oversized images.
- [ ] **State Management Conventions Documented:** Canonical target documentation specifies: watch/select only smallest state slice, avoid broad watches in high-level widgets, derived state is computed/cached outside `build()`.
- [ ] **Theming Protocols Documented (D.2):** Documentation explains `ThemeData`/`ColorScheme` defaults, supported overrides via `TabBarThemeData` or `TabBar` props, and explicitly forbids hardcoding colors inside tab label widgets.
- [ ] **Common Pitfalls Documented:** Canonical target documentation lists: `shrinkWrap` in scrolling lists, `IntrinsicWidth`/`IntrinsicHeight` in hot paths, `saveLayer` from `Clip.antiAliasWithSaveLayer`, oversized images missing `cacheWidth`/`cacheHeight`.
- [ ] **Golden Tests (D.3):** Pixel-perfect golden tests (`matchesGoldenFile`) exist for empty, populated, active/inactive, and interactive states. Updated via `flutter test --update-goldens`.
- [ ] **Visual Regression Gate:** Golden file comparisons run in CI with a tightly controlled tolerance threshold.

---

## Works Cited

### Source Quality Policy

- Authoritative sources (Flutter docs, API references, DevTools docs) are primary for thresholds and normative guidance.
- Third-party content may provide supporting patterns, but cannot be the sole basis for hard pass/fail gates.
- Any non-authoritative claim used in findings must be corroborated by tooling evidence or official documentation.

- Flutter performance profiling - [https://docs.flutter.dev/perf/ui-performance](https://docs.flutter.dev/perf/ui-performance)
- Performance best practices - [https://docs.flutter.dev/perf/best-practices](https://docs.flutter.dev/perf/best-practices)
- Rename MaterialState to WidgetState - [https://docs.flutter.dev/release/breaking-changes/material-state](https://docs.flutter.dev/release/breaking-changes/material-state)
- Measure performance with an integration test - [https://docs.flutter.dev/cookbook/testing/integration/profiling](https://docs.flutter.dev/cookbook/testing/integration/profiling)
- DevTools release notes - [https://docs.flutter.dev/tools/devtools/release-notes/](https://docs.flutter.dev/tools/devtools/release-notes/)
- Supplemental: Practical Accessibility in Flutter (and Code You'll Actually Use) - DCM.dev
