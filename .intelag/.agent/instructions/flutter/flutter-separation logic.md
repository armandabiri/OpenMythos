# Flutter Architectural Guidelines: Controller-Delegate-View Pattern

> **Scope**: All Intelag Flutter modules.
> **Goal**: Pure functional UI — every widget is a stateless function of its payload, with zero knowledge of controllers, service locators, or widget-tree ancestry.
> **Dependency policy**: Use Flutter framework primitives first. **No third-party state management packages by default** — this includes `provider`, `bloc`, `flutter_bloc`, `get`, `mobx`, or any equivalent. State management is handled through the Controller-Delegate-View pattern using Flutter-native tools (`ValueNotifier`, `ValueListenableBuilder`, `InheritedWidget`, `StatefulWidget`). **Exception**: `riverpod` / `flutter_riverpod` may be used under the conditions defined in section 4.1.

---

## 1. Why this pattern exists

Third-party state management libraries (`provider`, `riverpod`, `bloc`) introduce implicit dependencies on the widget tree. A widget that calls `ref.watch(myProvider)` or `context.read<MyBloc>()` cannot be rendered outside the specific ancestor tree that supplies that dependency. This creates four recurring problems:

1. **Testing friction.** Every widget test must reconstruct a provider/riverpod scope or bloc provider wrapper, even for a simple UI assertion.
2. **Layout lock-in.** Moving a widget to a different part of the tree (e.g., from a sidebar to a dialog) breaks it if the required provider isn't an ancestor in the new location.
3. **Hidden coupling.** The import of `flutter_riverpod` or `provider` in a builder file signals that the file reaches outside its own inputs — making code review harder and refactors riskier.
4. **Dependency bloat.** Third-party packages introduce versioning constraints, upgrade churn, and API surface that the team must track — for functionality Flutter already provides natively.

**Flutter's own primitives are sufficient.** `ValueNotifier` + `ValueListenableBuilder` cover reactive rebuilds. `InheritedWidget` covers scoped dependency injection when explicit payload passing is impractical (e.g., deep trees). `StatefulWidget` covers local ephemeral state. There is no state management problem in our codebase that requires a third-party solution.

This standard eliminates third-party state management entirely and enforces a single rule: **the view receives everything it needs through a typed, immutable data object — nothing else.**

---

## 2. Core architecture

Every module follows four layers, in strict dependency order:

| Layer          | Role                                                                                | Knows about                            |
| :------------- | :---------------------------------------------------------------------------------- | :------------------------------------- |
| **Controller** | Manages business logic, holds mutable state, binds callbacks to specific entity IDs | Domain models, repositories            |
| **Payload**    | Immutable snapshot of state + pre-bound callbacks                                   | Nothing (it *is* the contract)         |
| **Builder**    | Pure function: `(BuildContext, Payload) → Widget`                                   | Only the payload type                  |
| **View**       | Composes builders into a final layout                                               | Builder signatures, layout constraints |

Dependencies flow **downward only**. The view never imports the controller. The builder never resolves a service.

### Flutter-native state propagation

When the controller's state changes, use Flutter's built-in reactivity to rebuild the view and produce fresh payloads:

```dart
/// The controller exposes a ValueNotifier for rebuild signalling.
class PanelController {
  final revision = ValueNotifier<int>(0);
  // ... state and methods ...
  void _notify() => revision.value++;
}

/// The view listens with ValueListenableBuilder — no packages needed.
class PanelView extends StatelessWidget {
  final PanelController controller;
  const PanelView({required this.controller});

  @override
  Widget build(BuildContext context) {
    return ValueListenableBuilder<int>(
      valueListenable: controller.revision,
      builder: (context, _, __) {
        final data = controller.headerPayload('main');
        return panelHeader(context, data);
      },
    );
  }
}
```

For scoped injection across deep widget trees (where passing the controller manually would be impractical), use a custom `InheritedWidget` — not `Provider`, not `Riverpod`:

```dart
class PanelScope extends InheritedWidget {
  final PanelController controller;

  const PanelScope({
    required this.controller,
    required super.child,
  });

  static PanelController of(BuildContext context) {
    return context.dependOnInheritedWidgetOfExactType<PanelScope>()!.controller;
  }

  @override
  bool updateShouldNotify(PanelScope oldWidget) =>
      controller != oldWidget.controller;
}
```

**Important:** `InheritedWidget` is permitted only at the **view composition level** — never inside a builder. Builders must still receive their data exclusively through the payload parameter.

---

## 3. Payload rules

The payload (data object) is the single point of contact between engine and presentation.

### 3.1 Immutable state only

Every field must be `final`. The payload is a snapshot, not a live reference.

```dart
// ✗ Leaks the controller into the presentation layer
class HeaderData {
  final PanelController controller;
}

// ✓ Exposes only the resolved values the view needs
class HeaderData {
  final String title;
  final Color accentColor;
  final bool isCollapsed;

  const HeaderData({
    required this.title,
    required this.accentColor,
    this.isCollapsed = false,
  });
}
```

### 3.2 Pre-bound functional callbacks

Actions are closures that the controller binds before constructing the payload. The builder never knows *which* entity it's acting on or *how* the action is fulfilled.

```dart
class HeaderData {
  final String title;
  final VoidCallback? onClose;         // pre-bound to a specific panelId
  final void Function(String)? onRename; // accepts new name, hides target identity

  const HeaderData({
    required this.title,
    this.onClose,
    this.onRename,
  });
}
```

### 3.3 Resolved booleans over conditional logic

If the builder would need an `if` statement to derive a display flag, resolve it in the controller and pass the result as a `bool`.

```dart
// ✗ Forces the builder to understand render-mode rules
if (controller.mode == RenderMode.compact && controller.childCount > 3) ...

// ✓ Controller resolves the condition; builder gets a simple flag
class PanelData {
  final bool shouldShowOverflowIndicator;
}
```

### 3.4 No reactive primitives in the payload

Passing `ChangeNotifier`, `ValueNotifier`, `Stream`, or `StreamController` into a builder violates snapshot semantics. The controller listens to reactive sources internally and emits a new payload on each change.

---

## 4. Prohibited patterns

### 4.1 Package policy

#### Permanently banned

These packages must **never** appear in any `pubspec.yaml` or import statement:

| Package                                   | Reason                                                  |
| :---------------------------------------- | :------------------------------------------------------ |
| `provider` / `flutter_provider`           | Service-locator pattern couples UI to widget tree       |
| `bloc` / `flutter_bloc` / `hydrated_bloc` | Context-dependent `BlocProvider`/`BlocBuilder` pattern  |
| `get` / `getx`                            | Magic singletons with global mutable state              |
| `mobx` / `flutter_mobx`                   | Observable annotations leak reactive primitives into UI |
| `stacked`                                 | ViewModel locator pattern                               |

If a third-party package (e.g., a UI library) internally depends on one of these, it must be wrapped behind an adapter that exposes only payload-compatible interfaces to our builders.

#### Conditionally permitted: Riverpod

`riverpod` / `flutter_riverpod` / `hooks_riverpod` may be used **at the view composition level only** when it demonstrably improves readability without violating the payload contract. This is not a blanket approval — it is a controlled exception with strict boundaries.

**When Riverpod is permitted:**

1. **Readability gain is clear.** The Riverpod version is meaningfully easier to read, maintain, or onboard new developers into compared to the equivalent `ValueNotifier` + `ValueListenableBuilder` or `InheritedWidget` approach. Examples of qualifying scenarios: coordinating multiple async data sources where nested `ValueListenableBuilder` widgets become deeply indented and hard to follow, or scoped dependency overrides across subtrees that would require multiple custom `InheritedWidget` classes.

2. **Performance impact is negligible.** The Riverpod usage introduces no measurable frame-time regression. A minor overhead from the ref-lookup mechanism is acceptable. A pattern that triggers unnecessary full-subtree rebuilds or adds measurable jank is not.

3. **The payload contract is preserved.** Riverpod providers live in the **view layer** — they produce payloads, they do not replace them. Builders must still receive a typed, immutable data object as their only input. `ref.watch` and `ref.read` must **never** appear inside a builder function.

4. **The builder remains testable without Riverpod.** If you can't widget-test the builder by constructing a payload by hand (no `ProviderScope`, no `ProviderContainer`), the Riverpod usage has leaked into the presentation layer and must be refactored.

**Riverpod usage pattern (correct):**

```dart
// ✓ Riverpod at the view level — produces a payload for the builder
final panelDataProvider = Provider.family<HeaderData, String>((ref, panelId) {
  final controller = ref.watch(panelControllerProvider);
  return controller.headerPayload(panelId);
});

class PanelView extends ConsumerWidget {
  final String panelId;
  const PanelView({required this.panelId});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    // Riverpod resolves the payload HERE — at the view level
    final data = ref.watch(panelDataProvider(panelId));
    // Builder receives only the payload — no ref, no provider
    return panelHeader(context, data);
  }
}
```

**Riverpod usage pattern (violation):**

```dart
// ✗ ref.watch inside a builder — breaks isolation and testability
Widget panelHeader(BuildContext context, WidgetRef ref) {
  final title = ref.watch(titleProvider);       // VIOLATION
  final onClose = ref.read(closeActionProvider); // VIOLATION
  return Row(children: [Text(title)]);
}
```

**Approval process:**

- Riverpod usage requires a code comment explaining why the Flutter-native alternative was insufficient (e.g., `// Riverpod: coordinating 3 async sources; nested ValueListenableBuilder was unreadable`).
- The PR reviewer must verify that the payload contract and builder isolation are intact.
- If the same pattern can be achieved with `ValueNotifier` + `ValueListenableBuilder` at comparable readability, the Flutter-native approach is preferred.

### 4.2 Banned code patterns

| Pattern                                              | Why it's prohibited                                   | Replacement                                             |
| :--------------------------------------------------- | :---------------------------------------------------- | :------------------------------------------------------ |
| `Provider.of<T>(context)`                            | Couples widget to provider tree                       | `data.propertyOrCallback`                               |
| `context.read<T>()` / `context.watch<T>()`           | Provider-specific context extension                   | `data.propertyOrCallback`                               |
| `ref.watch(provider)` / `ref.read(provider)`         | Riverpod ref-based lookup                             | `data.propertyOrCallback`                               |
| `context.select<T, R>(...)`                          | Provider-specific selective rebuild                   | Resolved field on the payload                           |
| `BlocBuilder<B, S>` / `BlocListener<B, S>`           | Bloc-specific context coupling                        | `ValueListenableBuilder` + payload                      |
| `Consumer` / `ConsumerWidget` (Riverpod)             | Riverpod widget-tree dependency                       | `StatelessWidget` + payload parameter                   |
| `context.findAncestorStateOfType<T>()`               | Fragile dependency on nesting order                   | Inject the action through the payload                   |
| `controller.mutate(entityId)` in a builder           | Exposes internal identity; builder shouldn't know IDs | `data.onUpdate()` (pre-bound)                           |
| Complex `if/switch` on state enums in UI             | Scatters business logic across the presentation layer | `data.isConditionMet` (resolved bool)                   |
| `ChangeNotifier` or `Stream` in a payload            | Turns the snapshot into a live subscription           | Static fields + callback closures                       |
| Helper methods or utility functions in builder files | Disguises engine logic as UI convenience              | Move to the payload class or a dedicated engine utility |

---

## 5. Full implementation example

### Controller (engine layer)

```dart
class PanelController {
  final _panels = <String, PanelState>{};
  final ValueNotifier<int> _revision = ValueNotifier(0);

  /// Produces an immutable payload for the given panel.
  /// All callbacks are pre-bound to [panelId].
  HeaderData headerPayload(String panelId) {
    final panel = _panels[panelId]!;
    return HeaderData(
      title: panel.title,
      accentColor: panel.theme.accent,
      isCollapsed: panel.isCollapsed,
      onClose: () => _removePanel(panelId),
      onRename: (name) => _renamePanel(panelId, name),
      onToggleCollapse: () => _toggleCollapse(panelId),
    );
  }

  void _removePanel(String id) { /* ... */ _revision.value++; }
  void _renamePanel(String id, String name) { /* ... */ _revision.value++; }
  void _toggleCollapse(String id) { /* ... */ _revision.value++; }
}
```

### Payload (contract)

```dart
class HeaderData {
  final String title;
  final Color accentColor;
  final bool isCollapsed;
  final VoidCallback? onClose;
  final void Function(String)? onRename;
  final VoidCallback? onToggleCollapse;

  const HeaderData({
    required this.title,
    required this.accentColor,
    this.isCollapsed = false,
    this.onClose,
    this.onRename,
    this.onToggleCollapse,
  });
}
```

### Builder (presentation layer)

```dart
/// Pure function. No imports from the engine layer.
Widget panelHeader(BuildContext context, HeaderData data) {
  return Row(
    children: [
      Expanded(
        child: Text(
          data.title,
          style: TextStyle(color: data.accentColor),
        ),
      ),
      if (data.onToggleCollapse != null)
        IconButton(
          icon: Icon(data.isCollapsed ? Icons.expand_more : Icons.expand_less),
          onPressed: data.onToggleCollapse,
        ),
      if (data.onClose != null)
        IconButton(
          icon: const Icon(Icons.close),
          onPressed: data.onClose,
        ),
    ],
  );
}
```

---

## 6. Testing strategy

The payload pattern makes widget testing trivial — no mocks, no provider trees, no pump-and-settle rituals.

### Unit-testing a builder

```dart
testWidgets('header shows close button only when onClose is provided', (tester) async {
  // No controller, no context setup — just a payload.
  await tester.pumpWidget(
    MaterialApp(
      home: Scaffold(
        body: panelHeader(
          tester.element(find.byType(Scaffold)),
          const HeaderData(title: 'Test', accentColor: Colors.blue),
        ),
      ),
    ),
  );

  expect(find.byIcon(Icons.close), findsNothing);

  await tester.pumpWidget(
    MaterialApp(
      home: Scaffold(
        body: panelHeader(
          tester.element(find.byType(Scaffold)),
          HeaderData(
            title: 'Test',
            accentColor: Colors.blue,
            onClose: () {},
          ),
        ),
      ),
    ),
  );

  expect(find.byIcon(Icons.close), findsOneWidget);
});
```

### Verifying callback binding

```dart
test('controller binds onClose to the correct panel', () {
  final controller = PanelController();
  controller.addPanel('abc', PanelState(title: 'Demo'));

  final payload = controller.headerPayload('abc');
  payload.onClose!(); // fires _removePanel('abc') internally

  expect(controller.hasPanel('abc'), isFalse);
});
```

---

## 7. Migration guide

### 7a. Removing Provider / Riverpod / Bloc

**Step 1 — Audit dependencies.** Run `grep -r "provider\|riverpod\|bloc\|getx\|mobx" lib/` to find every import. List each file and what it consumes from the package.

**Step 2 — Replace state containers.** Convert each `ChangeNotifierProvider`, `StateNotifierProvider`, `BlocProvider`, or `Riverpod provider` into a plain Dart class with a `ValueNotifier` for rebuild signalling:

```dart
// ✗ Before (Riverpod)
final panelProvider = StateNotifierProvider<PanelNotifier, PanelState>((ref) {
  return PanelNotifier();
});

// ✓ After (Flutter-native)
class PanelController {
  final revision = ValueNotifier<int>(0);
  PanelState _state = PanelState.initial();

  PanelState get state => _state;
  void updateTitle(String title) {
    _state = _state.copyWith(title: title);
    revision.value++;
  }
}
```

**Step 3 — Replace consumer widgets.** Every `Consumer`, `ConsumerWidget`, `BlocBuilder`, or `context.watch` call becomes a `ValueListenableBuilder` that reads from the controller and constructs a payload:

```dart
// ✗ Before (Provider)
Widget build(BuildContext context) {
  final ctrl = context.watch<PanelController>();
  return Text(ctrl.title);
}

// ✓ After (Flutter-native)
Widget build(BuildContext context) {
  return ValueListenableBuilder<int>(
    valueListenable: controller.revision,
    builder: (context, _, __) {
      final data = controller.headerPayload();
      return panelHeader(context, data);
    },
  );
}
```

**Step 4 — Replace scoped injection.** If a `Provider`/`Riverpod` scope was supplying a controller deep into the tree, replace it with a custom `InheritedWidget` (see section 2). Keep the `InheritedWidget` at the view level — never inside a builder.

**Step 5 — Remove packages.** Delete `provider`, `flutter_riverpod`, `flutter_bloc`, etc. from `pubspec.yaml`. Run `flutter pub get` and fix remaining compile errors.

**Step 6 — Add lint rule.** Add a custom lint or CI grep check that fails the build if any banned package import reappears.

### 7b. Converting existing coupled builders

For builders that already use direct controller access (without a third-party package):

**Step 1 — Identify the contract.** List every property and method the builder currently reads from the controller. These become fields on the new payload class.

**Step 2 — Create the payload.** Define an immutable class with `final` fields for state and `Function` fields for actions.

**Step 3 — Add a factory on the controller.** Write a method that produces the payload, binding closures to the relevant entity IDs.

**Step 4 — Update the builder signature.** Replace `(BuildContext context)` with `(BuildContext context, PayloadType data)`. Remove all direct controller access.

**Step 5 — Wire the view.** At the composition point (the view), call the controller's factory method and pass the result to the builder.

**Step 6 — Verify isolation.** Confirm the builder file has zero imports from the engine layer.

---

## 8. PR checklist

Before approving any PR, verify:

- [ ] **No banned packages.** No import of `provider`, `riverpod`, `bloc`, `getx`, `mobx`, or any third-party state management library anywhere in the diff.
- [ ] **Zero context lookups in builders.** No `Provider.of`, `context.read`, `context.watch`, `context.select`, `ref.watch`, `ref.read`, `BlocBuilder`, or `context.findAncestorStateOfType` in any builder file.
- [ ] **Flutter-native reactivity only.** State propagation uses `ValueNotifier` + `ValueListenableBuilder`, `InheritedWidget`, or `StatefulWidget` — never a third-party wrapper.
- [ ] **Pre-bound callbacks.** Every user action is a closure on the payload. No entity IDs, no controller references leaked to the presentation layer.
- [ ] **Immutable payloads.** Every field on the data class is `final`. No `ChangeNotifier`, `StateNotifier`, `Stream`, or mutable collection passed through.
- [ ] **Resolved conditionals.** No multi-branch `if/switch` on state enums inside builders. Complex conditions are resolved in the controller and passed as booleans.
- [ ] **Layout independence.** The builder makes no assumptions about parent orientation (horizontal vs. vertical), scroll context, or sibling widgets.
- [ ] **Replaceable builders.** The builder can be swapped with an alternative implementation from a different package without touching the controller or view.
- [ ] **Testable in isolation.** The builder can be widget-tested with a hand-constructed payload — no app context, no DI container, no provider scope, no mock framework required.

---

## 9. FAQ

**Q: Where does the `BuildContext` go?**
Builders still receive `BuildContext` as the first argument — it's needed for `Theme.of(context)`, `MediaQuery.of(context)`, and similar framework-level queries. These are Flutter framework lookups, not state management — they're permitted.

**Q: Why not just use Riverpod? It's the community standard.**
Riverpod solves a real problem (scoped state without `BuildContext`), but it solves it by introducing `ref` — a new implicit dependency that every widget must carry. Our pattern achieves the same decoupling without any third-party API surface: the controller produces a payload, the builder consumes it, and Flutter's own `ValueListenableBuilder` handles reactivity. Fewer dependencies, smaller API surface, zero upgrade churn from package authors.

**Q: What about `flutter_bloc`? Our team already uses it.**
`BlocBuilder` and `BlocListener` are context-coupled widgets — they look up a `Bloc` from the ancestor tree, which is the exact pattern this standard eliminates. Migrate existing blocs to plain controller classes with `ValueNotifier` signalling. The business logic inside the bloc stays the same; only the delivery mechanism changes.

**Q: Can I use `InheritedWidget` inside a builder?**
No. `InheritedWidget` is permitted only at the **view composition level** to inject a controller into a deep subtree. Inside a builder, all data must come from the payload parameter. If a builder needs something from an `InheritedWidget`, the view should read it and pass it through the payload.

**Q: What about `InheritedModel` or `InheritedNotifier`?**
Same rule — permitted at the view level for selective rebuilds, never inside builders. `InheritedNotifier<ValueNotifier<T>>` is a convenient Flutter-native alternative when you want subtree-scoped reactive rebuilds without manual `ValueListenableBuilder` wiring.

**Q: How do I handle async operations (API calls, database reads)?**
The controller owns all async work. When the operation completes, the controller updates its internal state and increments its `ValueNotifier`. The view rebuilds, produces a fresh payload, and the builder renders the new state. The builder never awaits anything — it receives the current snapshot and renders it.

**Q: What if a third-party UI package requires `provider` as a transitive dependency?**
Wrap the third-party widget in an adapter at the view level. The adapter supplies whatever the package needs internally, but exposes only a payload-compatible interface to our builders. Document the wrapper and revisit it when the dependency is no longer needed.
