You are a senior Flutter architect working inside the Intelag monorepo. Your goal is to build a strict, centralized Intelag UI architecture where every essential visual component—buttons, text, fields, layout shells, effects, and interactions—is controlled through a single reusable UI kit in packages/intelag_ui_kit, ensuring full visual consistency, enforceability, performance optimization, and long-term scalability. You want to eliminate ad-hoc styling, prevent design drift, standardize behavior across all apps, integrate cleanly with Riverpod and routing for fine-grained rendering, and make future design changes possible from one source of truth—so the UI becomes predictable, performant, and fully under your control as Intelag grows.

REQUEST 1:
Study the packages/intelag_ui_kit and create a README.md file that outlines the architecture, guidelines, and instructions for using the Intelag UI Kit, including which Flutter standard components to replace and how to handle styling and component creation. The README should serve as a comprehensive guide for all developers working on the Intelag monorepo to ensure consistency and adherence to the UI architecture. It is important to create a table that lists the Flutter standard components and their corresponding Intelag UI Kit equivalents, along with instructions on how to create new components when necessary. 

REQUEST 2:
Flutter Standard Components to Replace:
- Buttons: ElevatedButton, OutlinedButton, TextButton, FilledButton, IconButton, FloatingActionButton
- Text: Text, RichText (when used as common labels), DefaultTextStyle (where appropriate)
- Fields: TextField, TextFormField, DropdownButton, Checkbox, Switch, Radio
- Layout shells: Scaffold, AppBar, Drawer, BottomNavigationBar (or your dashboard shell)
- Lists/items: ListTile, Card, Divider (if used as basic building blocks)
- Feedback: SnackBar, Dialog, BottomSheet (create Intelag wrappers if used commonly)
- Indicators: CircularProgressIndicator, LinearProgressIndicator (if used commonly)
- Chips/tags: Chip, FilterChip, ChoiceChip (if used commonly)
If these analogous components do not exist in the Intelag UI Kit, you MUST create them first in the kit before replacing usage in the app. For more complex custom-built widgets that are repeated 2+ times with similar structure, you MUST also extract them into reusable Intelag components in the kit. Update the readme.md.

REQUEST 3:
Study packages/flutter_utils and move some of the widgets that can be considered reusable or essential into the Intelag UI Kit, ensuring they adhere to the centralized styling and component guidelines. Update the examples, and tests accordingly to reflect the changes and ensure that all components are consistent with the Intelag design system.

REQUEST 4:
No ad-hoc styling in feature code in packages/flutter_utils and no direct use of Flutter standard components if an Intelag equivalent exists or should exist. This means:
   - no ButtonStyle/styleFrom, no inline TextStyle, no inline hardcoded colors/radius/spacing
   - all styling is done through Intelag UI Kit tokens/theme/extensions or Intelag components 
   
ALL basic and essential components must go through Intelag UI Kit so we control:
   - shape, size, padding, density
   - typography hierarchy
   - colors and theming
   - behavior (disabled/loading, accessibility defaults, tap areas, etc.)

The target code MUST NOT use raw Flutter/Material basic controls directly if an Intelag equivalent exists or should exist. If you find repeated custom-built widget patterns used 2+ times with similar structure, you MUST:
   - extract them into a reusable Intelag component named IntelagXXX
   - save it under: packages/intelag_ui_kit/lib/src/widgets
   - replace all occurrences in the app with that new IntelagXXX component


REQUEST 5:
No ad-hoc styling in feature code in packages/intelag_auth and no direct use of Flutter standard components if an Intelag equivalent exists or should exist.

---
RULES:
You must enforce my rules in 
.intelag/.agent/instructions/ai-instruction.md
.intelag/.agent/instructions/flutter-ai.md
.intelag/.agent/instructions/folder-ai.md