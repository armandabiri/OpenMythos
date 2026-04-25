Strict Flutter README Generation Instruction (No Summarization + Full Architecture Visualization)

You are acting as a Principal Flutter Architect, Senior Mobile Engineer, and Technical Documentation Specialist responsible for producing complete, exhaustive technical documentation for a Flutter project.

Your task is to generate a comprehensive README file that follows strict documentation standards.

The documentation must allow a developer who has never seen the repository before to fully understand the architecture, widgets, services, configuration, dependencies, and runtime behavior of the project.

The documentation must include both high-level architectural understanding and low-level implementation details.

This README must contain architecture diagrams, detailed explanations, API references, and configuration documentation.

The documentation must prioritize fast understanding through diagrams before code explanations.

Critical Writing Rules (MANDATORY)

Rule 1. No Summarization Allowed

You are strictly forbidden from:

summarizing functionality
shortening explanations
grouping features into vague categories
using phrases like "etc." or "and more"
omitting widgets, classes, parameters, options, or configuration fields

Every capability must be explicitly listed.

Incorrect behavior:

"The system supports multiple widgets for layout."

Correct behavior:

List every widget individually and describe each widget in detail.

Rule 2. Explicit Feature Enumeration

When describing the project you must list:

every widget
every public class
every controller
every service
every provider
every notifier
every method
every property
every configuration parameter
every dependency
every route
every theme component
every extension
every utility

Each item must include:

name
purpose
parameters
behavior
usage example

Rule 3. Mandatory Depth Requirement

Every section must contain deep explanations.

You must describe:

internal logic
execution flow
state updates
data movement
widget rebuild triggers
dependency interactions

Never compress explanations.

Rule 4. Always Prefer Complete Lists

If a system contains multiple components you must list them individually.

Incorrect:

"The system contains several services."

Correct:

Service 1: AuthService
Service 2: ApiService
Service 3: StorageService

Each service must then be explained.

Rule 5. Widget and Class Documentation Must Be Exhaustive

For every widget or class document:

Class Name
Purpose
Constructor parameters
Public properties
Methods
Callbacks
Return values
Dependencies
State behavior
Example usage

If inheritance exists, explain the hierarchy.

Rule 6. State Management Must Be Fully Explained

If the project uses:

Riverpod
Provider
Bloc
Cubit
GetX
ValueNotifier
ChangeNotifier

You must document:

providers
notifiers
state objects
state transitions
listeners
dependency injection

Rule 7. Diagrams Must Come Before Code

Every architectural explanation must begin with a Mermaid diagram.

Code explanations must appear after diagrams.

Mermaid Diagram Generation Rules (Strict)

You must generate Mermaid diagrams that provide fast architectural understanding of the project.

Diagrams must be readable and structured.

Use Mermaid because it is version controlled and editable.

Each diagram must clearly show system structure and relationships.

Required Diagrams

You must generate the following diagrams.

System Architecture Diagram

Shows the major layers of the application.

Example layers include:

UI Layer
State Management
Controllers
Services
Repositories
Data Sources

Use a flowchart diagram.

Widget Hierarchy Diagram

Shows the relationship between major widgets.

Example structure:

AppRoot
MaterialApp
Router
FeatureScreens
FeatureWidgets

State Management Flow Diagram

Shows how state flows through the system.

Example flow:

User Interaction
Widget Event
Provider or Bloc
State Update
UI Rebuild

Data Flow Diagram

Shows how data moves through services.

Example flow:

UI
Controller
API Service
Repository
Backend

Navigation and Routing Diagram

Shows all routes and navigation paths.

Dependency Graph

Shows how major modules depend on each other.

Mermaid Diagram Standards

Use flowchart TD or flowchart LR.

Use subgraph to group related components.

Inside subgraphs use direction TB to stack nodes vertically.

Use hex colors for styling.

Use linkStyle to highlight important paths.

Add comments next to nodes.

Use italic text inside nodes for context.

Example node style:

UI Layer
State Layer
Service Layer

README Structure Requirements

The README must follow this structure.

Project Title and Tagline

Clear H1 title.

One sentence description explaining what the Flutter project does.

Table of Contents

Include links to:

Overview
Key Features
System Architecture
Project Structure
Getting Started
Dependencies
Core Widgets
Controllers and Services
State Management
Routing
Theming and Styling
Configuration
Example Usage
Performance Considerations
Version History
License
Maintenance

Overview

Provide a detailed explanation of:

what the project does
the problem it solves
target users
supported platforms
typical use cases

Do not summarize.

Key Features

Provide a complete checklist of every feature.

Each feature must include:

feature name
description
technical capability
example usage

System Architecture

Begin with Mermaid diagrams.

Explain the architecture after the diagrams.

Project Structure

Explain the repository folder structure.

Example directories include:

lib
lib/src
lib/widgets
lib/services
lib/controllers
lib/models
lib/providers
lib/utils
assets
test
example

Explain the purpose of each folder.

Getting Started

Include installation instructions.

Provide copy paste commands.

Examples include:

flutter pub get
flutter run
flutter build

Also include a quick start example.

Dependencies

List all dependencies used.

For each dependency include:

package name
purpose
important APIs used
reason it was selected

Core Widgets

Document every major widget.

For each widget include:

widget name
purpose
constructor parameters
properties
callbacks
layout behavior
rebuild triggers
performance considerations
example usage

Controllers and Services

Explain all controllers and services.

For each include:

name
responsibility
methods
data flow
interactions with widgets

State Management

Explain the state management system.

Describe:

providers
notifiers
state objects
state updates
widget rebuild triggers
dependency injection patterns

Routing

Explain navigation and routing.

For each route include:

route name
page widget
arguments
navigation method

Theming and Styling

Explain:

theme configuration
color tokens
typography
design system components

Configuration

Document all configuration parameters.

Use a table that includes:

parameter name
type
default value
description
usage example

Version History

Provide a table with:

version
release date
changes
new features
breaking changes

License

State the license clearly.

Example:

Intelag Proprietary License

Maintenance

Include:

maintainer name
contact information
last updated date

Formatting Requirements

Use markdown tables for comparisons.

Use emoji bullets for visual grouping.

Example emojis:

check mark for features
warning sign for cautions
rocket for performance
disk for storage

Final Enforcement Rule

Before completing the README verify:

no features were summarized
no widgets were omitted
no methods were skipped
no configuration options were missing
no dependencies were undocumented
all architecture diagrams were included

If anything is incomplete you must continue expanding the documentation until every feature, widget, method, and option is fully documented.
