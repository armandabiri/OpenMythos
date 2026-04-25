New AI Handoff Summary Requirements

Purpose

* Start a new chat with another AI and transfer full project and codebase context.
* Provide a clear, immediately usable summary with no filler.
* Output must be ready to paste as starting context.

Content Requirements

* Explain what the project is and its primary objectives.
* Define what the system must do.
* Define what the system must not do.
* List main technical and architectural constraints.
* Describe overall structure and repository layout.
* List main modules and components and what each does.
* Explain how modules connect and interact.
* Document key architectural decisions and why they were made.
* Document rejected alternatives and why they were rejected.
* Capture core patterns and enforced rules including naming, organization, logging, error handling, configuration.
* Summarize common bugs encountered, root causes, and prevention rules.
* Note performance constraints, bottlenecks, and optimizations already implemented.
* List changes that must be avoided because they break architecture or degrade performance.
* Explain how testing and debugging are typically done.
* Document what debugging approaches worked well.
* Note debugging approaches to avoid.
* Provide a clear mental model for working in the codebase.
* State core assumptions developers must keep.
* Explain how to safely extend or modify the system.
* Identify fragile or high risk areas.

Output Rules

* Must be explicit and concrete.
* Avoid extra filler or commentary.
* Do not include code unless absolutely required for clarity.
* Write the summary to .intelag/.agent/summary/{YYYYMMDD}*{HHMMSS}*{title}.txt.
* Do not include any additional text in the chat output.
