# Full-Scene-Agents review rules

- Focus review comments on correctness, security, backward compatibility, and maintainability.
- This repository hosts agent plugins implemented in C++, Python, and TypeScript, so reviewers should pay extra attention to cross-language boundaries and shared data contracts.
- Treat plugin registration, manifest files, exported APIs, configuration schemas, and message formats as public contracts. Flag silent breakage or implicit behavior changes.
- Prefer deterministic, reproducible automation. Flag workflows or scripts that depend on mutable global state, undeclared tools, or unchecked downloads.
- Flag unsafe shell usage, unchecked subprocess calls, path traversal risks, insecure temporary file handling, and unvalidated external input.
- Flag code that weakens observability, error handling, or cleanup for long-running agent processes.
- Ignore formatting-only churn unless it hides a functional issue.
