# Data

Includes the example tasks and the codebase as well as any data I generated with LLMs.

- `hvm3_real_tasks.json` contains the example refactoring tasks Victor posted on Discord. I used a mixture of AI and manual curation to decide which symbols were related to each task. I trimmed each list to a minimal set of high-confidence related symbols.
- I split the code between `hvm-code.hs` and `hvm-code.c` to make them parseable by Tree-sitter.