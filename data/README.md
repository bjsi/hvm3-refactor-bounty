# Data

This folder includes the example tasks and the codebase as well as any datasets I generated with LLMs.

- `hvm3_real_tasks.json` contains the example refactoring tasks Victor posted on Discord. I used an active learning approach involving an LLM judge to figure out which symbols are related to each task and which blocks need to be modified.
- I split the code between `hvm-code.hs` and `hvm-code.c` and changed the `BLOCK` lines to comments to make the code parseable with tree-sitter.