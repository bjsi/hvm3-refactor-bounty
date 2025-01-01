# HVM Large Refactor Bounty

Develop an AI tool that, given the current snapshot of HVM3's codebase and a refactor request, correctly assigns which chunks must be updated, in order to fulfill that request. The AI tool must use at most $1 to complete this task, while achieving a recall of at least 90% and a precision of at least 90%.

## Setup

- Clone the repo + submodules.
- `python3 -m venv env`
- `source env/bin/activate`
- `pip install -r requirements.txt`
- `npm install tree-sitter`
- Create a `.env` file with an `OPENROUTER_API_KEY`.

## Approach

1. Separate the C code and Haskell code into two different files.
2. Use tree-sitter to extract the 270 or so important code symbols (function names, type names, etc).
3. For a given refactoring task, perform the following in parallel for every code symbol:
    - Extract the code and related context for the code symbol using tree-sitter.
    - Send to DeepSeek to determine whether the code symbol is relevant to the refactoring task.
    - Write a list of code symbol names that would be affected by the refactoring task.
    - Finally, decide which specific chunk numbers are affected by the refactoring task.

## Timings and costs

- There are ~270 code symbols
- ...