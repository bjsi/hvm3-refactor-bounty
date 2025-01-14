# HVM Large Refactor Bounty

Develop an AI tool that, given the current snapshot of HVM3's codebase and a refactor request, correctly assigns which chunks must be updated, in order to fulfill that request. The AI tool must use at most $1 to complete this task, while achieving a recall of at least 90% and a precision of at least 90%.

## Setup

- `git clone --recurse-submodules`
- `python3 -m venv env`
- `source env/bin/activate`
- `pip install -r requirements.txt`
- `npm install tree-sitter`
- Create a `.env` file with your `GEMINI_API_KEY`.

To run the pipeline on a task run:

`python -m src.main`

## High-level Approach

I wanted to see how good a small, cheap model could do at this task, so I used Gemini Flash 8b. I wrote a `FileContext` class that uses tree-sitter to parse the C and Haskell code to provide accurate context. I used the `dspy` Python library to write and optimize the prompts instructions and few-shot examples. Finally, I developed a semi-automated active learning approach to mining synthetic task data based on the refactoring examples Victor shared on Discord.

### Details
