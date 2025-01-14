# HVM Large Refactor Bounty

Develop an AI tool that, given the current snapshot of HVM3's codebase and a refactor request, correctly assigns which chunks must be updated, in order to fulfill that request. The AI tool must use at most $1 to complete this task, while achieving a recall of at least 90% and a precision of at least 90%.

## Setup

- `git clone --recurse-submodules https://github.com/bjsi/hvm3-refactor-bounty`
- `cd hvm3-refactor-bounty`
- `python3 -m venv env`
- `source env/bin/activate`
- `pip install -r requirements.txt`
- Create a `.env` file with your `GEMINI_API_KEY`.

To run the pipeline on a task run:

`python -m src.main <task>`

NOTE: The rate limit for Gemini 8b is 4000/min and this pipeline makes < 1000 requests per task but I was occasionally hitting rate limit errors so I had to turn down `ASYNC_MAX_WORKERS` to 20 in `src/main.py` which is making things really slow :(. Try turning this up to 300 and see if it works?

## High-level Approach

I wanted to see how good a small, cheap model could do at this task, so I used Gemini Flash 8b. I wrote a `FileContext` class that uses tree-sitter to parse the C and Haskell code to provide accurate context. I used the `dspy` Python library to write and optimize the prompts instructions and few-shot examples. Finally, I developed a semi-automated active learning-ish approach to mining synthetic task data.

### Details

#### Symbol Explanations

- I used `tree-sitter` to create a list of the most important functions and types in the codebase.
- For every symbol I extracted its definition and lines where it is used.
- I used this to create high quality explanations for every symbol in the codebase.
- NOTE: I cache this in a JSON file and do not recompute it because the codebase is frozen for the competition. I think this is reasonable because this approach could work in production too - simply cache symbol explanations and recompute symbol explanations only after their code (as retrieved using tree-sitter) changes instead of creating explanations from scratch every time you run the pipeline.

#### Symbol Classification

- Given a refactoring request, the first step is to get the most related code symbol explanations.
- I start by using `classify_symbols` to get a list of all symbol explanations that are relevant to the refactoring request.
- This can be done in parallel for every symbol (~270) in the codebase.
- For prompt optimization I used Victor's list of refactoring examples and DeepSeek V3 to get a list of symbols that are relevant to the refactoring request. I then manually screened the symbols to check if they were relevant or not. This was quite tedious and error prone because I am not very familiar with the codebase and inspired my later attempts to mine synthetic data in a more principled way.
- After optimizing the prompt performance rose >25% from 60% to 86% on accuracy on my balanced dataset of 50:50 relevant and irrelevant task-symbol pairs.

#### Block Classification

- Given a list of relevant symbols, I used the `classify_blocks` to get a list of all blocks that must be edited during the refactoring.
- This can be done in parallel for every block (~500) in the codebase.
- Determining whether a block must be edited is quite challenging. I struggled to gather data here manually.
- I came up with an automated way to get data - I repeatedly ran `classify_blocks` over Victor's list of refactoring examples and used an LLM judge to decide in cases where there were classification disagreements between runs. I made a simple CLI function to review disagreements and optimized the LLM judge using `dspy` on the disagreement dataset.
- I only figured out how to do this later in the competition and had to move on to other things, but by iterating in this manner and optimizing on the final dataset I was able to raise Gemini 8b's performance from ~55% to 87% accuracy.

## Thoughts

- Firstly, this was a lot of fun and I want to attempt more things like this! (lmk if you have some tasks for me!)
- I ran out of time to do a really rigorous evaluation of the full pipeline. I will probably evaluate it on my own codebase instead because I am not a good judge of the output for HVM3. I am curious what Victor thinks. You can swap models and re-optimize on different models. I tried a bunch but found that Gemini 8b was the best model for this task because of its low cost, low latency, high RPM rate limit and decent reasoning ability.
- There is significant ambiguity in determining what is relevant to a refactoring request if you are being picky. It definitely requires strong reasoning.
- `dspy` is excellent, though it is not a panacea and you still need to do a lot of work to get good results. `dspy` performs search over the space of possible prompts, but you still have to search over the space of possible `dspy` programs. I am sure someone with more experience could improve the results further. It is interesting to scroll through the optimized prompts in the `optimized_programs` directory.
- I wasted a lot of time fooling myself with evaluation metrics because I didn't balance the classification datasets. I am still somewhat nervous about them...
- I am quite excited about the potential of active learning with LLM as a judge techniques in `dspy`! It could be used to align end-user preferences with a prompt's behavior without fine-tuning and has great potential in synthetic data collection.
