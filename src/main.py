import dspy
import json
import asyncio
import sys
from src.file_context import create_contexts_for_blocks, format_block_context, get_all_block_numbers, hide_block_numbers
from src.my_datasets import load_symbol_explanations
from src.prompts.classify_blocks import classify_blocks
from src.llms import gemini_8b
from src.prompts.classify_symbols import classify_symbols

# NOTE: i was hitting rate limits, but may have been daily-usage related
# 8b should support 4000/min and this pipeline makes < 1000 requests
# try turning this up to 300 or so
ASYNC_MAX_WORKERS = 20

def predict_blocks(task: str, model=gemini_8b):
    block_numbers = sorted(get_all_block_numbers())
    sym_exps = load_symbol_explanations()
    sym_exps = {exp["name"]: exp["explanation"] for exp in sym_exps}

    ## get relevant symbols (fn names, types etc)
    examples = []
    classify_symbols_input_keys = ['task', 'symbol', 'explanation']
    for sym, exp in sym_exps.items():
        examples.append(dspy.Example(task=task, symbol=sym, explanation=exp).with_inputs(*classify_symbols_input_keys))
    related_symbols = classify_symbols(examples=examples, model=model, async_max_workers=ASYNC_MAX_WORKERS, cache=False)

    ## classify blocks
    examples = []
    classify_blocks_input_keys = ['codebase_symbol_explanations', 'task', 'specific_context', 'block_number']
    for block_number in block_numbers:
        task_specific_context = format_block_context(*create_contexts_for_blocks([block_number]))
        task_specific_context = hide_block_numbers(set(block_numbers) - set([block_number]), task_specific_context)
        examples.append(dspy.Example(
            codebase_symbol_explanations="\n".join([f"{name}: {sym_exps[name]}" for name in related_symbols]),
            task=task,
            specific_context=task_specific_context,
            block_number=block_number,
        ).with_inputs(*classify_blocks_input_keys))
    blocks = classify_blocks(model=model, examples=examples, async_max_workers=ASYNC_MAX_WORKERS, cache=False)
    print(json.dumps(blocks))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide a task as the first argument.")
        sys.exit(1)
    task = sys.argv[1]
    asyncio.run(predict_blocks(task))
