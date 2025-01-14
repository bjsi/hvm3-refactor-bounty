import json
from litellm import acompletion
import asyncio
import sys
from src.file_context import create_contexts_for_blocks, format_block_context, get_all_block_numbers, hide_block_numbers
# from src.my_datasets import load_symbol_explanations
# from src.prompts.classify_blocks_fast import create_classify_blocks_messages, parse_classify_blocks_response

async def abatch_completion(batches):
    tasks = [acompletion(model="gemini/gemini-1.5-flash-8b", messages=msgs) for msgs in batches]
    results = await asyncio.gather(*tasks)
    return [result["choices"][0]["message"]["content"] for result in results]

def determine_relevancy():
    sym_exps = load_symbol_explanations()
    sym_exps = {exp["name"]: exp["explanation"] for exp in sym_exps}
    pass

async def classify_blocks(task: str, related_symbols: list[str]):
    block_numbers = sorted(get_all_block_numbers())
    sym_exps = load_symbol_explanations()
    sym_exps = {exp["name"]: exp["explanation"] for exp in sym_exps}
    message_batches = []
    for block_number in block_numbers:
        task_specific_context = format_block_context(*create_contexts_for_blocks([block_number]))
        task_specific_context = hide_block_numbers(set(block_numbers) - set([block_number]), task_specific_context)
        message_batches.append(create_classify_blocks_messages(
            codebase_symbol_explanations="\n".join([f"{name}: {sym_exps[name]}" for name in related_symbols]),
            task=task,
            specific_context=task_specific_context,
            block_number=block_number,
        ))
    print(message_batches)
    # results = await abatch_completion(message_batches)
    # return [parse_classify_blocks_response(result) for result in results]

async def predict_blocks(task: str):
    relevant_symbols = await determine_relevancy(task)
    blocks = await classify_blocks(task, relevant_symbols)
    return blocks

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide a task as the first argument.")
        sys.exit(1)
    task = sys.argv[1]
    blocks = asyncio.run(predict_blocks(task))
    sys.stdout.write(json.dumps(blocks))
