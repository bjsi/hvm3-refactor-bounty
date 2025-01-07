import asyncio
import json
import time
from typing import Any, Callable
import tiktoken
import tqdm
import dspy

def count_tokens(text: str) -> int:
    return len(tiktoken.get_encoding("cl100k_base").encode(text))

async def run_parallel_tasks_with_progress(tasks: list[Callable], desc: str = "Tasks") -> list[Any]:
    time_start = time.time()
    async def wrapped_task(task: Callable, index: int, pbar: tqdm) -> tuple[int, Any]:
        result = await task()
        pbar.update(1)
        return index, result
    with tqdm.tqdm(total=len(tasks), desc=desc) as pbar:
        results = await asyncio.gather(*[
            wrapped_task(task, i, pbar) 
            for i, task in enumerate(tasks)
        ])
    time_end = time.time()
    print(f"{desc}: time taken: {time_end - time_start}")
    return [result for _, result in sorted(results, key=lambda x: x[0])]

async def run_dspy_parallel(predictor, examples: dspy.Example, desc: str = None):
    tasks = [
        lambda x=x: dspy.asyncify(predictor)(**x.inputs().toDict())
        for x in examples
    ]
    return await run_parallel_tasks_with_progress(tasks, desc=desc)

def load_jsonl(path: str):
    with open(path, "r") as f: return [json.loads(line) for line in f]

def save_jsonl(data: list[dict], path: str):
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

def load_json(path: str):
    with open(path, "r") as f: return json.load(f)

def save_json(data: dict, path: str):
    with open(path, "w") as f: json.dump(data, f)

def load_text(path: str):
    with open(path, "r") as f: return f.read()