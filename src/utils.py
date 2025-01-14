import asyncio
import json
import time
from typing import Any, Callable, Literal
from openai import RateLimitError
import tiktoken
import tqdm

def count_tokens(text: str) -> int:
    return len(tiktoken.get_encoding("cl100k_base").encode(text))

async def run_parallel_tasks_with_progress(tasks: list[Callable], desc: str = "Tasks") -> list[Any]:
    time_start = time.time()
    async def wrapped_task(task: Callable, index: int, pbar: tqdm) -> tuple[int, Any]:
        try:
            result = await task()
            if isinstance(result, RateLimitError):
                print(f"Rate limit error: {result}")
            pbar.update(1)
            return index, result
        except Exception as e:
            pbar.update(1)
            return index, e
    with tqdm.tqdm(total=len(tasks), desc=desc) as pbar:
        results = await asyncio.gather(*[
            wrapped_task(task, i, pbar) 
            for i, task in enumerate(tasks)
        ])
    time_end = time.time()
    print(f"{desc}: time taken: {time_end - time_start}")
    return [result for _, result in sorted(results, key=lambda x: x[0])]

async def run_dspy_parallel(predictor, examples, desc: str = None):
    import dspy # slow
    tasks = [lambda x=x: dspy.asyncify(predictor)(**x.inputs().toDict()) for x in examples]
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
    with open(path, "w") as f: json.dump(data, f, indent=2)

def load_text(path: str):
    with open(path, "r") as f: return f.read()

def convert_confidence_to_num(confidence: float | Literal["low", "medium", "high", "very high"]):
    if isinstance(confidence, float): return confidence
    return {"low": 0.25, "medium": 0.5, "high": 0.75, "very high": 0.9, "certain": 1.0}[confidence]

def convert_num_to_confidence(num: float) -> Literal["low", "medium", "high", "very high"]:
    if num < 0.5: return "low"
    elif num < 0.75: return "medium"
    elif num < 0.9: return "high"
    else: return "very high"

def parse_confidence(x: str | float) -> Literal["low", "medium", "high", "very high"]:
    if isinstance(x, float): return convert_num_to_confidence(x)
    return x