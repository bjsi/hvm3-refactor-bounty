import asyncio
import random
from typing import Callable, Literal
import dspy
import dspy.teleprompt
from src.my_datasets import load_binary_classification_judge_data, load_real_tasks, load_symbol_explanations
from src.file_context import create_contexts_for_blocks, format_block_context, get_all_block_numbers, find_block_numbers, hide_block_numbers
from src.filesystem import get_optimized_program_path
from src.utils import convert_confidence_to_num, parse_confidence, run_dspy_parallel
from src.llms import gemini_8b, deepseek_chat, gpt_4o

class ClassifyBlock(dspy.Signature):
    """
    Determine whether code in the given BLOCKs require direct modification.
    BLOCKs only require direct modification if the code directly visible inside them needs to be changed.
    Rules you must follow:
    - Empty blocks never require direct modification.
    """
    codebase_symbol_explanations: str = dspy.InputField()
    task: str = dspy.InputField()
    specific_context: str = dspy.InputField()
    block_number: int = dspy.InputField()
    task_reflection: str = dspy.OutputField(desc="Think step-by-step about the refactoring task")
    reasoning: str = dspy.OutputField(desc="Think step-by-step about whether or not the code in the BLOCK must be directly modified")
    requires_direct_modification: bool = dspy.OutputField()
    confidence: Literal["low", "medium", "high", "very high"] = dspy.OutputField(desc="Your confidence that this block must be directly modified")

##########
# Training
##########

def load_judged_edge_cases():
    real_tasks = load_real_tasks()
    sym_exps = load_symbol_explanations()
    sym_exps = {exp["name"]: exp["explanation"] for exp in sym_exps}
    judge_data = load_binary_classification_judge_data()
    input_keys = [ "codebase_symbol_explanations", "specific_context", "task", "block_number"]
    output_keys = ["task_reflection", "reasoning", "requires_direct_modification", "confidence"]
    all_keys = set(input_keys + output_keys)
    examples = []
    for data in judge_data:
        task = real_tasks[data["task"]]
        if not data.get("block_number"): raise ValueError(f"block_number not found for task {data['task']} {data["reasoning"]}")
        block_number = data["block_number"]
        reasoning = data["reasoning_for_modification"] if data["requires_direct_modification"] else data["reasoning_against_modification"]
        examples.append(dspy.Example({
            **{k: data[k] for k in all_keys if k in data},
            "block_number": block_number,
            "codebase_symbol_explanations": "\n".join([f"{name}: {sym_exps[name]}" for name in task["related_symbols"]]),
            "reasoning": reasoning,
            "confidence": parse_confidence(data["confidence"]),
        }).with_inputs(*input_keys))
    random.shuffle(examples)
    return examples

def load_balanced_trainset():
    edge_cases = load_judged_edge_cases()
    real_tasks_examples = load_trainset(lambda tasks: tasks[:6])
    
    # Split real tasks examples into positive and negative cases
    true_cases = [ex for ex in real_tasks_examples if ex.requires_direct_modification]
    false_cases = [ex for ex in real_tasks_examples if not ex.requires_direct_modification]

    print(f"total positive_cases: {len(true_cases)} total negative_cases: {len(false_cases)}")
    
    # Calculate how many examples we need from each class to balance
    k = min(len(true_cases), len(false_cases))
    
    true_cases = random.choices(true_cases, k=k)
    false_cases = random.choices(false_cases, k=k)

    # Combine edge cases with balanced real task examples
    balanced_examples = true_cases + false_cases # + edge_cases
    print(f"balanced_examples: {len(balanced_examples)}\npositive: {len(true_cases)}\nnegative: {len(false_cases)}")

    random.seed(42)
    random.shuffle(balanced_examples)

    for ex in balanced_examples:
        if isinstance(ex.confidence, float):
            raise ValueError(f"confidence is a float: {ex.confidence}")
    return balanced_examples

def load_trainset(filter_tasks: Callable[[list], bool] = lambda _: True):
    real_tasks = load_real_tasks()
    sym_exps = load_symbol_explanations()
    sym_exps = {exp["name"]: exp["explanation"] for exp in sym_exps}
    all_block_numbers = sorted(get_all_block_numbers())
    # TODO: simplify single block
    block_numbers_batched = [all_block_numbers[i:i+1] for i in range(0, len(all_block_numbers), 1)]
    all_examples = []
    for task in filter_tasks(list(real_tasks.values())):
        task_examples = []
        # Check for duplicate block numbers in blocks_to_edit
        block_nums = [b["block_number"] for b in task["blocks_to_edit"]]
        if len(block_nums) != len(set(block_nums)):
            duplicates = [num for num in block_nums if block_nums.count(num) > 1]
            raise ValueError(f"Duplicate block numbers found in blocks_to_edit: {duplicates}")
        for block_numbers_batch in block_numbers_batched:
            # specific context for this example - a contiguous window of blocks for evaluation
            task_specific_context = format_block_context(*create_contexts_for_blocks(block_numbers_batch))
            task_specific_context = hide_block_numbers(set(all_block_numbers) - set(block_numbers_batch), task_specific_context)
            # check that the correct blocks are being evaluated
            block_numbers: set[int] = find_block_numbers(task_specific_context)
            assert block_numbers == set(block_numbers_batch), f"expected {block_numbers_batch}, got {block_numbers}"
            if task["related_symbols"]:
                block = block_numbers_batch[0]
                block_info = next((b for b in task["blocks_to_edit"] if b["block_number"] == block), None)
                task_examples.append(dspy.Example(
                    codebase_symbol_explanations="\n".join([f"{name}: {sym_exps[name]}" for name in task["related_symbols"]]),
                    specific_context=task_specific_context,
                    task=task["task"],
                    block_number=block,
                    task_reflection=task.get("task_reflection", None),
                    reasoning=block_info.get("reasoning", None) if block_info else None,
                    requires_direct_modification=block_info.get("requires_direct_modification", False) if block_info else False,
                    confidence=parse_confidence(block_info.get("confidence", "low")) if block_info else "low",
                ).with_inputs("codebase_symbol_explanations", "specific_context", "task", "block_number"))
        all_examples.extend(task_examples)
    return all_examples

def direct_score(example, pred, trace=None):
    predicted = pred.requires_direct_modification and convert_confidence_to_num(pred.confidence) >= 0.75
    actual = example.requires_direct_modification
    return int(predicted == actual)

def f1_score(predicted_blocks_to_edit, actual_blocks_to_edit):
    predicted = set(predicted_blocks_to_edit)
    actual = set(actual_blocks_to_edit)
    true_positives = len(predicted & actual)
    false_positives = len(predicted - actual)
    false_negatives = len(actual - predicted)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

def optimize(devset, task_lm, prompt_lm, teacher_lm, dataset_summary_lm):
    program = dspy.Predict(ClassifyBlock)
    with dspy.context(lm=task_lm, async_max_workers=10):
        optimizer = dspy.teleprompt.MIPROv2(
            metric=direct_score,
            auto="medium",
            prompt_model=prompt_lm,
            task_model=task_lm,
            max_bootstrapped_demos=1,
            max_labeled_demos=6,
            num_threads=10,
            hide_demo_fields=[
                "codebase_summary",
                "codebase_symbol_explanations",
                "specific_context",
            ],
            dataset_summary_model=dataset_summary_lm, # TODO: deepseek hangs
            teacher_settings=dict(lm=teacher_lm), # causing hangs with deepseek
        )
        optimized_program = optimizer.compile(program, trainset=devset)
        optimized_program.save(get_optimized_program_path(__file__))

###########
# Inference
###########

def classify_blocks(model, examples, async_max_workers: int = 50, cache=True):
    program = dspy.Predict(ClassifyBlock)
    if get_optimized_program_path(__file__).exists():
        program.load(get_optimized_program_path(__file__))
    with dspy.context(lm=model, async_max_workers=async_max_workers, cache=cache):
        results = asyncio.run(run_dspy_parallel(program, examples))
    return results

if __name__ == "__main__":
    optimize(load_balanced_trainset(), gemini_8b, deepseek_chat, deepseek_chat, gpt_4o)