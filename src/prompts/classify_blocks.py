import asyncio
import json
import re
import sys
from typing import Callable, Literal, Optional
import dspy
import dspy.teleprompt
from pydantic import BaseModel
from src.my_datasets import load_binary_classification_judge_data, load_codebase_summary, load_real_tasks, load_symbol_explanations
from src.file_context import FileContext, get_all_block_numbers, find_block_numbers, format_contexts, get_all_names, get_block_code, hide_block_numbers
from src.filesystem import data_dir, get_optimized_program_path
from src.utils import run_dspy_parallel
from src.llms import gemini_8b, deepseek_chat, claude_sonnet

class ClassifyBlock(dspy.Signature):
    """Determine whether code in the given BLOCKs require direct modification. BLOCKs only require direct modification if the code directly visible inside them needs to be changed."""
    codebase_summary: str = dspy.InputField()
    codebase_symbol_explanations: str = dspy.InputField()
    task: str = dspy.InputField()
    specific_context: str = dspy.InputField()
    block_number: int = dspy.InputField()
    task_reflection: str = dspy.OutputField(desc="Think step-by-step about the refactoring task")
    reasoning: str = dspy.OutputField(desc="Debate whether or not the code in the BLOCK must be directly modified")
    requires_direct_modification: bool = dspy.OutputField()
    confidence: Literal["low", "medium", "high", "very high"] = dspy.OutputField(desc="Your confidence that this block must be directly modified")

##########
# Training
##########

def merge_contexts(*contexts: tuple[FileContext, FileContext]):
    merged_hs_ctx, merged_c_ctx = None, None
    for hs_ctx, c_ctx in contexts:
        if merged_hs_ctx is None: merged_hs_ctx = hs_ctx
        else: merged_hs_ctx = merged_hs_ctx.shallow_copy().merge(hs_ctx)
        if merged_c_ctx is None: merged_c_ctx = c_ctx
        else: merged_c_ctx = merged_c_ctx.shallow_copy().merge(c_ctx)
    return merged_hs_ctx, merged_c_ctx

def merge_contexts_for_names(names, code_contexts):
    contexts = []
    for name in names:
        hs_ctx, c_ctx = code_contexts[name]
        contexts.append((hs_ctx, c_ctx))
    return merge_contexts(*contexts)

def create_contexts_for_blocks(block_numbers: list[int]):
    hs_ctx = FileContext(data_dir / "hvm-code.hs")
    c_ctx = FileContext(data_dir / "hvm-code.c")
    hs_ctx.show_blocks(block_numbers).show_parents()
    c_ctx.show_blocks(block_numbers).show_parents()
    return hs_ctx, c_ctx

def format_block_context(hs_ctx, c_ctx):
    """Add an END BLOCK comment to the end of the block."""
    s = format_contexts(hs_ctx, c_ctx)
    match = re.search(r"(//|--)\s+BLOCK \d+", s)
    if match: match = match.group().split()[0]  # Get just the // or -- part
    lines = s.split('\n')
    end_block_line = f"{match} BLOCK END"
    lines = lines[:-2] + [end_block_line] + lines[-2:]
    return "\n".join(lines)

# take the hardest cases from the judge data to use for training
def load_judged_edge_cases():
    codebase_summary = load_codebase_summary()
    real_tasks = load_real_tasks()
    sym_exps = load_symbol_explanations()
    sym_exps = {exp["name"]: exp["explanation"] for exp in sym_exps}
    judge_data = load_binary_classification_judge_data()
    input_keys = [
        "codebase_summary",
        "codebase_symbol_explanations",
        "specific_context",
        "task",
        "block_number",
    ]
    output_keys = [
        "task_reflection",
        "reasoning",
        "requires_direct_modification",
        "confidence",
    ]
    all_keys = set(input_keys + output_keys)
    examples = []
    for data in judge_data:
        task = real_tasks[data["task"]]
        block_number = data["block_number"]
        reasoning = data["reasoning_for_modification"] if data["requires_direct_modification"] else data["reasoning_against_modification"]
        examples.append(dspy.Example({
            **{k: data[k] for k in all_keys if k in data},
            "block_number": block_number,
            "codebase_summary": codebase_summary,
            "codebase_symbol_explanations": "\n".join([f"{name}: {sym_exps[name]}" for name in task["related_symbols"]]),
            "reasoning": reasoning
        }).with_inputs(*input_keys))
    return examples

def load_trainset(filter_tasks: Callable[[list], bool]):
    real_tasks = load_real_tasks()
    codebase_summary = load_codebase_summary()
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
                    codebase_summary=codebase_summary,
                    codebase_symbol_explanations="\n".join([f"{name}: {sym_exps[name]}" for name in task["related_symbols"]]),
                    specific_context=task_specific_context,
                    task=task["task"],
                    block_number=block,
                    task_reflection=task["task_reflection"],
                    reasoning=block_info.get("reasoning", None) if block_info else None,
                    requires_direct_modification=block_info.get("requires_direct_modification", False) if block_info else False,
                    confidence=block_info.get("confidence", "low") if block_info else "low",
                ).with_inputs("codebase_summary", "codebase_symbol_explanations", "specific_context", "task", "block_number"))
        all_examples.extend(task_examples)
    return all_examples

def convert_confidence_to_num(confidence: float | Literal["low", "medium", "high", "very high"]):
    if isinstance(confidence, float): return confidence
    return {"low": 0.25, "medium": 0.5, "high": 0.75, "very high": 0.9, "certain": 1.0}[confidence]

def convert_num_to_confidence(num: float) -> Literal["low", "medium", "high", "very high"]:
    if num < 0.5: return "low"
    elif num < 0.75: return "medium"
    elif num < 0.9: return "high"
    else: return "very high"

def get_confidence(x: str | float) -> Literal["low", "medium", "high", "very high"]:
    if isinstance(x, float): return convert_num_to_confidence(x)
    return x

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

def optimize(devset, task_lm, prompt_lm, teacher_lm):
    program = dspy.Predict(ClassifyBlock)
    if get_optimized_program_path(__file__).exists():
        print("Using optimized classify_blocks program")
        program.load(get_optimized_program_path(__file__))
    with dspy.context(lm=task_lm):
        optimizer = dspy.teleprompt.MIPROv2(
            metric=direct_score,
            auto="light",
            prompt_model=prompt_lm,
            task_model=task_lm,
            max_bootstrapped_demos=1,
            max_labeled_demos=len(program.demos) + 2,
            num_threads=6,
            hide_demo_fields=[
                "codebase_summary",
                "codebase_symbol_explanations",
            ],
            # teacher_settings=dict(lm=teacher_lm),
        )
        optimized_program = optimizer.compile(program, trainset=devset)
        optimized_program.save(get_optimized_program_path(__file__))

###########
# Inference
###########

def classify_blocks(model, examples):
    program = dspy.Predict(ClassifyBlock)
    if get_optimized_program_path(__file__).exists():
        print("Using optimized classify_blocks program")
        program.load(get_optimized_program_path(__file__))
    with dspy.context(lm=model, async_max_workers=20):
        results = asyncio.run(run_dspy_parallel(program, examples))
    return results

# 1 block, gemini, no optimizing, with filtering, accuracy 94%, f1 37%
# -- USING END BLOCK --
# 1 block, gemini, no optimizing, with filtering, accuracy 94%, f1 43%
# -- CLEANED DATA --
# 1 block, gemini, no optimizing, with filtering, accuracy 95%, f1 65%
# 1 block, deepseek, no optimizing, with filtering, accuracy 96%, f1 75%
# -- CLEANED DATA WITH TIEBREAKER --
# 1 block, gemini, no optimizing, with filtering, accuracy 96%, f1 61%
# 1 block, deepseek, no optimizing, with filtering, accuracy 98%, f1 80%
# -- OPTIMIZATION v1 -- only edge cases identified w/ tiebreaker
# 1 block, gemini, optimizing, with filtering, accuracy 96%, f1 60% - hmm pretty much the same - the instructions are overfitting
# -- OPTIMIZATION v2 -- only edge cases identified w/ tiebreaker

if __name__ == "__main__":
    # optimize(load_judged_edge_cases(), gemini_8b, deepseek_chat, deepseek_chat)
    # real_tasks = load_real_tasks()
    # task = list(real_tasks.values())[0]
    # dataset = load_trainset(lambda tasks: tasks[0:1])
    # results = classify_blocks(gemini_8b, dataset)
    # scores = []
    # wrong_reasons = []
    # for (result, datapoint) in list(zip(results, dataset)):
    #     score = direct_score(datapoint, result)
    #     scores.append(score)
    #     if score == 0:
    #         reason = "wrong" if result.requires_direct_modification != datapoint.requires_direct_modification else "low confidence"
    #         if reason == "low confidence": continue
    #         print(f"block {datapoint.block_number}")
    #         print(f"code: {get_block_code(datapoint.block_number)}")
    #         print(f"reasoning: {result.reasoning}")
    #         print(f"prediction: {result.requires_direct_modification}")
    #         print(f"confidence: {result.confidence}")
    #         print(f"score: {score}")
    #         wrong_reasons.append(reason)
    #         print(f"reason for fail {reason}")
    #         print('---')

    # expected_blocks_to_edit = set([example.block_number for example in dataset if example.requires_direct_modification and convert_confidence_to_num(example.confidence) >= 0.75])
    # actual_blocks_to_edit = set([example.block_number for example, result in zip(dataset, results) if result.requires_direct_modification and convert_confidence_to_num(result.confidence) >= 0.75])
    # print(f"f1 score: {f1_score(expected_blocks_to_edit, actual_blocks_to_edit)}")
    # print(f"accuracy: {sum(scores) / len(scores)}")