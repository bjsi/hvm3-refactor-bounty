import asyncio
from typing import Literal, Optional
import dspy
import dspy.teleprompt
from pydantic import BaseModel
from src.my_datasets import load_codebase_summary, load_real_tasks, load_symbol_explanations
from src.file_context import FileContext, get_all_block_numbers, create_contexts_for_name, find_block_numbers, format_contexts, get_all_names, get_block_code, hide_block_numbers
from src.filesystem import data_dir, get_optimized_program_path
from src.utils import run_dspy_parallel
from src.llms import gemini_8b, deepseek_chat, claude_sonnet

#################
# Multiple Blocks
#################

class Block(BaseModel):
    block_number: int = dspy.OutputField()
    reasoning: str = dspy.OutputField(desc="Debate whether this block must be directly modified")
    requires_direct_modification: bool = dspy.OutputField() 
    confidence: Literal["low", "medium", "high", "certain"] = dspy.OutputField(desc="Your confidence that this block must be directly modified")
    def __getitem__(self, key): return getattr(self, key)

class ClassifyBlocksToEdit(dspy.Signature):
    """Determine whether code in the given BLOCKs require direct modification. BLOCKs only require direct modification if the code directly visible inside them needs to be changed."""
    codebase_summary: str = dspy.InputField()
    codebase_symbol_explanations: str = dspy.InputField()
    general_context: str = dspy.InputField()
    task: str = dspy.InputField()
    specific_context: str = dspy.InputField()
    block_numbers: str = dspy.InputField(desc="BLOCK numbers to evaluate")
    task_reflection: str = dspy.OutputField()
    blocks_to_edit: list[Block] = dspy.OutputField(desc="Empty if no blocks need to be edited")

#################
# Single Block
#################

class ClassifyBlockToEdit(dspy.Signature):
    """
    Determine whether a block of code requires direct modification during the given task. A block only requires direct modification if the code directly visible inside it needs to be changed.
    """
    codebase_summary: str = dspy.InputField()
    codebase_symbol_explanations: str = dspy.InputField()
    general_context: str = dspy.InputField()
    task: str = dspy.InputField()
    specific_context: str = dspy.InputField()
    task_reflection: str = dspy.OutputField(desc="Think step-by-step about what the task is asking for")
    reasoning: str = dspy.OutputField(desc="Debate whether or not this block must be directly modified during the task")
    requires_direct_modification: bool = dspy.OutputField()
    confidence: Literal["low", "medium", "high", "certain"] = dspy.OutputField(desc="Your confidence that this block must be directly modified")

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

def load_trainset_batched(n_tasks: Optional[int] = None, window_size: int = 6):
    """
    window_size: number of adjacent block numbers to include per prompt.
    """
    hvm_names, hs_nodes, c_nodes = get_all_names()
    only_definitions_code_contexts = {name: create_contexts_for_name(name, hs_nodes, c_nodes, definitions_only=True) for name in hvm_names}
    real_tasks = load_real_tasks()
    codebase_summary = load_codebase_summary()
    sym_exps = load_symbol_explanations()
    sym_exps = {exp["name"]: exp["explanation"] for exp in sym_exps}
    all_block_numbers = sorted(get_all_block_numbers())
    all_examples = []
    blocks_batches = [all_block_numbers[i:i + window_size] for i in range(0, len(all_block_numbers), window_size)]
    for _, task in list(real_tasks.items())[:n_tasks]:
        task_examples = []
        for blocks_batch in blocks_batches:
            # NOTE: important
            # general context about the task gets shown with a window of blocks that are currently being evaluated
            # the idea is to ground the assessment of specific blocks in general context that is likely to be highly relevant
            # this is expected to prevent the model from making poor judgements about which blocks likely require direct modification
            task_general_context = format_contexts(*merge_contexts_for_names(task["related_symbols"], only_definitions_code_contexts))
            task_general_context = hide_block_numbers(set(all_block_numbers) - set(blocks_batch), task_general_context)
            # specific context for this example - a contiguous window of blocks for evaluation
            task_specific_context = format_contexts(*create_contexts_for_blocks(blocks_batch))
            task_specific_context = hide_block_numbers(set(all_block_numbers) - set(blocks_batch), task_specific_context)
            # check that the correct blocks are being evaluated
            block_numbers: set[int] = find_block_numbers(task_specific_context) | find_block_numbers(task_general_context)
            assert block_numbers == set(blocks_batch), f"expected {blocks_batch}, got {block_numbers}"
            expected_blocks_to_edit = [x for x in task["blocks_to_edit"] if x["block_number"] in blocks_batch]
            if task["related_symbols"]:
                task_examples.append(dspy.Example(
                    codebase_summary=codebase_summary,
                    codebase_symbol_explanations="\n".join([f"{name}: {sym_exps[name]}" for name in task["related_symbols"]]),
                    general_context=task_general_context,
                    specific_context=task_specific_context,
                    task=task["task"],
                    # TODO:!!!
                    # task_reflection=task["task_reflection"],
                    blocks_to_edit=expected_blocks_to_edit,
                    block_numbers=", ".join([str(num) for num in blocks_batch])
                ).with_inputs("codebase_summary", "codebase_symbol_explanations", "general_context", "specific_context", "task", "block_numbers")
            )
        all_examples.extend(task_examples)
    return all_examples

def convert_confidence_to_num(confidence: float | Literal["low", "medium", "high", "certain"]):
    if isinstance(confidence, float): return confidence
    return {"low": 0.25, "medium": 0.5, "high": 0.75, "certain": 0.9}[confidence]

def filter_block_nums(blocks: list[Block]):
    return sorted([block["block_number"] for block in blocks if block["requires_direct_modification"] and convert_confidence_to_num(block["confidence"]) >= 0.75])

def jaccard_similarity(example, pred, trace=None):
    expected_blocks_to_edit = set(filter_block_nums(example.blocks_to_edit))
    actual_blocks_to_edit = set(filter_block_nums(pred.blocks_to_edit))
    union = expected_blocks_to_edit | actual_blocks_to_edit
    if not union: return 1.0
    intersection = expected_blocks_to_edit & actual_blocks_to_edit
    return len(intersection) / len(union)

def optimize(devset, task_lm, prompt_lm, teacher_lm):
    program = dspy.Predict(ClassifyBlocksToEdit)
    with dspy.context(lm=task_lm):
        optimizer = dspy.teleprompt.MIPROv2(
            metric=jaccard_similarity,
            auto="light",
            prompt_model=prompt_lm,
            task_model=task_lm,
            max_bootstrapped_demos=1,
            max_labeled_demos=6,
            num_threads=6,
            hide_demo_fields=[
                "codebase_summary",
                "codebase_symbol_explanations",
                "general_context",
                "specific_context"
            ],
            teacher_settings=dict(lm=teacher_lm),
        )
        optimized_program = optimizer.compile(program, trainset=devset)
        optimized_program.save(get_optimized_program_path(__file__))

###########
# Inference
###########

def classify_block(model, example):
    program = dspy.Predict(ClassifyBlocksToEdit)
    with dspy.context(lm=model):
        result = program(**example.inputs().toDict())
    return result

def classify_blocks(model, examples):
    program = dspy.Predict(ClassifyBlocksToEdit)
    with dspy.context(lm=model, async_max_workers=20):
        results = asyncio.run(run_dspy_parallel(program, examples))
    blocks_to_edit = []
    for result in results: blocks_to_edit.extend(result.blocks_to_edit)
    return results, sorted(blocks_to_edit, key=lambda b: b.block_number)
    
def classify_blocks_single(model, examples):
    program = dspy.Predict(ClassifyBlockToEdit)
    with dspy.context(lm=model):
        results = asyncio.run(run_dspy_parallel(program, examples))
    blocks_to_edit = []
    for result in results: blocks_to_edit.extend(result.blocks_to_edit)
    return results, sorted(blocks_to_edit, key=lambda b: b.block_number)

## window = 6, confidence >= 0.9
# deepseek = [21, 87, 116, 121, 124, 140, 146, 158, 189, 211, 213, 231, 232, 242, 243, 245, 284, 285, 286, 287, 288, 289, 294, 295, 296, 297, 298, 300, 303, 304, 305, 307, 308, 322, 324, 330, 331, 332, 333, 334, 344, 345, 346, 347, 360, 366, 394, 395, 403, 408, 420, 421, 422, 423, 438, 480, 484, 485, 486, 491, 507]
# gemini = [21, 27, 37, 78, 79, 80, 81, 82, 83, 87, 88, 90, 91, 92, 94, 95, 96, 97, 98, 99, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 120, 121, 122, 123, 124, 125, 140, 146, 180, 181, 182, 183, 184, 185, 188, 189, 190, 192, 194, 204, 205, 206, 207, 210, 211, 213, 224, 231, 237, 238, 242, 243, 246, 251, 252, 253, 267, 282, 284, 285, 286, 287, 288, 289, 294, 295, 296, 297, 298, 299, 300, 301, 302, 305, 306, 307, 308, 309, 310, 311, 324, 330, 331, 332, 333, 348, 349, 350, 360, 361, 362, 364, 365, 366, 367, 370, 371, 396, 397, 398, 408, 414, 415, 416, 417, 418, 419, 435, 436, 437, 438, 439, 450, 460, 480, 484, 490, 491]
# the potential ones gem missed were 322 and 158, then included a lot of false positives
## window = 3, confidence >= 0.9
# deepseek = [21, 27, 87, 92, 98, 99, 108, 116, 124, 138, 139, 140, 141, 146, 189, 216, 231, 242, 243, 253, 255, 280, 285, 286, 287, 288, 289, 294, 295, 296, 300, 304, 305, 306, 307, 308, 321, 322, 324, 330, 331, 332, 333, 349, 350, 355, 360, 366, 396, 397, 398, 408, 419, 438, 480, 484, 485, 486, 491, 510]
# gemini = [17, 21, 22, 27, 37, 44, 69, 70, 71, 77, 84, 85, 86, 87, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 116, 117, 120, 121, 122, 124, 136, 138, 139, 140, 141, 146, 158, 186, 187, 188, 189, 190, 192, 193, 204, 205, 206, 210, 211, 212, 216, 217, 218, 219, 220, 221, 222, 223, 224, 231, 232, 233, 237, 238, 239, 242, 243, 244, 245, 246, 252, 253, 254, 267, 268, 269, 282, 284, 285, 286, 287, 294, 295, 296, 300, 301, 302, 309, 310, 311, 321, 322, 323, 324, 330, 331, 332, 333, 348, 349, 350, 355, 360, 362, 363, 364, 365, 366, 367, 375, 376, 378, 379, 380, 393, 394, 395, 396, 397, 398, 399, 400, 401, 408, 409, 423, 438, 439, 450, 459, 460, 461, 483, 484, 489, 490, 491, 507]

# 3 blocks, no optimizing, no filtering, jsim ~54%
# 3 blocks, instructions only, no filtering, jsim ~75% (but instructions are overfitted)
# 1 block, gemini, no optimizing, no filtering, jsim ~75% (note - using the batched version, just batch of one), ~76% with conf filtering
# 1 block, gemini, no optimizing, WITH filtering, jsim ~76%, batch size of one, ~88% with conf filtering!!!
# 2 blocks, gemini, no optimizing, WITH filtering, jsim ~76%, batch size of one, ~81% with conf filtering - so 7% worse than 1 block
# 1 block, deepseek, no optimizing, no filtering, jsim ~73% (note - using the batched version, just batch of one), 


if __name__ == "__main__":
    examples = load_trainset_batched(n_tasks=1, window_size=1)
    real_tasks = load_real_tasks()
    task = list(real_tasks.values())[0]
    results1, blocks1 = classify_blocks(gemini_8b, examples)
    scores = []
    # print(blocks1)
    # for block in blocks1:
    #     print(block)
    #     print(get_block_code(block.block_number))
    #     print('---')
    for (result, example) in zip(results1, examples):
        score = jaccard_similarity(example, result)
        scores.append(score)
    print(f"score: {sum(scores) / len(scores)}")