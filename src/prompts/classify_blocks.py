import dspy
from pydantic import BaseModel
from src.my_datasets import load_codebase_summary, load_real_tasks, load_symbol_explanations
from src.file_context import FileContext, get_all_block_numbers, create_contexts_for_name, find_block_numbers, format_contexts, get_all_names, hide_block_numbers

class Block(BaseModel):
    block_number: int = dspy.OutputField()
    reasoning: str = dspy.OutputField(desc="The reason why this block must be directly modified")
    requires_direct_modification: bool = dspy.OutputField(desc="True if this block must be directly modified, False otherwise")
    confidence: float = dspy.OutputField(ge=0, le=1)

class ClassifyBlocksToEdit(dspy.Signature):
    """Determine which BLOCK numbers require direct modification."""
    codebase_summary: str = dspy.InputField()
    codebase_symbol_explanations: str = dspy.InputField()
    codebase_context: str = dspy.InputField()
    task: str = dspy.InputField()
    block_numbers: str = dspy.InputField()
    task_reflection: str = dspy.OutputField()
    blocks_to_edit: list[Block] = dspy.OutputField(desc="Empty if no blocks need to be edited")

##########
# Training
##########

def merge_contexts(*contexts: tuple[FileContext, FileContext]):
    merged_hs_ctx = None
    merged_c_ctx = None
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

def load_trainset():
    hvm_names, hs_nodes, c_nodes = get_all_names()
    full_code_contexts = {name: create_contexts_for_name(name, hs_nodes, c_nodes) for name in hvm_names}
    only_definitions_code_contexts = {name: create_contexts_for_name(name, hs_nodes, c_nodes, definitions_only=True) for name in hvm_names}
    real_tasks = load_real_tasks()
    codebase_summary = load_codebase_summary()
    sym_exps = load_symbol_explanations()
    sym_exps = {exp["name"]: exp["explanation"] for exp in sym_exps}
    all_block_numbers = get_all_block_numbers()
    all_examples = []
    for _, task in real_tasks.items():
        task_examples = []
        for name in hvm_names:
            # NOTE: important
            # general context about the task based on the AI's assessment of codebase symbol explanations
            # gets merged with specific context about the symbol that is currently being evaluated
            # the idea is to ground the assessment of specific symbols in general context that is likely to be highly relevant
            # this is expected to prevent the model from making poor judgements about which blocks are likely to be edited
            task_general_context = merge_contexts_for_names(task["related_symbols"], only_definitions_code_contexts)
            blocks_in_general_context = find_block_numbers(format_contexts(*task_general_context))
            task_specific_context = merge_contexts_for_names([name], full_code_contexts)
            blocks_in_specific_context = find_block_numbers(format_contexts(*task_specific_context))
            task_full_context = hide_block_numbers(
                # hide blocks numbers in the general context that are not in specific context
                # ie. focus on the context of the specific symbol that is being evaluated
                blocks_in_general_context - blocks_in_specific_context,
                format_contexts(*merge_contexts(task_general_context, task_specific_context))
            )
            block_numbers = find_block_numbers(task_full_context)
            if task["related_symbols"]:
                task_examples.append(dspy.Example(
                    codebase_summary=codebase_summary,
                    codebase_symbol_explanations="\n".join([f"{name}: {sym_exps[name]}" for name in task["related_symbols"] + [name]]),
                    codebase_context=task_full_context,
                    task=task["task"],
                    block_numbers=", ".join([str(num) for num in block_numbers]),
                    blocks_to_edit=task["blocks_to_edit"]
                ).with_inputs("codebase_summary", "codebase_symbol_explanations", "codebase_context", "task", "block_numbers")
            )
        # check that all block numbers are in the dataset
        all_blocks_seen_in_task = set()
        for task_example in task_examples:
            found_block_numbers = find_block_numbers(task_example.codebase_context)
            all_blocks_seen_in_task.update(found_block_numbers)
        difference = all_block_numbers - all_blocks_seen_in_task
        if difference:
            print(f"WARNING: {difference} blocks not seen in task {task['task']}")
        all_examples.extend(task_examples)
    return all_examples

###########
# Inference
###########

def classify_block(model, example):
    program = dspy.Predict(ClassifyBlocksToEdit)
    with dspy.context(lm=model):
        result = program(**example.inputs().toDict())
    return result