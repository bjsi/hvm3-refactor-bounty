import asyncio
import random
import dspy
import dspy.teleprompt
from src.my_datasets import load_real_tasks, load_symbol_explanations
from src.filesystem import get_optimized_program_path
from src.file_context import get_all_names
from src.utils import run_dspy_parallel

class DetermineRelatedness(dspy.Signature):
    """Decide whether the codebase symbol is essential context for the task"""
    symbol: str = dspy.InputField()
    explanation: str = dspy.InputField()
    task: str = dspy.InputField()
    is_related: bool = dspy.OutputField()
    confidence: float = dspy.OutputField(ge=0.0, le=1.0)

##########
# Training
##########

def load_symbol_task_relatedness_trainset(shuffle=True):
    devset = []
    context_explanations = load_symbol_explanations()
    context_explanations = {row['name']: row['explanation'] for row in context_explanations}
    for task, task_data in load_real_tasks().items():
        for sym, exp in context_explanations.items():
            devset.append(
                dspy.Example(
                    task=task,
                    symbol=sym,
                    explanation=exp,
                    is_related=sym in task_data["related_symbols"]
                ).with_inputs('task', 'symbol', 'explanation')
            )
    if shuffle: random.shuffle(devset)
    return devset

def score_response(example, pred, trace=None):
    if example.is_related == pred.is_related: score = 1
    elif not example.is_related and pred.is_related: score = 0.25 # lenient to false positives
    else: score = 0.0
    return score

def optimize_for(program, devset, task_lm, prompt_lm):
    with dspy.context(lm=task_lm):
        tp = dspy.teleprompt.MIPROv2(
            metric=score_response,
            auto="auto",
            prompt_model=prompt_lm,
            task_model=task_lm,
            max_bootstrapped_demos=0,
            max_labeled_demos=0,
            num_threads=20
        )
        optimized_classify = tp.compile(program, trainset=devset)
        optimized_classify.save(get_optimized_program_path(__file__))
    return optimized_classify

def evaluate(model, program, devset):
    with dspy.context(lm=model, cache=False):
        dspy.Evaluate(devset=devset, metric=score_response, num_threads=20, display_progress=True, display_table=True)(program)

###########
# Inference
###########

def get_related_symbols(program, model, tasks, explanations):
    symbols = get_all_names()[0]
    examples = []
    for task in tasks:
        for sym in symbols:
            examples.append(dspy.Example(task=task, symbol=sym, explanation=explanations[sym]).with_inputs('task', 'symbol', 'explanation'))
    dspy.configure(lm=model, async_max_workers=300, cache=False)
    results = asyncio.run(run_dspy_parallel(program, examples))
    task_to_related_symbols = {}
    for example, result in zip(examples, results):
        print(f"example: {example.symbol} related {result.is_related} reason {result.reasoning if 'reasoning' in result else result.rationale}")
        if example.task not in task_to_related_symbols:
            task_to_related_symbols[example.task] = {"related_symbols": []}
        if result.is_related and result.confidence >= 0.95:
            task_to_related_symbols[example.task]["related_symbols"].append(example.symbol)
    return task_to_related_symbols