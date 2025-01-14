import asyncio
import random
from typing import Literal
import dspy
import dspy.teleprompt
from src.my_datasets import load_real_tasks, load_symbol_explanations
from src.filesystem import get_optimized_program_path
from src.utils import convert_confidence_to_num, run_dspy_parallel

class ClassifySymbol(dspy.Signature):
    """Decide whether the codebase symbol is essential context for the task"""
    symbol: str = dspy.InputField()
    explanation: str = dspy.InputField()
    task: str = dspy.InputField()
    reasoning: str = dspy.OutputField()
    is_related: bool = dspy.OutputField()
    confidence: Literal["low", "medium", "high", "very high"] = dspy.OutputField()

##########
# Training
##########

def load_symbol_task_relatedness_trainset(shuffle=True):
    devset = []
    sym_exps = load_symbol_explanations()
    sym_exps = {row['name']: row['explanation'] for row in sym_exps}
    for task, task_data in load_real_tasks().items():
        for sym, exp in sym_exps.items():
            devset.append(
                dspy.Example(
                    task=task,
                    symbol=sym,
                    explanation=exp,
                    is_related=sym in task_data["related_symbols"]
                ).with_inputs('task', 'symbol', 'explanation')
            )
    false_items = [example for example in devset if not example.is_related]
    true_items = [example for example in devset if example.is_related]
    k = min(len(false_items), len(true_items))
    devset = random.sample(false_items, k) + random.sample(true_items, k)
    print(f"balanced devset: {len(devset)} items, {len(false_items)} false, {len(true_items)} true")
    if shuffle: random.shuffle(devset)
    return devset

def direct_score(example, pred, trace=None):
    predicted = pred.is_related and convert_confidence_to_num(pred.confidence) >= 0.75
    actual = example.is_related
    return int(predicted == actual)

def optimize(devset, task_lm, prompt_lm, teacher_lm):
    program = dspy.Predict(ClassifySymbol)
    with dspy.context(lm=task_lm):
        optimizer = dspy.teleprompt.MIPROv2(
            metric=direct_score,
            auto="light",
            prompt_model=prompt_lm,
            task_model=task_lm,
            max_bootstrapped_demos=6,
            max_labeled_demos=0,
            num_threads=12,
            teacher_settings=dict(lm=teacher_lm)
        )
        optimized_classify = optimizer.compile(program, trainset=devset)
        optimized_classify.save(get_optimized_program_path(__file__))
    return optimized_classify

###########
# Inference
###########

def classify_symbols(examples: list[dspy.Example], model: dspy.LM, async_max_workers=50, cache=True):
    program = dspy.Predict(ClassifySymbol)
    if get_optimized_program_path(__file__).exists():
        program.load(get_optimized_program_path(__file__))
    with dspy.context(lm=model, async_max_workers=async_max_workers, cache=cache):
        results = asyncio.run(run_dspy_parallel(program, examples))
    related_symbols = [example.symbol for example, result in zip(examples, results)
                       if result.is_related and convert_confidence_to_num(result.confidence) >= 0.75]
    return related_symbols