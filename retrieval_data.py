import asyncio
import random
import dspy
import dspy.teleprompt
from llms import get_lm
from utils import load_json, load_jsonl, run_dspy_parallel, save_json

###############################
# determine relatedness
# works w/ 8b models
# works best w/ gemini-flash-8b
###############################

class DetermineRelatedness(dspy.Signature):
    """Use the codebase symbol explanation to decide whether it is directly involved in the task"""
    symbol: str = dspy.InputField()
    explanation: str = dspy.InputField()
    task: str = dspy.InputField()
    is_related: bool = dspy.OutputField()
    confidence: float = dspy.OutputField(ge=0.0, le=1.0)

def load_symbol_task_relatedness_trainset(shuffle=True):
    devset = []
    symbol_explanations = [(row['name'], row['explanation']) for row in load_jsonl("symbol_explanations.jsonl")]
    for task, task_data in load_json("hvm3_real_tasks.json").items():
        for sym_exp in symbol_explanations:
            symbol, explanation = sym_exp
            devset.append(
                dspy.Example(
                    task=task,
                    symbol=symbol,
                    explanation=explanation,
                    is_related=symbol in task_data["related_symbols"]
                ).with_inputs('task', 'symbol', 'explanation')
            )
    if shuffle: random.shuffle(devset)
    return devset

def score_response(example, pred, trace=None):
    return 1 if example.is_related == pred.is_related else 0

# llama8b gets ~73% accuracy on this unoptimized
# gemini-flash-8b gets ~83% accuracy on this unoptimized
# gemini-flash-8b gets ~90% accuracy on this optimized

def optimize_for(program, devset, task_lm, prompt_lm, save_name):
    with dspy.context(lm=task_lm):
        tp = dspy.teleprompt.MIPROv2(metric=score_response, auto="light", prompt_model=prompt_lm, task_model=task_lm, max_bootstrapped_demos=0, max_labeled_demos=0, num_threads=20)
        optimized_classify = tp.compile(program, trainset=devset)
        optimized_classify.save(f"optimized_classify_{save_name}.json")
    return optimized_classify

def evaluate(model, program, devset):
    with dspy.context(lm=model):
        dspy.Evaluate(devset=devset, metric=score_response, num_threads=8, display_progress=True, display_table=True)(program)

def get_related_symbols(tasks):
    symbol_explanations = [(row['name'], row['explanation']) for row in load_jsonl("symbol_explanations.jsonl")]
    examples = []
    for task in tasks:
        for symbol, explanation in symbol_explanations:
            examples.append(
                dspy.Example(
                    task=task,
                    symbol=symbol,
                    explanation=explanation,
                ).with_inputs('task', 'symbol', 'explanation')
            )
    dspy.configure(lm=get_lm("gemini/gemini-1.5-flash-8b"), async_max_workers=100)
    program = dspy.ChainOfThoughtWithHint(DetermineRelatedness)
    program.load(f"optimized_classify_gemini.pkl")
    results = asyncio.run(run_dspy_parallel(program, examples))
    task_to_related_symbols = {}
    for example, result in zip(examples, results):
        if example.task not in task_to_related_symbols:
            task_to_related_symbols[example.task] = {"related_symbols": []}
        if result.is_related:
            task_to_related_symbols[example.task]["related_symbols"].append(example.symbol)
    return task_to_related_symbols

###########################################################################
# create fake tasks
# works fine with deepseek
# tasks only used for training retriever - don't need to be ultra-realistic
###########################################################################

class CreateFakeTasks(dspy.Signature):
    """
    Use the codebase symbol explanation to create a new tasks in the same style as the examples.
    Don't refer directly to the symbol names. Only refer to them indirectly.
    """
    examples: str = dspy.InputField()
    explanations: str = dspy.InputField()
    tasks: list[str] = dspy.OutputField(desc="A list of tasks referring indirectly to the symbol names")

def load_create_fake_tasks_trainset(shuffle=True):
    devset = []
    symbol_explanations = [(row['name'], row['explanation']) for row in load_jsonl("symbol_explanations.jsonl")]
    batch_size = 10
    symbol_explanations_batched = [symbol_explanations[i:i+batch_size] for i in range(0, len(symbol_explanations), batch_size)]
    real_tasks = load_json("hvm3_real_tasks.json").keys()
    for sym_exp_batch in symbol_explanations_batched:
        devset.append(
            dspy.Example(
                examples="\n".join(real_tasks),
                explanations="\n\n".join([f"{sym_exp[0]}: {sym_exp[1]}" for sym_exp in sym_exp_batch]),
            ).with_inputs('examples', 'explanations')
        )
    if shuffle: random.shuffle(devset)
    return devset

def load_fake_tasks():
    return load_json("hvm3_fake_tasks.json")

def create_fake_tasks(trainset, save=False):
    with dspy.context(lm=get_lm("deepseek/deepseek-chat"), async_max_workers=100):
        program = dspy.Predict(CreateFakeTasks)
        results = asyncio.run(run_dspy_parallel(program, trainset))
    fake_tasks = load_fake_tasks()
    for result in results:
        for task in result.tasks:
            fake_tasks[task] = {}
    if save: save_json(fake_tasks, "hvm3_fake_tasks.json")
    return results

###########################################################################
# train retriever and evaluate vs LM
###########################################################################



if __name__ == "__main__":
    fake_tasks = load_fake_tasks()
    related_symbols = get_related_symbols(fake_tasks.keys())
    save_json(related_symbols, "related_symbols.json")
