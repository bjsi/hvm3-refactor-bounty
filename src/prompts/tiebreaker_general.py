import asyncio
import random
from typing import Literal
import dspy

from src.filesystem import get_optimized_program_path
from src.utils import run_dspy_parallel

class BinaryClassificationTiebreaker(dspy.Signature):
    """You are a judge tasked with resolving a tie between two binary predictions"""
    # problem context
    context: str = dspy.InputField()
    # resolve tie
    reasoning_for_true: str = dspy.OutputField()
    reasoning_for_false: str = dspy.OutputField()
    prediction: bool = dspy.OutputField()
    confidence: Literal["low", "medium", "high", "very high"] = dspy.OutputField()

def tiebreak(model, disagreements: list[tuple[dspy.Example, dspy.Prediction]]):
    input_keys = [
        "context",
        "reasoning_for_true",
        "reasoning_for_false",
    ]
    examples = [dspy.Example(
        context=example.context,
        reasoning_for_true=example.reasoning_for_true,
        reasoning_for_false=example.reasoning_for_false
    ).with_inputs(*input_keys) for example, prediction in disagreements]
    program = dspy.Predict(BinaryClassificationTiebreaker)
    with dspy.context(lm=model, async_max_workers=15):
        return asyncio.run(run_dspy_parallel(program, examples, desc="Tiebreaking"))

def direct_score(example, prediction, trace=None):
    return example.requires_direct_modification == prediction.requires_direct_modification

def optimize_judge(model, dataset):
    """Optimize the judge itself on the most challenging pairs."""
    input_keys = [
        "context",
        "reasoning_for_true",
        "reasoning_for_false",
    ]
    examples = [dspy.Example(dataset[i]).with_inputs(*input_keys) for i in range(len(dataset))]
    random.seed(42)
    random.shuffle(examples) # important eg. for data aware proposer, but still get the cache
    program = dspy.Predict(BinaryClassificationTiebreaker)
    with dspy.context(lm=model, async_max_workers=30):
        optimizer = dspy.teleprompt.MIPROv2(
            metric=direct_score,
            prompt_model=model,
            task_model=model,
            max_bootstrapped_demos=1,
            max_labeled_demos=2 if len(examples) < 10 else 4,
            num_threads=6,
            hide_demo_fields=[
                "codebase_summary",
                "codebase_symbol_explanations",
            ]
        )
        optimized_program = optimizer.compile(program, trainset=examples)
        # TODO:
        optimized_program.save(get_optimized_program_path(__file__))

# basic cli for human in the loop data collection
# it will prompt you to review challenging cases
def review_case(incumbent, challenger, judgement):
    print("\033[H\033[J", end="")  # Clear console using ANSI escape codes
    print("\nTask:")
    print(incumbent.task)
    print("\nContext:")
    print(incumbent.specific_context)
    print("\nJudge Decision:")
    def color(s, color):
        if color == "green": return f"\033[32m{s}\033[0m"
        elif color == "red": return f"\033[31m{s}\033[0m"
        else: return s
    print(f"Requires modification: {color(judgement.requires_direct_modification, 'green' if judgement.requires_direct_modification else 'red')}")
    print(f"Confidence: {judgement.confidence}")
    print(f"Reasoning for: {color(judgement.reasoning_for_modification, 'green' if judgement.requires_direct_modification else '')}")
    print(f"Reasoning against: {color(judgement.reasoning_against_modification, 'red' if not judgement.requires_direct_modification else '')}")
    while True:
        agree = input("\nDo you agree with the judge? (y/n/s to skip): ").lower()
        if agree == 'y':
            choice = incumbent if judgement.requires_direct_modification == incumbent.requires_direct_modification else challenger
            # use the judge's reasoning
            reasoning = judgement.reasoning_for_modification if judgement.requires_direct_modification else judgement.reasoning_against_modification
            choice.reasoning = reasoning
            return choice
        elif agree == 'n':
            choice = incumbent if judgement.requires_direct_modification != incumbent.requires_direct_modification else challenger
            # use the judge's other reasoning
            reasoning = judgement.reasoning_against_modification if judgement.requires_direct_modification else judgement.reasoning_for_modification
            choice.reasoning = reasoning
            return choice
        elif agree == 's':
            return None