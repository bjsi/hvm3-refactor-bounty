import asyncio
import random
import sys
import time
import dspy
from typing import Literal
import dspy.teleprompt
from src.filesystem import get_optimized_program_path
from src.my_datasets import load_binary_classification_judge_data, save_binary_classification_judge_data, save_new_incumbent_data
from src.prompts.classify_blocks import classify_blocks, convert_confidence_to_num, load_trainset
from src.utils import run_dspy_parallel
from src.llms import deepseek_chat

class BinaryClassificationTiebreaker(dspy.Signature):
    """You are a judge tasked with resolving a tie between two programmers' predictions about whether a code block requires direct modification.  You will be given the problem context, the two programmers' reasoning, and their predictions.  You will then need to determine which prediction is correct, and provide your reasoning for doing so."""
    # problem context
    task: str = dspy.InputField()
    task_reflection: str = dspy.InputField()
    specific_context: str = dspy.InputField()
    # current dataset
    programmer_1_reasoning: str = dspy.InputField()
    programmer_1_requires_direct_modification: bool = dspy.InputField()
    # new predictions
    programmer_2_reasoning: str = dspy.InputField()
    programmer_2_requires_direct_modification: bool = dspy.InputField()
    # resolve tie
    reasoning_for_modification: str = dspy.OutputField()
    reasoning_against_modification: str = dspy.OutputField()
    requires_direct_modification: bool = dspy.OutputField()
    confidence: Literal["low", "medium", "high", "very high"] = dspy.OutputField(desc="Your confidence that code in the block must be directly modified")

def tiebreak(model, disagreements: list[tuple[dspy.Example, dspy.Prediction]]):
    input_keys = [
        "codebase_summary",
        "codebase_symbol_explanations",
        "task",
        "specific_context",
        "task_reflection",
        "programmer_1_reasoning",
        "programmer_1_requires_direct_modification",
        "programmer_2_reasoning",
        "programmer_2_requires_direct_modification"
    ]
    examples = [dspy.Example(
        codebase_summary=example.codebase_summary,
        codebase_symbol_explanations=example.codebase_symbol_explanations,
        task=example.task,
        specific_context=example.specific_context,
        task_reflection=example.task_reflection,
        # example
        programmer_1_reasoning=example.reasoning,
        programmer_1_requires_direct_modification=example.requires_direct_modification,
        # prediction
        programmer_2_reasoning=prediction.reasoning,
        programmer_2_requires_direct_modification=prediction.requires_direct_modification,
    ).with_inputs(*input_keys) for (example, prediction) in disagreements]
    program = dspy.Predict(BinaryClassificationTiebreaker)
    # load the optimized judge if it exists
    if get_optimized_program_path(__file__).exists():
        print("Using optimized judge")
        program.load(get_optimized_program_path(__file__))
    with dspy.context(lm=model, async_max_workers=15):
        return asyncio.run(run_dspy_parallel(program, examples, desc="Tiebreaking"))
    
def direct_score(example, prediction, trace=None):
    return example.requires_direct_modification == prediction.requires_direct_modification

def optimize_judge(model):
    """Optimize the judge itself on the most challenging pairs."""
    judge_dataset = load_binary_classification_judge_data()
    input_keys = [
        "task",
        "specific_context",
        "task_reflection",
        "programmer_1_reasoning",
        "programmer_1_requires_direct_modification",
        "programmer_2_reasoning",
        "programmer_2_requires_direct_modification"
    ]
    examples = [dspy.Example(judge_dataset[i]).with_inputs(*input_keys) for i in range(len(judge_dataset))]
    random.seed(42)
    random.shuffle(examples) # important eg. for data aware proposer, but still get the cache
    program = dspy.Predict(BinaryClassificationTiebreaker)
    # hmmm, I don't know if this benefits the optimization process vs doing it from scratch
    # if get_optimized_program_path(__file__).exists(): program.load(get_optimized_program_path(__file__))
    with dspy.context(lm=model, async_max_workers=30):
        optimizer = dspy.teleprompt.MIPROv2(
            metric=direct_score,
            auto="light",
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

# if no data:
    # new_incumbent_data = []
    # for (incumbent_prediction, challenger_prediction) in zip(incumbent_dataset, challenger_dataset):
    #     new_incumbent_data.append(
    #         {
    #             "block_number": incumbent_prediction.block_number,
    #             "reasoning": challenger_prediction.reasoning,
    #             "requires_direct_modification": challenger_prediction.requires_direct_modification,
    #             "confidence": convert_confidence_to_num(challenger_prediction.confidence)
    #         })
    # save_new_incumbent_data(new_incumbent_data)

if __name__ == "__main__":
    # optimize_judge(deepseek_chat)
    incumbent_dataset = load_trainset(lambda tasks: tasks[3:4])
    # deepseek_chat.cache = False
    challenger_dataset = classify_blocks(deepseek_chat, incumbent_dataset)
    ties = [
        (incumbent_prediction, challenger_prediction)
        for incumbent_prediction, challenger_prediction in zip(incumbent_dataset, challenger_dataset)
        if incumbent_prediction.requires_direct_modification != challenger_prediction.requires_direct_modification
    ]
    print(f"Number of ties: {len(ties)}")
    print("Tiebreaking...")
    judgements = tiebreak(deepseek_chat, ties)
    scores = []
    losses = []
    not_confident = []
    for (incumbent_prediction, challenger_prediction), judgement in zip(ties, judgements):
        print(f"Incumbent: {incumbent_prediction.reasoning}\n")
        print(f"Challenger: {challenger_prediction.reasoning}\n")
        print(f"Judge for: {judgement.reasoning_for_modification} against: {judgement.reasoning_against_modification}")
        if convert_confidence_to_num(judgement.confidence) >= 0.75:
            if judgement.requires_direct_modification == incumbent_prediction.requires_direct_modification:
                scores.append(1)
                print("Incumbent wins")
            else:
                scores.append(0)
                print("Challenger wins")
                losses.append((incumbent_prediction, challenger_prediction, judgement))
        else:
            print("Judge is not confident")
            not_confident.append((incumbent_prediction, challenger_prediction, judgement))
    print(f"Incumbent win rate: {sum(scores) / max(len(scores), 1)}")
    print(f"Judge not confident: {len([s for s in scores if s == 0])}")

    cases_to_review = losses + not_confident
    print(f"Number of cases to review: {len(cases_to_review)}")
    time.sleep(5)
    # sys.exit()
    new_incumbent_data = []
    new_judge_data = []
    for i, (incumbent_prediction, challenger_prediction, judgement) in list(enumerate(cases_to_review)):
        review = review_case(incumbent_prediction, challenger_prediction, judgement)
        if review is None: continue
        judge_data = {
            "block_number": incumbent_prediction.block_number,
            "task": incumbent_prediction.task,
            "specific_context": incumbent_prediction.specific_context,
            "task_reflection": incumbent_prediction.task_reflection,
            "programmer_1_reasoning": incumbent_prediction.reasoning,
            "programmer_1_requires_direct_modification": incumbent_prediction.requires_direct_modification,
            "programmer_2_reasoning": challenger_prediction.reasoning,
            "programmer_2_requires_direct_modification": challenger_prediction.requires_direct_modification,
            "reasoning_for_modification": judgement.reasoning_for_modification,
            "reasoning_against_modification": judgement.reasoning_against_modification,
            "requires_direct_modification": review.requires_direct_modification,
            "confidence": "very high" # because we've reviewed it
        }
        new_judge_data.append(judge_data)
        # TODO: should prob account for confidence
        if review.requires_direct_modification != incumbent_prediction.requires_direct_modification:
            new_incumbent_data.append(
                {
                    "block_number": incumbent_prediction.block_number,
                    "reasoning": review.reasoning,
                    "requires_direct_modification": review.requires_direct_modification,
                    "confidence": review.confidence,
            })

    save_new_incumbent_data(new_incumbent_data)
    existing_judge_data = load_binary_classification_judge_data()
    updated_judge_data = existing_judge_data + new_judge_data
    save_binary_classification_judge_data(updated_judge_data)

    # Check for duplicates in judge data by task + block number
    seen_pairs = {}
    for entry in updated_judge_data:
        key = (entry["task"], entry["block_number"])
        if key in seen_pairs:
            # Check if the decisions conflict
            if entry["requires_direct_modification"] != seen_pairs[key]["requires_direct_modification"]:
                print(f"\nWARNING: Conflicting judge decisions found:")
                print(f"Task: {entry['task']}")
                print(f"Block: {entry['block_number']}")
                print("Please review these cases manually.\n")
        else:
            seen_pairs[key] = entry