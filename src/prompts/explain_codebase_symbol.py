import asyncio
from typing import Literal
import dspy
import dspy.teleprompt
from src.filesystem import get_optimized_program_path, data_dir
from src.file_context import create_contexts_for_name, format_contexts, get_all_names
from src.llms import deepseek_chat, gemini_8b
from src.utils import load_codebase_summary, load_jsonl, run_dspy_parallel

# my first attempt at a judge, ./classify_blocks_tiebreaker.py is better
# that should be cleaned, generalized and replace the one here too

class ExplainCodebaseSymbol(dspy.Signature):
    codebase_summary: str = dspy.InputField()
    codebase_symbol: str = dspy.InputField()
    codebase_context: str = dspy.InputField()
    explanation: str = dspy.OutputField(desc="A concise but detailed paragraph for a technical audience that explains the codebase symbol's role, purpose and behavior.")

##########
# Training
##########

def load_trainset(explanations_version: Literal["v1", "v2", "v3"]):
    hvm_names, hs_nodes, c_nodes = get_all_names()
    summary = load_codebase_summary()
    code_contexts = {name: format_contexts(*create_contexts_for_name(name, hs_nodes, c_nodes)) for name in hvm_names}
    sym_exps = load_jsonl(data_dir / f"symbol_explanations_{explanations_version}.jsonl")
    sym_exps = {exp["name"]: exp["explanation"] for exp in sym_exps}
    examples = [dspy.Example(explanation=sym_exps[name], codebase_summary=summary, codebase_symbol=name, codebase_context=code_contexts[name]).with_inputs("codebase_summary", "codebase_symbol", "codebase_context") for name in hvm_names]
    return examples

class ExplanationJudge(dspy.Signature):
    """Judge the quality of the explanation for a technical audience"""
    context: str = dspy.InputField()
    explanation0: str = dspy.InputField()
    explanation1: str = dspy.InputField()
    best: Literal[0, 1] = dspy.OutputField(desc="0 if explanation0 is better, 1 if explanation1 is better")

def judge_quality(example, pred, trace=None):
    with dspy.context(lm=deepseek_chat):
        judge = dspy.ChainOfThoughtWithHint(ExplanationJudge)
        judgement = judge(
            context=example.codebase_context,
            explanation0=example.explanation,
            explanation1=pred.explanation,
            hint="Debate which explanation is better."
        )
        print(f"judgement: {judgement}")
        return judgement.best

def optimize(devset, task_lm, prompt_lm, teacher_lm):
    program = dspy.ChainOfThoughtWithHint(ExplainCodebaseSymbol)
    with dspy.context(lm=task_lm):
        optimizer = dspy.teleprompt.MIPROv2(
            metric=judge_quality,
            auto="light",
            prompt_model=prompt_lm,
            task_model=task_lm,
            max_bootstrapped_demos=6,
            max_labeled_demos=2,
            num_threads=6,
            hide_demo_fields=["codebase_summary", "codebase_context"],
            teacher_settings=dict(lm=teacher_lm),
        )
        optimized_program = optimizer.compile(program, trainset=devset)
        optimized_program.save(get_optimized_program_path(__file__))
    return optimized_program

###########
# Inference
###########

def explain_symbols(names: list[str]):
    with dspy.context(lm=gemini_8b):
        codebase_summary = load_codebase_summary()
        _, hs_nodes, c_nodes = get_all_names()
        program = dspy.ChainOfThoughtWithHint(ExplainCodebaseSymbol)
        program.load(get_optimized_program_path(__file__))
        examples = [dspy.Example(
            codebase_summary=codebase_summary,
            codebase_symbol=name,
            codebase_context=format_contexts(*create_contexts_for_name(name, hs_nodes, c_nodes))
        ).with_inputs("codebase_summary", "codebase_symbol", "codebase_context") for name in names]
        results = asyncio.run(run_dspy_parallel(program, examples))
        print(results)
        sym_exps = {name: result.explanation for name, result in zip(names, results)}
        return sym_exps