import dspy
import dspy.teleprompt
import pydantic
from src.file_context import most_mentioned_names
from src.llms import get_lm
from src.utils import load_jsonl

class WriteSection(dspy.Signature):
    """The section should be a single paragraph for a technical audience."""
    title: str = dspy.InputField()
    description: str = dspy.InputField()
    symbol_explanations: str = dspy.InputField()
    section: str = dspy.OutputField()

class Section(pydantic.BaseModel):
    title: str = dspy.OutputField()
    description: str = dspy.OutputField()
    top_symbols: list[str] = dspy.OutputField()

class UpdateOutline(dspy.Signature):
    """Update the codebase summary outline.
    Focus on the most important parts of the codebase.
    The summary should have a maximum of 3 sections.
    """
    existing_outline: str = dspy.InputField()
    symbol_explanations: str = dspy.InputField()
    updated_outline: list[Section] = dspy.OutputField()

class WriteCodebaseSummary(dspy.Module):
    def __init__(self):
        self.write_outline = dspy.ChainOfThought(UpdateOutline)
        self.write_section = dspy.ChainOfThoughtWithHint(WriteSection)

    def forward(self, sym_exps, top_exps):
        def format_outline(sections):
            return "\n\n".join([f"{section.title}\n{section.description}\n{section.top_symbols}" for section in sections])

        batch_size = 25
        top_exps_batched = [top_exps[i:i+batch_size] for i in range(0, len(top_exps), batch_size)]
        outline = None
        for batch in top_exps_batched:
            outline = self.write_outline(existing_outline=format_outline(outline.updated_outline) if outline else "", symbol_explanations="\n\n".join(batch))

        def symbol_explanations_for_section(symbols):
            return "\n\n".join([x for x in [sym_exps.get(symbol) for symbol in symbols] if x])

        results = []
        for section in outline.updated_outline:
            symbol_explanations=symbol_explanations_for_section(section.top_symbols)
            result = self.write_section(title=section.title, description=section.description, symbol_explanations=symbol_explanations)
            results.append(result)

        summary = "\n\n".join([f"{section.title}\n{result.section}" for (result, section) in zip(results, outline.updated_outline)])
        return summary

def write_summary():
    dspy.configure(lm=get_lm("gemini/gemini-1.5-flash-8b"))
    program = WriteCodebaseSummary()
    result = program()
    return result

class SummaryJudge(dspy.Signature):
    """Judge the quality of the summary for a technical audience"""
    summary: str = dspy.InputField()
    quality: int = dspy.OutputField(ge=0, le=10, desc="The quality of the summary between 0 and 10")

def quality_metric(example, pred, trace=None):
    with dspy.context(lm=get_lm("gemini/gemini-1.5-pro")):
        judge = dspy.ChainOfThoughtWithHint(SummaryJudge)
        criteria = """Does the summary focus on the most important parts of the codebase? Is there anything vague, confusing or unexplained?"""
        judgement = judge(summary=pred, hint=criteria)
        return (judgement.quality / 10)
    
def load_dataset():
    top_names = most_mentioned_names()
    sym_exps = load_jsonl("symbol_explanations.jsonl")
    sym_exps = {sym_exp["name"]: sym_exp["explanation"] for sym_exp in sym_exps}
    top_exps = [f"{name}: {sym_exps[name]}" for name in top_names][:100]
    return [dspy.Example(sym_exps=sym_exps, top_exps=top_exps).with_inputs("sym_exps", "top_exps")]

def optimize_for(program, task_lm, prompt_lm):
    with dspy.context(lm=task_lm):
        tp = dspy.teleprompt.MIPROv2(metric=quality_metric, auto="light", prompt_model=prompt_lm, task_model=task_lm, max_bootstrapped_demos=0, max_labeled_demos=0, num_threads=20)
        dataset = load_dataset()
        optimized_classify = tp.compile(program, trainset=dataset, valset=dataset)
        optimized_classify.save(f"optimized_summary.pkl")
    return optimized_classify

if __name__ == "__main__":
    program = WriteCodebaseSummary()
    program.load("optimized_summary.pkl")
    dspy.configure(lm=get_lm("gemini/gemini-1.5-flash-8b"))
    result = program(**load_dataset()[0].inputs().toDict())
    print(result)
