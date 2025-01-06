import dspy

class SummarizeCodebase(dspy.Signature):
    """Summarize the main flow of the codebase into a paragraph for a technical audience."""
    codebase_name: str = dspy.InputField()
    codebase_context: str = dspy.InputField()
    symbol_explanations: list[str] = dspy.InputField()
    detailed_summary: str = dspy.OutputField()

summarize_codebase = dspy.ChainOfThoughtWithHint(SummarizeCodebase)
