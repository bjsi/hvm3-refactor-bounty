import asyncio
import dspy
from src.filesystem import data_dir
from src.file_context import get_all_names

class ExplainCodebaseSymbol(dspy.Signature):
    """Explain the purpose and role of the given codebase symbol within the codebase. The explanation should be one concise paragraph that explains the symbol's role, purpose and behavior. """
    codebase_summary: str = dspy.InputField()
    codebase_symbol: str = dspy.InputField()
    codebase_context: str = dspy.InputField()
    explanation: str = dspy.OutputField()

def load_codebase_summary():
    with open(data_dir / "codebase_summary.txt", "r") as f: return f.read()

def explain_symbols():
    hvm_names, hs_nodes, c_nodes = get_all_names()
    codebase_summary = load_codebase_summary()
    explanations = asyncio.run()
    symbol_explanations = [(name, explanation.explanation) for name, explanation in zip(hvm_names, explanations)]

def load_trainset():
    hvm_names, hs_nodes, c_nodes = get_all_names()

def optimize_for():
    pass