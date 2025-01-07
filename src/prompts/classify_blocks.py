import dspy
from pydantic import BaseModel
from src.filesystem import get_optimized_program_path

class Block(BaseModel):
    number: int
    reasoning: str
    confidence: float

class ClassifyBlocksToEdit(dspy.Signature):
    """Classify which blocks must be edited during a refactor."""
    codebase_summary: str = dspy.InputField()
    refactoring_task: str = dspy.InputField()
    codebase_symbol: str = dspy.InputField(desc="The name of the codebase symbol that may or may not be relevant to the refactoring task")
    codebase_symbol_explanation: str = dspy.InputField()
    codebase_context: str = dspy.InputField()
    blocks_to_edit: list[Block] = dspy.OutputField(desc="Each block that must be edited including the reason why it must be edited and your confidence.")

# filter high confidece
# filter dups
