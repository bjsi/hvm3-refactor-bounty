import dspy
from file_context import get_all_names
from utils import load_json, load_jsonl
import pydantic

hvm_names = get_all_names()[0]

class CodebaseSymbolNumber(pydantic.BaseModel):
    symbol_index: int = dspy.OutputField(desc="The index of the codebase symbol that is strongly related to the task.")
    reasoning: str = dspy.OutputField(desc="Reasoning why the symbol is related to the task.")
    confidence: float = dspy.OutputField(ge=0.0, le=1.0, desc="Confidence that the symbol is related to the task.")

class DetermineCodeSymbolRelevancy(dspy.Signature):
    """Use the symbol explanations to decide whether they are related to the task"""
    symbol_explanations: str = dspy.InputField()
    task: str = dspy.InputField()
    symbols: str = dspy.InputField(desc="Numbered codebase symbols")
    strongly_related_symbols: list[CodebaseSymbolNumber] = dspy.OutputField(desc="Numbers corresponding to codebase symbols that are strongly related to the task.")

class ClassifyRelatednessProgram(dspy.Module):
    def __init__(self):
        self.classify_relatedness = dspy.ChainOfThoughtWithHint(DetermineCodeSymbolRelevancy)

    def forward(self, task: str, symbols: list[str], symbol_explanations: str) -> list[CodebaseSymbolNumber]:
        hint = "For each symbol, think step-by-step about whether or not it is directly related to the task and why."
        return self.classify_relatedness(symbol_explanations=symbol_explanations, task=task, symbols=symbols, hint=hint)
    
class GenerateRefactoringTasks(dspy.Signature):
    """Generate 20 refactoring tasks in a similar style to the examples. Do not use direct names for symbols in the task descriptions. They should be indirect/colloquial names."""
    codebase_summary: str = dspy.InputField()
    examples: list[str] = dspy.InputField()
    refactoring_tasks: list[str] = dspy.OutputField()

def load_trainset():
    devset = []
    symbol_explanations = [(row['name'], row['explanation']) for row in load_jsonl("symbol_explanations.jsonl")]
    batch_size = 1
    symbol_explanations_batched = [symbol_explanations[i:i+batch_size] for i in range(0, len(symbol_explanations), batch_size)]
    tasks = load_json("hvm3_real_tasks.json")
    for task, task_data in list(tasks.items()):
        for symbol_explanations in symbol_explanations_batched:
            symbols = [name for name, _ in symbol_explanations]
            related_symbols = [s for s in task_data["related_symbols"] if s in symbols]
            devset.append(
                dspy.Example(
                    task=task,
                    strongly_related_symbols=related_symbols,
                    symbols=[f"{i}: {name}" for i, name in enumerate(symbols)],
                    symbol_explanations="\n\n".join([f"{i}: {name}: {explanation}" for i, (name, explanation) in enumerate(symbol_explanations)])).with_inputs('task', 'symbols', 'symbol_explanations')
                )
    return devset

def filter_symbols(symbols: list[CodebaseSymbolNumber]) -> list[str]:
    return sorted([symbol.symbol for symbol in symbols if symbol.symbol in hvm_names and symbol.relevancy_score >= 0.9])

def score_response(example, pred, trace=None):
    # Calculate similarity with reduced penalty for extra predictions
    # Basically a modified Jaccard similarity 0-1 between predicted and example symbols
    # Uses intersection over union: |A ∩ B| / |A ∪ B|
    example_symbols = set(example.strongly_related_symbols)
    pred_symbols = set(filter_symbols(pred.strongly_related_symbols))
    intersection = len(example_symbols & pred_symbols)
    extra_predictions = len(pred_symbols - example_symbols)
    if not example_symbols | pred_symbols: return 1.0
    return intersection / (len(example_symbols) + (1 * extra_predictions))

def evaluate(devset):
    evaluator = dspy.Evaluate(devset=devset, num_threads=10, display_progress=True, display_table=5)
    evaluator(ClassifyRelatednessProgram(), metric=score_response)

def optimize_for(devset, task_lm, prompt_lm, save_name):
    with dspy.context(lm=task_lm):
        tp = dspy.MIPROv2(metric=score_response, auto="light", prompt_model=prompt_lm, task_model=task_lm)
        program = ClassifyRelatednessProgram()
        optimized_classify = tp.compile(program, trainset=devset, max_labeled_demos=3, max_bootstrapped_demos=3)
        optimized_classify.save(f"optimized_classify_{save_name}.pkl")

fake_tasks = [
    "implement path optimization by merging overlapping bit-strings in the recursive tree structure when their patterns match",
    "modify the atomic value container to use 48-bit addresses and 16-bit type tags for better memory utilization",
    "refactor the state tracking system to use a persistent map instead of mutable references",
    "add support for circular reference detection during parallel execution",
    "modify the tree flattening algorithm to use an iterative approach instead of recursion",
    "implement a caching system for frequently accessed metadata lookups",
    "refactor the parallel execution engine to use work-stealing queues",
    "add runtime statistics collection for parallel path execution patterns",
    "implement automatic garbage collection for unused tree branches",
    "modify the name resolution system to support hierarchical namespaces",
    "implement a more efficient string interning mechanism for identifier storage",
    "refactor the error handling system to provide more detailed stack traces",
    "implement lazy evaluation for tree branch expansion",
    "add support for custom memory allocators in the atomic value container",
    "refactor the state tracking system to use a persistent map instead of mutable references",
    "add support for circular reference detection during parallel execution",
    "modify the tree flattening algorithm to use an iterative approach instead of recursion",
    "implement a caching system for frequently accessed metadata lookups",
    "refactor the parallel execution engine to use work-stealing queues",
    "add runtime statistics collection for parallel path execution patterns",
    "implement automatic garbage collection for unused tree branches",
    "modify the name resolution system to support hierarchical namespaces",
    "implement a more efficient string interning mechanism for identifier storage",
    "refactor the error handling system to provide more detailed stack traces",
    "implement lazy evaluation for tree branch expansion",
    "add support for custom memory allocators in the atomic value container",
    "refactor the state tracking system to use a persistent map instead of mutable references",
    "add support for circular reference detection during parallel execution",
    "modify the tree flattening algorithm to use an iterative approach instead of recursion",
    "implement a caching system for frequently accessed metadata lookups",
    "refactor the parallel execution engine to use work-stealing queues",
    "add runtime statistics collection for parallel path execution patterns",
    "implement automatic garbage collection for unused tree branches",
    "modify the name resolution system to support hierarchical namespaces",
    "implement a more efficient string interning mechanism for identifier storage",
    "refactor the error handling system to provide more detailed stack traces",
    "implement lazy evaluation for tree branch expansion",
    "add support for custom memory allocators in the atomic value container",
    "modify the compilation pipeline to support incremental compilation",
    "implement a more efficient pattern matching algorithm for tree traversal",
    "refactor the state management system to use immutable data structures",
    "add support for concurrent metadata updates during runtime",
    "implement a more efficient serialization format for stored definitions",
]
