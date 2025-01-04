import json
import dspy
from file_context import create_contexts_for_name, format_contexts, get_all_names
from llms import model_to_provider, provider_to_api_key, provider_to_base_url
import asyncio
from pydantic import BaseModel

from utils import run_parallel_tasks_with_progress

codebase_summary = """
The HVM3 codebase is a highly parallel, functional runtime system designed to execute programs efficiently on massively parallel hardware. It is built around the Interaction Combinator model, which enables parallel evaluation of terms through a graph-based computational model. The codebase is divided into two main parts: the Haskell frontend (`hvm.hs`) and the C backend (`hvm.c`). The Haskell code handles high-level operations like parsing, compilation, and term manipulation, while the C code provides low-level runtime support for memory management, term reduction, and parallel execution.
The core of the system revolves around the `Term` data type, which represents nodes in the computational graph. Each `Term` encodes a tag, label, and location, allowing the runtime to efficiently manage and process terms. The `reduce` function is the backbone of the evaluation mechanism, applying reduction rules based on the term's type. The system also includes a `Collapse` monad for managing parallel computations and a `Book` data structure for storing function definitions and metadata.
The compilation process translates high-level `Core` terms into low-level C code, which is then executed by the runtime. The runtime uses a memory model based on Interaction Combinators, with functions like `allocNode` and `set` managing memory allocation and term manipulation. The system supports both strict and lazy evaluation modes, with optimizations for parallel execution.
Overall, the codebase is designed to handle complex, parallel computations efficiently, leveraging the Interaction Combinator model to achieve high performance on modern hardware.

### Key Components:
1. **Term Representation**:
    - The `Term` data type is the core of the system, representing nodes in the computational graph. Each `Term` encodes a tag, label, and location, allowing the runtime to efficiently manage and process terms.
    - Tags identify the type of the term (e.g., `ERA`, `REF`, `NUM`, `CON`, `DUP`), while labels provide additional metadata (e.g., function IDs, constructor IDs).
    - Locations point to memory addresses where terms are stored, enabling efficient access and manipulation.

2. **Reduction Engine**:
    - The `reduce` function is the backbone of the evaluation mechanism. It applies reduction rules based on the term's type, handling operations like function application (`APP`), pattern matching (`MAT`), and duplication (`DUP`).
    - The `reduceAt` function is a higher-level reduction engine that recursively reduces terms to their normal form, handling different term types with specific reduction rules.

3. **Memory Management**:
    - The `allocNode` function allocates memory for nodes in the runtime, ensuring efficient memory usage and supporting the massively parallel execution model.
    - The `set` and `got` functions are used to write and retrieve terms from specific memory locations, enabling dynamic term manipulation.

4. **Compilation**:
    - The `compile` function orchestrates the compilation process, translating high-level `Core` terms into low-level C code. It supports different compilation modes (`compileFull`, `compileFast`, `compileSlow`) for various evaluation strategies.
    - The `compileFastCore` function optimizes the compilation of terms for parallel execution, generating efficient C code for constructs like `Lam`, `App`, `Sup`, and `Dup`.

5. **Parallel Computation**:
    - The `Collapse` monad manages parallel computations, handling multiple possible outcomes or states and reducing them to a single value or a list of results.
    - The `Sup` operation allows for the combination of two terms into a single superposed term, enabling parallel evaluation.

6. **Book Data Structure**:
    - The `Book` data structure stores function definitions and metadata, providing quick access to the necessary information for compilation and execution.
    - It includes mappings for function IDs, names, labels, and constructors, ensuring efficient lookup and management of runtime resources.

7. **Interaction Combinators**:
    - The runtime is built around the Interaction Combinator model, which enables parallel evaluation of terms through a graph-based computational model.
    - Functions like `reduce_ref_sup`, `reduce_dup_lam`, and `reduce_mat_ctr` handle specific interaction rules, ensuring correct and efficient execution.

### Logical Flow:
1. **Parsing and Compilation**:
    - The input program is parsed into a high-level `Core` representation.
    - The `compile` function translates the `Core` terms into low-level C code, optimizing for parallel execution.

2. **Runtime Initialization**:
    - The runtime initializes the memory model and sets up the necessary data structures (e.g., `Book`, `State`).

3. **Term Reduction**:
    - The `reduceAt` function reduces the main term to its normal form, applying reduction rules based on the term's type.
    - The `reduce` function handles specific reduction operations, ensuring that all subterms are fully evaluated.

4. **Parallel Execution**:
    - The `Collapse` monad manages parallel computations, reducing multiple outcomes to a single result.
    - The `Sup` operation enables parallel evaluation of terms, leveraging the massively parallel hardware.

5. **Memory Management**:
    - The `allocNode` function allocates memory for new nodes, while `set` and `got` manage term manipulation and access.
    - The runtime ensures efficient memory usage, supporting the parallel execution model.

6. **Output and Debugging**:
    - The `print_term` function provides debugging and diagnostic output, allowing developers to inspect the state of the computation.
""".strip()


if __name__ == "__main__":
    hvm_names, hs_nodes, c_nodes = get_all_names()

    model = "deepseek/deepseek-chat"
    lm = dspy.LM(
        model=model,
        api_key=provider_to_api_key[model_to_provider[model]],
        api_base=provider_to_base_url[model_to_provider[model]],
        max_tokens=3000
        #cache=False
    )

    dspy.configure(lm=lm, async_max_workers=300)

    class ExplainCodebaseSymbol(dspy.Signature):
        """Explain the purpose and role of the given codebase symbol within the codebase."""
        codebase_summary: str = dspy.InputField()
        codebase_symbol: str = dspy.InputField()
        codebase_context: str = dspy.InputField()
        explanation: str = dspy.OutputField()

    class SummarizeCodebase(dspy.Signature):
        """Summarize the main flow of the codebase into notes for a technical audience."""
        codebase_name: str = dspy.InputField()
        codebase_context: str = dspy.InputField()
        symbol_explanations: list[str] = dspy.InputField()
        detailed_summary: str = dspy.OutputField()

    summarize_codebase = dspy.ChainOfThoughtWithHint(SummarizeCodebase)
    explain_symbol = dspy.asyncify(dspy.ChainOfThought(ExplainCodebaseSymbol))

    async def explain_symbols_async(names: list[str]):
        tasks = [
            lambda name=name: explain_symbol(
                codebase_summary=codebase_summary,
                codebase_symbol=name, 
                codebase_context=format_contexts(*create_contexts_for_name(name, hs_nodes, c_nodes))
            )
            for name in names
        ]
        return await run_parallel_tasks_with_progress(tasks, desc="Explaining symbols")

    explanations = asyncio.run(explain_symbols_async(hvm_names))
    symbol_explanations = [(name, explanation.explanation) for name, explanation in zip(hvm_names, explanations)]
    print(symbol_explanations[0:3])
    symbol_explanation_map = {name: explanation for name, explanation in symbol_explanations}
    with open("symbol_explanations.jsonl", "w") as f:
        for name, explanation in symbol_explanations:
            f.write(json.dumps({"name": name, "explanation": explanation}) + "\n")

    # sorted_symbol_explanations = sorted(symbol_explanations, key=lambda x: num_mentions(x[0]), reverse=True)[:50]
    # skeleton = codebase_skeleton()
    # new_codebase_summary = summarize_codebase(
    #     codebase_name="HVM3",
    #     codebase_context=skeleton,
    #     symbol_explanations=[f"{name}: {explanation}" for name, explanation in sorted_symbol_explanations],
    #     hint="Summarize the main logical flow of the codebase"
    # )
    # print(new_codebase_summary)
    # print(lm.history[-1])

    # model = "openrouter/meta-llama/llama-3.1-8b-instruct"
    lm = dspy.LM(
        model=model,
        api_key=provider_to_api_key[model_to_provider[model]],
        api_base=provider_to_base_url[model_to_provider[model]],
        max_tokens=3000
        # cache=False
    )

    dspy.configure(lm=lm, async_max_workers=300)
    
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

    classify_blocks_async = dspy.asyncify(dspy.ChainOfThoughtWithHint(ClassifyBlocksToEdit))

    async def classify_blocks_parallel(query: str, names: list[str]):
        tasks = [
            lambda name=name: classify_blocks_async(
                codebase_summary=codebase_summary,
                codebase_context=format_contexts(*create_contexts_for_name(name, hs_nodes, c_nodes)),
                codebase_symbol=name,
                codebase_symbol_explanation=symbol_explanation_map[name],
                refactoring_task=query,
                hint="Debate whether or not a block in the codebase context needs to be edited.",
            )
            for name in names
        ]
        return await run_parallel_tasks_with_progress(tasks, desc="Classifying blocks")

    results = asyncio.run(classify_blocks_parallel(hvm3_dataset[0].task, hvm_names))
    print(results[0])
    high_confidence_blocks = []
    for result in results:
        for block in result.blocks_to_edit:
            if block.confidence >= 0.8:
                high_confidence_blocks.append(block)

    unique_blocks = {}
    for block in high_confidence_blocks:
        # Keep the block with highest confidence if there are duplicates
        if block.number not in unique_blocks or block.confidence > unique_blocks[block.number].confidence:
            unique_blocks[block.number] = block
    high_confidence_blocks = list(unique_blocks.values())
    
    for block in high_confidence_blocks:
        print(block.number)
        print(block.reasoning)
        print(block.confidence)
        print('----')
    print(f"number of high confidence blocks: {len(high_confidence_blocks)}")

    # Check if we found all the expected positive blocks
    expected_positives = set(block_num for block_num, _ in hvm3_dataset[0].positives)
    found_positives = set(block.number for block in high_confidence_blocks)
    
    missing_positives = expected_positives - found_positives
    if missing_positives:
        print("\nMissing expected positive blocks:")
        for block_num in missing_positives:
            # Find the explanation for this block from the dataset
            explanation = next(exp for num, exp in hvm3_dataset[0].positives if num == block_num)
            print(f"Block {block_num}: {explanation}")
            
    # output = classify_blocks_async(
    #     codebase_summary=codebase_summary,
    #     codebase_context=format_contexts(*create_contexts_for_name(name, hs_nodes, c_nodes)),
    #     codebase_symbol=name,
    #     codebase_symbol_explanation=symbol_explanation_map[name],
    #     refactoring_task=example_queries[0],
    #     hint="Debate whether or not a block in the codebase context needs to be edited.",
    # )
    # print(lm.history[-1])

