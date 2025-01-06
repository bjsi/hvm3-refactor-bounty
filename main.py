import json
import dspy
from file_context import create_contexts_for_name, format_contexts, get_all_names
from llms import model_to_provider, provider_to_api_key, provider_to_base_url
import asyncio
from pydantic import BaseModel

from utils import load_text, run_parallel_tasks_with_progress

codebase_summary = load_text("codebase_summary.txt")

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

