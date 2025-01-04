import asyncio
import json
import dspy
from llms import get_lm
from utils import load_jsonl, run_dspy_parallel

explanations = [dspy.Example(symbol=x["name"], explanation=x["explanation"]).with_inputs("symbol", "explanation") for x in load_jsonl("symbol_explanations.jsonl")]

class UpdateExplanation(dspy.Signature):
    """
    Update the explanation of the code symbol. The explanation should be one concise paragraph that explains the symbol's role, purpose and behavior.
    """
    symbol: str = dspy.InputField()
    explanation: str = dspy.InputField()
    updated_explanation: str = dspy.OutputField()

update_explanation = dspy.Predict(UpdateExplanation)

dspy.configure(lm=get_lm("deepseek/deepseek-chat"), async_max_workers=100)
results = asyncio.run(run_dspy_parallel(update_explanation, explanations))
updated_explanations = [result.updated_explanation for result in results]

with open("updated_explanations.jsonl", "w") as f:
    for symbol, explanation in zip(explanations, updated_explanations):
        f.write(json.dumps({"name": symbol.symbol, "explanation": explanation}) + "\n")

