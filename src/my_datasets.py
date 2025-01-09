from typing import Literal
from src.utils import load_json, load_jsonl
from src.filesystem import data_dir


def load_real_tasks():
    return load_json(data_dir / "hvm3_real_tasks.json")

def load_symbol_explanations(version: Literal["v1", "v2", "v3"] = "v3"):
    return load_jsonl(data_dir / f"symbol_explanations_{version}.jsonl")

def load_codebase_summary():
    return (data_dir / "codebase_summary.txt").read_text()