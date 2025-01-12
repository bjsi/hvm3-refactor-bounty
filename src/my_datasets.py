from typing import Literal
from src.utils import load_json, load_jsonl, save_json
from src.filesystem import data_dir


def load_real_tasks():
    return load_json(data_dir / "hvm3_real_tasks.json")

def load_symbol_explanations(version: Literal["v1", "v2", "v3"] = "v3"):
    return load_jsonl(data_dir / f"symbol_explanations_{version}.jsonl")

def load_codebase_summary():
    return (data_dir / "codebase_summary.txt").read_text()

def load_binary_classification_judge_data():
    file = data_dir / "new_judge_data.json"
    return load_json(file) if file.exists() else []

def save_binary_classification_judge_data(data: list[dict]):
    save_json(data, data_dir / "new_judge_data.json")

def save_new_incumbent_data(data: list[dict]):
    save_json(data, data_dir / "new_incumbent_data.json")

def load_new_incumbent_data():
    return load_json(data_dir / "new_incumbent_data.json")
