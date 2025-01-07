from pathlib import Path

def find_project_root() -> Path:
    current = Path.cwd().resolve()
    root_indicator = Path(".project-root")
    while current != current.parent:
        if (current / root_indicator).exists(): return current
        current = current.parent
    raise Exception("Could not find project root")

project_root = find_project_root()
data_dir = project_root / "data"
tree_sitter_haskell_dir = project_root / "tree-sitter-haskell"
optimized_programs_dir = project_root / "optimized_programs"
build_dir = project_root / "build"
tree_sitter_haskell_lib = build_dir / "haskell.so"

assert data_dir.exists()
assert tree_sitter_haskell_dir.exists()
assert optimized_programs_dir.exists()

def get_optimized_program_path(file_path: str):
    return optimized_programs_dir / f"{Path(file_path).stem}_optimized.json"