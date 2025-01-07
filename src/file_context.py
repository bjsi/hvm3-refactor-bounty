import re
from typing import Literal, Optional
from tree_sitter import Language, Node, Parser
from tree_sitter_languages import get_language, get_parser
from src.filesystem import tree_sitter_haskell_dir, tree_sitter_haskell_lib, data_dir


def setup_tree_sitter(file: str):
    ext = file.split('.')[1]
    if ext == "hs":
        if not tree_sitter_haskell_lib.exists():
            Language.build_library(tree_sitter_haskell_lib, [tree_sitter_haskell_dir])
        HASKELL_LANGUAGE = Language(tree_sitter_haskell_lib, 'haskell')
        parser = Parser()
        parser.set_language(HASKELL_LANGUAGE)
        return parser, HASKELL_LANGUAGE
    elif ext == "c":
        return get_parser("c"), get_language("c")
    
def snake_to_camel(snake_str):
    """Convert snake_case to camelCase.
    Example: "hello_world" -> "helloWorld"
    Useful for converting haskell names <-> c names.
    """
    if not '_' in snake_str: return snake_str
    components = snake_str.split('_')
    return components[0] + ''.join(x.title() for x in components[1:])

def camel_to_snake(camel_str):
    """Convert camelCase to snake_case.
    Example: "helloWorld" -> "hello_world"
    Useful for converting haskell names <-> c names.
    """
    if not any(c.isupper() for c in camel_str[1:]): return camel_str
    result = [camel_str[0].lower()]
    for char in camel_str[1:]:
        if char.isupper():
            result.extend(['_', char.lower()])
        else:
            result.append(char)
    return ''.join(result)

def node_size(node: Optional[Node]): return (node.end_point[0] - node.start_point[0]) if node else float("-inf")
    
class EmptyNode:
    start_point: tuple[int, int]
    end_point: tuple[int, int]
    type: str = "empty"
    parent: None = None

    def __init__(self, start_point: tuple[int, int], end_point: tuple[int, int]):
        self.start_point = start_point
        self.end_point = end_point

# NOTE: should get reset between LLM calls in case of changes to the codebase
_parse_cache = {}

class FileContext:
    """Tree of context for a single source code file. Show nodes using the `show` methods.
    Originally forked from: https://github.com/Aider-AI/grep-ast
    """
    def __init__(self, file: Optional[str] = None, content: Optional[str] = None):
        self.file = file
        if content: self.code = content
        else:
            with open(file, 'r') as f: self.code = f.read()
        self.lines = self.code.splitlines()
        self.parser, self.language = setup_tree_sitter(file)
        self.tree = _parse_cache[file]["tree"] if file in _parse_cache else self.parser.parse(bytes(self.code, "utf8"))
        self.root_node = self.tree.root_node
        self.show_indices = set()
        self.nodes = _parse_cache[file]["nodes"] if file in _parse_cache else [[] for _ in range(len(self.lines) + 1)]
        self.scopes = _parse_cache[file]["scopes"] if file in _parse_cache else [set() for _ in range(len(self.lines) + 1)]
        if not file in _parse_cache: self.walk_tree(self.root_node)
        _parse_cache[file] = {"tree": self.tree, "nodes": self.nodes, "scopes": self.scopes}

    def shallow_copy(self):
        return FileContext(file=self.file, tree=self.tree)

    def show_lines_mentioning(
        self,
        name: str,
        scope: Literal["full", "line", "partial"] = "line",
        parents: Literal["none", "all"] = "none",
        padding: int = 0
    ):
        line_matches = []
        for i, line in enumerate(self.lines):
            if name in line or camel_to_snake(name) in line or snake_to_camel(name) in line:
                # some extra context
                self.show(lines=list(range(max(0, i + 1 - padding), min(len(self.lines), i + padding + 2))), scope="line", last_line=False, parents=parents)
                line_matches.append(i + 1)
        self.show(lines=line_matches, scope=scope, last_line=False, parents=parents)
        return self
    
    def types_query(self):
        if self.language.name == "haskell":
            return "(data_type name: (name) @type.definition) @type.body"
        elif self.language.name == "c":
            return"""
(type_definition declarator: (type_identifier) @name) @definition.type
(declaration type: (union_specifier name: (type_identifier) @name)) @definition.class
(struct_specifier name: (type_identifier) @name body:(_)) @definition.class
(enum_specifier name: (type_identifier) @name) @definition.type
"""
        else: raise ValueError(f"Unsupported language: {self.language.name}")

    def definition_query(self, name: Optional[str] = None, include_body: bool = False):
        query_hs = f"""
(function name: (variable) @function.definition {f'(#match? @function.definition "{name}")' if name else ''}) {f'@function.body' if include_body else ''}
(signature name: (variable) @function.definition {f'(#match? @function.definition "{name}")' if name else ''}) {f'@signature.body' if include_body else ''}
(data_type name: (name) @type.definition {f'(#match? @type.definition "{name}")' if name else ''}) {f'@type.body' if include_body else ''}
"""
        query_c = f"""
(function_declarator declarator: (identifier) @name {f'(#match? @name "{name}")' if name else ''}) @definition.function
(type_definition declarator: (type_identifier) @name {f'(#match? @name "{name}")' if name else ''}) @definition.type
(declaration type: (union_specifier name: (type_identifier) @name {f'(#match? @name "{name}")' if name else ''})) @definition.class
(struct_specifier name: (type_identifier) @name {f'(#match? @name "{name}")' if name else ''} body:(_)) @definition.class
(enum_specifier name: (type_identifier) @name {f'(#match? @name "{name}")' if name else ''}) @definition.type
"""
        if self.language.name == "haskell": return query_hs
        elif self.language.name == "c": return query_c
        else: raise ValueError(f"Unsupported language: {self.language.name}")

    def is_definition(self, node: Node):
        if self.language.name == "haskell":
            return node.type in ["function", "bind", "data_type"]
        elif self.language.name == "c":
            return node.type in ["function_definition", "declaration", "struct_specifier", "enum_specifier"]
        else: raise ValueError(f"Unsupported language: {self.language.name}")

    def find_all_children(self, node):
        children = [node]
        for child in node.children: children += self.find_all_children(child)
        return children
    
    def node_for_line(self, line: int, definition: bool = False):
        if not self.lines[line - 1].strip(): return EmptyNode((line - 1, 0), (line - 1, 0))
        largest = None
        for node in self.nodes[line - 1]:
            if (node_size(node) > node_size(largest)) and (not definition or self.is_definition(node)):
                largest = node
        ret = largest or EmptyNode((line - 1, 0), (line - 1, 0))
        return ret
        
    def query(self, query_string: str):
        query = self.language.query(query_string)
        captures = query.captures(self.tree.root_node)
        nodes = [node for node, _ in captures]
        return nodes
    
    def show_nearby_block_nums(self):
        """Required to show the block numbers Victor added to the codebase.
        Note that I think the task is doable without this simply by outputting code symbols with their full scopes.
        But this helps satisfy the competition requirements.
        This conveniently also shows the comments above important code blocks.
        """
        prev_idx = None
        non_consecutive_idxs = []
        for idx in sorted(self.show_indices):
            if prev_idx is None or idx - prev_idx > 1:
                non_consecutive_idxs.append(idx)
            prev_idx = idx
        for idx in non_consecutive_idxs:
            line = self.lines[idx]
            m = re.match(r'^(--|\/\/) BLOCK (\d+):', line)
            if m: continue
            else: # search backwards to find the nearest block number
                reverse_idx = idx
                block = None
                while reverse_idx > 0 and not (block := re.match(r'^(--|\/\/) BLOCK (\d+):', self.lines[reverse_idx - 1])):
                    reverse_idx -= 1
                if block:
                    # include all lines between the previous block number and the current line
                    self.show_indices.update(range(reverse_idx - 1, idx + 1))

    def show(
        self,
        line_range: Optional[tuple[int, int]] = None,
        lines: Optional[list[int]] = None,
        query: Optional[str] = None,
        names: Optional[list[str]] = None,
        scope: Literal["full", "line", "partial"] = "line",
        parents: Literal["none", "all"] = "none",
        last_line: bool = True,
    ):
        if lines:
            nodes = [self.node_for_line(line, definition = True) for line in lines]
            self.show_indices.update([l - 1 for l in lines])
        elif query:
            nodes = [n for n in self.query(query)]
        elif names: nodes = self.query(self.definition_query("|".join([f"^{name}$" for name in names]), include_body=scope=="full"))
        elif line_range:
            if line_range[1] == -1: line_range = (line_range[0], len(self.lines) - 1)
            if line_range[0] == -1: line_range = (len(self.lines) - 1, line_range[1])
            nodes = [self.node_for_line(line) for line in range(line_range[0], line_range[1] + 1)]
            self.show_indices.update(range(line_range[0] -1, line_range[1]))
            # scope = "full"
        else: return self
        for node in nodes:
            if scope == "full": self.show_indices.update(range(node.start_point[0], node.end_point[0] + 1))
            elif scope == "line": self.show_indices.add(node.start_point[0])
            elif scope == "partial": self.show_indices.update((node.start_point[0], node.end_point[0]))
            if parents == "all":
                while node.parent:
                    self.show_indices.add(node.parent.start_point[0])
                    node = node.parent
        if last_line: self.show_indices.add(len(self.lines) - 1)
        return self
    
    def show_all(self):
        return self.show(line_range=(0, -1), scope="line", parents="none")
    
    def show_skeleton(self, full_types: bool = False):
        if full_types: self.show(query=self.types_query(), scope="full", parents="none")
        return self.show(query=self.definition_query(), scope="line", parents="none")
    
    def merge(self, other: "FileContext"):
        self.show_indices.update(other.show_indices)
        return self
    
    def walk_tree(self, node, depth=0):
        start_line = node.start_point[0]
        end_line = node.end_point[0]
        self.nodes[start_line].append(node)
        for i in range(start_line, end_line + 1): self.scopes[i].add(start_line)
        for child in node.children: self.walk_tree(child, depth + 1)
        return start_line, end_line
    
    def stringify(self, line_number=True):
        if not self.show_indices: return ""
        small_gap_size = 2 # close small gaps
        closed_show = set(self.show_indices)
        sorted_show = sorted(self.show_indices)
        for i in range(len(sorted_show) - 1):
            if sorted_show[i + 1] - sorted_show[i] == small_gap_size:
                closed_show.add(sorted_show[i] + 1)
        self.show_indices = closed_show
        output = ""
        dots = not (0 in self.show_indices)
        for i, line in enumerate(self.code.splitlines()):
            if i not in self.show_indices:
                if dots:
                    if line_number: output += "....⋮...\n"
                    else: output += "....⋮...\n"
                    dots = False
                continue
            spacer = "│"
            line_output = f"{spacer}{line}"
            if line_number: line_output = f"{i+1:4}" + line_output
            output += line_output + "\n"
            dots = True
        return output.rstrip()
    
def get_all_names():
    """Get all useful code symbol names across the codebase."""
    ctx = FileContext(data_dir / "hvm-code.c")
    c_nodes = ctx.query(ctx.definition_query(include_body=False))

    c_names = set([node.text.decode("utf-8") for node in c_nodes if node.type == "identifier" or node.type == "type_identifier"])
    print(f"c file found {len(c_names)} names")

    ctx = FileContext(data_dir / "hvm-code.hs")
    hs_nodes = ctx.query(ctx.definition_query())
    hs_names = set([node.text.decode("utf-8") for node in hs_nodes])

    all_names = c_names.union(hs_names)
    print(f"all files found {len(all_names)} names")
    all_names = sorted(all_names)
    return all_names, hs_nodes, c_nodes

def create_contexts_for_name(name: str, hs_nodes: list[Node], c_nodes: list[Node], definitions_only: bool = False):
    """To create context for a particular name, we simply show every line that mentions the name.
    We also include variations of the name - camel case, snake case, etc.
    We also include nearby block numbers.
    """
    hs_ctx = FileContext(data_dir / "hvm-code.hs")
    c_ctx = FileContext(data_dir / "hvm-code.c")

    # show the main definition
    # avoiding re-querying through tree-sitter because it's slow if you do it 200+ times
    name_query = "|".join([f"^{name}$", f"^{camel_to_snake(name)}$", f"^{snake_to_camel(name)}$"])
    hs_nodes_for_name = [n.start_point[0] + 1 for n in hs_nodes if re.match(name_query, n.text.decode("utf-8"))]
    c_nodes_for_name = [n.start_point[0] + 1 for n in c_nodes if re.match(name_query, n.text.decode("utf-8"))]
    hs_ctx.show(lines=hs_nodes_for_name, scope="full", parents="all", last_line=False)
    c_ctx.show(lines=c_nodes_for_name, scope="full", parents="all", last_line=False)

    if not definitions_only:
        # show all lines mentioning the name, including some context
        hs_ctx.show_lines_mentioning(name, scope="line")
        c_ctx.show_lines_mentioning(name, scope="line")

        # include block numbers
        hs_ctx.show_nearby_block_nums()
        c_ctx.show_nearby_block_nums()
    return hs_ctx, c_ctx

def format_contexts(hs_ctx: FileContext, c_ctx: FileContext, line_number: bool = False):
    skeleton1 = hs_ctx.stringify(line_number=line_number)
    skeleton2 = c_ctx.stringify(line_number=line_number)
    context = ""
    if skeleton1: context += f"./hvm.hs:\n{skeleton1}\n---------\n"
    if skeleton2: context += f"./hvm.c:\n{skeleton2}\n---------\n"
    return context.strip()

def codebase_skeleton():
    hs_ctx = FileContext(data_dir / "hvm-code.hs")
    c_ctx = FileContext(data_dir / "hvm-code.c")
    hs_ctx.show_skeleton(full_types=True)
    c_ctx.show_skeleton(full_types=True)
    return format_contexts(hs_ctx, c_ctx)

# use these functions to inspect the tree created by tree-sitter
# useful for debugging and understanding why queries are not working

def print_tree(node, level=0, source_code=None):
    """Print the tree structure with node types and text."""
    indent = "  " * level
    
    # Get the text for this node if source code is provided
    text = ""
    if source_code:
        start_byte = node.start_byte
        end_byte = node.end_byte
        text = f" -> {source_code[start_byte:end_byte]!r}"
    
    print(f"{indent}{node.type}{text}")
    
    # Recursively print children
    for child in node.children:
        print_tree(child, level + 1, source_code)

def inspect_code(code: str, lang: Literal[".hs", ".c"]):
    """Parse and inspect Haskell code."""
    # Setup parser
    parser, _ = setup_tree_sitter(lang)

    # Parse code
    tree = parser.parse(bytes(code, "utf8"))
    
    # Print full tree
    print("Full Parse Tree:")
    print_tree(tree.root_node, source_code=code)

def num_mentions(name: str):
    with open(data_dir / "hvm-code.hs", "r") as f:
        hs_code = f.read()
    with open(data_dir / "hvm-code.c", "r") as f:
        c_code = f.read()
    return hs_code.count(name) + c_code.count(name) + hs_code.count(camel_to_snake(name)) + hs_code.count(snake_to_camel(name)) + c_code.count(camel_to_snake(name)) + c_code.count(snake_to_camel(name))

def most_mentioned_names():
    all_names = get_all_names()[0]
    return sorted(all_names, key=lambda x: num_mentions(x), reverse=True)

if __name__ == "__main__":
    all_names, hs_nodes, c_nodes = get_all_names()
    hs_ctx, c_ctx = create_contexts_for_name("u12v2_y", hs_nodes, c_nodes, definitions_only=True)
    s = format_contexts(hs_ctx, c_ctx)
    print(s)