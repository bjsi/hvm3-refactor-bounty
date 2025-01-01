from tqdm import tqdm
import time
import tiktoken
import asyncio
from dataclasses import dataclass, field
import datetime
import shutil
import textwrap
from typing import Any, Callable, Generator, Literal, Optional
from openai import AsyncOpenAI, OpenAI, NOT_GIVEN
from dotenv import load_dotenv
import os
from tree_sitter import Node, Language, Parser
from tree_sitter_languages import get_language, get_parser
import re
import warnings
from llms import model_to_provider, provider_to_api_key, provider_to_base_url
from pydantic import BaseModel

##########################
## LLM SETUP
##########################

openrouter_headers = {
    "HTTP-Referer": "https://github.com/HigherOrderCO/HVM",
    "X-Title": "victor-hvm-refactoring",
}

TEMPERATURE = 0
TOP_P = 0.1

warnings.filterwarnings("ignore", category=FutureWarning, module="tree_sitter")

@dataclass(frozen=True, eq=False)
class Message:
    role: Literal["system", "user", "assistant"]
    content: str
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)

    def __repr__(self):
        content = textwrap.shorten(self.content, 20, placeholder="...")
        return f"<Message role={self.role} content={content}>"

    def to_dict(self) -> dict:
        d: dict = {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
        }
        return d
    
def format_msgs(
    msgs: list[Message],
    oneline: bool = False,
    highlight: bool = False,
    indent: int = 0,
) -> list[str]:
    """Formats messages for printing to the console."""
    outputs = []
    for msg in msgs:
        userprefix = msg.role.capitalize()
        if highlight:
            userprefix = f"{userprefix.upper()}"
        # get terminal width
        max_len = shutil.get_terminal_size().columns - len(userprefix)
        output = ""
        if oneline:
            output += textwrap.shorten(
                msg.content.replace("\n", "\\n"), width=max_len, placeholder="..."
            )
            if len(output) < 20:
                output = msg.content.replace("\n", "\\n")[:max_len] + "..."
        else:
            multiline = len(msg.content.split("\n")) > 1
            output += "\n" + indent * " " if multiline else ""
            for i, block in enumerate(msg.content.split("```")):
                if i % 2 == 0:
                    output += textwrap.indent(block, prefix=indent * " ")
                    continue
                elif highlight:
                    # lang = block.split("\n")[0]
                    # block = rich_to_str(Syntax(block.rstrip(), lang))
                    block = block.rstrip()
                output += f"```{block.rstrip()}\n```"
        outputs.append(f"{userprefix}: {output.rstrip()}")
    return outputs

def print_msg(
    msg: Message | list[Message],
    oneline: bool = False,
    highlight: bool = True,
) -> None:
    """Prints the log to the console."""
    msgs = msg if isinstance(msg, list) else [msg]
    msgstrs = format_msgs(msgs, highlight=highlight, oneline=oneline)
    for m, s in zip(msgs, msgstrs):
        print(s)

def _prep_o1(msgs: list[Message]) -> Generator[Message, None, None]:
    # prepare messages for OpenAI O1, which doesn't support the system role
    # and requires the first message to be from the user
    for msg in msgs:
        if msg.role == "system":
            msg = msg.replace(
                role="user", content=f"<system>\n{msg.content}\n</system>"
            )
        yield msg

def msgs2dicts(msgs: list[Message]) -> list[dict]:
    return [msg.to_dict() for msg in msgs]

async def achat(messages: list[Message], model: str, timeout: float = 40.0) -> str:
    """Async version of chat()"""
    provider = model_to_provider[model]
    aopenai = AsyncOpenAI(api_key=provider_to_api_key[provider], base_url=provider_to_base_url[provider])
    async def _chat(messages: list[Message]):
        is_o1 = model.startswith("o1")
        if is_o1:
            messages = list(_prep_o1(messages))

        messages_dicts = msgs2dicts(messages)

        response = await aopenai.chat.completions.create(
            model=model,
            messages=messages_dicts,
            temperature=TEMPERATURE if not is_o1 else NOT_GIVEN,
            top_p=TOP_P if not is_o1 else NOT_GIVEN,
            tools=NOT_GIVEN,
            extra_headers=(
                openrouter_headers if "openrouter.ai" in str(aopenai.base_url) else {}
            ),
        )
        content = response.choices[0].message.content
        assert content
        return content

    try:
        return await asyncio.wait_for(_chat(messages), timeout=timeout)
    except asyncio.TimeoutError:
        print(f"Chat completion timed out after {timeout} seconds")
        return ""

##########################
## TREE SITTER
##########################

def setup_tree_sitter(file: str):
    ext = file.split('.')[1]
    if ext == "hs": # outdated in tree-sitter-languages
        if not os.path.exists('build/haskell.so'):
            Language.build_library(
                'build/haskell.so',
                [
                    './tree-sitter-haskell'
                ]
            )
        HASKELL_LANGUAGE = Language('build/haskell.so', 'haskell')
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

# should get reset between LLM calls in case of changes to the codebase
_parse_cache = {}

class FileContext:
    """Tree of context for a single source code file. Show nodes using the `show` methods."""
    def __init__(self, file: str | None = None, content: str | None = None):
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
        self.show(line_range=(0, -1), scope="line", parents="none")
        return self
    
    def show_skeleton(self, full_types: bool = False):
        if full_types: self.show(query=self.types_query(), scope="full", parents="none")
        self.show(query=self.definition_query(), scope="line", parents="none")
        return self
    
    def merge(self, other: "FileContext"):
        self.show_indices.update(other.show_indices)
        return self
    
    def walk_tree(self, node, depth=0):
        start = node.start_point
        end = node.end_point
        start_line = start[0]
        end_line = end[0]
        self.nodes[start_line].append(node)
        for i in range(start_line, end_line + 1):
            self.scopes[i].add(start_line)
        for child in node.children:
            self.walk_tree(child, depth + 1)
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
    ctx = FileContext("hvm-code.c")
    c_nodes = ctx.query(ctx.definition_query(include_body=False))

    c_names = set([node.text.decode("utf-8") for node in c_nodes if node.type == "identifier" or node.type == "type_identifier"])
    print(f"c file found {len(c_names)} names")

    ctx = FileContext("hvm-code.hs")
    hs_nodes = ctx.query(ctx.definition_query())
    hs_names = set([node.text.decode("utf-8") for node in hs_nodes])
    print(f"hs file found {len(hs_names)} names")

    all_names = c_names.union(hs_names)
    print(f"all files found {len(all_names)} names")
    ret = sorted(all_names)
    print(ret)
    return ret, hs_nodes, c_nodes

def create_contexts_for_name(name: str, hs_nodes: list[Node], c_nodes: list[Node]):
    """To create context for a particular name, we simply show every line that mentions the name.
    We also include variations of the name - camel case, snake case, etc.
    We also include nearby block numbers.
    """
    hs_ctx = FileContext("hvm-code.hs")
    c_ctx = FileContext("hvm-code.c")

    # show the main definition
    # avoiding re-querying through tree-sitter because it's slow if you do it 200+ times
    name_query = "|".join([f"^{name}$", f"^{camel_to_snake(name)}$", f"^{snake_to_camel(name)}$"])
    hs_nodes_for_name = [n.start_point[0] for n in hs_nodes if re.match(name_query, n.text.decode("utf-8"))]
    c_nodes_for_name = [n.start_point[0] for n in c_nodes if re.match(name_query, n.text.decode("utf-8"))]
    hs_ctx.show(lines=hs_nodes_for_name, scope="full", parents="all", last_line=False)
    c_ctx.show(lines=c_nodes_for_name, scope="full", parents="all", last_line=False)

    # show all lines mentioning the name, including some context
    hs_ctx.show_lines_mentioning(name, scope="line")
    c_ctx.show_lines_mentioning(name, scope="line")

    # include block numbers
    hs_ctx.show_nearby_block_nums()
    c_ctx.show_nearby_block_nums()
    return hs_ctx, c_ctx

def codebase_skeleton():
    hs_ctx = FileContext("hvm-code.hs")
    c_ctx = FileContext("hvm-code.c")
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
    parser, language = setup_tree_sitter(lang)

    # Parse code
    tree = parser.parse(bytes(code, "utf8"))
    
    # Print full tree
    print("Full Parse Tree:")
    print_tree(tree.root_node, source_code=code)

def num_mentions(name: str):
    with open("hvm-code.hs", "r") as f:
        hs_code = f.read()
    with open("hvm-code.c", "r") as f:
        c_code = f.read()
    return hs_code.count(name) + c_code.count(name) + hs_code.count(camel_to_snake(name)) + hs_code.count(snake_to_camel(name)) + c_code.count(camel_to_snake(name)) + c_code.count(snake_to_camel(name))

##########################
## PROMPTS
##########################

example_queries = [
    "make CTRs store only the CID in the Lab field, and move the arity to a global static object in C",
    "replace the Lab field on Ctr nodes to store just the CID, and move the arity to a C struct stored on the Runtime State",
    "extend the size of the addr field on runtime nodes from 32 to 40 bits, and reduce the label field from 24 to 16 bits",
    "completely remove native numbers as a feature",
    "remove the list/string pretty printers",
    "measure interactions by interaction type instead of just storing the total count. report results segmented by interaction type",
    "implement a very simple #import file.hvml feature. an import will just load and inline a different file into the current file.",
    "implement a feature that prevents the user from creating two constructors with the same name. show a helpful error when that happens.",
    "clean up every commented-out line of code (\"garbage collect\" the codebase)",
    "add Tup and Get constructors. Tup behaves similarly to a superposition of label 0, and is represented as (a,b). Get behaves similarly to a duplication with label 0, and is represented as ! (a,b) = x",
    "extend Lam and App nodes to also store a label, just like Sups and Dups. the App-Lam rule must be updated so that, when the labels are different, the nodes will commute instead of beta-reducing",
    "replace the 'λx body' syntax by '\\x body'",
]

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

def format_contexts(hs_ctx: FileContext, c_ctx: FileContext, line_number: bool = False):
    skeleton1 = hs_ctx.stringify(line_number=line_number)
    skeleton2 = c_ctx.stringify(line_number=line_number)
    context = ""
    if skeleton1: context += f"./hvm.hs:\n{skeleton1}\n---------\n"
    if skeleton2: context += f"./hvm.c:\n{skeleton2}\n---------\n"
    return context.strip()

@dataclass
class CodeSymbol:
    name: str
    relevancy: float

@dataclass
class BlockNumber:
    reasoning: str
    number: int
    confidence: float

@dataclass
class RefactoringAnalysis:
    explanation: str
    thinking: str
    relevancy: float
    block_numbers: list[BlockNumber]

def parse_explanation(text: str) -> str:
    """Parse the explanation section into list of (title, description) pairs"""
    match = re.search(r'<explanation>(.*?)</explanation>', text, re.DOTALL)
    if not match: return ""
    content = match.group(1).strip()
    return content

def parse_block_numbers(text: str) -> list[BlockNumber]:
    """Parse the block numbers section into list of (number, confidence) pairs"""
    match = re.search(r'<relevant_block_numbers>(.*?)</relevant_block_numbers>', text, re.DOTALL)
    if not match: return []
    content = match.group(1).strip()
    return [BlockNumber(reasoning, int(num), float(conf)) for reasoning, num, conf in re.findall(r'<reasoning>(.*?)</reasoning><block_number>(.*?)</block_number><confidence>(.*?)</confidence>', content)]

def parse_refactoring_analysis(text: str) -> RefactoringAnalysis:
    """Parse the complete analysis output format"""
    explanation = parse_explanation(text)
    relevancy = 0.0
    rel_match = re.search(r'<relevancy>(.*?)</relevancy>', text)
    if rel_match:
        try:
            relevancy = float(rel_match.group(1))
        except ValueError:
            pass
    thinking = re.search(r'<thinking>(.*?)</thinking>', text)
    if thinking: thinking = thinking.group(1)
    block_numbers = parse_block_numbers(text)
    return RefactoringAnalysis(
        explanation=explanation,
        thinking=thinking,
        relevancy=relevancy,
        block_numbers=block_numbers
    )

async def understand_and_classify_symbols_parallel(query: str, names: list[str], hs_nodes: list[Node], c_nodes: list[Node], model: str):
    """Run multiple symbol classifications in parallel"""
    tasks = []
    for name in names:
        task = asyncio.create_task(understand_and_classify_symbol_async(query, name, hs_nodes, c_nodes, model))
        tasks.append(task)
    results = await asyncio.gather(*tasks)
    return results

async def understand_and_classify_symbol_async(query: str, name: str, hs_nodes: list[Node], c_nodes: list[Node], model: str):
    """Single mega-prompt called in parallel for each name."""
    hs_ctx, c_ctx = create_contexts_for_name(name, hs_nodes, c_nodes)
    codebase_context = format_contexts(hs_ctx, c_ctx)
    messages = [
        Message(role="system", content=f"""
- You are an expert codebase explainer assisting a user with a refactoring task.
- You are given code related to "{name}" in the HVM3 codebase.

1. Summarize the role and purpose of "{name}" into a concise markdown tree within <explanation>...</explanation> tags.
2. Think about whether any of the code is relevant to the refactoring query in <thinking>...</thinking> tags. Rate the overall relevancy from 0 to 1 within <relevancy>...</relevancy> tags.
3. Only list the BLOCK numbers for code blocks that will definitely require editing in the refactoring. If there are none, simply return an empty list.

# Output Format

<explanation>
- **title 1**
  - description 1
- **title 2**
  - description 2
- **Refactoring Context**
  - The query involves refactoring ...
  - This change will affect...
</explanation>
<thinking>Are there any blocks that will definitely require editing? ...</thinking>
<relevancy>0-1</relevancy>
<relevant_block_numbers>
- <reasoning>This block will definitely require editing because...</reasoning><block_number>1</block_number><confidence>0-1</confidence>
- ...
</relevant_block_numbers>

If there are no relevant blocks, simply return an empty list.
""".strip()),
        Message(role="user", content=f"""
# Codebase Summary
{codebase_summary}

# Query
{query}

# Codebase Context
{codebase_context}
""".strip()),
    ]
    print_msg(messages)
    resp_msg = Message(role="assistant", content=await achat(messages=messages, model=model))
    print_msg(resp_msg)
    parsed = parse_refactoring_analysis(resp_msg.content)
    print(parsed)
    return parsed

def count_tokens(text: str) -> int:
    return len(tiktoken.get_encoding("cl100k_base").encode(text))

if __name__ == "__main__":
    hvm_names, hs_nodes, c_nodes = get_all_names()
    # hs_ctx, c_ctx = create_contexts_for_name(hvm_names[0], hs_nodes, c_nodes)
    # print(format_contexts(hs_ctx, c_ctx))

    # results: list[RefactoringAnalysis] = asyncio.run(
    #     understand_and_classify_symbols_parallel(
    #         example_queries[5],
    #         hvm_names,
    #         hs_nodes,
    #         c_nodes,
    #         model="deepseek-chat"
    #     )
    # )
    # all_blocks: list[BlockNumber] = [block for result in results for block in result.block_numbers]
    # high_confidence_blocks: list[BlockNumber] = sorted({block.number: block for block in all_blocks if block.confidence >= 0.9}.values(), key=lambda x: x.number)
    # print(f"number of high confidence blocks: {len(high_confidence_blocks)}")
    # for block in high_confidence_blocks:
    #     print(block.number)
    #     print(block.reasoning)
    #     print(block.confidence)
    #     print('----')

    import dspy
    from llms import model_to_provider, provider_to_api_key, provider_to_base_url
    from typing import Literal

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

    async def run_parallel_tasks_with_progress(tasks: list[Callable], desc: str = "Tasks") -> list[Any]:
        time_start = time.time()
        async def wrapped_task(task: Callable, index: int, pbar: tqdm) -> tuple[int, Any]:
            result = await task()
            pbar.update(1)
            return index, result
        with tqdm(total=len(tasks), desc=desc) as pbar:
            results = await asyncio.gather(*[
                wrapped_task(task, i, pbar) 
                for i, task in enumerate(tasks)
            ])
        time_end = time.time()
        print(f"{desc}: time taken: {time_end - time_start}")
        return [result for _, result in sorted(results, key=lambda x: x[0])]

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
        """Classify which blocks **must** be edited during a refactor."""
        codebase_summary: str = dspy.InputField()
        refactoring_task: str = dspy.InputField()
        codebase_symbol: str = dspy.InputField(desc="The name of the codebase symbol that may or may not be relevant to the refactoring task")
        codebase_symbol_explanation: str = dspy.InputField()
        codebase_context: str = dspy.InputField()

        blocks: list[Block] = dspy.OutputField(desc="A list of each block that must be edited along with the reason why it must be edited and your confidence.")

    name = hvm_names[0]
    classify = dspy.ChainOfThoughtWithHint(ClassifyBlocksToEdit)
    output = classify(
        codebase_summary=codebase_summary,
        codebase_context=format_contexts(*create_contexts_for_name(name, hs_nodes, c_nodes)),
        codebase_symbol=name,
        codebase_symbol_explanation=symbol_explanation_map[name],
        refactoring_task=example_queries[0],
        hint="Debate whether or not a block in the codebase context needs to be edited.",
    )
    print(output)
    # print(lm.history[-1])

