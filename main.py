import tiktoken
import asyncio
from typing import Generator, Literal, Optional, List
from dataclasses import dataclass, field
import datetime
import shutil
import textwrap
from typing import Generator, Literal, Optional, List
from openai import AsyncOpenAI, OpenAI, NOT_GIVEN
from dotenv import load_dotenv
import os
from tree_sitter import Node, Language, Parser
from tree_sitter_languages import get_language, get_parser
import re
import warnings

##########################
## LLM SETUP
##########################

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

openai = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
aopenai = AsyncOpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

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
                    lang = block.split("\n")[0]
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

def chat(messages: list[Message], model: str) -> str:
    is_o1 = model.startswith("o1")
    if is_o1:
        messages = list(_prep_o1(messages))

    messages_dicts = msgs2dicts(messages)

    response = openai.chat.completions.create(
        model=model,
        messages=messages_dicts,
        temperature=TEMPERATURE if not is_o1 else NOT_GIVEN,
        top_p=TOP_P if not is_o1 else NOT_GIVEN,
        tools=NOT_GIVEN,
        extra_headers=(
            openrouter_headers if "openrouter.ai" in str(openai.base_url) else {}
        ),
    )
    content = response.choices[0].message.content
    assert content
    return content

async def achat(messages: list[Message], model: str, timeout: float = 40.0) -> str:
    """Async version of chat()"""
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

deepseek = 'deepseek/deepseek-chat'

##########################
## TREE SITTER
##########################

def setup_tree_sitter(file: str):
    ext = file.split('.')[1]
    if ext == "hs": # outdated in tree-sitter-languages
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
    
class EmptyNode:
    start_point: tuple[int, int]
    end_point: tuple[int, int]
    type: str = "empty"
    parent: None = None

    def __init__(self, start_point: tuple[int, int], end_point: tuple[int, int]):
        self.start_point = start_point
        self.end_point = end_point

class FileContext:
    """Tree of context for a single source code file. Show nodes using the `show` methods."""
    def __init__(self, file: str | None = None, content: str | None = None):
        self.file = file
        if content: self.code = content
        else:
            with open(file, 'r') as f: self.code = f.read()
        self.lines = self.code.splitlines()
        self.parser, self.language = setup_tree_sitter(file)
        self.tree = self.parser.parse(bytes(self.code, "utf8"))
        self.root_node = self.tree.root_node
        self.show_indices = set()
        self.nodes = [[] for _ in range(len(self.lines) + 1)]
        self.scopes = [set() for _ in range(len(self.lines) + 1)]
        self.walk_tree(self.root_node)

    def show_scopes_mentioning(
        self,
        name: str,
        scope: Literal["full", "line", "partial"] = "line",
        parents: Literal["none", "all"] = "none"
    ):
        line_matches = []
        for i, line in enumerate(self.lines):
            if name in line or camel_to_snake(name) in line or snake_to_camel(name) in line:
                line_matches.append(i + 1)
                if scope != "full": continue
                signature_node: Optional[Node] = next((n for n in self.nodes[i] if n.type == "signature"), None)
                if signature_node:
                    try:
                        fn_name = signature_node.named_child(0).text.decode("utf-8")
                        self.show(query=self.definition_query(fn_name, include_body=True), scope="full", last_line=False, parents=parents)
                    except: pass
        self.show(lines=line_matches, scope=scope, last_line=False, parents=parents)
        return self

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
            return node.type in ["function", "signature", "bind", "data_type", "variable"]
        elif self.language.name == "c":
            return node.type in ["function_definition", "declaration","struct_specifier", "enum_specifier"]
        else: raise ValueError(f"Unsupported language: {self.language.name}")

    def show_nearest_definition(self, line: int):
        node = self.node_for_line(line, definition=True)
        if self.is_definition(node) or node.type == "empty": return node
        parent = node.parent
        while parent and not self.is_definition(parent): parent = parent.parent
        x = parent or node
        self.show_indices.update(range(x.start_point[0], x.end_point[0] + 1))
        return self

    def find_all_children(self, node):
        children = [node]
        for child in node.children: children += self.find_all_children(child)
        return children
    
    def node_for_line(self, line: int, definition: bool = False):
        if not self.lines[line - 1].strip(): return EmptyNode((line - 1, 0), (line - 1, 0))
        largest = None
        def size(node: Optional[Node]): return (node.end_point[0] - node.start_point[0]) if node else float("-inf")
        for node in self.nodes[line - 1]:
            if (size(node) > size(largest)) and (not definition or self.is_definition(node)):
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
    
    def show_skeleton(self):
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
    nodes = ctx.query(ctx.definition_query(include_body=False))

    c_names = set([node.text.decode("utf-8") for node in nodes if node.type == "identifier" or node.type == "type_identifier"])
    print(f"c file found {len(c_names)} names")

    ctx = FileContext("hvm-code.hs")
    nodes = ctx.query(ctx.definition_query())
    hs_names = set([node.text.decode("utf-8") for node in nodes])
    print(f"hs file found {len(hs_names)} names")

    all_names = c_names.union(hs_names)
    print(f"all files found {len(all_names)} names")
    ret = sorted(all_names)
    print(ret)
    return ret

def create_contexts_for_name(name: str):
    """To create context for a particular name, we simply show every line that mentions the name.
    We also include variations of the name - camel case, snake case, etc.
    We also include nearby block numbers.
    """
    hs_ctx = FileContext("hvm-code.hs")
    c_ctx = FileContext("hvm-code.c")
    hs_ctx.show_scopes_mentioning(name)
    c_ctx.show_scopes_mentioning(name)
    # also include block numbers
    hs_ctx.show_nearby_block_nums()
    c_ctx.show_nearby_block_nums()
    return hs_ctx, c_ctx

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

##########################
## PROMPTS
##########################

example_query1 = "make CTRs store only the CID in the Lab field, and move the arity to a global static object in C"
example_query2 = "replace the 'λx body' syntax by '\\x body'"
example_query3 = ""

codebase_summary = """
Higher-order Virtual Machine 2 (HVM2) is a massively parallel Interaction Combinator evaluator.
By compiling programs from high-level languages (such as Python and Haskell) to HVM, one can run these languages directly on massively parallel hardware, like GPUs, with near-ideal speedup.
HVM2 is the successor to HVM1, a 2022 prototype of this concept. Compared to its predecessor, HVM2 is simpler, faster and, most importantly, more correct. HOC provides long-term support for all features listed on its PAPER.
This repository provides a low-level IR language for specifying the HVM2 nets and a compiler from that language to C and CUDA. It is not meant for direct human usage. If you're looking for a high-level language to interface with HVM2, check Bend instead.
""".strip()

def format_contexts(hs_ctx: FileContext, c_ctx: FileContext):
    skeleton1 = hs_ctx.stringify(line_number=True)
    skeleton2 = c_ctx.stringify(line_number=True)
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
    number: int
    confidence: float

@dataclass
class RefactoringAnalysis:
    explanation: str
    relevancy: float
    code_symbols: List[CodeSymbol]
    reasoning: str
    block_numbers: list[BlockNumber]

def parse_explanation(text: str) -> str:
    """Parse the explanation section into list of (title, description) pairs"""
    match = re.search(r'<explanation>(.*?)</explanation>', text, re.DOTALL)
    if not match: return ""
    content = match.group(1).strip()
    return content

def parse_block_numbers(text: str) -> list[BlockNumber]:
    """Parse the block numbers section into list of (number, confidence) pairs"""
    match = re.search(r'<block_numbers>(.*?)</block_numbers>', text, re.DOTALL)
    if not match: return []
    content = match.group(1).strip()
    return [BlockNumber(int(num), float(conf)) for num, conf in re.findall(r'<block_number>(.*?)</block_number><confidence>(.*?)</confidence>', content)]

def parse_code_symbols(text: str) -> tuple[List[CodeSymbol], str]:
    """Parse the code symbols section, returns (symbols, reasoning)"""
    # Extract content between <relevant_code_symbols> tags
    match = re.search(r'<relevant_code_symbols>(.*?)</relevant_code_symbols>', text, re.DOTALL)
    if not match:
        return [], ""
    
    content = match.group(1).strip()
    symbols = []
    reasoning = ""
    
    # Extract reasoning
    reason_match = re.search(r'<reasoning>(.*?)</reasoning>', content, re.DOTALL)
    if reason_match:
        reasoning = reason_match.group(1).strip()
    
    # Parse symbols
    symbol_pattern = r'<name>(.*?)</name><relevancy>(.*?)</relevancy>'
    for name_match, rel_match in re.findall(symbol_pattern, content):
        try:
            relevancy = float(rel_match)
            symbols.append(CodeSymbol(name_match.strip(), relevancy))
        except ValueError:
            continue
            
    return symbols, reasoning

def parse_refactoring_analysis(text: str) -> RefactoringAnalysis:
    """Parse the complete analysis output format"""
    explanation = parse_explanation(text)
    
    # Parse relevancy
    relevancy = 0.0
    rel_match = re.search(r'<relevancy>(.*?)</relevancy>', text)
    if rel_match:
        try:
            relevancy = float(rel_match.group(1))
        except ValueError:
            pass
    code_symbols, reasoning = parse_code_symbols(text)
    block_numbers = parse_block_numbers(text)
    return RefactoringAnalysis(
        explanation=explanation,
        relevancy=relevancy,
        reasoning=reasoning,
        code_symbols=code_symbols,
        block_numbers=block_numbers
    )

async def understand_and_classify_symbols_parallel(query: str, names: list[str]):
    """Run multiple symbol classifications in parallel"""
    tasks = []
    for name in names:
        task = asyncio.create_task(understand_and_classify_symbol_async(query, name))
        tasks.append(task)
    results = await asyncio.gather(*tasks)
    return results

async def understand_and_classify_symbol_async(query: str, name: str):
    """Single mega-prompt called in parallel for each name."""
    hs_ctx, c_ctx = create_contexts_for_name(name)
    codebase_context = format_contexts(hs_ctx, c_ctx)
    messages = [
        Message(role="system", content=f"""
- You are an expert codebase explainer assisting a user with a refactoring task.
- You are given code related to "{name}" in the HVM3 codebase.

1. Summarize the role and purpose of "{name}" into a concise markdown tree within <explanation>...</explanation> tags.
2. Think about whether any of the related code is relevant to the refactoring query. Rate the overall relevancy from 0 to 1 within <relevancy>...</relevancy> tags.
3. List the functions or types that contain code which would get changed by the refactoring by reasoning step-by-step within <reasoning>...</reasoning> tags.
4. If some code inside a function or type will get changed by the refactoring, list the name of the function or type exactly as it appears in the codebase.
5. Finally, list the specific block numbers of the code that you are sure will get changed in the refactoring. Only list blocks that you are confident are relevant to the refactoring.

# Output Format

<explanation>
- **title 1**
  - description 1
- **title 2**
  - description 2
- ...
  - ...
</explanation>
<relevancy>0-1</relevancy>
<relevant_code_symbols>
<reasoning>If the refactoring is performed, the following code symbols or code within their scope will/will not need to change because...</reasoning>
- <name>name 1</name><relevancy>0-1</relevancy>
- <name>name 2</name><relevancy>0-1</relevancy>
- ...
</relevant_code_symbols>
<block_numbers>
- <block_number>1</block_number><confidence>0-1</confidence>
- <block_number>2</block_number><confidence>0-1</confidence>
- ...
</block_numbers>
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
    resp_msg = Message(role="assistant", content=await achat(messages=messages, model=deepseek))
    print_msg(resp_msg)
    parsed = parse_refactoring_analysis(resp_msg.content)
    print(parsed)
    return parsed

def count_tokens(text: str) -> int:
    return len(tiktoken.get_encoding("cl100k_base").encode(text))

if __name__ == "__main__":
    hvm_names = get_all_names()
    name = hvm_names[0]
    hs_ctx, c_ctx = create_contexts_for_name(name)
    formatted = format_contexts(hs_ctx, c_ctx)
    print("---")
    print(f"NAME: {name}\n{formatted}")
    print(f"TOKENS: {count_tokens(formatted)}")
    print("---")
    # inspect_code()