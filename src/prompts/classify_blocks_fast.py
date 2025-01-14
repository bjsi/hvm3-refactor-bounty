from src.utils import convert_confidence_to_num

# exported from DSPy to run in litellm

system_prompt = """
Your input fields are:
1. `codebase_symbol_explanations` (str)
2. `task` (str)
3. `specific_context` (str)
4. `block_number` (int)

Your output fields are:
1. `task_reflection` (str): Think step-by-step about the refactoring task
2. `reasoning` (str): Think step-by-step about whether or not the code in the BLOCK must be directly modified
3. `requires_direct_modification` (bool)
4. `confidence` (Literal[low, medium, high, very high]): Your confidence that this block must be directly modified

All interactions will be structured in the following way, with the appropriate values filled in.

[[ ## codebase_symbol_explanations ## ]]
{codebase_symbol_explanations}

[[ ## task ## ]]
{task}

[[ ## specific_context ## ]]
{specific_context}

[[ ## block_number ## ]]
{block_number}

[[ ## task_reflection ## ]]
{task_reflection}

[[ ## reasoning ## ]]
{reasoning}

[[ ## requires_direct_modification ## ]]
{requires_direct_modification}        # note: the value you produce must be True or False

[[ ## confidence ## ]]
{confidence}        # note: the value you produce must be one of: low; medium; high; very high

[[ ## completed ## ]]

In adhering to this structure, your objective is: 
        Determine whether code in the given BLOCKs require direct modification.
        BLOCKs only require direct modification if the code directly visible inside them needs to be changed.
        Rules you must follow:
        - Empty blocks never require direct modification.
"""

user1 = """
This is an example of the task, though some input or output fields are not supplied.

[[ ## codebase_symbol_explanations ## ]]
None

[[ ## task ## ]]
measure interactions by interaction type instead of just storing the total count. report results segmented by interaction type

[[ ## specific_context ## ]]
None

[[ ## block_number ## ]]
151

Respond with the corresponding output fields, starting with the field `[[ ## task_reflection ## ]]`, then `[[ ## reasoning ## ]]`, then `[[ ## requires_direct_modification ## ]]` (must be formatted as a valid Python bool), then `[[ ## confidence ## ]]` (must be formatted as a valid Python Literal[low, medium, high, very high]), and then ending with the marker for `[[ ## completed ## ]]`.
""".strip(),

assistant1 = """
[[ ## task_reflection ## ]]
The task involves modifying the interaction counting mechanism in the HVM3 runtime to track interactions by type (e.g., `APP`, `MAT`, `DUP`, etc.) instead of just maintaining a total count. This will require changes to the interaction counting logic, likely in the C backend, to categorize and store counts for each interaction type. Additionally, the Haskell frontend will need updates to report these segmented results, possibly by extending the `State` structure or adding new data structures to hold the segmented counts.

[[ ## reasoning ## ]]
The provided block (BLOCK 151) contains the `liftDups` function, which handles the lifting of `Var` terms. This function is unrelated to the interaction counting mechanism or the global interaction counter (`HVM.itrs`). Since the task focuses on modifying how interactions are counted and reported, and this block does not involve interaction counting or reporting logic, it does not require direct modification. The changes should instead be made in functions like `inc_itr`, `reduce`, and other reduction-related functions where interaction counting occurs.

[[ ## requires_direct_modification ## ]]
False

[[ ## confidence ## ]]
high

[[ ## completed ## ]]
""".strip()

user2 = """
This is an example of the task, though some input or output fields are not supplied.

[[ ## codebase_symbol_explanations ## ]]
None

[[ ## task ## ]]
make CTRs store only the CID in the Lab field, and move the arity to a global static object in C

[[ ## specific_context ## ]]
None

[[ ## block_number ## ]]
345

Respond with the corresponding output fields, starting with the field `[[ ## task_reflection ## ]]`, then `[[ ## reasoning ## ]]`, then `[[ ## requires_direct_modification ## ]]` (must be formatted as a valid Python bool), then `[[ ## confidence ## ]]` (must be formatted as a valid Python Literal[low, medium, high, very high]), and then ending with the marker for `[[ ## completed ## ]]`.
""".strip()

assistant2 = """
Assistant message:

[[ ## task_reflection ## ]]
The task involves modifying how constructor terms (CTRs) store their metadata. Currently, the `Lab` field of a CTR term stores both the constructor ID (CID) and the arity, encoded together using the `u12v2` encoding scheme. The task requires moving the arity out of the `Lab` field and into a global static object in C, so that the `Lab` field only stores the CID. This change will require updates to the code that interacts with CTR terms, particularly where the arity is extracted or used.

[[ ## reasoning ## ]]
The `print_term` function in BLOCK 345 currently uses `term_lab(term)` to extract the label, which contains both the CID and arity. Since the task involves moving the arity to a global static object, the `print_term` function will need to be updated to reflect this change. Specifically, the `term_lab(term)` call will now return only the CID, and the arity will need to be fetched from the global static object. This means the function will require direct modification to correctly print the term's metadata, ensuring that the arity is displayed accurately alongside the CID.

[[ ## requires_direct_modification ## ]]
True

[[ ## confidence ## ]]
very high

[[ ## completed ## ]]
""".strip()

user3 = """
This is an example of the task, though some input or output fields are not supplied.

[[ ## codebase_symbol_explanations ## ]]
None

[[ ## task ## ]]
measure interactions by interaction type instead of just storing the total count. report results segmented by interaction type

[[ ## specific_context ## ]]
None

[[ ## block_number ## ]]
286

Respond with the corresponding output fields, starting with the field `[[ ## task_reflection ## ]]`, then `[[ ## reasoning ## ]]`, then `[[ ## requires_direct_modification ## ]]` (must be formatted as a valid Python bool), then `[[ ## confidence ## ]]` (must be formatted as a valid Python Literal[low, medium, high, very high]), and then ending with the marker for `[[ ## completed ## ]]`.
""".strip()

assistant3 = """
[[ ## task_reflection ## ]]
The task involves modifying the interaction counting mechanism in the HVM3 runtime to track interactions by type (e.g., `APP`, `MAT`, `DUP`, etc.) instead of just maintaining a total count. This will require changes to the interaction counting logic, likely in the C backend, to categorize and store counts for each interaction type. Additionally, the Haskell frontend will need updates to report these segmented results, possibly by extending the `State` structure or adding new data structures to hold the segmented counts.

[[ ## reasoning ## ]]
The provided block (BLOCK 286) handles the reduction of `OPX` terms but does not directly interact with the global interaction counter (`inc_itr` or `get_itr`). The task of tracking interactions by type primarily involves modifying the interaction counting mechanism, which is likely handled elsewhere in the codebase (e.g., in the `State` structure or the C backend). Since the block itself does not currently handle interaction counting, it does not require direct modification to achieve the task. The changes should instead focus on the `State` structure and the interaction counting mechanism, leaving this block largely unchanged.

[[ ## requires_direct_modification ## ]]
False

[[ ## confidence ## ]]
very high

[[ ## completed ## ]]
""".strip()

user4 = """
This is an example of the task, though some input or output fields are not supplied.

[[ ## codebase_symbol_explanations ## ]]
None

[[ ## task ## ]]
completely remove native numbers as a feature

[[ ## specific_context ## ]]
None

[[ ## block_number ## ]]
344

Respond with the corresponding output fields, starting with the field `[[ ## task_reflection ## ]]`, then `[[ ## reasoning ## ]]`, then `[[ ## requires_direct_modification ## ]]` (must be formatted as a valid Python bool), then `[[ ## confidence ## ]]` (must be formatted as a valid Python Literal[low, medium, high, very high]), and then ending with the marker for `[[ ## completed ## ]]`.
""".strip()

assistant4 = """
[[ ## task_reflection ## ]]
The task involves completely removing native numbers as a feature from the HVM3 codebase. This means that all references to native number types (e.g., `u32`, `u64`, `W32`, etc.) and their associated operations must be eliminated. This will require identifying and modifying all parts of the codebase that rely on native numbers, including data structures, reduction functions, and compilation logic. The removal of native numbers will likely necessitate the introduction of alternative representations for numerical values, such as Church numerals or other functional encodings.

[[ ## reasoning ## ]]
The `print_tag` function is a utility function that maps `Tag` enum values to their string representations. Since the task involves removing native numbers, the `W32` tag (which represents 32-bit words) must be removed from the `Tag` enum. Consequently, the corresponding case in the `print_tag` function must also be removed. This is a direct modification because the `W32` case is explicitly present in the code and must be deleted to align with the task's requirements.

[[ ## requires_direct_modification ## ]]
True

[[ ## confidence ## ]]
very high

[[ ## completed ## ]]
""".strip()

user5 = """
This is an example of the task, though some input or output fields are not supplied.

[[ ## codebase_symbol_explanations ## ]]
None

[[ ## task ## ]]
completely remove native numbers as a feature

[[ ## specific_context ## ]]
None

[[ ## block_number ## ]]
108

Respond with the corresponding output fields, starting with the field `[[ ## task_reflection ## ]]`, then `[[ ## reasoning ## ]]`, then `[[ ## requires_direct_modification ## ]]` (must be formatted as a valid Python bool), then `[[ ## confidence ## ]]` (must be formatted as a valid Python Literal[low, medium, high, very high]), and then ending with the marker for `[[ ## completed ## ]]`.
""".strip()

assistant5 = """
[[ ## task_reflection ## ]]
The task involves completely removing native numbers as a feature from the HVM3 codebase. This means that all references to native number types (e.g., `u32`, `u64`, `W32`, etc.) and their associated operations must be eliminated. This will require identifying and modifying all parts of the codebase that rely on native numbers, including data structures, reduction functions, and compilation logic. The removal of native numbers will likely necessitate the introduction of alternative representations for numerical values, such as Church numerals or other functional encodings.

[[ ## reasoning ## ]]
The `compileFastCore` function is responsible for compiling `Core` terms in fast mode, which includes handling numeric literals and operations. If native numbers are to be removed, this function would need to be modified to either exclude or replace the compilation logic for numeric terms. However, the specific context provided does not show the implementation details of `compileFastCore`, so it is unclear whether this block directly handles native numbers. If it does, direct modification would be necessary; otherwise, the block might not require changes. Given the task's scope, it is likely that this block is involved in compiling numeric terms, but without seeing the full implementation, this remains uncertain.

[[ ## requires_direct_modification ## ]]
True

[[ ## confidence ## ]]
medium

[[ ## completed ## ]]
""".strip()

user6 = """
This is an example of the task, though some input or output fields are not supplied.

[[ ## codebase_symbol_explanations ## ]]
None

[[ ## task ## ]]
measure interactions by interaction type instead of just storing the total count. report results segmented by interaction type

[[ ## specific_context ## ]]
None

[[ ## block_number ## ]]
297

Respond with the corresponding output fields, starting with the field `[[ ## task_reflection ## ]]`, then `[[ ## reasoning ## ]]`, then `[[ ## requires_direct_modification ## ]]` (must be formatted as a valid Python bool), then `[[ ## confidence ## ]]` (must be formatted as a valid Python Literal[low, medium, high, very high]), and then ending with the marker for `[[ ## completed ## ]]`.
""".strip()

assistant6 = """
[[ ## task_reflection ## ]]
The task involves modifying the interaction counting mechanism in the HVM3 runtime to track interactions by type (e.g., `APP`, `MAT`, `DUP`, etc.) instead of just maintaining a total count. This will require changes to the interaction counting logic, likely in the C backend, to categorize and store counts for each interaction type. Additionally, the Haskell frontend will need updates to report these segmented results, possibly by extending the `State` structure or adding new data structures to hold the segmented counts.

[[ ## reasoning ## ]]
The `reduceRefAt_LogF` function in BLOCK 297 is responsible for handling the `@LOG` primitive, which logs messages during runtime. It currently increments the interaction counter using `incItr` without specifying the interaction type. To implement the task, this function must be modified to pass the interaction type (e.g., `LOG`) to `incItr`. This change is necessary to enable segmentation of interaction counts by type. The modification is straightforward and involves updating the `incItr` call to include the interaction type. The rest of the function's logic (e.g., logging messages, handling arity checks) remains unchanged.

[[ ## requires_direct_modification ## ]]
True

[[ ## confidence ## ]]
very high

[[ ## completed ## ]]
""".strip()

user7 = """
This is an example of the task, though some input or output fields are not supplied.

[[ ## codebase_symbol_explanations ## ]]
None

[[ ## task ## ]]
measure interactions by interaction type instead of just storing the total count. report results segmented by interaction type

[[ ## specific_context ## ]]
None

[[ ## block_number ## ]]
366

Respond with the corresponding output fields, starting with the field `[[ ## task_reflection ## ]]`, then `[[ ## reasoning ## ]]`, then `[[ ## requires_direct_modification ## ]]` (must be formatted as a valid Python bool), then `[[ ## confidence ## ]]` (must be formatted as a valid Python Literal[low, medium, high, very high]), and then ending with the marker for `[[ ## completed ## ]]`.
""".strip()

assistant7 = """
[[ ## task_reflection ## ]]
The task involves modifying the interaction counting mechanism in the HVM3 runtime to track interactions by type (e.g., `APP`, `MAT`, `DUP`, etc.) instead of just maintaining a total count. This will require changes to the interaction counting logic, likely in the C backend, to categorize and store counts for each interaction type. Additionally, the Haskell frontend will need updates to report these segmented results, possibly by extending the `State` structure or adding new data structures to hold the segmented counts.

[[ ## reasoning ## ]]
The `reduce_mat_ctr` function is responsible for handling `MAT-CTR` interactions, which are a specific type of reduction operation. To implement the task, we need to replace the generic `inc_itr()` call with a more specific counter increment that tracks `MAT-CTR` interactions. This would involve modifying the function to increment a dedicated counter for `MAT-CTR` interactions, such as `inc_itr_mat_ctr()`. Since the function directly performs a `MAT-CTR` interaction, it must be modified to ensure accurate tracking of this interaction type. The change is localized to this block and does not require modifications to other parts of the codebase.

[[ ## requires_direct_modification ## ]]
True

[[ ## confidence ## ]]
very high

[[ ## completed ## ]]
""".strip()

user8 = """
This is an example of the task, though some input or output fields are not supplied.

[[ ## codebase_symbol_explanations ## ]]
None

[[ ## task ## ]]
remove the list/string pretty printers

[[ ## specific_context ## ]]
None

[[ ## block_number ## ]]
407

Respond with the corresponding output fields, starting with the field `[[ ## task_reflection ## ]]`, then `[[ ## reasoning ## ]]`, then `[[ ## requires_direct_modification ## ]]` (must be formatted as a valid Python bool), then `[[ ## confidence ## ]]` (must be formatted as a valid Python Literal[low, medium, high, very high]), and then ending with the marker for `[[ ## completed ## ]]`.
""".strip()

assistant8 = """
[[ ## task_reflection ## ]]
The task involves removing the list/string pretty printers from the codebase. These pretty printers are likely part of the debugging or visualization utilities, such as `pretty`, `prettyLst`, and `prettyStr`, which convert `Core` terms into human-readable string representations. Removing these utilities will require identifying and eliminating their usage throughout the codebase, ensuring that no functionality depends on them. This task may also involve updating or replacing debugging tools that rely on these pretty printers.

[[ ## reasoning ## ]]
The block in question (BLOCK 407) is part of the `normal` function, which handles term normalization in the HVM3 runtime. This block specifically deals with `DP0` and `DP1` tags, which are related to term duplication. The code retrieves a term from memory, normalizes it, and updates the memory location. There is no direct interaction with list/string pretty printers (`pretty`, `prettyLst`, `prettyStr`) in this block. The block is focused on term reduction and memory management, which are core functionalities of the runtime and unrelated to the pretty-printing utilities targeted by the task.

[[ ## requires_direct_modification ## ]]
False

[[ ## confidence ## ]]
very high

[[ ## completed ## ]]
""".strip()

user9 = """
This is an example of the task, though some input or output fields are not supplied.

[[ ## codebase_symbol_explanations ## ]]
None

[[ ## task ## ]]
extend the size of the addr field on runtime nodes from 32 to 40 bits, and reduce the label field from 24 to 16 bits

[[ ## specific_context ## ]]
None

[[ ## block_number ## ]]
348

Respond with the corresponding output fields, starting with the field `[[ ## task_reflection ## ]]`, then `[[ ## reasoning ## ]]`, then `[[ ## requires_direct_modification ## ]]` (must be formatted as a valid Python bool), then `[[ ## confidence ## ]]` (must be formatted as a valid Python Literal[low, medium, high, very high]), and then ending with the marker for `[[ ## completed ## ]]`.
""".strip()

assistant9 = """
[[ ## task_reflection ## ]]
The task involves modifying the bit allocation for the `addr` and `label` fields in runtime nodes. Specifically, we need to increase the `addr` field from 32 to 40 bits and reduce the `label` field from 24 to 16 bits. This change will likely require adjustments to the data structures and functions that interact with these fields, particularly in the C backend where memory layout and bit manipulation are handled. The Haskell frontend may also need updates to reflect these changes, especially in functions that construct or manipulate `Term` instances.

[[ ## reasoning ## ]]
The provided block (BLOCK 348) is labeled as "Evaluation" but does not contain any visible code. Since the task involves modifying the bit allocation of fields in runtime nodes, direct modification would only be necessary if this block contains code that interacts with the `addr` or `label` fields. Without visible code, it is impossible to determine whether this block requires direct modification. The task is more likely to affect blocks that define the runtime node structure, memory allocation functions, or functions that manipulate `Loc` and `Lab` values.

[[ ## requires_direct_modification ## ]]
False

[[ ## confidence ## ]]
high

[[ ## completed ## ]]
""".strip()

user10 = """
This is an example of the task, though some input or output fields are not supplied.

[[ ## codebase_symbol_explanations ## ]]
None

[[ ## task ## ]]
extend the size of the addr field on runtime nodes from 32 to 40 bits, and reduce the label field from 24 to 16 bits

[[ ## specific_context ## ]]
None

[[ ## block_number ## ]]
235

Respond with the corresponding output fields, starting with the field `[[ ## task_reflection ## ]]`, then `[[ ## reasoning ## ]]`, then `[[ ## requires_direct_modification ## ]]` (must be formatted as a valid Python bool), then `[[ ## confidence ## ]]` (must be formatted as a valid Python Literal[low, medium, high, very high]), and then ending with the marker for `[[ ## completed ## ]]`.
""".strip()

assistant10 = """
[[ ## task_reflection ## ]]
The task involves modifying the bit allocation for the `addr` and `label` fields in runtime nodes. Specifically, we need to increase the `addr` field from 32 to 40 bits and reduce the `label` field from 24 to 16 bits. This change will likely require adjustments to the data structures and functions that interact with these fields, particularly in the C backend where memory layout and bit manipulation are handled. The Haskell frontend may also need updates to reflect these changes, especially in functions that construct or manipulate `Term` instances.

[[ ## reasoning ## ]]
Block 235 contains a parser function (`parseEscapedChar`) that handles escaped characters in the input source code. This function is part of the frontend's parsing logic and does not interact with the runtime's memory layout or the `Term` data structure. The task focuses on modifying the runtime node structure, which is implemented in the C backend and the parts of the Haskell code that interface with it. Since this block is purely concerned with parsing and does not involve memory management or term representation, it does not require direct modification to achieve the task.

[[ ## requires_direct_modification ## ]]
False

[[ ## confidence ## ]]
very high

[[ ## completed ## ]]
""".strip()

user11 = """
This is an example of the task, though some input or output fields are not supplied.

[[ ## codebase_symbol_explanations ## ]]
None

[[ ## task ## ]]
implement a feature that prevents the user from creating two constructors with the same name. show a helpful error when that happens.

[[ ## specific_context ## ]]
None

[[ ## block_number ## ]]
360

Respond with the corresponding output fields, starting with the field `[[ ## task_reflection ## ]]`, then `[[ ## reasoning ## ]]`, then `[[ ## requires_direct_modification ## ]]` (must be formatted as a valid Python bool), then `[[ ## confidence ## ]]` (must be formatted as a valid Python Literal[low, medium, high, very high]), and then ending with the marker for `[[ ## completed ## ]]`.
""".strip()

assistant11 = """
[[ ## task_reflection ## ]]
None

[[ ## reasoning ## ]]
None

[[ ## requires_direct_modification ## ]]
False

[[ ## confidence ## ]]
low

[[ ## completed ## ]]
""".strip()

user12 = """
This is an example of the task, though some input or output fields are not supplied.

[[ ## codebase_symbol_explanations ## ]]
None

[[ ## task ## ]]
completely remove native numbers as a feature

[[ ## specific_context ## ]]
None

[[ ## block_number ## ]]
218

Respond with the corresponding output fields, starting with the field `[[ ## task_reflection ## ]]`, then `[[ ## reasoning ## ]]`, then `[[ ## requires_direct_modification ## ]]` (must be formatted as a valid Python bool), then `[[ ## confidence ## ]]` (must be formatted as a valid Python Literal[low, medium, high, very high]), and then ending with the marker for `[[ ## completed ## ]]`.
""".strip()

assistant12 = """
[[ ## task_reflection ## ]]
The task involves completely removing native numbers as a feature from the HVM3 codebase. This means that all references to native number types (e.g., `u32`, `u64`, `W32`, etc.) and their associated operations must be eliminated. This will require identifying and modifying all parts of the codebase that rely on native numbers, including data structures, reduction functions, and compilation logic. The removal of native numbers will likely necessitate the introduction of alternative representations for numerical values, such as Church numerals or other functional encodings.

[[ ## reasoning ## ]]
BLOCK 218 contains logic for parsing superposition terms (`Sup`) and references (`Ref`). Specifically, it handles the case where a superposition term is defined with a numeric label (e.g., `&123{...}`). This numeric label is parsed using `reads name`, which converts the string representation of a number into a `Word64` value. Since the task is to remove native numbers as a feature, this logic must be modified or removed. The block also handles the case where the name is not a number, which does not directly involve native numbers and can remain unchanged. Therefore, the part of the block that parses numeric labels must be directly modified to align with the task.

[[ ## requires_direct_modification ## ]]
True

[[ ## confidence ## ]]
high

[[ ## completed ## ]]
""".strip()

user13 = """
This is an example of the task, though some input or output fields are not supplied.

[[ ## codebase_symbol_explanations ## ]]
None

[[ ## task ## ]]
measure interactions by interaction type instead of just storing the total count. report results segmented by interaction type

[[ ## specific_context ## ]]
None

[[ ## block_number ## ]]
19

Respond with the corresponding output fields, starting with the field `[[ ## task_reflection ## ]]`, then `[[ ## reasoning ## ]]`, then `[[ ## requires_direct_modification ## ]]` (must be formatted as a valid Python bool), then `[[ ## confidence ## ]]` (must be formatted as a valid Python Literal[low, medium, high, very high]), and then ending with the marker for `[[ ## completed ## ]]`.
""".strip()

assistant13 = """
[[ ## task_reflection ## ]]
The task involves modifying the interaction counting mechanism in the HVM3 runtime to track interactions by type (e.g., `APP`, `MAT`, `DUP`, etc.) instead of just maintaining a total count. This will require changes to the interaction counting logic, likely in the C backend, to categorize and store counts for each interaction type. Additionally, the Haskell frontend will need updates to report these segmented results, possibly by extending the `State` structure or adding new data structures to hold the segmented counts.

[[ ## reasoning ## ]]
The provided block (BLOCK 19) handles the reduction of `DP0` terms, which are part of the dynamic duplication mechanism. Currently, the block does not include any interaction counting logic. To implement the task, we would need to add a mechanism to increment a counter specific to `DP0` interactions. This would involve modifying the block to include a call to a new function (e.g., `incInteractionType "DP0"`) or extending the existing `inc_itr` function to accept an interaction type parameter. Since the block does not currently include any interaction counting logic, it requires direct modification to support the new functionality.

[[ ## requires_direct_modification ## ]]
True

[[ ## confidence ## ]]
high

[[ ## completed ## ]]
""".strip()

user14 = """
This is an example of the task, though some input or output fields are not supplied.

[[ ## codebase_symbol_explanations ## ]]
None

[[ ## task ## ]]
extend the size of the addr field on runtime nodes from 32 to 40 bits, and reduce the label field from 24 to 16 bits

[[ ## specific_context ## ]]
None

[[ ## block_number ## ]]
461

Respond with the corresponding output fields, starting with the field `[[ ## task_reflection ## ]]`, then `[[ ## reasoning ## ]]`, then `[[ ## requires_direct_modification ## ]]` (must be formatted as a valid Python bool), then `[[ ## confidence ## ]]` (must be formatted as a valid Python Literal[low, medium, high, very high]), and then ending with the marker for `[[ ## completed ## ]]`.
""".strip()

assistant14 = """
[[ ## task_reflection ## ]]
The task involves modifying the bit allocation for the `addr` and `label` fields in runtime nodes. Specifically, we need to increase the `addr` field from 32 to 40 bits and reduce the `label` field from 24 to 16 bits. This change will likely require adjustments to the data structures and functions that interact with these fields, particularly in the C backend where memory layout and bit manipulation are handled. The Haskell frontend may also need updates to reflect these changes, especially in functions that construct or manipulate `Term` instances.

[[ ## reasoning ## ]]
Block 461 does not directly interact with the runtime node structure or the bit-level representation of `addr` and `label` fields. Instead, it focuses on renaming variables and processing the AST during the pretty-printing phase. The changes to the `addr` and `label` fields are low-level modifications that would primarily affect the C backend and runtime memory management, not the high-level AST manipulation in the Haskell frontend. Therefore, this block does not require direct modification to accommodate the task. The task's impact would be felt in the runtime's memory allocation and term representation, which are handled elsewhere in the codebase.

[[ ## requires_direct_modification ## ]]
False

[[ ## confidence ## ]]
very high

[[ ## completed ## ]]
""".strip()

user15 = """
This is an example of the task, though some input or output fields are not supplied.

[[ ## codebase_symbol_explanations ## ]]
None

[[ ## task ## ]]
measure interactions by interaction type instead of just storing the total count. report results segmented by interaction type

[[ ## specific_context ## ]]
None

[[ ## block_number ## ]]
375

Respond with the corresponding output fields, starting with the field `[[ ## task_reflection ## ]]`, then `[[ ## reasoning ## ]]`, then `[[ ## requires_direct_modification ## ]]` (must be formatted as a valid Python bool), then `[[ ## confidence ## ]]` (must be formatted as a valid Python Literal[low, medium, high, very high]), and then ending with the marker for `[[ ## completed ## ]]`.
""".strip()

assistant15 = """
[[ ## task_reflection ## ]]
The task involves modifying the interaction counting mechanism in the HVM3 runtime to track interactions by type (e.g., `APP`, `MAT`, `DUP`, etc.) instead of just maintaining a total count. This will require changes to the interaction counting logic, likely in the C backend, to categorize and store counts for each interaction type. Additionally, the Haskell frontend will need updates to report these segmented results, possibly by extending the `State` structure or adding new data structures to hold the segmented counts.

[[ ## reasoning ## ]]
The `reduce_opy_sup` function in BLOCK 375 is a specific reduction rule for handling `OPY-SUP` interactions. Currently, it increments the global interaction counter using `inc_itr()`. To implement the task, this block would need to be modified to use a new version of `inc_itr` that accepts an interaction type parameter (e.g., `inc_itr_by_type(OPY_SUP)`). This change would allow the runtime to track `OPY-SUP` interactions separately from other types. Since the task requires categorizing interactions, this block must be directly modified to use the new interaction counting mechanism.

[[ ## requires_direct_modification ## ]]
True

[[ ## confidence ## ]]
very high

[[ ## completed ## ]]
""".strip()

user16 = """
This is an example of the task, though some input or output fields are not supplied.

[[ ## codebase_symbol_explanations ## ]]
None

[[ ## task ## ]]
make CTRs store only the CID in the Lab field, and move the arity to a global static object in C

[[ ## specific_context ## ]]
None

[[ ## block_number ## ]]
408

Respond with the corresponding output fields, starting with the field `[[ ## task_reflection ## ]]`, then `[[ ## reasoning ## ]]`, then `[[ ## requires_direct_modification ## ]]` (must be formatted as a valid Python bool), then `[[ ## confidence ## ]]` (must be formatted as a valid Python Literal[low, medium, high, very high]), and then ending with the marker for `[[ ## completed ## ]]`.
""".strip()

assistant16 = """
[[ ## task_reflection ## ]]
The task involves modifying how constructor terms (CTRs) store their metadata. Currently, the `Lab` field of a CTR term stores both the constructor ID (CID) and the arity, encoded together using the `u12v2` encoding scheme. The task requires moving the arity out of the `Lab` field and into a global static object in C, so that the `Lab` field only stores the CID. This change will require updates to the code that interacts with CTR terms, particularly where the arity is extracted or used.

[[ ## reasoning ## ]]
The block in question (BLOCK 408) directly interacts with the `Lab` field of a CTR term. It uses `u12v2_x` to extract the CID and `u12v2_y` to extract the arity from the `Lab` field. If the arity is moved to a global static object, this block will need to be modified to retrieve the arity from the new location instead of extracting it from the `Lab` field. This is a direct modification because the block explicitly relies on the current encoding scheme to access the arity. The change will require replacing the `u12v2_y(lab)` call with a lookup into the global static object that stores the arity.

[[ ## requires_direct_modification ## ]]
True

[[ ## confidence ## ]]
very high

[[ ## completed ## ]]
""".strip()

def user_prompt(
    codebase_symbol_explanations: str,
    task: str,
    specific_context: str,
    block_number: int,
) -> str:
    return f"""
[[ ## codebase_symbol_explanations ## ]]
{codebase_symbol_explanations}

[[ ## task ## ]]
{task}

[[ ## specific_context ## ]]
{specific_context}

[[ ## block_number ## ]]
{block_number}

Respond with the corresponding output fields, starting with the field `[[ ## task_reflection ## ]]`, then `[[ ## reasoning ## ]]`, then `[[ ## requires_direct_modification ## ]]` (must be formatted as a valid Python bool), then `[[ ## confidence ## ]]` (must be formatted as a valid Python Literal[low, medium, high, very high]), and then ending with the marker for `[[ ## completed ## ]]`.
""".strip()

def create_classify_blocks_messages(codebase_symbol_explanations: str, task: str, specific_context: str, block_number: int) -> list[dict]:
    return [
        system_prompt,
        user1,
        assistant1,
        user2,
        assistant2,
        user3,
        assistant3,
        user4,
        assistant4,
        user5,
        assistant5,
        user6,
        assistant6,
        user7,
        assistant7,
        user8,
        assistant8,
        user9,
        assistant9,
        user10,
        assistant10,
        user11,
        assistant11,
        user12,
        assistant12,
        user13,
        assistant13,
        user14,
        assistant14,
        user15,
        assistant15,
        user16,
        assistant16,
        user_prompt(codebase_symbol_explanations, task, specific_context, block_number),
    ]

def parse_classify_blocks_response(response: str) -> dict:
    data = {
        "task_reflection": response.split("[[ ## task_reflection ## ]]")[1].split("[[ ## reasoning ## ]]")[0].strip(),
        "reasoning": response.split("[[ ## reasoning ## ]]")[1].split("[[ ## requires_direct_modification ## ]]")[0].strip(),
        "requires_direct_modification": eval(response.split("[[ ## requires_direct_modification ## ]]")[1].split("[[ ## confidence ## ]]")[0].strip()),
        "confidence": response.split("[[ ## confidence ## ]]")[1].split("[[ ## completed ## ]]")[0].strip(),
    }
    return data["task_reflection"] and convert_confidence_to_num(data["confidence"]) >= 0.75