# Veloc IR Instruction System

## Overview

Veloc IR (Intermediate Representation) is a low-level intermediate representation designed for stack-based virtual machines with register allocation. It adopts SSA (Static Single Assignment) form and supports multiple data types and operations.

---

## Type System

| Type | Description | Size (bytes) |
|------|-------------|--------------|
| `I8`, `I16`, `I32`, `I64` | Signed integers | 1, 2, 4, 8 |
| `F32`, `F64` | IEEE 754 floating-point | 4, 8 |
| `Bool` | Boolean | 1 |
| `Ptr` | Pointer type | 8 |
| `Void` | No return value | 0 |

---

## Entity Identifiers

| Type | Purpose |
|------|---------|
| `Inst` | Instruction identifier |
| `Value` | SSA value identifier |
| `Block` | Basic block identifier |
| `StackSlot` | Stack slot identifier |
| `FuncId` | Function identifier |
| `SigId` | Function signature identifier |
| `BlockCall` | Block call (with arguments) |
| `JumpTable` | Jump table identifier |

---

## Instruction List

### 1. Unary Operations

#### `Unary`
| Attribute | Description |
|-----------|-------------|
| **Operands** | `opcode: Opcode`, `arg: Value` |
| **Return Type** | `ty` (specified in instruction data) |
| **Description** | Unary operations: negation, abs, type conversion, bit counting, etc. |

**Constraints:**
- The type of `arg` must match the expected input type of `opcode`
- For integer extension/truncation operations, `ty` must be an integer type
- For floating-point operations, `arg` must be a floating-point type

**Supported Opcodes:**
- Arithmetic: `Ineg`, `Fneg`, `Abs`, `Sqrt`, `Ceil`, `Floor`
- Conversion: `ExtendS/U`, `Wrap`, `TruncS/U`, `ConvertS/U`, `Demote`, `Promote`, `Reinterpret`
- Bitwise: `Clz` (count leading zeros), `Ctz` (count trailing zeros), `Popcnt` (population count), `Eqz` (is zero)

---

### 2. Binary Operations

#### `Binary`
| Attribute | Description |
|-----------|-------------|
| **Operands** | `opcode: Opcode`, `args: [Value; 2]` |
| **Return Type** | `ty` (specified in instruction data) |
| **Description** | Binary operations: arithmetic, logical, bitwise |

**Constraints:**
- Both operands must have the same type
- Operand types must be compatible with `opcode` (integer opcodes require integer types, floating-point opcodes require float types)
- For signed/unsigned variants (e.g., `DivS` vs `DivU`), operand types must be integers

**Supported Opcodes:**
- Integer: `Iadd`, `Isub`, `Imul`, `DivS/U`, `RemS/U`, `And`, `Or`, `Xor`, `Shl`, `ShrS/U`, `Rotl`, `Rotr`
- Float: `Fadd`, `Fsub`, `Fmul`, `Fdiv`, `Min`, `Max`, `Copysign`

---

### 3. Memory Access Instructions

#### `Load`
| Attribute | Description |
|-----------|-------------|
| **Operands** | `ptr: Value`, `offset: u32`, `flags: MemFlags` |
| **Return Type** | `ty` |
| **Description** | Load value of type `ty` from memory address `ptr + offset` |

**Constraints:**
- `ptr` must be of `Ptr` type
- `ty` must be one of: `I8`, `I16`, `I32`, `I64`, `F32`, `F64`
- `offset` must ensure the final address is aligned to `ty`'s natural alignment boundary (unless `flags` specify unaligned access)
- Load address must be within valid memory range

---

#### `Store`
| Attribute | Description |
|-----------|-------------|
| **Operands** | `ptr: Value`, `value: Value`, `offset: u32`, `flags: MemFlags` |
| **Return Type** | `Void` (no return value) |
| **Description** | Store `value` to memory address `ptr + offset` |

**Constraints:**
- `ptr` must be of `Ptr` type
- Type of `value` must be one of: `I8`, `I16`, `I32`, `I64`, `F32`, `F64`
- `offset` must ensure the final address is aligned to `value` type's natural alignment boundary (unless `flags` specify unaligned access)
- Store address must be within valid writable memory range

---

#### `StackLoad`
| Attribute | Description |
|-----------|-------------|
| **Operands** | `slot: StackSlot`, `offset: u32` |
| **Return Type** | `ty` |
| **Description** | Load value from stack slot |

**Constraints:**
- `slot` must be a valid stack slot declared by the current function
- `offset + sizeof(ty)` must be within the stack slot size
- `ty` supports: `I8`, `I16`, `I32`, `I64`, `F32`, `F64`

---

#### `StackStore`
| Attribute | Description |
|-----------|-------------|
| **Operands** | `slot: StackSlot`, `value: Value`, `offset: u32` |
| **Return Type** | `Void` |
| **Description** | Store value to stack slot |

**Constraints:**
- `slot` must be a valid stack slot declared by the current function
- `offset + sizeof(value)` must be within the stack slot size

---

#### `StackAddr`
| Attribute | Description |
|-----------|-------------|
| **Operands** | `slot: StackSlot`, `offset: u32` |
| **Return Type** | `Ptr` |
| **Description** | Get address of stack slot (`&slot[offset]`) |

**Constraints:**
- `slot` must be a valid stack slot declared by the current function

---

### 4. Constant Instructions

#### `Iconst`
| Attribute | Description |
|-----------|-------------|
| **Operands** | `value: i64` |
| **Return Type** | `ty` (must be integer type `I8`, `I16`, `I32`, `I64`) |
| **Description** | Integer constant |

**Constraints:**
- `ty` must be an integer type
- `value` must be within the representable range of `ty` (if `ty` is `I8`, `value` must be within -128~127)

---

#### `Fconst`
| Attribute | Description |
|-----------|-------------|
| **Operands** | `value: u64` (bit representation) |
| **Return Type** | `ty` (must be `F32` or `F64`) |
| **Description** | Floating-point constant |

**Constraints:**
- `ty` must be `F32` or `F64`
- For `F32`, only the low 32 bits of `value` are used

---

#### `Bconst`
| Attribute | Description |
|-----------|-------------|
| **Operands** | `value: bool` |
| **Return Type** | `Bool` |
| **Description** | Boolean constant |

---

### 5. Control Flow Instructions

#### `Jump`
| Attribute | Description |
|-----------|-------------|
| **Operands** | `dest: BlockCall` |
| **Return Type** | `Void` |
| **Description** | Unconditional jump to target basic block |

**Constraints:**
- `dest` must be a valid basic block within the current function
- Argument types in `dest` must match the target block's parameter types (count and type)
- Must be the last instruction in a basic block

---

#### `Br` (Conditional Branch)
| Attribute | Description |
|-----------|-------------|
| **Operands** | `condition: Value`, `then_dest: BlockCall`, `else_dest: BlockCall` |
| **Return Type** | `Void` |
| **Description** | Jump to `then_dest` if `condition` is true, otherwise to `else_dest` |

**Constraints:**
- `condition` must be of `Bool` type
- `then_dest` and `else_dest` must be valid basic blocks within the current function
- Both target block argument types must match the provided arguments
- Must be the last instruction in a basic block

---

#### `BrTable` (Indirect Jump Table)
| Attribute | Description |
|-----------|-------------|
| **Operands** | `index: Value`, `table: JumpTable` |
| **Return Type** | `Void` |
| **Description** | Select jump target based on `index` (similar to switch) |

**Constraints:**
- `index` must be an integer type (typically `I32`)
- `table` contains default target and all case targets
- If `index` is out of bounds, jump to default target
- All target blocks must have compatible argument types
- Must be the last instruction in a basic block

---

#### `Return`
| Attribute | Description |
|-----------|-------------|
| **Operands** | `value: Option<Value>` |
| **Return Type** | `Void` |
| **Description** | Function return, optional return value |

**Constraints:**
- If function return type is not `Void`, `value` must exist and match the function signature
- If function return type is `Void`, `value` must be `None`
- Must be the last instruction in a basic block

---

#### `Unreachable`
| Attribute | Description |
|-----------|-------------|
| **Operands** | None |
| **Return Type** | `Void` |
| **Description** | Mark unreachable code |

**Constraints:**
- Must be the last instruction in a basic block
- Code after this should not be executed

---

### 6. Comparison and Selection Instructions

#### `IntCompare` (Integer Comparison)
| Attribute | Description |
|-----------|-------------|
| **Operands** | `kind: IntCC`, `args: [Value; 2]` |
| **Return Type** | `Bool` |
| **Description** | Compare two integers |

**Constraints:**
- Both operands must be of the same integer type
- `kind` specifies comparison mode: `Eq`, `Ne`, `LtS`, `LtU`, `LeS`, `LeU`, `GtS`, `GtU`, `GeS`, `GeU`
- Note: Signed vs unsigned comparisons (`S` vs `U` suffix)

---

#### `FloatCompare` (Floating-point Comparison)
| Attribute | Description |
|-----------|-------------|
| **Operands** | `kind: FloatCC`, `args: [Value; 2]` |
| **Return Type** | `Bool` |
| **Description** | Compare two floating-point numbers |

**Constraints:**
- Both operands must be of the same floating-point type (`F32` or `F64`)
- `kind` specifies comparison mode: `Eq`, `Ne`, `Lt`, `Le`, `Gt`, `Ge`
- Note: IEEE 754 special value handling (NaN comparisons always return false)

---

#### `Select` (Conditional Selection)
| Attribute | Description |
|-----------|-------------|
| **Operands** | `condition: Value`, `then_val: Value`, `else_val: Value` |
| **Return Type** | `ty` (same as `then_val`/`else_val`) |
| **Description** | Return `then_val` if `condition` is true, otherwise `else_val` |

**Constraints:**
- `condition` must be of `Bool` type
- `then_val` and `else_val` must have the same type
- `ty` must match the type of `then_val`/`else_val`

---

### 7. Function Call Instructions

#### `Call` (Direct Call)
| Attribute | Description |
|-----------|-------------|
| **Operands** | `func_id: FuncId`, `args: ValueList` |
| **Return Type** | Determined by the callee's signature |
| **Description** | Call the specified function |

**Constraints:**
- `func_id` must be a valid function declared in the module
- Count and types of `args` must match the callee's signature
- Return type is determined by the function signature

---

#### `CallIndirect` (Indirect Call)
| Attribute | Description |
|-----------|-------------|
| **Operands** | `ptr: Value`, `args: ValueList`, `sig_id: SigId` |
| **Return Type** | Determined by signature specified by `sig_id` |
| **Description** | Call via function pointer |

**Constraints:**
- `ptr` must be of `Ptr` type
- `sig_id` must be a valid signature declared in the module
- Count and types of `args` must match the signature
- At runtime, `ptr` must point to a function compatible with the signature

---

#### `CallIntrinsic` (Intrinsic Function Call)
| Attribute | Description |
|-----------|-------------|
| **Operands** | `intrinsic: Intrinsic`, `args: ValueList`, `sig_id: SigId` |
| **Return Type** | Determined by signature specified by `sig_id` |
| **Description** | Call compiler built-in special functions |

**Constraints:**
- `intrinsic` must be a valid intrinsic function identifier
- `sig_id` must match the expected signature of the intrinsic
- Different intrinsics have specific argument requirements (see below)

**Intrinsic Functions:**

| Intrinsic | Arguments | Return Type | Description |
|-----------|-----------|-------------|-------------|
| `SIN_F32/F64` | `x: F32/F64` | `F32/F64` | Sine function |
| `COS_F32/F64` | `x: F32/F64` | `F32/F64` | Cosine function |
| `POW_F32/F64` | `base, exp: F32/F64` | `F32/F64` | Power function |
| `EXP_F32/F64` | `x: F32/F64` | `F32/F64` | Natural exponent |
| `LOG_F32/F64` | `x: F32/F64` | `F32/F64` | Natural logarithm |
| `LOG2_F32/F64` | `x: F32/F64` | `F32/F64` | Base-2 logarithm |
| `LOG10_F32/F64` | `x: F32/F64` | `F32/F64` | Base-10 logarithm |
| `MEMCPY` | `dest: Ptr`, `src: Ptr`, `len: I64` | `Void` | Memory copy |
| `MEMMOVE` | `dest: Ptr`, `src: Ptr`, `len: I64` | `Void` | Memory move (allows overlap) |
| `MEMSET` | `dest: Ptr`, `val: I32`, `len: I64` | `Void` | Memory fill |
| `MEMCMP` | `a: Ptr`, `b: Ptr`, `len: I64` | `I32` | Memory compare |
| `FENCE` | None | `Void` | Full memory barrier |
| `ASSUME` | `cond: Bool` | `Void` | Optimization hint: condition is true |
| `EXPECT` | `val: T`, `expected: T` | `T` | Optimization hint: expected value |
| `TRAP` | None | `Void` | Trigger trap/exception |

---

### 8. Pointer Operations

#### `IntToPtr`
| Attribute | Description |
|-----------|-------------|
| **Operands** | `arg: Value` |
| **Return Type** | `Ptr` |
| **Description** | Convert integer to pointer |

**Constraints:**
- `arg` must be an integer type (typically `I64`)
- The converted integer value should be a valid pointer address

---

#### `PtrToInt`
| Attribute | Description |
|-----------|-------------|
| **Operands** | `arg: Value` |
| **Return Type** | `ty` (must be integer type) |
| **Description** | Convert pointer to integer |

**Constraints:**
- `arg` must be of `Ptr` type
- `ty` must be an integer type (typically `I64`)

---

#### `PtrOffset`
| Attribute | Description |
|-----------|-------------|
| **Operands** | `ptr: Value`, `offset: i32` |
| **Return Type** | `Ptr` |
| **Description** | Pointer offset (in bytes): `ptr + offset` |

**Constraints:**
- `ptr` must be of `Ptr` type
- `offset` is in bytes and can be negative

---

#### `PtrIndex`
| Attribute | Description |
|-----------|-------------|
| **Operands** | `ptr: Value`, `index: Value`, `scale: u32`, `offset: i32` |
| **Return Type** | `Ptr` |
| **Description** | Pointer index calculation: `ptr + index * scale + offset` |

**Constraints:**
- `ptr` must be of `Ptr` type
- `index` must be an integer type
- `scale` is typically the element size (e.g., 4 for `I32` arrays)
- `offset` is in bytes and can be negative

---

### 9. Other Instructions

#### `Nop`
| Attribute | Description |
|-----------|-------------|
| **Operands** | None |
| **Return Type** | `Void` |
| **Description** | No operation, has no effect |

---

## Instruction Attributes

### `is_terminator` - Terminator Instructions
The following instructions must be the last instruction in a basic block:
- `Jump`
- `Br`
- `BrTable`
- `Return`
- `Unreachable`

### `has_side_effects` - Side Effects
Instructions with side effects (cannot be arbitrarily deleted or reordered):
- `Store`, `StackStore` - Modify memory
- `Call`, `CallIndirect`, `CallIntrinsic` - May produce arbitrary side effects
- `Return` - Control flow transfer
- `Jump`, `Br`, `BrTable` - Control flow transfer
- `Unreachable` - Control flow marker

### `visit_operands`
Traverses all `Value`s used by the instruction, used for:
- Liveness analysis
- Register allocation
- SSA deconstruction

---

## Basic Block Structure

Each basic block:
1. Starts with a list of parameters (can be empty)
2. Contains zero or more non-terminator instructions
3. Ends with exactly one terminator instruction

**Constraints:**
- Parameters are SSA values within the block
- When jumping, arguments must match the target block's parameters
- In the control flow graph, block predecessors must be consistent with their parameter definitions
