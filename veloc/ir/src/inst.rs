use super::dfg::DataFlowGraph;
use crate::opspec::{MemoryEffect, OpFormat};
use crate::types::{BlockCall, FuncId, JumpTable, StackSlot, Value, ValueList};
use crate::{FloatCC, IntCC, Intrinsic, MemFlags, Opcode, SigId};
use core::fmt;
use cranelift_entity::entity_impl;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PtrIndexImm {
    pub offset: i32,
    pub scale: u32,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PtrIndexImmId(pub u32);
entity_impl!(PtrIndexImmId, "ptr_index_imm");

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Inst(pub u32);
entity_impl!(Inst, "inst");

// =============================================================================
// Vector Extension IDs (用于指向辅助数据池)
// =============================================================================

/// 向量操作扩展信息 ID (指向 DFG.vector_ext_pool)
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VectorExtId(pub u32);
entity_impl!(VectorExtId, "vext");

/// 常量池 ID (用于 Shuffle 掩码等)
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ConstantPoolId(pub u32);
entity_impl!(ConstantPoolId, "const");

/// 常量池中的数据
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ConstantPoolData {
    /// 原始字节数据 (用于向量常量、掩码等)
    Bytes(alloc::vec::Vec<u8>),
}

// =============================================================================
// 向量操作辅助数据结构 (存储在 DFG 的 Arena 中)
// =============================================================================

/// 向量操作扩展信息
/// 用于存储带 Mask 和 EVL 的向量操作（RISC-V V / AVX-512）
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VectorExtData {
    /// 谓词/掩码 (boolean vector)
    pub mask: Value,
    /// 显式向量长度 (Type::I32), None 表示使用默认 VL
    pub evl: Option<Value>,
}

/// Optional configuration shared by vector memory operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VectorMemOptions {
    /// 立即数偏移
    pub offset: i32,
    /// 内存标志 (对齐、Volatile等)
    pub flags: MemFlags,
    /// 索引缩放因子 (用于 Gather/Scatter，如 index * scale)
    pub scale: u8,
    /// 掩码 (可选)
    pub mask: Option<Value>,
    /// 显式向量长度 (可选)
    pub evl: Option<Value>,
}

impl Default for VectorMemOptions {
    fn default() -> Self {
        Self {
            offset: 0,
            flags: MemFlags::new(),
            scale: 1,
            mask: None,
            evl: None,
        }
    }
}

/// 扩展配置 ID
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VectorMemExtId(pub u32);
entity_impl!(VectorMemExtId, "vmem_ext");

impl Inst {
    pub fn visit_operands<F>(self, dfg: &DataFlowGraph, f: F)
    where
        F: FnMut(Value),
    {
        dfg.instructions[self].visit_operands(dfg, f)
    }
}

#[derive(Debug, Clone)]
pub enum InstructionData {
    /// 一元运算
    Unary {
        opcode: Opcode,
        arg: Value,
    },
    /// 二元运算
    Binary {
        opcode: Opcode,
        args: [Value; 2],
    },
    /// 从内存加载
    Load {
        ptr: Value,
        offset: u32,
        flags: MemFlags,
    },
    /// 存储到内存
    Store {
        ptr: Value,
        value: Value,
        offset: u32,
        flags: MemFlags,
    },
    /// 从栈槽加载
    StackLoad {
        slot: StackSlot,
        offset: u32,
    },
    /// 存储到栈槽
    StackStore {
        slot: StackSlot,
        value: Value,
        offset: u32,
    },
    /// 获取栈槽地址
    StackAddr {
        slot: StackSlot,
        offset: u32,
    },
    /// 整数常量
    Iconst {
        value: u64,
    },
    /// 浮点常量
    Fconst {
        value: u64,
    },
    /// 布尔常量
    Bconst {
        value: bool,
    },
    /// 向量常量 (数据在常量池中)
    Vconst {
        pool_id: ConstantPoolId,
    },
    /// 直接函数调用
    Call {
        func_id: FuncId,
        args: ValueList,
    },
    /// 无条件跳转
    Jump {
        dest: BlockCall,
    },
    /// 条件分支
    Br {
        condition: Value,
        then_dest: BlockCall,
        else_dest: BlockCall,
    },
    /// 跳转表
    BrTable {
        index: Value,
        table: JumpTable,
    },
    /// 函数返回（支持多返回值）
    Return {
        values: ValueList,
    },
    /// 整数比较
    IntCompare {
        kind: IntCC,
        args: [Value; 2],
    },
    /// 浮点比较
    FloatCompare {
        kind: FloatCC,
        args: [Value; 2],
    },
    /// 不可达代码
    Unreachable,
    /// 间接函数调用
    CallIndirect {
        ptr: Value,
        args: ValueList,
        sig_id: SigId,
    },
    /// 整数转指针
    IntToPtr {
        arg: Value,
    },
    /// 指针转整数
    PtrToInt {
        arg: Value,
    },
    /// 指针偏移
    PtrOffset {
        ptr: Value,
        offset: i32,
    },
    /// 带有立即数的指针索引
    PtrIndex {
        ptr: Value,
        index: Value,
        imm_id: PtrIndexImmId,
    },
    /// 内建函数调用
    CallIntrinsic {
        intrinsic: Intrinsic,
        args: ValueList,
        sig_id: SigId,
    },
    Ternary {
        opcode: Opcode,
        args: [Value; 3],
    },
    /// 带扩展信息的向量操作
    /// 适用于：RISC-V V / AVX-512 带 Mask 或 EVL 的运算
    VectorOpWithExt {
        opcode: Opcode,
        args: ValueList,
        ext: VectorExtId,
    },

    // ==========================================
    // 向量内存操作 - Strided (步长访问)
    // ==========================================
    /// 固定步长向量加载
    /// ptr + stride * i
    VectorLoadStrided {
        ptr: Value,
        stride: Value,
        ext: VectorMemExtId,
    },

    /// 固定步长向量存储
    VectorStoreStrided {
        args: ValueList, // [ptr, stride, value]
        ext: VectorMemExtId,
    },

    /// 离散向量加载 (Gather)
    /// base_ptr + index[i] * scale
    VectorGather {
        ptr: Value,
        index: Value,
        ext: VectorMemExtId,
    },

    /// 离散向量存储 (Scatter)
    VectorScatter {
        args: ValueList, // [ptr, index, value]
        ext: VectorMemExtId,
    },

    /// Shuffle 操作
    /// 两个输入向量 + 常量掩码
    Shuffle {
        args: [Value; 2],
        mask: ConstantPoolId,
    },

    /// 空操作
    Nop,
}

// Keep the physical instruction layout in one declarative table. The table is
// expanded into format checks, operand walking/rewrite, and opcode extraction;
// adding a new layout therefore cannot update only one of those views.
macro_rules! schema_opcode {
    (dynamic($opcode:ident)) => {
        *$opcode
    };
    (fixed($opcode:ident)) => {
        Opcode::$opcode
    };
}

macro_rules! schema_matches_format {
    ($dfg:ident, $format:ident, fixed($expected:ident)) => {
        $format == OpFormat::$expected
    };
    ($dfg:ident, $format:ident, arity($args:ident)) => {
        match $dfg.get_value_list(*$args).len() {
            1 => $format == OpFormat::Unary,
            2 => $format == OpFormat::Binary,
            3 => $format == OpFormat::Ternary,
            _ => false,
        }
    };
}

macro_rules! schema_visit_operand {
    ($dfg:ident, $f:ident, value($value:ident)) => {
        $f(*$value);
    };
    ($dfg:ident, $f:ident, array($values:ident)) => {
        for &value in $values.iter() {
            $f(value);
        }
    };
    ($dfg:ident, $f:ident, value_list($values:ident)) => {
        $dfg.visit_value_list(*$values, &mut $f);
    };
    ($dfg:ident, $f:ident, block_call($call:ident)) => {
        $dfg.visit_block_call(*$call, &mut $f);
    };
    ($dfg:ident, $f:ident, jump_table($table:ident)) => {
        $dfg.visit_jump_table(*$table, &mut $f);
    };
    ($dfg:ident, $f:ident, vector_ext($ext:ident)) => {
        $dfg.visit_vector_ext(*$ext, &mut $f);
    };
    ($dfg:ident, $f:ident, vector_mem_ext($ext:ident)) => {
        $dfg.visit_vector_mem_ext(*$ext, &mut $f);
    };
}

macro_rules! schema_replace_operand {
    ($dfg:ident, $old:ident, $new:ident, value($value:ident)) => {
        if *$value == $old {
            *$value = $new;
        }
    };
    ($dfg:ident, $old:ident, $new:ident, array($values:ident)) => {
        for value in $values.iter_mut() {
            if *value == $old {
                *value = $new;
            }
        }
    };
    ($dfg:ident, $old:ident, $new:ident, value_list($values:ident)) => {
        $dfg.replace_value_list($values, $old, $new);
    };
    ($dfg:ident, $old:ident, $new:ident, block_call($call:ident)) => {
        $dfg.replace_block_call(*$call, $old, $new);
    };
    ($dfg:ident, $old:ident, $new:ident, jump_table($table:ident)) => {
        $dfg.replace_jump_table(*$table, $old, $new);
    };
    ($dfg:ident, $old:ident, $new:ident, vector_ext($ext:ident)) => {
        $dfg.replace_vector_ext($ext, $old, $new);
    };
    ($dfg:ident, $old:ident, $new:ident, vector_mem_ext($ext:ident)) => {
        $dfg.replace_vector_mem_ext($ext, $old, $new);
    };
}

macro_rules! define_instruction_schema {
    ($(
        $pattern:pat => {
            opcode: $opcode_kind:ident($opcode_arg:ident),
            format: $format_kind:ident($format_arg:ident),
            primary: [$($primary_kind:ident($primary:ident)),* $(,)?],
            auxiliary: [$($aux_kind:ident($aux:ident)),* $(,)?]
        }
    ),* $(,)?) => {
        #[allow(unused_variables)]
        pub fn matches_format(&self, dfg: &DataFlowGraph, format: OpFormat) -> bool {
            match self {
                $($pattern => schema_matches_format!(dfg, format, $format_kind($format_arg))),*
            }
        }

        /// Visit operands that participate in the opcode's type scheme.
        ///
        /// Auxiliary mask/EVL operands belong to the predication envelope and
        /// are validated separately from the core operation.
        #[allow(unused_variables)]
        pub fn visit_type_operands<F>(&self, dfg: &DataFlowGraph, mut f: F)
        where
            F: FnMut(Value),
        {
            match self {
                $($pattern => {
                    $(schema_visit_operand!(dfg, f, $primary_kind($primary));)*
                }),*
            }
        }

        #[allow(unused_variables)]
        pub fn visit_operands<F>(&self, dfg: &DataFlowGraph, mut f: F)
        where
            F: FnMut(Value),
        {
            match self {
                $($pattern => {
                    $(schema_visit_operand!(dfg, f, $primary_kind($primary));)*
                    $(schema_visit_operand!(dfg, f, $aux_kind($aux));)*
                }),*
            }
        }

        #[allow(unused_variables)]
        pub fn replace_value(
            &mut self,
            dfg: &mut DataFlowGraph,
            old_val: Value,
            new_val: Value,
        ) {
            match self {
                $($pattern => {
                    $(schema_replace_operand!(dfg, old_val, new_val, $primary_kind($primary));)*
                    $(schema_replace_operand!(dfg, old_val, new_val, $aux_kind($aux));)*
                }),*
            }
        }

        #[allow(unused_variables)]
        pub fn opcode(&self) -> Opcode {
            match self {
                $($pattern => schema_opcode!($opcode_kind($opcode_arg))),*
            }
        }
    };
}

impl InstructionData {
    define_instruction_schema! {
        Self::Unary { opcode, arg } => {
            opcode: dynamic(opcode),
            format: fixed(Unary),
            primary: [value(arg)],
            auxiliary: []
        },
        Self::Binary { opcode, args } => {
            opcode: dynamic(opcode),
            format: fixed(Binary),
            primary: [array(args)],
            auxiliary: []
        },
        Self::Load { ptr, .. } => {
            opcode: fixed(Load),
            format: fixed(Load),
            primary: [value(ptr)],
            auxiliary: []
        },
        Self::Store { ptr, value, .. } => {
            opcode: fixed(Store),
            format: fixed(Store),
            primary: [value(ptr), value(value)],
            auxiliary: []
        },
        Self::StackLoad { .. } => {
            opcode: fixed(StackLoad),
            format: fixed(StackLoad),
            primary: [],
            auxiliary: []
        },
        Self::StackStore { value, .. } => {
            opcode: fixed(StackStore),
            format: fixed(StackStore),
            primary: [value(value)],
            auxiliary: []
        },
        Self::StackAddr { .. } => {
            opcode: fixed(StackAddr),
            format: fixed(StackAddr),
            primary: [],
            auxiliary: []
        },
        Self::Iconst { .. } => {
            opcode: fixed(Iconst),
            format: fixed(Iconst),
            primary: [],
            auxiliary: []
        },
        Self::Fconst { .. } => {
            opcode: fixed(Fconst),
            format: fixed(Fconst),
            primary: [],
            auxiliary: []
        },
        Self::Bconst { .. } => {
            opcode: fixed(Bconst),
            format: fixed(Bconst),
            primary: [],
            auxiliary: []
        },
        Self::Vconst { .. } => {
            opcode: fixed(Vconst),
            format: fixed(Vconst),
            primary: [],
            auxiliary: []
        },
        Self::Call { args, .. } => {
            opcode: fixed(Call),
            format: fixed(Call),
            primary: [value_list(args)],
            auxiliary: []
        },
        Self::Jump { dest } => {
            opcode: fixed(Jump),
            format: fixed(Jump),
            primary: [block_call(dest)],
            auxiliary: []
        },
        Self::Br {
            condition,
            then_dest,
            else_dest,
        } => {
            opcode: fixed(Br),
            format: fixed(Br),
            primary: [value(condition), block_call(then_dest), block_call(else_dest)],
            auxiliary: []
        },
        Self::BrTable { index, table } => {
            opcode: fixed(BrTable),
            format: fixed(BrTable),
            primary: [value(index), jump_table(table)],
            auxiliary: []
        },
        Self::Return { values } => {
            opcode: fixed(Return),
            format: fixed(Return),
            primary: [value_list(values)],
            auxiliary: []
        },
        Self::IntCompare { args, .. } => {
            opcode: fixed(Icmp),
            format: fixed(IntCompare),
            primary: [array(args)],
            auxiliary: []
        },
        Self::FloatCompare { args, .. } => {
            opcode: fixed(Fcmp),
            format: fixed(FloatCompare),
            primary: [array(args)],
            auxiliary: []
        },
        Self::Unreachable => {
            opcode: fixed(Unreachable),
            format: fixed(Unreachable),
            primary: [],
            auxiliary: []
        },
        Self::CallIndirect { ptr, args, .. } => {
            opcode: fixed(CallIndirect),
            format: fixed(CallIndirect),
            primary: [value(ptr), value_list(args)],
            auxiliary: []
        },
        Self::IntToPtr { arg } => {
            opcode: fixed(IntToPtr),
            format: fixed(IntToPtr),
            primary: [value(arg)],
            auxiliary: []
        },
        Self::PtrToInt { arg } => {
            opcode: fixed(PtrToInt),
            format: fixed(PtrToInt),
            primary: [value(arg)],
            auxiliary: []
        },
        Self::PtrOffset { ptr, .. } => {
            opcode: fixed(PtrOffset),
            format: fixed(PtrOffset),
            primary: [value(ptr)],
            auxiliary: []
        },
        Self::PtrIndex { ptr, index, .. } => {
            opcode: fixed(PtrIndex),
            format: fixed(PtrIndex),
            primary: [value(ptr), value(index)],
            auxiliary: []
        },
        Self::CallIntrinsic { args, .. } => {
            opcode: fixed(CallIntrinsic),
            format: fixed(CallIntrinsic),
            primary: [value_list(args)],
            auxiliary: []
        },
        Self::Ternary { opcode, args } => {
            opcode: dynamic(opcode),
            format: fixed(Ternary),
            primary: [array(args)],
            auxiliary: []
        },
        Self::VectorOpWithExt { opcode, args, ext } => {
            opcode: dynamic(opcode),
            format: arity(args),
            primary: [value_list(args)],
            auxiliary: [vector_ext(ext)]
        },
        Self::VectorLoadStrided { ptr, stride, ext } => {
            opcode: fixed(LoadStride),
            format: fixed(VectorLoadStrided),
            primary: [value(ptr), value(stride)],
            auxiliary: [vector_mem_ext(ext)]
        },
        Self::VectorStoreStrided { args, ext } => {
            opcode: fixed(StoreStride),
            format: fixed(VectorStoreStrided),
            primary: [value_list(args)],
            auxiliary: [vector_mem_ext(ext)]
        },
        Self::VectorGather { ptr, index, ext } => {
            opcode: fixed(Gather),
            format: fixed(VectorGather),
            primary: [value(ptr), value(index)],
            auxiliary: [vector_mem_ext(ext)]
        },
        Self::VectorScatter { args, ext } => {
            opcode: fixed(Scatter),
            format: fixed(VectorScatter),
            primary: [value_list(args)],
            auxiliary: [vector_mem_ext(ext)]
        },
        Self::Shuffle { args, .. } => {
            opcode: fixed(Shuffle),
            format: fixed(Shuffle),
            primary: [array(args)],
            auxiliary: []
        },
        Self::Nop => {
            opcode: fixed(Nop),
            format: fixed(Nop),
            primary: [],
            auxiliary: []
        }
    }
    pub fn is_terminator(&self) -> bool {
        self.opcode().spec().is_terminator()
    }

    pub fn memory_effect(&self, dfg: &DataFlowGraph) -> MemoryEffect {
        let effect = self.opcode().spec().memory_effect;
        let flags = match self {
            Self::Load { flags, .. } | Self::Store { flags, .. } => Some(*flags),
            Self::VectorLoadStrided { ext, .. }
            | Self::VectorStoreStrided { ext, .. }
            | Self::VectorGather { ext, .. }
            | Self::VectorScatter { ext, .. } => Some(
                dfg.vector_mem_ext(*ext)
                    .expect("instruction refers to a missing vector memory extension")
                    .flags,
            ),
            _ => None,
        };
        if flags.is_some_and(|flags| flags.is_volatile()) {
            effect.with_volatile()
        } else {
            effect
        }
    }

    pub fn has_side_effects(&self, dfg: &DataFlowGraph) -> bool {
        let spec = self.opcode().spec();
        spec.is_terminator() || spec.may_trap() || self.memory_effect(dfg).has_side_effects()
    }
}

impl fmt::Display for InstructionData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.opcode())
    }
}
