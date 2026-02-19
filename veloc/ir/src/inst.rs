use super::dfg::DataFlowGraph;
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
    /// 谓词/掩码 (Type::Predicate)
    pub mask: Value,
    /// 显式向量长度 (Type::EVL), None 表示使用默认 VL
    pub evl: Option<Value>,
}

/// 向量内存操作的扩展配置
/// 存储在 DFG 的 vector_mem_ext_pool 中
/// 包含静态配置和可选的高级特性（Mask/EVL）
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VectorMemExtData {
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
    /// 条件选择
    Select {
        condition: Value,
        then_val: Value,
        else_val: Value,
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

impl InstructionData {
    pub fn visit_operands<F>(&self, dfg: &DataFlowGraph, mut f: F)
    where
        F: FnMut(Value),
    {
        match self {
            InstructionData::Unary { arg, .. } => f(*arg),
            InstructionData::Binary { args, .. } => {
                f(args[0]);
                f(args[1]);
            }
            InstructionData::Load { ptr, .. } => f(*ptr),
            InstructionData::Store { ptr, value, .. } => {
                f(*ptr);
                f(*value);
            }
            InstructionData::StackLoad { .. } => {}
            InstructionData::StackStore { value, .. } => f(*value),
            InstructionData::StackAddr { .. } => {}
            InstructionData::Iconst { .. } => {}
            InstructionData::Fconst { .. } => {}
            InstructionData::Bconst { .. } => {}
            InstructionData::Vconst { .. } => {}
            InstructionData::Call { args, .. } => {
                for &arg in dfg.get_value_list(*args) {
                    f(arg);
                }
            }
            InstructionData::CallIndirect { ptr, args, .. } => {
                f(*ptr);
                for &arg in dfg.get_value_list(*args) {
                    f(arg);
                }
            }
            InstructionData::Jump { dest } => {
                let call_data = &dfg.block_calls[*dest];
                for &arg in dfg.get_value_list(call_data.args) {
                    f(arg);
                }
            }
            InstructionData::Br {
                condition,
                then_dest,
                else_dest,
            } => {
                f(*condition);
                let then_data = &dfg.block_calls[*then_dest];
                for &arg in dfg.get_value_list(then_data.args) {
                    f(arg);
                }
                let else_data = &dfg.block_calls[*else_dest];
                for &arg in dfg.get_value_list(else_data.args) {
                    f(arg);
                }
            }
            InstructionData::BrTable { index, table } => {
                f(*index);
                let table_data = &dfg.jump_tables[*table];
                for &dest in table_data.targets.iter() {
                    let call_data = &dfg.block_calls[dest];
                    for &arg in dfg.get_value_list(call_data.args) {
                        f(arg);
                    }
                }
            }
            InstructionData::Return { values } => {
                for &v in dfg.get_value_list(*values) {
                    f(v);
                }
            }
            InstructionData::Select {
                condition,
                then_val,
                else_val,
                ..
            } => {
                f(*condition);
                f(*then_val);
                f(*else_val);
            }
            InstructionData::IntCompare { args, .. } => {
                f(args[0]);
                f(args[1]);
            }
            InstructionData::FloatCompare { args, .. } => {
                f(args[0]);
                f(args[1]);
            }
            InstructionData::Unreachable => {}
            InstructionData::IntToPtr { arg } => f(*arg),
            InstructionData::PtrToInt { arg, .. } => f(*arg),
            InstructionData::PtrOffset { ptr, .. } => f(*ptr),
            InstructionData::PtrIndex { ptr, index, .. } => {
                f(*ptr);
                f(*index);
            }
            InstructionData::CallIntrinsic { args, .. } => {
                for &arg in dfg.get_value_list(*args) {
                    f(arg);
                }
            }
            // Vector operations
            InstructionData::Ternary { args, .. } => {
                f(args[0]);
                f(args[1]);
                f(args[2]);
            }
            InstructionData::VectorOpWithExt { args, ext, .. } => {
                for &arg in dfg.get_value_list(*args) {
                    f(arg);
                }
                // 还需要访问 mask 和 evl
                let ext_data = &dfg.vector_ext_pool[*ext];
                f(ext_data.mask);
                if let Some(evl_val) = ext_data.evl {
                    f(evl_val);
                }
            }
            // Strided 操作
            InstructionData::VectorLoadStrided { ptr, stride, ext } => {
                f(*ptr);
                f(*stride);
                let ext_data = &dfg.vector_mem_ext_pool[*ext];
                if let Some(mask) = ext_data.mask {
                    f(mask);
                }
                if let Some(evl) = ext_data.evl {
                    f(evl);
                }
            }
            InstructionData::VectorStoreStrided { args, ext } => {
                for &arg in dfg.get_value_list(*args) {
                    f(arg);
                }
                let ext_data = &dfg.vector_mem_ext_pool[*ext];
                if let Some(mask) = ext_data.mask {
                    f(mask);
                }
                if let Some(evl) = ext_data.evl {
                    f(evl);
                }
            }
            // Gather/Scatter 操作
            InstructionData::VectorGather { ptr, index, ext } => {
                f(*ptr);
                f(*index);
                let ext_data = &dfg.vector_mem_ext_pool[*ext];
                if let Some(mask) = ext_data.mask {
                    f(mask);
                }
                if let Some(evl) = ext_data.evl {
                    f(evl);
                }
            }
            InstructionData::VectorScatter { args, ext } => {
                for &arg in dfg.get_value_list(*args) {
                    f(arg);
                }
                let ext_data = &dfg.vector_mem_ext_pool[*ext];
                if let Some(mask) = ext_data.mask {
                    f(mask);
                }
                if let Some(evl) = ext_data.evl {
                    f(evl);
                }
            }
            InstructionData::Shuffle { args, .. } => {
                f(args[0]);
                f(args[1]);
            }
            InstructionData::Nop => {}
        }
    }

    pub fn opcode(&self) -> Opcode {
        match self {
            InstructionData::Unary { opcode, .. } => *opcode,
            InstructionData::Binary { opcode, .. } => *opcode,
            InstructionData::Load { .. } => Opcode::Load,
            InstructionData::Store { .. } => Opcode::Store,
            InstructionData::StackLoad { .. } => Opcode::StackLoad,
            InstructionData::StackStore { .. } => Opcode::StackStore,
            InstructionData::StackAddr { .. } => Opcode::StackAddr,
            InstructionData::Iconst { .. } => Opcode::Iconst,
            InstructionData::Fconst { .. } => Opcode::Fconst,
            InstructionData::Bconst { .. } => Opcode::Bconst,
            InstructionData::Vconst { .. } => Opcode::Vconst,
            InstructionData::Call { .. } => Opcode::Call,
            InstructionData::Jump { .. } => Opcode::Jump,
            InstructionData::Br { .. } => Opcode::Br,
            InstructionData::BrTable { .. } => Opcode::BrTable,
            InstructionData::Return { .. } => Opcode::Return,
            InstructionData::Select { .. } => Opcode::Select,
            InstructionData::IntCompare { .. } => Opcode::Icmp,
            InstructionData::FloatCompare { .. } => Opcode::Fcmp,
            InstructionData::Unreachable => Opcode::Unreachable,
            InstructionData::CallIndirect { .. } => Opcode::CallIndirect,
            InstructionData::IntToPtr { .. } => Opcode::IntToPtr,
            InstructionData::PtrToInt { .. } => Opcode::PtrToInt,
            InstructionData::PtrOffset { .. } => Opcode::PtrOffset,
            InstructionData::PtrIndex { .. } => Opcode::PtrIndex,
            InstructionData::CallIntrinsic { .. } => Opcode::CallIntrinsic,
            // Vector operations
            InstructionData::Ternary { opcode, .. } => *opcode,
            InstructionData::VectorOpWithExt { opcode, .. } => *opcode,
            // Vector memory operations
            InstructionData::VectorLoadStrided { .. } => Opcode::LoadStride,
            InstructionData::VectorStoreStrided { .. } => Opcode::StoreStride,
            InstructionData::VectorGather { .. } => Opcode::Gather,
            InstructionData::VectorScatter { .. } => Opcode::Scatter,
            InstructionData::Shuffle { .. } => Opcode::Shuffle,
            InstructionData::Nop => Opcode::Nop,
        }
    }

    pub fn is_terminator(&self) -> bool {
        matches!(
            self.opcode(),
            Opcode::Jump | Opcode::Br | Opcode::BrTable | Opcode::Return | Opcode::Unreachable
        )
    }

    pub fn has_side_effects(&self) -> bool {
        // 首先检查是否是向量存储操作
        if matches!(
            self,
            InstructionData::VectorStoreStrided { .. } | InstructionData::VectorScatter { .. }
        ) {
            return true;
        }

        match self.opcode() {
            Opcode::Store
            | Opcode::StackStore
            | Opcode::Call
            | Opcode::CallIndirect
            | Opcode::CallIntrinsic
            | Opcode::Return
            | Opcode::Jump
            | Opcode::Br
            | Opcode::BrTable
            | Opcode::Unreachable => true,
            _ => false,
        }
    }

    /// 检查是否是向量操作
    pub fn is_vector_op(&self) -> bool {
        matches!(
            self,
            InstructionData::Ternary { .. }
                | InstructionData::VectorOpWithExt { .. }
                | InstructionData::VectorLoadStrided { .. }
                | InstructionData::VectorStoreStrided { .. }
                | InstructionData::VectorGather { .. }
                | InstructionData::VectorScatter { .. }
                | InstructionData::Shuffle { .. }
        )
    }
}

impl fmt::Display for InstructionData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.opcode())
    }
}
