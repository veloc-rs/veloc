use super::Reg;
use alloc::string::String;
use alloc::string::ToString;
use alloc::vec::Vec;
use veloc_ir::Type;

/// 目标架构标识
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TargetArch {
    X86_64,
    AArch64,
    Riscv64,
    Wasm32,
    Wasm64,
}

impl TargetArch {
    /// 获取架构名称
    pub fn name(&self) -> &'static str {
        match self {
            TargetArch::X86_64 => "x86_64",
            TargetArch::AArch64 => "aarch64",
            TargetArch::Riscv64 => "riscv64",
            TargetArch::Wasm32 => "wasm32",
            TargetArch::Wasm64 => "wasm64",
        }
    }

    /// 是否是小端
    pub fn is_little_endian(&self) -> bool {
        match self {
            TargetArch::X86_64
            | TargetArch::AArch64
            | TargetArch::Riscv64
            | TargetArch::Wasm32
            | TargetArch::Wasm64 => true,
        }
    }
}

/// 寄存器类
///
/// 不同架构的寄存器类可能有不同的数量和大小
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegClass {
    /// 通用目的寄存器（整数）
    GPR,
    /// 浮点/向量寄存器
    FPR,
    /// 向量寄存器（专用）
    VR,
    /// 谓词/掩码寄存器
    PR,
    /// 特殊寄存器（如栈指针、帧指针）
    Special,
}

/// 物理寄存器信息
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegInfo {
    /// 物理寄存器 ID
    pub preg: Reg,
    /// 寄存器名称
    pub name: &'static str,
    /// 寄存器大小（位）
    pub size: u16,
    /// 硬件编码
    pub hw_encoding: u16,
}

/// 通用寄存器类到物理寄存器集合的映射。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegClassInfo {
    pub kind: RegClass,
    pub members: &'static [Reg],
    pub allocatable: &'static [Reg],
}

/// 目标架构中的特殊寄存器。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpecialRegs {
    pub stack_pointer: Reg,
    pub frame_pointer: Option<Reg>,
}

/// 目标架构的寄存器文件描述。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegisterFile {
    pub regs: &'static [RegInfo],
    pub reg_classes: &'static [RegClassInfo],
    pub reserved_regs: &'static [Reg],
    pub special_regs: SpecialRegs,
}

impl RegisterFile {
    pub fn allocatable_regs_in_class(&self, class: RegClass) -> &'static [Reg] {
        self.reg_classes
            .iter()
            .find(|info| info.kind == class)
            .map(|info| info.allocatable)
            .unwrap_or(&[])
    }
}

/// 目标数据布局。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DataLayout {
    pub pointer_size: u8,
    pub little_endian: bool,
}

impl DataLayout {
    pub fn type_size(&self, ty: &Type) -> u32 {
        ty.size_bytes() as u32
    }

    pub fn type_align(&self, ty: &Type) -> u32 {
        self.type_size(ty)
    }
}

/// CPU 级别的描述信息。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuDescription {
    pub name: &'static str,
    pub features: &'static [&'static str],
    pub limitations: &'static [&'static str],
}

impl CpuDescription {
    pub fn has_feature(&self, feature: &str) -> bool {
        self.features.contains(&feature)
    }
}

/// 当前 target instance 的完整描述。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TargetDescription {
    pub arch: TargetArch,
    pub registers: RegisterFile,
    pub data_layout: DataLayout,
    pub cpu: CpuDescription,
}

impl TargetDescription {
    pub fn allocatable_regs_in_class(&self, class: RegClass) -> &'static [Reg] {
        self.registers.allocatable_regs_in_class(class)
    }

    pub fn reg_class_for_type(&self, ty: &Type) -> RegClass {
        if ty.is_float() || ty.is_vector() {
            RegClass::FPR
        } else {
            RegClass::GPR
        }
    }
}

/// 目标配置
#[derive(Debug, Clone)]
pub struct TargetConfig {
    pub arch: TargetArch,
    pub features: Vec<String>,
    /// CPU 型号，用于从 ISLE 获取特定 CPU 的优化信息
    pub cpu: String,
    /// 调优目标，可能与 cpu 不同（如编译在通用 CPU 上运行但针对特定 CPU 优化）
    pub tune: String,
}

impl Default for TargetConfig {
    fn default() -> Self {
        Self {
            arch: TargetArch::X86_64,
            features: Vec::new(),
            cpu: "generic".to_string(),
            tune: "generic".to_string(),
        }
    }
}
