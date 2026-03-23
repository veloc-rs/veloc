use super::Reg;
use crate::regalloc::regbank_select::RegisterBank;
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

impl RegClass {
    pub fn matches_type(self, ty: &Type) -> bool {
        match self {
            RegClass::GPR => *ty == Type::BOOL || ty.is_integer() || ty.is_ptr(),
            RegClass::FPR => ty.is_scalar() && ty.is_float(),
            RegClass::VR => ty.is_vector(),
            RegClass::PR => ty.is_predicate(),
            RegClass::Special => false,
        }
    }
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
    pub bank: RegisterBank,
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
    pub fn reg_class(&self, class: RegClass) -> Option<&RegClassInfo> {
        self.reg_classes.iter().find(|info| info.kind == class)
    }

    pub fn has_reg_class(&self, class: RegClass) -> bool {
        self.reg_class(class).is_some()
    }

    pub fn bank_for_reg_class(&self, class: RegClass) -> Option<RegisterBank> {
        self.reg_class(class).map(|info| info.bank)
    }

    pub fn default_reg_class_for_bank(&self, bank: RegisterBank, ty: &Type) -> Option<RegClass> {
        self.reg_classes
            .iter()
            .find(|info| info.bank == bank && info.kind.matches_type(ty))
            .or_else(|| self.reg_classes.iter().find(|info| info.bank == bank))
            .map(|info| info.kind)
    }

    pub fn allocatable_regs_in_class(&self, class: RegClass) -> &'static [Reg] {
        self.reg_class(class)
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

    pub fn bank_for_reg_class(&self, class: RegClass) -> Option<RegisterBank> {
        self.registers.bank_for_reg_class(class)
    }

    pub fn default_reg_class_for_bank(&self, bank: RegisterBank, ty: &Type) -> Option<RegClass> {
        self.registers.default_reg_class_for_bank(bank, ty)
    }

    pub fn reg_class_for_type(&self, ty: &Type) -> RegClass {
        if ty.is_predicate() {
            self.default_reg_class_for_bank(RegisterBank::PR, ty)
                .or_else(|| self.default_reg_class_for_bank(RegisterBank::GPR, ty))
        } else if ty.is_vector() {
            self.default_reg_class_for_bank(RegisterBank::VR, ty)
                .or_else(|| self.default_reg_class_for_bank(RegisterBank::FPR, ty))
        } else if ty.is_float() {
            self.default_reg_class_for_bank(RegisterBank::FPR, ty)
        } else {
            self.default_reg_class_for_bank(RegisterBank::GPR, ty)
        }
        .unwrap_or_else(|| {
            panic!(
                "target {:?} has no register class for type {:?}",
                self.arch, ty
            )
        })
    }

    pub fn reg_class_for_vreg(&self, ty: &Type, bank: Option<RegisterBank>) -> RegClass {
        if let Some(bank) = bank {
            self.default_reg_class_for_bank(bank, ty)
                .unwrap_or_else(|| {
                    panic!(
                        "target {:?} has no register class for bank {:?} and type {:?}",
                        self.arch, bank, ty
                    )
                })
        } else {
            self.reg_class_for_type(ty)
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

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_GPRS: &[Reg] = &[Reg(1)];
    const TEST_FPRS: &[Reg] = &[Reg(32)];
    const TEST_VRS: &[Reg] = &[Reg(64)];
    const TEST_PRS: &[Reg] = &[Reg(96)];

    const TEST_REGS: &[RegInfo] = &[
        RegInfo {
            preg: Reg(1),
            name: "r0",
            size: 64,
            hw_encoding: 0,
        },
        RegInfo {
            preg: Reg(32),
            name: "f0",
            size: 128,
            hw_encoding: 0,
        },
        RegInfo {
            preg: Reg(64),
            name: "v0",
            size: 128,
            hw_encoding: 0,
        },
        RegInfo {
            preg: Reg(96),
            name: "p0",
            size: 16,
            hw_encoding: 0,
        },
    ];

    const TEST_CLASSES_WITHOUT_VR: &[RegClassInfo] = &[
        RegClassInfo {
            kind: RegClass::GPR,
            bank: RegisterBank::GPR,
            members: TEST_GPRS,
            allocatable: TEST_GPRS,
        },
        RegClassInfo {
            kind: RegClass::FPR,
            bank: RegisterBank::FPR,
            members: TEST_FPRS,
            allocatable: TEST_FPRS,
        },
        RegClassInfo {
            kind: RegClass::PR,
            bank: RegisterBank::PR,
            members: TEST_PRS,
            allocatable: TEST_PRS,
        },
    ];

    const TEST_CLASSES_WITH_VR: &[RegClassInfo] = &[
        RegClassInfo {
            kind: RegClass::GPR,
            bank: RegisterBank::GPR,
            members: TEST_GPRS,
            allocatable: TEST_GPRS,
        },
        RegClassInfo {
            kind: RegClass::FPR,
            bank: RegisterBank::FPR,
            members: TEST_FPRS,
            allocatable: TEST_FPRS,
        },
        RegClassInfo {
            kind: RegClass::VR,
            bank: RegisterBank::VR,
            members: TEST_VRS,
            allocatable: TEST_VRS,
        },
        RegClassInfo {
            kind: RegClass::PR,
            bank: RegisterBank::PR,
            members: TEST_PRS,
            allocatable: TEST_PRS,
        },
    ];

    const TEST_DATA_LAYOUT: DataLayout = DataLayout {
        pointer_size: 8,
        little_endian: true,
    };

    const TEST_CPU: CpuDescription = CpuDescription {
        name: "test",
        features: &[],
        limitations: &[],
    };

    const TEST_DESC_WITHOUT_VR: TargetDescription = TargetDescription {
        arch: TargetArch::Riscv64,
        registers: RegisterFile {
            regs: TEST_REGS,
            reg_classes: TEST_CLASSES_WITHOUT_VR,
            reserved_regs: &[],
            special_regs: SpecialRegs {
                stack_pointer: Reg(2),
                frame_pointer: None,
            },
        },
        data_layout: TEST_DATA_LAYOUT,
        cpu: TEST_CPU,
    };

    const TEST_DESC_WITH_VR: TargetDescription = TargetDescription {
        arch: TargetArch::Riscv64,
        registers: RegisterFile {
            regs: TEST_REGS,
            reg_classes: TEST_CLASSES_WITH_VR,
            reserved_regs: &[],
            special_regs: SpecialRegs {
                stack_pointer: Reg(2),
                frame_pointer: None,
            },
        },
        data_layout: TEST_DATA_LAYOUT,
        cpu: TEST_CPU,
    };

    #[test]
    fn reg_class_info_carries_bank_mapping() {
        assert_eq!(
            TEST_DESC_WITHOUT_VR.bank_for_reg_class(RegClass::GPR),
            Some(RegisterBank::GPR)
        );
        assert_eq!(
            TEST_DESC_WITHOUT_VR.bank_for_reg_class(RegClass::FPR),
            Some(RegisterBank::FPR)
        );
        assert_eq!(
            TEST_DESC_WITHOUT_VR.bank_for_reg_class(RegClass::PR),
            Some(RegisterBank::PR)
        );
    }

    #[test]
    fn reg_class_for_type_falls_back_from_vr_to_fpr() {
        assert_eq!(
            TEST_DESC_WITHOUT_VR.reg_class_for_type(&Type::F32X4),
            RegClass::FPR
        );
    }

    #[test]
    fn reg_class_for_type_prefers_vr_when_available() {
        assert_eq!(
            TEST_DESC_WITH_VR.reg_class_for_type(&Type::F32X4),
            RegClass::VR
        );
    }

    #[test]
    fn reg_class_for_type_uses_predicate_bank_when_available() {
        let mask_ty = Type::new_predicate(8, false);
        assert_eq!(TEST_DESC_WITH_VR.reg_class_for_type(&mask_ty), RegClass::PR);
    }

    #[test]
    fn reg_class_for_vreg_prefers_explicit_bank_over_type_default() {
        assert_eq!(
            TEST_DESC_WITH_VR.reg_class_for_vreg(&Type::F64, Some(RegisterBank::VR)),
            RegClass::VR
        );
    }
}
