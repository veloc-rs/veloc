//! Register identities, value types, banks and physical clobber masks.
use crate::Type;
use cranelift_entity::entity_impl;

/// Abstract register bank; target-specific selection belongs to codegen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegisterBank {
    GPR,
    FPR,
    VR,
    PR,
    Special,
}

/// 虚拟寄存器索引
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VReg(u32);
entity_impl!(VReg, "vreg");

/// A physical register location. Unlike `Reg`, it cannot contain a virtual value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PReg(u32);

impl PReg {
    pub const fn new(index: u32) -> Self {
        assert!(
            index < Reg::VREG_MARK,
            "physical register index out of range"
        );
        Self(index)
    }

    pub const fn index(self) -> u32 {
        self.0
    }
}

impl From<PReg> for Reg {
    fn from(reg: PReg) -> Self {
        Self(reg.0)
    }
}

/// 寄存器标识符 (虚拟或物理)
///
/// 最高位为 1 表示虚拟寄存器 (VReg)，为 0 表示物理寄存器 (PReg)
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Reg(pub u32);

impl Reg {
    const VREG_MARK: u32 = 1 << 31;

    /// 创建一个虚拟寄存器
    pub fn new_vreg(index: u32) -> Self {
        debug_assert!(index < Self::VREG_MARK);
        Self(index | Self::VREG_MARK)
    }

    /// 创建一个物理寄存器
    pub fn new_preg(index: u32) -> Self {
        debug_assert!(index < Self::VREG_MARK);
        Self(index)
    }

    /// 检查是否为虚拟寄存器
    pub fn is_vreg(&self) -> bool {
        (self.0 & Self::VREG_MARK) != 0
    }

    /// 检查是否为物理寄存器
    pub fn is_preg(&self) -> bool {
        (self.0 & Self::VREG_MARK) == 0
    }

    pub fn as_preg(self) -> Option<PReg> {
        self.is_preg().then_some(PReg(self.0))
    }

    pub fn as_vreg(self) -> Option<VReg> {
        self.is_vreg().then(|| VReg::from_u32(self.index()))
    }

    /// 获取原始索引 (去掉 VReg 标记)
    pub fn index(&self) -> u32 {
        self.0 & !Self::VREG_MARK
    }
}

impl core::fmt::Debug for Reg {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if self.is_vreg() {
            write!(f, "v{}", self.index())
        } else {
            write!(f, "p{}", self.index())
        }
    }
}

/// Typed, allocatable value. Hardware state is represented by physical operands.
#[derive(Debug, Clone)]
pub struct VRegData {
    pub ty: Type,
    /// None leaves bank selection to the target and operand constraints.
    pub bank: Option<RegisterBank>,
}

/// Shared static clobber set emitted by the target's ABI definitions.
#[derive(Debug, Clone, Copy)]
pub struct RegMask(&'static [u64]);
impl Default for RegMask {
    fn default() -> Self {
        Self::from_static(&[])
    }
}
impl PartialEq for RegMask {
    fn eq(&self, other: &Self) -> bool {
        self.iter().eq(other.iter())
    }
}
impl Eq for RegMask {}
impl RegMask {
    pub const fn from_static(words: &'static [u64]) -> Self {
        Self(words)
    }
    pub fn contains(&self, root: PReg) -> bool {
        let index = root.index() as usize;
        self.0
            .get(index / 64)
            .is_some_and(|word| word & (1 << (index % 64)) != 0)
    }
    pub fn iter(&self) -> impl Iterator<Item = Reg> + '_ {
        self.0.iter().enumerate().flat_map(|(index, &word)| {
            let mut bits = word;
            core::iter::from_fn(move || {
                if bits == 0 {
                    return None;
                }
                let bit = bits.trailing_zeros();
                bits &= bits - 1;
                Some(Reg::new_preg((index * 64) as u32 + bit))
            })
        })
    }
}
