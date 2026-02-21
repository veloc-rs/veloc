pub(crate) mod codegen;
mod snippets;

pub use codegen::X86_64Backend;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Reg {
    RAX = 0,
    RCX = 1,
    RDX = 2,
    RBX = 3,
    RSP = 4,
    RBP = 5,
    RSI = 6,
    RDI = 7,
    R8 = 8,
    R9 = 9,
    R10 = 10,
    R11 = 11,
    R12 = 12,
    R13 = 13,
    R14 = 14,
    R15 = 15,

    XMM0 = 16,
    XMM1 = 17,
    XMM2 = 18,
    XMM3 = 19,
    XMM4 = 20,
    XMM5 = 21,
    XMM6 = 22,
    XMM7 = 23,
}

impl From<Reg> for crate::Reg {
    fn from(reg: Reg) -> Self {
        let val = reg as u8;
        // 如果是 XMM 寄存器 (16-31)，转换为其硬件索引 (0-15)
        if val >= 16 {
            crate::Reg(val - 16)
        } else {
            crate::Reg(val)
        }
    }
}
