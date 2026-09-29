//! RV64 fixed-width instruction formats. Register operands are hardware numbers.
use crate::Error;
pub fn r(op: u32, rd: u32, funct3: u32, rs1: u32, rs2: u32, funct7: u32) -> u32 {
    assert!(rd < 32 && rs1 < 32 && rs2 < 32 && funct3 < 8 && funct7 < 128);
    op | rd << 7 | funct3 << 12 | rs1 << 15 | rs2 << 20 | funct7 << 25
}
pub fn i(op: u32, rd: u32, funct3: u32, rs1: u32, imm: i64) -> u32 {
    assert!((-2048..2048).contains(&imm));
    r(
        op,
        rd,
        funct3,
        rs1,
        (imm as u32) & 31,
        ((imm as u32) >> 5) & 127,
    )
}
pub fn s(op: u32, funct3: u32, base: u32, src: u32, imm: i64) -> u32 {
    assert!((-2048..2048).contains(&imm));
    r(
        op,
        (imm as u32) & 31,
        funct3,
        base,
        src,
        ((imm as u32) >> 5) & 127,
    )
}
pub fn b(funct3: u32, rs1: u32, rs2: u32, offset: i32) -> u32 {
    assert!((-4096..4096).contains(&offset) && offset % 2 == 0);
    let v = offset as u32;
    0x63 | ((v >> 11) & 1) << 7
        | ((v >> 1) & 15) << 8
        | funct3 << 12
        | rs1 << 15
        | rs2 << 20
        | ((v >> 5) & 63) << 25
        | ((v >> 12) & 1) << 31
}
pub fn j(rd: u32, offset: i32) -> u32 {
    assert!((-1048576..1048576).contains(&offset) && offset % 2 == 0);
    let v = offset as u32;
    0x6f | rd << 7
        | ((v >> 12) & 255) << 12
        | ((v >> 11) & 1) << 20
        | ((v >> 1) & 1023) << 21
        | ((v >> 20) & 1) << 31
}
/// Patch an AUIPC/JALR pair, retaining its register fields.
pub fn patch_jump(bytes: &mut [u8], offset: i64) -> Result<(), Error> {
    if bytes.len() != 8 || offset % 2 != 0 || !(-2147483648..2147481600).contains(&offset) {
        return Err(Error::Immediate);
    }
    let high = (offset + 0x800) >> 12;
    let low = offset - (high << 12);
    let first = u32::from_le_bytes(bytes[..4].try_into().unwrap());
    let second = u32::from_le_bytes(bytes[4..].try_into().unwrap());
    bytes[..4].copy_from_slice(&((first & 0xfff) | ((high as u32) << 12)).to_le_bytes());
    bytes[4..].copy_from_slice(&((second & 0xfffff) | ((low as u32) << 20)).to_le_bytes());
    Ok(())
}

include!(concat!(env!("OUT_DIR"), "/riscv64.rs"));
impl Reg {
    pub fn hardware(self) -> u32 {
        self as u32 % 32
    }
    pub fn is_float(self) -> bool {
        self as u32 >= 32
    }
}
pub fn register(index: u32) -> Result<Reg, Error> {
    const REGS: [Reg; 64] = [
        Reg::X0,
        Reg::X1,
        Reg::X2,
        Reg::X3,
        Reg::X4,
        Reg::X5,
        Reg::X6,
        Reg::X7,
        Reg::X8,
        Reg::X9,
        Reg::X10,
        Reg::X11,
        Reg::X12,
        Reg::X13,
        Reg::X14,
        Reg::X15,
        Reg::X16,
        Reg::X17,
        Reg::X18,
        Reg::X19,
        Reg::X20,
        Reg::X21,
        Reg::X22,
        Reg::X23,
        Reg::X24,
        Reg::X25,
        Reg::X26,
        Reg::X27,
        Reg::X28,
        Reg::X29,
        Reg::X30,
        Reg::X31,
        Reg::F0,
        Reg::F1,
        Reg::F2,
        Reg::F3,
        Reg::F4,
        Reg::F5,
        Reg::F6,
        Reg::F7,
        Reg::F8,
        Reg::F9,
        Reg::F10,
        Reg::F11,
        Reg::F12,
        Reg::F13,
        Reg::F14,
        Reg::F15,
        Reg::F16,
        Reg::F17,
        Reg::F18,
        Reg::F19,
        Reg::F20,
        Reg::F21,
        Reg::F22,
        Reg::F23,
        Reg::F24,
        Reg::F25,
        Reg::F26,
        Reg::F27,
        Reg::F28,
        Reg::F29,
        Reg::F30,
        Reg::F31,
    ];
    REGS.get(index as usize).copied().ok_or(Error::Register)
}
struct Encoder(crate::Encoded<128>);
impl Encoder {
    fn word(&mut self, word: u32) -> Result<(), Error> {
        self.0.extend(&word.to_le_bytes())
    }
    fn constant(&mut self, d: u32, value: i64) -> Result<(), Error> {
        if (-2048..2048).contains(&value) {
            return self.word(i(0x13, d, 0, 0, value));
        }
        let low = (value << 52) >> 52;
        self.constant(d, ((value as i128 - low as i128) >> 12) as i64)?;
        self.word(i(0x13, d, 1, d, 12))?;
        if low != 0 {
            self.word(i(0x13, d, 0, d, low))?;
        }
        Ok(())
    }
    fn address(&mut self, d: u32, address: Address) -> Result<(), Error> {
        let base = address.base.hardware();
        if address.base.is_float() {
            return Err(Error::Address);
        }
        if (-2048..2048).contains(&address.offset) {
            self.word(i(0x13, d, 0, base, address.offset))
        } else {
            self.constant(31, address.offset)?;
            self.word(r(0x33, d, 0, base, 31, 0))
        }
    }
    fn memory_address(&mut self, address: Address) -> Result<(u32, i64), Error> {
        if address.base.is_float() {
            return Err(Error::Address);
        }
        if (-2048..2048).contains(&address.offset) {
            Ok((address.base.hardware(), address.offset))
        } else {
            self.address(31, address)?;
            Ok((31, 0))
        }
    }
}
pub fn encode(instruction: Instruction) -> Result<crate::Encoded<128>, Error> {
    let mut e = Encoder(Default::default());
    match instruction {
        Instruction::R(op, d, f3, a, b, f7) => {
            e.word(r(op, d.hardware(), f3, a.hardware(), b.hardware(), f7))?
        }
        Instruction::I(op, d, f3, a, imm) => e.word(i(op, d.hardware(), f3, a.hardware(), imm))?,
        Instruction::B(f3, a, rs2, offset) => e.word(b(
            f3,
            a.hardware(),
            rs2.hardware(),
            i32::try_from(offset).map_err(|_| Error::Immediate)?,
        ))?,
        Instruction::J(d, offset) => e.word(j(
            d.hardware(),
            i32::try_from(offset).map_err(|_| Error::Immediate)?,
        ))?,
        Instruction::Constant(d, value) => e.constant(d.hardware(), value)?,
        Instruction::Move(d, a, bits) => {
            let (rd, rs) = (d.hardware(), a.hardware());
            let word = match (d.is_float(), a.is_float()) {
                (false, false) => i(if bits == 32 { 0x1b } else { 0x13 }, rd, 0, rs, 0),
                (true, true) => r(0x53, rd, 0, rs, rs, if bits == 32 { 0x10 } else { 0x11 }),
                (true, false) => r(0x53, rd, 0, rs, 0, if bits == 32 { 0x78 } else { 0x79 }),
                (false, true) => r(0x53, rd, 0, rs, 0, if bits == 32 { 0x70 } else { 0x71 }),
            };
            e.word(word)?;
        }
        Instruction::Address(d, address) => e.address(d.hardware(), address)?,
        Instruction::Load(op, d, address, f3) => {
            let (base, offset) = e.memory_address(address)?;
            e.word(i(op, d.hardware(), f3, base, offset))?;
        }
        Instruction::Store(op, src, address, f3) => {
            let (base, offset) = e.memory_address(address)?;
            e.word(s(op, f3, base, src.hardware(), offset))?;
        }
    }
    Ok(e.0)
}
