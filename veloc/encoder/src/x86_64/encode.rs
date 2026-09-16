//! Prefix and addressing algorithms shared by all declared legacy forms.
use super::*;
use crate::{Error, Fixup};
type Result<T> = core::result::Result<T, Error>;

impl Scale {
    fn bits(self) -> u8 {
        match self {
            Self::One => 0,
            Self::Two => 1,
            Self::Four => 2,
            Self::Eight => 3,
        }
    }
}

#[derive(Default)]
struct Rex {
    r: bool,
    x: bool,
    b: bool,
    force: bool,
}
impl Rex {
    fn emit(&self, wide: bool, out: &mut Instruction) -> Result<()> {
        if wide || self.r || self.x || self.b || self.force {
            out.push(
                0x40 | (u8::from(wide) << 3)
                    | (u8::from(self.r) << 2)
                    | (u8::from(self.x) << 1)
                    | u8::from(self.b),
            )?;
        }
        Ok(())
    }
}

fn address(reg: u8, addr: Address, rex: &mut Rex, out: &mut Instruction) -> Result<Option<Fixup>> {
    match addr {
        Address::RipRelative(addend) => {
            out.push((reg << 3) | 5)?;
            let offset = out.len;
            out.extend(&[0; 4])?;
            Ok(Some(Fixup {
                offset,
                bytes: 4,
                base: 0,
                addend,
            }))
        }
        Address::BaseIndex(Memory {
            base,
            index,
            displacement,
        }) => {
            let displacement = i32::try_from(displacement).map_err(|_| Error::Address)?;
            let base = base.map(Reg::number);
            let index = index.map(|i| (i.reg.number(), i.scale));
            // rsp cannot be an index, but r12 can (REX.X distinguishes it).
            if index.is_some_and(|(r, _)| r == 4) {
                return Err(Error::Address);
            }
            rex.b = base.is_some_and(|r| r >= 8);
            rex.x = index.is_some_and(|(r, _)| r >= 8);
            let mode = if base.is_none() || (displacement == 0 && base.unwrap() & 7 != 5) {
                0
            } else if i8::try_from(displacement).is_ok() {
                1
            } else {
                2
            };
            let sib = index.is_some() || base.is_none() || base.is_some_and(|r| r & 7 == 4);
            out.push((mode << 6) | (reg << 3) | if sib { 4 } else { base.unwrap() & 7 })?;
            if sib {
                let (index, scale) = index.map(|(r, s)| (r & 7, s.bits())).unwrap_or((4, 0));
                out.push((scale << 6) | (index << 3) | base.map(|r| r & 7).unwrap_or(5))?;
            }
            if base.is_none() || mode == 2 {
                out.extend(&displacement.to_le_bytes())?;
            } else if mode == 1 {
                out.push(displacement as u8)?;
            }
            Ok(None)
        }
    }
}

pub fn encode(encoding: Legacy, form: Form, immediate: Immediate) -> Result<Instruction> {
    let mut rex = Rex::default();
    let mut operands = Instruction::default();
    let mut opcode = encoding.opcode;
    let mut fixup = match form {
        Form::None => None,
        Form::OpcodeReg(reg) => {
            let n = reg.number();
            rex.b = n >= 8;
            rex.force = false;
            if opcode & 7 != 0 {
                return Err(Error::Encoding);
            }
            opcode |= n & 7;
            None
        }
        Form::ModRm(reg, rm) => {
            let reg = match reg {
                RegField::Register(r) | RegField::ByteRegister(r) => {
                    let n = r.number();
                    rex.r = n >= 8;
                    rex.force |= matches!(reg, RegField::ByteRegister(_)) && n >= 4;
                    n & 7
                }
                RegField::Extension(n) => {
                    if n >= 8 {
                        return Err(Error::Encoding);
                    }
                    n
                }
            };
            match rm {
                Rm::Register(r) | Rm::ByteRegister(r) => {
                    let n = r.number();
                    rex.b = n >= 8;
                    rex.force |= matches!(rm, Rm::ByteRegister(_)) && n >= 4;
                    operands.push(0xc0 | (reg << 3) | (n & 7))?;
                    None
                }
                Rm::Memory(addr) => address(reg, addr, &mut rex, &mut operands)?,
            }
        }
    };
    match immediate {
        Immediate::None => {}
        Immediate::Bits8(n) => {
            if !(-128..=255).contains(&n) {
                return Err(Error::Immediate);
            }
            operands.push(n as u8)?;
        }
        Immediate::Bits32(n) => {
            if !(i32::MIN as i64..=u32::MAX as i64).contains(&n) {
                return Err(Error::Immediate);
            }
            operands.extend(&(n as u32).to_le_bytes())?;
        }
        Immediate::Signed32(n) => {
            let n = i32::try_from(n).map_err(|_| Error::Immediate)?;
            operands.extend(&n.to_le_bytes())?;
        }
        Immediate::Bits64(n) => operands.extend(&n.to_le_bytes())?,
        Immediate::Relative(addend) => {
            if fixup.is_some() {
                return Err(Error::Encoding);
            }
            fixup = Some(Fixup {
                offset: operands.len,
                bytes: 4,
                base: 0,
                addend,
            });
            operands.extend(&[0; 4])?;
        }
    }
    let mut bytes = Instruction::default();
    match encoding.prefix {
        Prefix::None => {}
        Prefix::P66 => bytes.push(0x66)?,
        Prefix::F2 => bytes.push(0xf2)?,
        Prefix::F3 => bytes.push(0xf3)?,
    }
    rex.emit(encoding.wide, &mut bytes)?;
    match encoding.map {
        OpcodeMap::Primary => {}
        OpcodeMap::Map0F => bytes.push(0x0f)?,
        OpcodeMap::Map0F38 => bytes.extend(&[0x0f, 0x38])?,
        OpcodeMap::Map0F3A => bytes.extend(&[0x0f, 0x3a])?,
    }
    bytes.push(opcode)?;
    if let Some(fixup) = &mut fixup {
        fixup.offset += bytes.len;
    }
    bytes.extend(operands.bytes())?;
    if let Some(fixup) = &mut fixup {
        fixup.base = bytes.len;
    }
    bytes.fixup = fixup;
    Ok(bytes)
}

/// Produce a concrete branch form. Layout chooses the width; encoding does not.
pub fn encode_branch(branch: Branch, short: bool) -> Result<Instruction> {
    let valid = match branch.map {
        OpcodeMap::Primary => branch.near == 0xe9 && branch.short == 0xeb,
        OpcodeMap::Map0F => {
            (0x80..=0x8f).contains(&branch.near) && branch.short == branch.near - 0x10
        }
        _ => false,
    };
    if !valid {
        return Err(Error::Encoding);
    }
    let mut out = Instruction::default();
    if !short && matches!(branch.map, OpcodeMap::Map0F) {
        out.push(0x0f)?;
    }
    out.push(if short { branch.short } else { branch.near })?;
    let offset = out.len;
    let bytes = if short { 1 } else { 4 };
    for _ in 0..bytes {
        out.push(0)?;
    }
    out.fixup = Some(Fixup {
        offset,
        bytes,
        base: out.len,
        addend: 0,
    });
    Ok(out)
}
