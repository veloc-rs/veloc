//! RV64C equivalents of individual, already allocated 32-bit instructions.
//! PC-relative encodings are handled separately by the layout relaxer.

fn compact(reg: u32) -> bool {
    (8..16).contains(&reg)
}
fn signed6(value: i32) -> bool {
    (-32..32).contains(&value)
}
fn ci(base: u32, rd: u32, imm: i32) -> u16 {
    (base | rd << 7 | (imm as u32 & 31) << 2 | (imm as u32 & 32) << 7) as u16
}

/// Return an equivalent instruction, or retain the original 32-bit encoding.
pub fn compress(word: u32) -> Option<u16> {
    let op = word & 127;
    let rd = word >> 7 & 31;
    let f3 = word >> 12 & 7;
    let rs1 = word >> 15 & 31;
    let rs2 = word >> 20 & 31;
    let f7 = word >> 25;
    let imm = (word as i32) >> 20;
    match op {
        0x13 | 0x1b if f3 == 0 && rd != 0 => {
            if op == 0x13 && rs1 == 0 && signed6(imm) {
                return Some(ci(0x4001, rd, imm));
            }
            if rd == rs1 && signed6(imm) && (op == 0x1b || imm != 0) {
                return Some(ci(if op == 0x13 { 0x0001 } else { 0x2001 }, rd, imm));
            }
            if op == 0x13
                && rd == 2
                && rs1 == 2
                && imm != 0
                && (-512..512).contains(&imm)
                && imm % 16 == 0
            {
                let n = imm as u32;
                return Some(
                    (0x6101
                        | (n & 0x200) << 3
                        | (n & 0x10) << 2
                        | (n & 0x40) >> 1
                        | (n & 0x180) >> 4
                        | (n & 0x20) >> 3) as u16,
                );
            }
            if op == 0x13 && rs1 == 2 && compact(rd) && (4..1024).contains(&imm) && imm % 4 == 0 {
                let n = imm as u32;
                return Some(
                    ((n & 0x30) << 7
                        | (n & 0x3c0) << 1
                        | (n & 4) << 4
                        | (n & 8) << 2
                        | (rd - 8) << 2) as u16,
                );
            }
            if op == 0x13 && imm == 0 && rs1 != 0 {
                return Some((0x8002 | rd << 7 | rs1 << 2) as u16);
            }
        }
        0x37 if rd != 0 && rd != 2 => {
            let n = (word as i32) >> 12;
            if n != 0 && signed6(n) {
                return Some(ci(0x6001, rd, n));
            }
        }
        0x13 if rd == rs1 && rd != 0 => {
            let shamt = (word >> 20 & 63) as i32;
            if f3 == 1 && word >> 26 == 0 && shamt != 0 {
                return Some(ci(0x0002, rd, shamt));
            }
            if compact(rd) {
                if f3 == 7 && signed6(imm) {
                    return Some(ci(0x8801, rd - 8, imm));
                }
                if f3 == 5 && shamt != 0 && matches!(word >> 26, 0 | 0x10) {
                    return Some(ci(
                        if word >> 26 == 0 { 0x8001 } else { 0x8401 },
                        rd - 8,
                        shamt,
                    ));
                }
            }
        }
        0x33 | 0x3b if rd != 0 => {
            if op == 0x33 && f3 == 0 && f7 == 0 {
                if rs1 == 0 && rs2 != 0 {
                    return Some((0x8002 | rd << 7 | rs2 << 2) as u16);
                }
                if rd == rs1 && rs2 != 0 {
                    return Some((0x9002 | rd << 7 | rs2 << 2) as u16);
                }
                if rd == rs2 && rs1 != 0 {
                    return Some((0x9002 | rd << 7 | rs1 << 2) as u16);
                }
            }
            if rd == rs1 && compact(rd) && compact(rs2) {
                let base = match (op, f3, f7) {
                    (0x33, 0, 0x20) => 0x8c01,
                    (0x33, 4, 0) => 0x8c21,
                    (0x33, 6, 0) => 0x8c41,
                    (0x33, 7, 0) => 0x8c61,
                    (0x3b, 0, 0x20) => 0x9c01,
                    (0x3b, 0, 0) => 0x9c21,
                    _ => return None,
                };
                return Some((base | (rd - 8) << 7 | (rs2 - 8) << 2) as u16);
            }
        }
        0x67 if f3 == 0 && imm == 0 && rs1 != 0 && matches!(rd, 0 | 1) => {
            return Some((0x8002 | rd << 12 | rs1 << 7) as u16);
        }
        0x03 | 0x07 | 0x23 | 0x27 => {
            let store = op & 0x20 != 0;
            let float = op & 4 != 0;
            if !matches!(f3, 2 | 3) || (float && f3 != 3) {
                return None;
            }
            let reg = if store { rs2 } else { rd };
            let n = if store {
                ((word as i32 >> 25) << 5) | (word >> 7 & 31) as i32
            } else {
                imm
            };
            let wide = f3 == 3;
            if n < 0 || n % (if wide { 8 } else { 4 }) != 0 {
                return None;
            }
            let n = n as u32;
            let funct3 = if float {
                1
            } else if wide {
                3
            } else {
                2
            };
            let base = (funct3 + if store { 4 } else { 0 }) << 13;
            if compact(rs1) && compact(reg) && n < if wide { 256 } else { 128 } {
                let bits = if wide {
                    (n & 0xc0) >> 1
                } else {
                    (n & 4) << 4 | (n & 0x40) >> 1
                };
                return Some(
                    (base | (n & 0x38) << 7 | bits | (rs1 - 8) << 7 | (reg - 8) << 2) as u16,
                );
            }
            if rs1 == 2 && n < if wide { 512 } else { 256 } {
                if store {
                    let bits = if wide {
                        (n & 0x38) << 7 | (n & 0x1c0) << 1
                    } else {
                        (n & 0x3c) << 7 | (n & 0xc0) << 1
                    };
                    return Some((base | 2 | bits | reg << 2) as u16);
                }
                if reg != 0 || float {
                    let bits = if wide {
                        (n & 0x18) << 2 | (n & 0x1c0) >> 4
                    } else {
                        (n & 0x1c) << 2 | (n & 0xc0) >> 4
                    };
                    return Some((base | 2 | reg << 7 | (n & 0x20) << 7 | bits) as u16);
                }
            }
        }
        _ => {}
    }
    None
}

pub fn jump(offset: i64) -> Result<u16, crate::Error> {
    if offset % 2 != 0 || !(-2048..2048).contains(&offset) {
        return Err(crate::Error::Immediate);
    }
    let n = offset as u16;
    Ok(0xa001
        | (n & 0x800) << 1
        | (n & 0x10) << 7
        | (n & 0x300) << 1
        | (n & 0x400) >> 2
        | (n & 0x40) << 1
        | (n & 0x80) >> 1
        | (n & 0xe) << 2
        | (n & 0x20) >> 3)
}

pub fn patch_jump(bytes: &mut [u8], offset: i64) -> Result<(), crate::Error> {
    if bytes.len() != 2 {
        return Err(crate::Error::Encoding);
    }
    bytes.copy_from_slice(&jump(offset)?.to_le_bytes());
    Ok(())
}

pub fn branch(nonzero: bool, reg: u32, offset: i64) -> Result<u16, crate::Error> {
    if !compact(reg) {
        return Err(crate::Error::Register);
    }
    if offset % 2 != 0 || !(-256..256).contains(&offset) {
        return Err(crate::Error::Immediate);
    }
    let n = offset as u16;
    Ok(0xc001
        | (nonzero as u16) << 13
        | ((reg - 8) as u16) << 7
        | (n & 0x100) << 4
        | (n & 0x18) << 7
        | (n & 0xc0) >> 1
        | (n & 6) << 2
        | (n & 0x20) >> 3)
}

pub fn patch_branch(bytes: &mut [u8], offset: i64) -> Result<(), crate::Error> {
    if bytes.len() != 2 {
        return Err(crate::Error::Encoding);
    }
    let old = u16::from_le_bytes(bytes.try_into().unwrap());
    bytes.copy_from_slice(
        &branch(old & 0x2000 != 0, 8 + ((old >> 7) & 7) as u32, offset)?.to_le_bytes(),
    );
    Ok(())
}
