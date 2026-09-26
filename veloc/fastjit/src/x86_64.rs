use crate::image;
use crate::stencil::{Code, Hole, Label, Patch, PatchKind, Stencil};
use crate::{Assembler, Error, Result};
use veloc_mir::{CallConv, FuncBody, InstView, Module, Opcode, Signature, Successor, Type, Value};

pub(super) struct Target;

impl image::Target for Target {
    const ARCH: object::Architecture = object::Architecture::X86_64;
    const ENDIAN: object::Endianness = object::Endianness::Little;

    fn compile<'a>(module: &'a Module, body: &'a FuncBody, sig: &Signature) -> Result<Code<'a>> {
        if sig.call_conv != CallConv::SystemV {
            return Err(unsupported("calling convention"));
        }
        compile_function(module, body, sig.params(), sig.returns())
    }
}

const I32_HOLE_3: &[Hole] = &[Hole {
    offset: 3,
    kind: PatchKind::I32,
}];
const I32_HOLE_2: &[Hole] = &[Hole {
    offset: 2,
    kind: PatchKind::I32,
}];
const U64_HOLE_2: &[Hole] = &[Hole {
    offset: 2,
    kind: PatchKind::U64,
}];
const REL32_HOLE_1: &[Hole] = &[Hole {
    offset: 1,
    kind: PatchKind::X86Rel32,
}];
const REL32_HOLE_2: &[Hole] = &[Hole {
    offset: 2,
    kind: PatchKind::X86Rel32,
}];

const PROLOGUE: Stencil = Stencil {
    bytes: &[0x55, 0x48, 0x89, 0xe5, 0x48, 0x81, 0xec, 0, 0, 0, 0],
    holes: &[Hole {
        offset: 7,
        kind: PatchKind::I32,
    }],
};
const EPILOGUE: Stencil = Stencil {
    bytes: &[0xc9, 0xc3],
    holes: &[],
};
const JUMP: Stencil = Stencil {
    bytes: &[0xe9, 0, 0, 0, 0],
    holes: REL32_HOLE_1,
};
const JUMP_ZERO: Stencil = Stencil {
    bytes: &[0x0f, 0x84, 0, 0, 0, 0],
    holes: REL32_HOLE_2,
};
const JUMP_EQUAL: Stencil = Stencil {
    bytes: &[0x0f, 0x84, 0, 0, 0, 0],
    holes: REL32_HOLE_2,
};
const TEST_EAX: Stencil = Stencil {
    bytes: &[0x85, 0xc0],
    holes: &[],
};
const COMPARE_EAX_IMM: Stencil = Stencil {
    bytes: &[0x3d, 0, 0, 0, 0],
    holes: &[Hole {
        offset: 1,
        kind: PatchKind::I32,
    }],
};
const MOV_RAX_IMM: Stencil = Stencil {
    bytes: &[0x48, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0],
    holes: U64_HOLE_2,
};
const MOV_RCX_IMM: Stencil = Stencil {
    bytes: &[0x48, 0xb9, 0, 0, 0, 0, 0, 0, 0, 0],
    holes: U64_HOLE_2,
};
const LOAD_EAX: Stencil = Stencil {
    bytes: &[0x8b, 0x85, 0, 0, 0, 0],
    holes: I32_HOLE_2,
};
const LOAD_ECX: Stencil = Stencil {
    bytes: &[0x8b, 0x8d, 0, 0, 0, 0],
    holes: I32_HOLE_2,
};
const LOAD_RAX: Stencil = Stencil {
    bytes: &[0x48, 0x8b, 0x85, 0, 0, 0, 0],
    holes: I32_HOLE_3,
};
const ADDR_RBP: Stencil = Stencil {
    bytes: &[0x48, 0x8d, 0x85, 0, 0, 0, 0],
    holes: I32_HOLE_3,
};
const LOAD_RCX: Stencil = Stencil {
    bytes: &[0x48, 0x8b, 0x8d, 0, 0, 0, 0],
    holes: I32_HOLE_3,
};
const LOAD_EDX: Stencil = Stencil {
    bytes: &[0x8b, 0x95, 0, 0, 0, 0],
    holes: I32_HOLE_2,
};
const STORE_RAX: Stencil = Stencil {
    bytes: &[0x48, 0x89, 0x85, 0, 0, 0, 0],
    holes: I32_HOLE_3,
};
const LOAD_MEM8: Stencil = Stencil {
    bytes: &[0x0f, 0xb6, 0x80, 0, 0, 0, 0],
    holes: I32_HOLE_3,
};
const LOAD_MEM16: Stencil = Stencil {
    bytes: &[0x0f, 0xb7, 0x80, 0, 0, 0, 0],
    holes: I32_HOLE_3,
};
const LOAD_MEM32: Stencil = Stencil {
    bytes: &[0x8b, 0x80, 0, 0, 0, 0],
    holes: I32_HOLE_2,
};
const LOAD_MEM64: Stencil = Stencil {
    bytes: &[0x48, 0x8b, 0x80, 0, 0, 0, 0],
    holes: I32_HOLE_3,
};
const STORE_MEM8: Stencil = Stencil {
    bytes: &[0x88, 0x88, 0, 0, 0, 0],
    holes: I32_HOLE_2,
};
const STORE_MEM16: Stencil = Stencil {
    bytes: &[0x66, 0x89, 0x88, 0, 0, 0, 0],
    holes: I32_HOLE_3,
};
const STORE_MEM32: Stencil = Stencil {
    bytes: &[0x89, 0x88, 0, 0, 0, 0],
    holes: I32_HOLE_2,
};
const STORE_MEM64: Stencil = Stencil {
    bytes: &[0x48, 0x89, 0x88, 0, 0, 0, 0],
    holes: I32_HOLE_3,
};
const PTR_OFFSET: Stencil = Stencil {
    bytes: &[0x48, 0x8d, 0x80, 0, 0, 0, 0],
    holes: I32_HOLE_3,
};
const PTR_INDEX: [Stencil; 4] = [
    Stencil {
        bytes: &[0x48, 0x8d, 0x84, 0x08, 0, 0, 0, 0],
        holes: &[Hole {
            offset: 4,
            kind: PatchKind::I32,
        }],
    },
    Stencil {
        bytes: &[0x48, 0x8d, 0x84, 0x48, 0, 0, 0, 0],
        holes: &[Hole {
            offset: 4,
            kind: PatchKind::I32,
        }],
    },
    Stencil {
        bytes: &[0x48, 0x8d, 0x84, 0x88, 0, 0, 0, 0],
        holes: &[Hole {
            offset: 4,
            kind: PatchKind::I32,
        }],
    },
    Stencil {
        bytes: &[0x48, 0x8d, 0x84, 0xc8, 0, 0, 0, 0],
        holes: &[Hole {
            offset: 4,
            kind: PatchKind::I32,
        }],
    },
];
const CMP32: Stencil = Stencil {
    bytes: &[0x39, 0xc8],
    holes: &[],
};
const CMP64: Stencil = Stencil {
    bytes: &[0x48, 0x39, 0xc8],
    holes: &[],
};
const ZERO_EXTEND_AL: Stencil = Stencil {
    bytes: &[0x0f, 0xb6, 0xc0],
    holes: &[],
};
const CALL: Stencil = Stencil {
    bytes: &[0xe8, 0, 0, 0, 0],
    holes: REL32_HOLE_1,
};
const CALL_ARGS: [Stencil; 6] = [
    Stencil {
        bytes: &[0x48, 0x8b, 0xbd, 0, 0, 0, 0],
        holes: I32_HOLE_3,
    },
    Stencil {
        bytes: &[0x48, 0x8b, 0xb5, 0, 0, 0, 0],
        holes: I32_HOLE_3,
    },
    Stencil {
        bytes: &[0x48, 0x8b, 0x95, 0, 0, 0, 0],
        holes: I32_HOLE_3,
    },
    Stencil {
        bytes: &[0x48, 0x8b, 0x8d, 0, 0, 0, 0],
        holes: I32_HOLE_3,
    },
    Stencil {
        bytes: &[0x4c, 0x8b, 0x85, 0, 0, 0, 0],
        holes: I32_HOLE_3,
    },
    Stencil {
        bytes: &[0x4c, 0x8b, 0x8d, 0, 0, 0, 0],
        holes: I32_HOLE_3,
    },
];
const OUTGOING_ARG: Stencil = Stencil {
    bytes: &[0x48, 0x89, 0x84, 0x24, 0, 0, 0, 0],
    holes: &[Hole {
        offset: 4,
        kind: PatchKind::I32,
    }],
};
const STORE_PARAM: [Stencil; 6] = [
    Stencil {
        bytes: &[0x48, 0x89, 0xbd, 0, 0, 0, 0],
        holes: I32_HOLE_3,
    },
    Stencil {
        bytes: &[0x48, 0x89, 0xb5, 0, 0, 0, 0],
        holes: I32_HOLE_3,
    },
    Stencil {
        bytes: &[0x48, 0x89, 0x95, 0, 0, 0, 0],
        holes: I32_HOLE_3,
    },
    Stencil {
        bytes: &[0x48, 0x89, 0x8d, 0, 0, 0, 0],
        holes: I32_HOLE_3,
    },
    Stencil {
        bytes: &[0x4c, 0x89, 0x85, 0, 0, 0, 0],
        holes: I32_HOLE_3,
    },
    Stencil {
        bytes: &[0x4c, 0x89, 0x8d, 0, 0, 0, 0],
        holes: I32_HOLE_3,
    },
];

fn unsupported(message: impl Into<String>) -> Error {
    Error::Unsupported(message.into())
}

fn slot(value: Value) -> Result<i32> {
    let offset = (u64::from(value.0) + 1) * 8;
    i32::try_from(offset)
        .map(|n| -n)
        .map_err(|_| unsupported("stack frame too large"))
}

fn integer_width(ty: Type) -> Result<bool> {
    match ty {
        Type::I32 | Type::BOOL => Ok(false),
        Type::I64 | Type::PTR => Ok(true),
        _ => Err(unsupported(format!("type {ty}"))),
    }
}

fn memory_width(ty: Type) -> Result<u32> {
    match ty {
        Type::I8 | Type::BOOL => Ok(1),
        Type::I16 => Ok(2),
        Type::I32 | Type::F32 => Ok(4),
        Type::I64 | Type::PTR | Type::F64 => Ok(8),
        _ => Err(unsupported(format!("memory type {ty}"))),
    }
}

#[derive(Clone, Copy)]
enum AbiArg {
    Gpr(usize),
    Xmm(usize),
    Stack(usize),
}

fn abi_args(types: &[Type]) -> Result<Vec<AbiArg>> {
    let (mut gpr, mut xmm, mut stack) = (0, 0, 0);
    types
        .iter()
        .map(|&ty| {
            let arg = match ty {
                Type::F32 | Type::F64 if xmm < 8 => {
                    let arg = AbiArg::Xmm(xmm);
                    xmm += 1;
                    arg
                }
                Type::F32 | Type::F64 => {
                    let arg = AbiArg::Stack(stack);
                    stack += 1;
                    arg
                }
                Type::I8 | Type::I16 | Type::I32 | Type::I64 | Type::PTR | Type::BOOL
                    if gpr < 6 =>
                {
                    let arg = AbiArg::Gpr(gpr);
                    gpr += 1;
                    arg
                }
                Type::I8 | Type::I16 | Type::I32 | Type::I64 | Type::PTR | Type::BOOL => {
                    let arg = AbiArg::Stack(stack);
                    stack += 1;
                    arg
                }
                _ => return Err(unsupported(format!("ABI type {ty}"))),
            };
            Ok(arg)
        })
        .collect()
}

fn emit_xmm_slot(
    asm: &mut Assembler<'_>,
    reg: usize,
    value: Value,
    ty: Type,
    store: bool,
) -> Result<()> {
    if reg >= 8 {
        return Err(unsupported("XMM register index"));
    }
    let prefix = match ty {
        Type::F32 => 0xf3,
        Type::F64 => 0xf2,
        _ => return Err(unsupported(format!("XMM type {ty}"))),
    };
    let mut bytes = [
        prefix,
        0x0f,
        if store { 0x11 } else { 0x10 },
        0x85 + (reg as u8) * 8,
        0,
        0,
        0,
        0,
    ];
    bytes[4..].copy_from_slice(&slot(value)?.to_le_bytes());
    asm.emit_raw(&bytes);
    Ok(())
}

fn emit_float(asm: &mut Assembler<'_>, opcode: Opcode, ty: Type) -> Result<()> {
    let prefix = if ty == Type::F32 {
        0xf3
    } else if ty == Type::F64 {
        0xf2
    } else {
        return Err(unsupported(format!("float type {ty}")));
    };
    let byte = match opcode {
        Opcode::FAdd => 0x58,
        Opcode::FSub => 0x5c,
        Opcode::FMul => 0x59,
        Opcode::FDiv => 0x5e,
        _ => return Err(unsupported(format!("float opcode {opcode:?}"))),
    };
    asm.emit_raw(&[prefix, 0x0f, byte, 0xc1]);
    Ok(())
}

fn emit_bytes(asm: &mut Assembler<'_>, bytes: &'static [u8]) -> Result<()> {
    asm.emit(&Stencil { bytes, holes: &[] }, &[])
}

fn emit_compare(asm: &mut Assembler<'_>, kind: veloc_mir::IntCC, wide: bool) -> Result<()> {
    asm.emit(if wide { &CMP64 } else { &CMP32 }, &[])?;
    let condition: &'static [u8] = match kind {
        veloc_mir::IntCC::Eq => &[0x0f, 0x94, 0xc0],
        veloc_mir::IntCC::Ne => &[0x0f, 0x95, 0xc0],
        veloc_mir::IntCC::LtS => &[0x0f, 0x9c, 0xc0],
        veloc_mir::IntCC::LtU => &[0x0f, 0x92, 0xc0],
        veloc_mir::IntCC::GtS => &[0x0f, 0x9f, 0xc0],
        veloc_mir::IntCC::GtU => &[0x0f, 0x97, 0xc0],
        veloc_mir::IntCC::LeS => &[0x0f, 0x9e, 0xc0],
        veloc_mir::IntCC::LeU => &[0x0f, 0x96, 0xc0],
        veloc_mir::IntCC::GeS => &[0x0f, 0x9d, 0xc0],
        veloc_mir::IntCC::GeU => &[0x0f, 0x93, 0xc0],
    };
    emit_bytes(asm, condition)?;
    asm.emit(&ZERO_EXTEND_AL, &[])
}

fn emit_one(asm: &mut Assembler<'_>, stencil: &Stencil, value: i32) -> Result<()> {
    asm.emit(stencil, &[Patch::I32(value)])
}

fn emit_call_args(asm: &mut Assembler<'_>, args: &[Value], types: &[Type]) -> Result<()> {
    let locations = abi_args(types)?;
    for (&value, &location) in args.iter().zip(&locations) {
        if let AbiArg::Stack(index) = location {
            emit_one(asm, &LOAD_RAX, slot(value)?)?;
            let offset = index
                .checked_mul(8)
                .and_then(|n| i32::try_from(n).ok())
                .ok_or_else(|| unsupported("outgoing arguments"))?;
            emit_one(asm, &OUTGOING_ARG, offset)?;
        }
    }
    for ((&value, &ty), &location) in args.iter().zip(types).zip(&locations) {
        match location {
            AbiArg::Gpr(index) => emit_one(asm, &CALL_ARGS[index], slot(value)?)?,
            AbiArg::Xmm(index) => emit_xmm_slot(asm, index, value, ty, false)?,
            AbiArg::Stack(_) => {}
        }
    }
    Ok(())
}

fn emit_edge(
    asm: &mut Assembler<'_>,
    dest: Successor<'_>,
    body: &FuncBody,
    labels: &[Option<Label>],
    first_temp: usize,
) -> Result<()> {
    let params = body.dfg().block_params(dest.block);
    if params.len() != dest.args.len() {
        return Err(unsupported("successor argument arity"));
    }
    // Stage all sources before overwriting destinations. This implements a
    // parallel block-parameter copy even when source and destination overlap.
    for (index, &source) in dest.args.iter().enumerate() {
        emit_one(asm, &LOAD_RAX, slot(source)?)?;
        let temp = Value(u32::try_from(first_temp + index).map_err(|_| unsupported("frame size"))?);
        emit_one(asm, &STORE_RAX, slot(temp)?)?;
    }
    for (index, &param) in params.iter().enumerate() {
        let temp = Value(u32::try_from(first_temp + index).map_err(|_| unsupported("frame size"))?);
        emit_one(asm, &LOAD_RAX, slot(temp)?)?;
        emit_one(asm, &STORE_RAX, slot(param)?)?;
    }
    let label = labels
        .get(dest.block.0 as usize)
        .copied()
        .flatten()
        .ok_or_else(|| unsupported("unplaced successor"))?;
    asm.emit(&JUMP, &[Patch::Label(label)])
}

fn compile_function<'a>(
    module: &'a Module,
    body: &FuncBody,
    params: &[Type],
    results: &[Type],
) -> Result<Code<'a>> {
    let dfg = body.dfg();
    let blocks: Vec<_> = body.layout().block_order().collect();
    if params.len() != body.params().len() {
        return Err(unsupported("parameter ABI"));
    }
    if results.len() > 1 {
        return Err(unsupported("multiple results"));
    }
    abi_args(params)?;
    for &ty in results {
        memory_width(ty)?;
    }

    let mut max_edge_args = 0;
    for &block in &blocks {
        for inst in body.layout().block_insts(block) {
            dfg.inst(inst).visit_successors(|edge| {
                max_edge_args = max_edge_args.max(edge.args.len());
            });
        }
    }
    let first_temp = dfg.values().len();
    let mut objects = vec![None; first_temp];
    let mut local_bytes = first_temp
        .checked_add(max_edge_args)
        .and_then(|n| n.checked_mul(8))
        .ok_or_else(|| unsupported("stack frame too large"))?;
    for &block in &blocks {
        for inst in body.layout().block_insts(block) {
            if let InstView::Alloca { size, align } = dfg.inst(inst) {
                let [result] = dfg.inst_results(inst) else {
                    return Err(unsupported("alloca result arity"));
                };
                let align = align as usize;
                if align == 0 || !align.is_power_of_two() || align > 16 {
                    return Err(unsupported("alloca alignment"));
                }
                local_bytes = local_bytes
                    .checked_add(size as usize)
                    .and_then(|n| n.checked_add(align - 1))
                    .map(|n| n & !(align - 1))
                    .ok_or_else(|| unsupported("stack frame too large"))?;
                objects[result.0 as usize] = Some(
                    -i32::try_from(local_bytes)
                        .map_err(|_| unsupported("stack frame too large"))?,
                );
            }
        }
    }
    let mut max_outgoing = 0;
    for &block in &blocks {
        for inst in body.layout().block_insts(block) {
            let sig = match dfg.inst(inst) {
                InstView::Call { func_id, .. } => {
                    Some(&module.signatures()[module.function(func_id).decl.signature])
                }
                InstView::CallIndirect { sig_id, .. } => Some(&module.signatures()[sig_id]),
                _ => None,
            };
            if let Some(sig) = sig {
                let stack = abi_args(sig.params())?
                    .into_iter()
                    .filter(|arg| matches!(arg, AbiArg::Stack(_)))
                    .count();
                max_outgoing = max_outgoing.max(stack);
            }
        }
    }
    let stack_size = max_outgoing
        .checked_mul(8)
        .and_then(|n| n.checked_add(local_bytes))
        .and_then(|n| n.checked_add(15))
        .map(|n| n & !15)
        .and_then(|n| i32::try_from(n).ok())
        .ok_or_else(|| unsupported("stack frame too large"))?;
    let mut asm = Assembler::new();
    let mut labels = vec![None; dfg.block_count()];
    for &block in &blocks {
        labels[block.0 as usize] = Some(asm.label());
    }
    emit_one(&mut asm, &PROLOGUE, stack_size)?;
    for ((&value, &ty), location) in body.params().iter().zip(params).zip(abi_args(params)?) {
        match location {
            AbiArg::Gpr(index) => emit_one(&mut asm, &STORE_PARAM[index], slot(value)?)?,
            AbiArg::Xmm(index) => emit_xmm_slot(&mut asm, index, value, ty, true)?,
            AbiArg::Stack(index) => {
                let offset = index
                    .checked_mul(8)
                    .and_then(|n| n.checked_add(16))
                    .and_then(|n| i32::try_from(n).ok())
                    .ok_or_else(|| unsupported("incoming arguments"))?;
                emit_one(&mut asm, &LOAD_RAX, offset)?;
                emit_one(&mut asm, &STORE_RAX, slot(value)?)?;
            }
        }
    }

    let mut returned = false;
    for &block in &blocks {
        asm.bind(labels[block.0 as usize].expect("placed block has label"))?;
        for inst in body.layout().block_insts(block) {
            let results_of_inst = dfg.inst_results(inst);
            match dfg.inst(inst) {
                InstView::Iconst { value } => {
                    let [result] = results_of_inst else {
                        return Err(unsupported("integer constant arity"));
                    };
                    memory_width(dfg.value_type(*result))?;
                    asm.emit(&MOV_RAX_IMM, &[Patch::U64(value.to_bits())])?;
                    emit_one(&mut asm, &STORE_RAX, slot(*result)?)?;
                }
                InstView::Bconst { value } => {
                    let [result] = results_of_inst else {
                        return Err(unsupported("boolean constant arity"));
                    };
                    asm.emit(&MOV_RAX_IMM, &[Patch::U64(u64::from(value))])?;
                    emit_one(&mut asm, &STORE_RAX, slot(*result)?)?;
                }
                InstView::Alloca { .. } => {
                    let [result] = results_of_inst else {
                        return Err(unsupported("alloca result arity"));
                    };
                    let offset =
                        objects[result.0 as usize].expect("placed alloca has stack object");
                    emit_one(&mut asm, &ADDR_RBP, offset)?;
                    emit_one(&mut asm, &STORE_RAX, slot(*result)?)?;
                }
                InstView::Fconst { value } => {
                    let [result] = results_of_inst else {
                        return Err(unsupported("float constant arity"));
                    };
                    match dfg.value_type(*result) {
                        Type::F32 | Type::F64 => {}
                        ty => return Err(unsupported(format!("float constant type {ty}"))),
                    }
                    asm.emit(&MOV_RAX_IMM, &[Patch::U64(value.to_bits())])?;
                    emit_one(&mut asm, &STORE_RAX, slot(*result)?)?;
                }
                InstView::Load { ptr, offset, .. } => {
                    let [result] = results_of_inst else {
                        return Err(unsupported("load result arity"));
                    };
                    let stencil = match memory_width(dfg.value_type(*result))? {
                        1 => &LOAD_MEM8,
                        2 => &LOAD_MEM16,
                        4 => &LOAD_MEM32,
                        8 => &LOAD_MEM64,
                        _ => unreachable!(),
                    };
                    emit_one(&mut asm, &LOAD_RAX, slot(ptr)?)?;
                    emit_one(
                        &mut asm,
                        stencil,
                        i32::try_from(offset).map_err(|_| unsupported("load offset"))?,
                    )?;
                    emit_one(&mut asm, &STORE_RAX, slot(*result)?)?;
                }
                InstView::Store {
                    ptr, value, offset, ..
                } => {
                    let stencil = match memory_width(dfg.value_type(value))? {
                        1 => &STORE_MEM8,
                        2 => &STORE_MEM16,
                        4 => &STORE_MEM32,
                        8 => &STORE_MEM64,
                        _ => unreachable!(),
                    };
                    emit_one(&mut asm, &LOAD_RAX, slot(ptr)?)?;
                    emit_one(&mut asm, &LOAD_RCX, slot(value)?)?;
                    emit_one(
                        &mut asm,
                        stencil,
                        i32::try_from(offset).map_err(|_| unsupported("store offset"))?,
                    )?;
                }
                InstView::PtrOffset { ptr, offset } => {
                    let [result] = results_of_inst else {
                        return Err(unsupported("pointer offset arity"));
                    };
                    emit_one(&mut asm, &LOAD_RAX, slot(ptr)?)?;
                    emit_one(&mut asm, &PTR_OFFSET, offset)?;
                    emit_one(&mut asm, &STORE_RAX, slot(*result)?)?;
                }
                InstView::PtrIndex { ptr, index, imm_id } => {
                    let [result] = results_of_inst else {
                        return Err(unsupported("pointer index arity"));
                    };
                    let scale = match imm_id.scale {
                        1 => 0,
                        2 => 1,
                        4 => 2,
                        8 => 3,
                        _ => return Err(unsupported(format!("pointer scale {}", imm_id.scale))),
                    };
                    emit_one(&mut asm, &LOAD_RAX, slot(ptr)?)?;
                    emit_one(&mut asm, &LOAD_RCX, slot(index)?)?;
                    emit_one(&mut asm, &PTR_INDEX[scale], imm_id.offset)?;
                    emit_one(&mut asm, &STORE_RAX, slot(*result)?)?;
                }
                InstView::IntCompare { kind, args } => {
                    let [result] = results_of_inst else {
                        return Err(unsupported("compare result arity"));
                    };
                    let wide = integer_width(dfg.value_type(args[0]))?;
                    emit_one(
                        &mut asm,
                        if wide { &LOAD_RAX } else { &LOAD_EAX },
                        slot(args[0])?,
                    )?;
                    emit_one(
                        &mut asm,
                        if wide { &LOAD_RCX } else { &LOAD_ECX },
                        slot(args[1])?,
                    )?;
                    emit_compare(&mut asm, kind, wide)?;
                    emit_one(&mut asm, &STORE_RAX, slot(*result)?)?;
                }
                InstView::Ternary {
                    opcode: Opcode::Select,
                    args,
                } => {
                    let [result] = results_of_inst else {
                        return Err(unsupported("select result arity"));
                    };
                    memory_width(dfg.value_type(*result))?;
                    emit_one(&mut asm, &LOAD_RAX, slot(args[2])?)?;
                    emit_one(&mut asm, &LOAD_RCX, slot(args[1])?)?;
                    emit_one(&mut asm, &LOAD_EDX, slot(args[0])?)?;
                    emit_bytes(&mut asm, &[0x85, 0xd2])?;
                    emit_bytes(&mut asm, &[0x48, 0x0f, 0x45, 0xc1])?;
                    emit_one(&mut asm, &STORE_RAX, slot(*result)?)?;
                }
                InstView::Call { func_id, args } => {
                    let target = module.function(func_id);
                    let sig = &module.signatures()[target.decl.signature];
                    if sig.params().len() != args.len()
                        || sig.returns().len() != results_of_inst.len()
                        || results_of_inst.len() > 1
                    {
                        return Err(unsupported("call signature or multiple results"));
                    }
                    abi_args(sig.params())?;
                    emit_call_args(&mut asm, args, sig.params())?;
                    asm.emit(&CALL, &[Patch::Symbol(&target.decl.name)])?;
                    if let [result] = results_of_inst {
                        match dfg.value_type(*result) {
                            ty @ (Type::F32 | Type::F64) => {
                                emit_xmm_slot(&mut asm, 0, *result, ty, true)?
                            }
                            _ => emit_one(&mut asm, &STORE_RAX, slot(*result)?)?,
                        }
                    }
                }
                InstView::CallIndirect { ptr, args, sig_id } => {
                    let sig = &module.signatures()[sig_id];
                    if sig.params().len() != args.len()
                        || sig.returns().len() != results_of_inst.len()
                        || results_of_inst.len() > 1
                    {
                        return Err(unsupported("indirect call signature"));
                    }
                    abi_args(sig.params())?;
                    emit_call_args(&mut asm, args, sig.params())?;
                    emit_one(&mut asm, &LOAD_RAX, slot(ptr)?)?;
                    emit_bytes(&mut asm, &[0xff, 0xd0])?;
                    if let [result] = results_of_inst {
                        match dfg.value_type(*result) {
                            ty @ (Type::F32 | Type::F64) => {
                                emit_xmm_slot(&mut asm, 0, *result, ty, true)?
                            }
                            _ => emit_one(&mut asm, &STORE_RAX, slot(*result)?)?,
                        }
                    }
                }
                InstView::FloatCompare { kind, args } => {
                    let [result] = results_of_inst else {
                        return Err(unsupported("float compare arity"));
                    };
                    let ty = dfg.value_type(args[0]);
                    emit_xmm_slot(&mut asm, 0, args[0], ty, false)?;
                    emit_xmm_slot(&mut asm, 1, args[1], ty, false)?;
                    emit_bytes(
                        &mut asm,
                        if ty == Type::F32 {
                            &[0x0f, 0x2e, 0xc1]
                        } else {
                            &[0x66, 0x0f, 0x2e, 0xc1]
                        },
                    )?;
                    use veloc_mir::FloatCC;
                    let condition: &'static [u8] = match kind {
                        FloatCC::Eq => &[0x0f, 0x94, 0xc0, 0x0f, 0x9b, 0xc2, 0x20, 0xd0],
                        FloatCC::Ne => &[0x0f, 0x95, 0xc0, 0x0f, 0x9a, 0xc2, 0x08, 0xd0],
                        FloatCC::Lt => &[0x0f, 0x92, 0xc0, 0x0f, 0x9b, 0xc2, 0x20, 0xd0],
                        FloatCC::Gt => &[0x0f, 0x97, 0xc0],
                        FloatCC::Le => &[0x0f, 0x96, 0xc0, 0x0f, 0x9b, 0xc2, 0x20, 0xd0],
                        FloatCC::Ge => &[0x0f, 0x93, 0xc0],
                    };
                    emit_bytes(&mut asm, condition)?;
                    asm.emit(&ZERO_EXTEND_AL, &[])?;
                    emit_one(&mut asm, &STORE_RAX, slot(*result)?)?;
                }
                InstView::Binary {
                    opcode: opcode @ (Opcode::FAdd | Opcode::FSub | Opcode::FMul | Opcode::FDiv),
                    args,
                } => {
                    let [result] = results_of_inst else {
                        return Err(unsupported("float binary arity"));
                    };
                    let ty = dfg.value_type(*result);
                    emit_xmm_slot(&mut asm, 0, args[0], ty, false)?;
                    emit_xmm_slot(&mut asm, 1, args[1], ty, false)?;
                    emit_float(&mut asm, opcode, ty)?;
                    emit_xmm_slot(&mut asm, 0, *result, ty, true)?;
                }
                InstView::Unary {
                    opcode: Opcode::IEqz,
                    arg,
                } => {
                    let [result] = results_of_inst else {
                        return Err(unsupported("eqz result arity"));
                    };
                    let wide = integer_width(dfg.value_type(arg))?;
                    emit_one(
                        &mut asm,
                        if wide { &LOAD_RAX } else { &LOAD_EAX },
                        slot(arg)?,
                    )?;
                    asm.emit(
                        if wide {
                            &Stencil {
                                bytes: &[0x48, 0x85, 0xc0],
                                holes: &[],
                            }
                        } else {
                            &TEST_EAX
                        },
                        &[],
                    )?;
                    emit_bytes(&mut asm, &[0x0f, 0x94, 0xc0])?;
                    asm.emit(&ZERO_EXTEND_AL, &[])?;
                    emit_one(&mut asm, &STORE_RAX, slot(*result)?)?;
                }
                InstView::Unary {
                    opcode: opcode @ (Opcode::FNeg | Opcode::FAbs),
                    arg,
                } => {
                    let [result] = results_of_inst else {
                        return Err(unsupported("float sign arity"));
                    };
                    let ty = dfg.value_type(*result);
                    let (wide, mask) = match (opcode, ty) {
                        (Opcode::FNeg, Type::F32) => (false, 0x8000_0000),
                        (Opcode::FNeg, Type::F64) => (true, 0x8000_0000_0000_0000),
                        (Opcode::FAbs, Type::F32) => (false, 0x7fff_ffff),
                        (Opcode::FAbs, Type::F64) => (true, 0x7fff_ffff_ffff_ffff),
                        _ => return Err(unsupported(format!("float sign type {ty}"))),
                    };
                    emit_one(
                        &mut asm,
                        if wide { &LOAD_RAX } else { &LOAD_EAX },
                        slot(arg)?,
                    )?;
                    asm.emit(&MOV_RCX_IMM, &[Patch::U64(mask)])?;
                    let bytes: &'static [u8] = match (opcode, wide) {
                        (Opcode::FNeg, false) => &[0x31, 0xc8],
                        (Opcode::FNeg, true) => &[0x48, 0x31, 0xc8],
                        (Opcode::FAbs, false) => &[0x21, 0xc8],
                        (Opcode::FAbs, true) => &[0x48, 0x21, 0xc8],
                        _ => unreachable!(),
                    };
                    emit_bytes(&mut asm, bytes)?;
                    emit_one(&mut asm, &STORE_RAX, slot(*result)?)?;
                }
                InstView::Unary {
                    opcode: opcode @ (Opcode::FloatToIntS | Opcode::FloatToIntU),
                    arg,
                } => {
                    let [result] = results_of_inst else {
                        return Err(unsupported("float-to-int arity"));
                    };
                    let src = dfg.value_type(arg);
                    let dst = dfg.value_type(*result);
                    emit_xmm_slot(&mut asm, 0, arg, src, false)?;
                    let prefix = if src == Type::F32 { 0xf3 } else { 0xf2 };
                    match (opcode, dst) {
                        (Opcode::FloatToIntS, Type::I32) => {
                            asm.emit_raw(&[prefix, 0x0f, 0x2c, 0xc0])
                        }
                        (Opcode::FloatToIntS, Type::I64) | (Opcode::FloatToIntU, Type::I32) => {
                            asm.emit_raw(&[prefix, 0x48, 0x0f, 0x2c, 0xc0])
                        }
                        // A 64-bit unsigned conversion needs a separate
                        // 2^63-range lowering; never use signed CVTT here.
                        _ => return Err(unsupported(format!("float-to-int {src} -> {dst}"))),
                    }
                    if opcode == Opcode::FloatToIntU && dst == Type::I32 {
                        // The signed 64-bit conversion covers the full u32
                        // domain. Truncate only after conversion.
                        emit_bytes(&mut asm, &[0x89, 0xc0])?;
                    }
                    emit_one(&mut asm, &STORE_RAX, slot(*result)?)?;
                }
                InstView::Unary {
                    opcode: opcode @ (Opcode::IntToFloatS | Opcode::IntToFloatU),
                    arg,
                } => {
                    let [result] = results_of_inst else {
                        return Err(unsupported("int-to-float arity"));
                    };
                    let src = dfg.value_type(arg);
                    let dst = dfg.value_type(*result);
                    let prefix = if dst == Type::F32 {
                        0xf3
                    } else if dst == Type::F64 {
                        0xf2
                    } else {
                        return Err(unsupported(format!("int-to-float result {dst}")));
                    };
                    match (opcode, src) {
                        (Opcode::IntToFloatS, Type::I32) => {
                            emit_one(&mut asm, &LOAD_EAX, slot(arg)?)?;
                            asm.emit_raw(&[prefix, 0x0f, 0x2a, 0xc0]);
                        }
                        (Opcode::IntToFloatS, Type::I64) => {
                            emit_one(&mut asm, &LOAD_RAX, slot(arg)?)?;
                            asm.emit_raw(&[prefix, 0x48, 0x0f, 0x2a, 0xc0]);
                        }
                        (Opcode::IntToFloatU, Type::I32) => {
                            emit_one(&mut asm, &LOAD_EAX, slot(arg)?)?;
                            asm.emit_raw(&[prefix, 0x48, 0x0f, 0x2a, 0xc0]);
                        }
                        _ => return Err(unsupported(format!("int-to-float {src} -> {dst}"))),
                    }
                    emit_xmm_slot(&mut asm, 0, *result, dst, true)?;
                }
                InstView::Unary {
                    opcode: opcode @ (Opcode::ICtz | Opcode::IClz),
                    arg,
                } => {
                    let [result] = results_of_inst else {
                        return Err(unsupported("bit count result arity"));
                    };
                    let wide = integer_width(dfg.value_type(arg))?;
                    emit_one(
                        &mut asm,
                        if wide { &LOAD_RAX } else { &LOAD_EAX },
                        slot(arg)?,
                    )?;
                    match (opcode, wide) {
                        (Opcode::ICtz, false) => emit_bytes(&mut asm, &[0x0f, 0xbc, 0xc0])?,
                        (Opcode::ICtz, true) => emit_bytes(&mut asm, &[0x48, 0x0f, 0xbc, 0xc0])?,
                        (Opcode::IClz, false) => emit_bytes(&mut asm, &[0x0f, 0xbd, 0xc0])?,
                        (Opcode::IClz, true) => emit_bytes(&mut asm, &[0x48, 0x0f, 0xbd, 0xc0])?,
                        _ => unreachable!(),
                    }
                    if opcode == Opcode::ICtz {
                        // BSF leaves its destination undefined for zero; CMOVZ
                        // supplies the exact width required by MIR.
                        emit_bytes(
                            &mut asm,
                            if wide {
                                &[0xb9, 64, 0, 0, 0]
                            } else {
                                &[0xb9, 32, 0, 0, 0]
                            },
                        )?;
                        emit_bytes(
                            &mut asm,
                            if wide {
                                &[0x48, 0x0f, 0x44, 0xc1]
                            } else {
                                &[0x0f, 0x44, 0xc1]
                            },
                        )?;
                    } else {
                        emit_bytes(&mut asm, &[0xb9, 0xff, 0xff, 0xff, 0xff])?;
                        emit_bytes(
                            &mut asm,
                            if wide {
                                &[0x48, 0x0f, 0x44, 0xc1]
                            } else {
                                &[0x0f, 0x44, 0xc1]
                            },
                        )?;
                        emit_bytes(
                            &mut asm,
                            if wide {
                                &[0xb9, 63, 0, 0, 0]
                            } else {
                                &[0xb9, 31, 0, 0, 0]
                            },
                        )?;
                        emit_bytes(&mut asm, &[0x29, 0xc1, 0x89, 0xc8])?;
                    }
                    emit_one(&mut asm, &STORE_RAX, slot(*result)?)?;
                }
                InstView::Unary {
                    opcode: Opcode::ExtendU | Opcode::Wrap | Opcode::Reinterpret,
                    arg,
                } => {
                    let [result] = results_of_inst else {
                        return Err(unsupported("conversion result arity"));
                    };
                    let width = memory_width(dfg.value_type(arg))?;
                    emit_one(
                        &mut asm,
                        if width == 8 { &LOAD_RAX } else { &LOAD_EAX },
                        slot(arg)?,
                    )?;
                    // Narrow SSA values occupy a full stack slot. A wrap or
                    // zero-extension must discard bits outside the source or
                    // destination type before another operation observes it.
                    let narrow = memory_width(dfg.value_type(*result))?.min(width);
                    match narrow {
                        1 => emit_bytes(&mut asm, &[0x0f, 0xb6, 0xc0])?,
                        2 => emit_bytes(&mut asm, &[0x0f, 0xb7, 0xc0])?,
                        _ => {}
                    }
                    emit_one(&mut asm, &STORE_RAX, slot(*result)?)?;
                }
                InstView::IntToPtr { arg } | InstView::PtrToInt { arg } => {
                    let [result] = results_of_inst else {
                        return Err(unsupported("pointer cast result arity"));
                    };
                    let wide = memory_width(dfg.value_type(arg))? == 8;
                    emit_one(
                        &mut asm,
                        if wide { &LOAD_RAX } else { &LOAD_EAX },
                        slot(arg)?,
                    )?;
                    emit_one(&mut asm, &STORE_RAX, slot(*result)?)?;
                }
                InstView::Unary {
                    opcode: Opcode::ExtendS,
                    arg,
                } => {
                    let [result] = results_of_inst else {
                        return Err(unsupported("extension result arity"));
                    };
                    let width = memory_width(dfg.value_type(arg))?;
                    emit_one(
                        &mut asm,
                        if width == 8 { &LOAD_RAX } else { &LOAD_EAX },
                        slot(arg)?,
                    )?;
                    match width {
                        1 => emit_bytes(&mut asm, &[0x48, 0x0f, 0xbe, 0xc0])?,
                        2 => emit_bytes(&mut asm, &[0x48, 0x0f, 0xbf, 0xc0])?,
                        4 => emit_bytes(&mut asm, &[0x48, 0x63, 0xc0])?,
                        8 => {}
                        _ => unreachable!(),
                    }
                    emit_one(&mut asm, &STORE_RAX, slot(*result)?)?;
                }
                InstView::Unreachable => emit_bytes(&mut asm, &[0x0f, 0x0b])?,
                InstView::Binary { opcode, args }
                    if matches!(
                        opcode,
                        Opcode::IAdd
                            | Opcode::ISub
                            | Opcode::IMul
                            | Opcode::IAnd
                            | Opcode::IOr
                            | Opcode::IXor
                    ) =>
                {
                    let [result] = results_of_inst else {
                        return Err(unsupported("binary result arity"));
                    };
                    let wide = integer_width(dfg.value_type(*result))?;
                    emit_one(
                        &mut asm,
                        if wide { &LOAD_RAX } else { &LOAD_EAX },
                        slot(args[0])?,
                    )?;
                    emit_one(
                        &mut asm,
                        if wide { &LOAD_RCX } else { &LOAD_ECX },
                        slot(args[1])?,
                    )?;
                    let bytes: &'static [u8] = match (opcode, wide) {
                        (Opcode::IAdd, false) => &[0x01, 0xc8],
                        (Opcode::IAdd, true) => &[0x48, 0x01, 0xc8],
                        (Opcode::ISub, false) => &[0x29, 0xc8],
                        (Opcode::ISub, true) => &[0x48, 0x29, 0xc8],
                        (Opcode::IMul, false) => &[0x0f, 0xaf, 0xc1],
                        (Opcode::IMul, true) => &[0x48, 0x0f, 0xaf, 0xc1],
                        (Opcode::IAnd, false) => &[0x21, 0xc8],
                        (Opcode::IAnd, true) => &[0x48, 0x21, 0xc8],
                        (Opcode::IOr, false) => &[0x09, 0xc8],
                        (Opcode::IOr, true) => &[0x48, 0x09, 0xc8],
                        (Opcode::IXor, false) => &[0x31, 0xc8],
                        (Opcode::IXor, true) => &[0x48, 0x31, 0xc8],
                        _ => unreachable!(),
                    };
                    asm.emit(&Stencil { bytes, holes: &[] }, &[])?;
                    emit_one(&mut asm, &STORE_RAX, slot(*result)?)?;
                }
                InstView::Binary { opcode, args }
                    if matches!(
                        opcode,
                        Opcode::IDivS | Opcode::IDivU | Opcode::IRemS | Opcode::IRemU
                    ) =>
                {
                    let [result] = results_of_inst else {
                        return Err(unsupported("division result arity"));
                    };
                    let wide = integer_width(dfg.value_type(*result))?;
                    let signed = matches!(opcode, Opcode::IDivS | Opcode::IRemS);
                    emit_one(
                        &mut asm,
                        if wide { &LOAD_RAX } else { &LOAD_EAX },
                        slot(args[0])?,
                    )?;
                    emit_one(
                        &mut asm,
                        if wide { &LOAD_RCX } else { &LOAD_ECX },
                        slot(args[1])?,
                    )?;
                    if signed {
                        emit_bytes(&mut asm, if wide { &[0x48, 0x99] } else { &[0x99] })?;
                    } else {
                        emit_bytes(&mut asm, &[0x31, 0xd2])?;
                    }
                    let op = match (signed, wide) {
                        (false, false) => &[0xf7, 0xf1][..],
                        (false, true) => &[0x48, 0xf7, 0xf1][..],
                        (true, false) => &[0xf7, 0xf9][..],
                        (true, true) => &[0x48, 0xf7, 0xf9][..],
                    };
                    emit_bytes(&mut asm, op)?;
                    if matches!(opcode, Opcode::IRemS | Opcode::IRemU) {
                        emit_bytes(
                            &mut asm,
                            if wide {
                                &[0x48, 0x89, 0xd0]
                            } else {
                                &[0x89, 0xd0]
                            },
                        )?;
                    }
                    emit_one(&mut asm, &STORE_RAX, slot(*result)?)?;
                }
                InstView::Binary { opcode, args }
                    if matches!(
                        opcode,
                        Opcode::IShl
                            | Opcode::IShrS
                            | Opcode::IShrU
                            | Opcode::IRotl
                            | Opcode::IRotr
                    ) =>
                {
                    let [result] = results_of_inst else {
                        return Err(unsupported("shift result arity"));
                    };
                    let wide = integer_width(dfg.value_type(*result))?;
                    emit_one(
                        &mut asm,
                        if wide { &LOAD_RAX } else { &LOAD_EAX },
                        slot(args[0])?,
                    )?;
                    emit_one(&mut asm, &LOAD_ECX, slot(args[1])?)?;
                    let operation = match opcode {
                        Opcode::IShl => 0xe0,
                        Opcode::IShrS => 0xf8,
                        Opcode::IShrU => 0xe8,
                        Opcode::IRotl => 0xc0,
                        Opcode::IRotr => 0xc8,
                        _ => unreachable!(),
                    };
                    if wide {
                        emit_bytes(&mut asm, &[0x48])?;
                    }
                    // The ModR/M byte is the only variant; the chosen bits encode
                    // one of the x86 shift/rotate operations with count in CL.
                    match operation {
                        0xe0 => emit_bytes(&mut asm, &[0xd3, 0xe0])?,
                        0xf8 => emit_bytes(&mut asm, &[0xd3, 0xf8])?,
                        0xe8 => emit_bytes(&mut asm, &[0xd3, 0xe8])?,
                        0xc0 => emit_bytes(&mut asm, &[0xd3, 0xc0])?,
                        0xc8 => emit_bytes(&mut asm, &[0xd3, 0xc8])?,
                        _ => unreachable!(),
                    }
                    emit_one(&mut asm, &STORE_RAX, slot(*result)?)?;
                }
                InstView::Return { values } if values.len() == results.len() => {
                    if let Some(&value) = values.first() {
                        match dfg.value_type(value) {
                            ty @ (Type::F32 | Type::F64) => {
                                emit_xmm_slot(&mut asm, 0, value, ty, false)?
                            }
                            ty => {
                                let wide = integer_width(ty)?;
                                emit_one(
                                    &mut asm,
                                    if wide { &LOAD_RAX } else { &LOAD_EAX },
                                    slot(value)?,
                                )?;
                            }
                        }
                    }
                    asm.emit(&EPILOGUE, &[])?;
                    returned = true;
                }
                InstView::Jump { dest } => {
                    emit_edge(&mut asm, dest, body, &labels, first_temp)?;
                }
                InstView::Br {
                    condition,
                    then_dest,
                    else_dest,
                } => {
                    emit_one(&mut asm, &LOAD_EAX, slot(condition)?)?;
                    asm.emit(&TEST_EAX, &[])?;
                    let else_label = asm.label();
                    asm.emit(&JUMP_ZERO, &[Patch::Label(else_label)])?;
                    emit_edge(&mut asm, then_dest, body, &labels, first_temp)?;
                    asm.bind(else_label)?;
                    emit_edge(&mut asm, else_dest, body, &labels, first_temp)?;
                }
                InstView::BrTable { index, table } => {
                    let (default, cases) = table
                        .split_last()
                        .ok_or_else(|| unsupported("empty branch table"))?;
                    emit_one(&mut asm, &LOAD_EAX, slot(index)?)?;
                    let case_labels: Vec<_> = cases.iter().map(|_| asm.label()).collect();
                    for (index, &label) in case_labels.iter().enumerate() {
                        emit_one(
                            &mut asm,
                            &COMPARE_EAX_IMM,
                            i32::try_from(index).map_err(|_| unsupported("branch table size"))?,
                        )?;
                        asm.emit(&JUMP_EQUAL, &[Patch::Label(label)])?;
                    }
                    emit_edge(&mut asm, default, body, &labels, first_temp)?;
                    for (label, case) in case_labels.into_iter().zip(cases.iter()) {
                        asm.bind(label)?;
                        emit_edge(&mut asm, case, body, &labels, first_temp)?;
                    }
                }
                InstView::Nop {} => {}
                other => return Err(unsupported(format!("instruction {:?}", other.opcode()))),
            }
        }
    }
    if !returned {
        return Err(unsupported("missing return"));
    }
    asm.finish()
}
