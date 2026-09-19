//! Assembly is a typed projection of an instruction, not its debug spelling.
use super::{FinalInstDef, generate::find_operand_info};
use crate::syntax::{Decl, DeclKind, Kind, Node};
use crate::target::OperandConstraint;
use std::collections::{BTreeMap, HashMap};
use std::fmt::Write;

#[derive(Debug, Clone)]
pub(super) struct Assembly(Vec<Line>);

#[derive(Debug, Clone)]
struct Line {
    mnemonic: String,
    operands: Vec<Operand>,
}

#[derive(Debug, Clone)]
enum Operand {
    Reg {
        result: bool,
        index: usize,
        bits: u32,
    },
    Imm(usize),
    Block(usize),
    Symbol(usize),
    Memory {
        result: bool,
        index: usize,
        offset: usize,
        address_index: Option<usize>,
        bits: u32,
    },
    Stack {
        index: usize,
        bits: u32,
    },
}

fn fields(node: &Node) -> Result<&BTreeMap<String, Node>, String> {
    match &node.kind {
        Kind::Record(fields) => Ok(fields),
        _ => Err("expected an assembly record".into()),
    }
}

fn width(node: &Node) -> Result<u32, String> {
    match node.kind {
        Kind::Number(bits) if bits > 0 && bits % 8 == 0 => Ok(bits),
        _ => Err("assembly width must be a positive multiple of 8".into()),
    }
}

fn operand(node: &Node, inst: &FinalInstDef) -> Result<Operand, String> {
    let Kind::Call(kind, args) = &node.kind else {
        return Err("expected an assembly operand constructor".into());
    };
    let field = |node: &Node| {
        let Kind::Name(name) = &node.kind else {
            return Err("assembly operand requires a field name".to_owned());
        };
        find_operand_info(name, &inst.operands)
            .ok_or_else(|| format!("unknown assembly field `{name}`"))
    };
    match (kind.as_str(), args.as_slice()) {
        ("reg", [name, bits]) => {
            let (index, role) = field(name)?;
            let result = match role {
                OperandConstraint::Def(_) => true,
                OperandConstraint::Use(_) | OperandConstraint::FixedUse { .. } => false,
                _ => return Err("reg requires a register operand".into()),
            };
            Ok(Operand::Reg {
                result,
                index,
                bits: width(bits)?,
            })
        }
        ("imm", [name]) => match field(name)? {
            (i, OperandConstraint::Imm(_)) => Ok(Operand::Imm(i)),
            _ => Err("imm requires an immediate".into()),
        },
        ("target", [name]) => match field(name)? {
            (i, OperandConstraint::Block(_)) => Ok(Operand::Block(i)),
            (i, OperandConstraint::Global(_)) => Ok(Operand::Symbol(i)),
            _ => Err("target requires a block or symbol".into()),
        },
        ("mem", [base, offset, bits]) | ("mem", [base, _, offset, bits]) => {
            let address_index = if args.len() == 4 {
                let (index, OperandConstraint::Use(_)) = field(&args[1])? else {
                    return Err("memory index must be an input register".into());
                };
                Some(index)
            } else {
                None
            };
            let (index, role) = field(base)?;
            let result = match role {
                OperandConstraint::Def(_) => true,
                OperandConstraint::Use(_) | OperandConstraint::FixedUse { .. } => false,
                _ => return Err("memory base must be a register".into()),
            };
            let (offset, OperandConstraint::Imm(_)) = field(offset)? else {
                return Err("memory offset must be an immediate".into());
            };
            Ok(Operand::Memory {
                address_index,
                result,
                index,
                offset,
                bits: width(bits)?,
            })
        }
        ("stack", [name, bits]) => match field(name)? {
            (index, OperandConstraint::StackSlot(_)) => Ok(Operand::Stack {
                index,
                bits: width(bits)?,
            }),
            _ => Err("stack requires a stack slot".into()),
        },
        _ => Err(format!("invalid assembly operand `{kind}`")),
    }
}

pub(super) fn compile(
    declarations: &[Decl],
    instructions: &mut HashMap<String, FinalInstDef>,
) -> Result<(), String> {
    for declaration in declarations {
        if !matches!(&declaration.kind, DeclKind::Fields(kind) if kind == "assembly") {
            continue;
        }
        let inst = instructions
            .get_mut(&declaration.name)
            .ok_or_else(|| format!("assembly `{}` has no instruction", declaration.name))?;
        if inst.assembly.is_some() {
            return Err(format!("duplicate assembly `{}`", declaration.name));
        }
        let parse = || -> Result<Assembly, String> {
            if declaration.fields.len() != 1 {
                return Err("assembly requires only a lines property".into());
            }
            let Some(Node {
                kind: Kind::List(lines),
                ..
            }) = declaration.fields.get("lines")
            else {
                return Err("assembly requires a lines list".into());
            };
            if lines.is_empty() {
                return Err("assembly must contain at least one line".into());
            }
            let lines = lines
                .iter()
                .map(|line| {
                    let fields = fields(line)?;
                    if fields.len() != 2 {
                        return Err("assembly line requires mnemonic and operands".into());
                    }
                    let Some(Node {
                        kind: Kind::Text(mnemonic),
                        ..
                    }) = fields.get("mnemonic")
                    else {
                        return Err("assembly mnemonic requires a string".into());
                    };
                    if mnemonic.is_empty()
                        || !mnemonic
                            .bytes()
                            .all(|b| b.is_ascii_alphanumeric() || b == b'.' || b == b'_')
                    {
                        return Err("invalid assembly mnemonic".into());
                    }
                    let Some(Node {
                        kind: Kind::List(operands),
                        ..
                    }) = fields.get("operands")
                    else {
                        return Err("assembly operands require a list".into());
                    };
                    Ok(Line {
                        mnemonic: mnemonic.clone(),
                        operands: operands
                            .iter()
                            .map(|node| operand(node, inst))
                            .collect::<Result<_, _>>()?,
                    })
                })
                .collect::<Result<_, String>>()?;
            Ok(Assembly(lines))
        };
        inst.assembly = Some(parse().map_err(|error| format!("{}: {error}", declaration.name))?);
    }
    for (name, inst) in instructions {
        if !inst.is_pseudo && inst.assembly.is_none() {
            return Err(format!("instruction `{name}` has no assembly declaration"));
        }
    }
    Ok(())
}

fn register(result: bool, index: usize) -> String {
    format!(
        "inst.{}()[{index}]",
        if result { "results" } else { "inputs" }
    )
}

fn field(index: usize, variant: &str) -> String {
    format!(
        "match inst.fields().read({index}) {{ veloc_lir::FieldValueRef::{variant}(value) => *value, _ => return Err(core::fmt::Error) }}"
    )
}

fn emit(out: &mut String, op: &Operand) {
    match op {
        Operand::Reg {
            result,
            index,
            bits,
        } => writeln!(out, "out.register({}, {bits})?;", register(*result, *index)).unwrap(),
        Operand::Imm(index) => writeln!(out, "out.immediate({})?;", field(*index, "Imm")).unwrap(),
        Operand::Block(index) => writeln!(
            out,
            "out.block(inst.edge({}).block)?;",
            field(*index, "Edge")
        )
        .unwrap(),
        Operand::Symbol(index) => {
            writeln!(out, "out.symbol({})?;", field(*index, "Global")).unwrap()
        }
        Operand::Stack { index, bits } => writeln!(
            out,
            "out.stack_slot({}, {bits})?;",
            field(*index, "StackSlot")
        )
        .unwrap(),
        Operand::Memory {
            address_index,
            result,
            index,
            offset,
            bits,
        } => {
            writeln!(
                out,
                "out.memory({}, {}, {}, {bits})?;",
                register(*result, *index),
                address_index
                    .map(|index| format!("Some({})", register(false, index)))
                    .unwrap_or_else(|| "None".into()),
                field(*offset, "Imm")
            )
            .unwrap();
        }
    }
}

pub(super) fn generate(out: &mut String, instructions: &HashMap<String, FinalInstDef>) {
    out.push_str("impl TargetInst { pub fn write_assembly(&self, inst: &veloc_lir::InstRef<'_>, out: &mut dyn crate::target::AssemblyWriter) -> core::fmt::Result { match self {\n");
    let mut instructions: Vec<_> = instructions.iter().collect();
    instructions.sort_by_key(|(name, _)| *name);
    for (name, inst) in instructions {
        writeln!(out, "Self::{name} => {{").unwrap();
        if let Some(Assembly(lines)) = &inst.assembly {
            for (index, line) in lines.iter().enumerate() {
                if index != 0 {
                    out.push_str("out.write_str(\"\\n\")?;\n");
                }
                writeln!(out, "out.write_str({:?})?;", line.mnemonic).unwrap();
                for (index, operand) in line.operands.iter().enumerate() {
                    writeln!(
                        out,
                        "out.write_str({:?})?;",
                        if index == 0 { " " } else { ", " }
                    )
                    .unwrap();
                    emit(out, operand);
                }
            }
            out.push_str("Ok(())\n");
        } else {
            out.push_str("Err(core::fmt::Error)\n");
        }
        out.push_str("},\n");
    }
    out.push_str("} } }\n");
}
