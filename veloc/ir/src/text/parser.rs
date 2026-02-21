//! IR 文本解析器
//!
//! 使用 chumsky 实现的高性能、高可读性解析器。
//! 这是 printer.rs 的逆操作。

use crate::{
    CallConv, Result,
    function::{Function, StackSlotData},
    inst::{ConstantPoolId, InstructionData, VectorExtData, VectorMemExtData},
    module::{Linkage, Module},
    opcode::{FloatCC, IntCC, MemFlags, Opcode},
    types::{
        Block, BlockCallData, FuncId, SigId, Signature, StackSlot, Type, Value, ValueData, ValueDef,
    },
};
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use chumsky::prelude::*;
use hashbrown::HashMap;

pub use super::format;
use super::format::*;

/// 解析错误
#[derive(Debug, Clone)]
pub struct ParseError(pub String);

impl core::fmt::Display for ParseError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// 模块解析步骤数据
pub struct ModuleParser {}

impl Default for ModuleParser {
    fn default() -> Self {
        Self::new()
    }
}

impl ModuleParser {
    pub fn new() -> Self {
        Self {}
    }

    pub fn parse(&mut self, input: &str) -> Result<Module> {
        let input = input.trim();
        if input.is_empty() {
            return Err(ParseError("Empty input".to_string()).into());
        }

        let mut func_parser = FuncParser::new();
        let func = func_parser.parse(input)?;
        let mut module_data = crate::module::ModuleData::default();
        let sig = Signature::new(vec![], vec![], CallConv::SystemV);
        let sig_id = module_data.intern_signature(sig);
        module_data.declare_function(func.name.clone(), sig_id, func.linkage);
        module_data.functions[FuncId(0)].dfg = func.dfg;
        module_data.functions[FuncId(0)].layout = func.layout;
        module_data.functions[FuncId(0)].stack_slots = func.stack_slots;
        module_data.functions[FuncId(0)].entry_block = func.entry_block;
        Ok(Module::new(module_data))
    }
}

/// 解析上下文：管理值和基本块的映射
pub struct ParseContext {
    pub value_map: HashMap<String, Value>,
    pub block_map: HashMap<String, Block>,
    pub next_value_idx: u32,
}

/// 函数解析器
pub struct FuncParser {
    pub ctx: ParseContext,
}

impl Default for FuncParser {
    fn default() -> Self {
        Self::new()
    }
}

fn identifier<'a>() -> impl Parser<'a, &'a str, String, extra::Err<Simple<'a, char>>> + Clone {
    any()
        .filter(|c: &char| c.is_ascii_alphabetic() || *c == '_')
        .then(
            any()
                .filter(|c: &char| c.is_ascii_alphanumeric() || matches!(*c, '_' | '.' | '-'))
                .repeated()
                .collect::<Vec<char>>(),
        )
        .map(|(c, mut rest)| {
            rest.insert(0, c);
            rest.into_iter().collect::<String>()
        })
}

fn type_parser<'a>() -> impl Parser<'a, &'a str, Type, extra::Err<Simple<'a, char>>> + Clone {
    let simple = choice((
        just("i8"),
        just("i16"),
        just("i32"),
        just("i64"),
        just("f32"),
        just("f64"),
        just("bool"),
        just("ptr"),
        just("void"),
    ));

    let vector = just('<')
        .ignore_then(
            any()
                .filter(|c: &char| *c != '>')
                .repeated()
                .collect::<String>(),
        )
        .then_ignore(just('>'))
        .map(|s| format!("<{}>", s));

    simple
        .map(|s| s.to_string())
        .or(vector)
        .try_map(|s, span| parse_type(&s).ok_or_else(|| Simple::new(None, span)))
}

fn value_name_parser<'a>() -> impl Parser<'a, &'a str, String, extra::Err<Simple<'a, char>>> + Clone
{
    identifier()
}

fn block_id_parser<'a>() -> impl Parser<'a, &'a str, u32, extra::Err<Simple<'a, char>>> + Clone {
    just("block").ignore_then(text::int(10).from_str().unwrapped())
}

fn block_call_parser<'a>()
-> impl Parser<'a, &'a str, (u32, Vec<String>), extra::Err<Simple<'a, char>>> + Clone {
    block_id_parser().then(
        value_name_parser()
            .padded()
            .separated_by(just(','))
            .allow_trailing()
            .collect::<Vec<String>>()
            .delimited_by(just('('), just(')')),
    )
}

fn stack_slot_id_parser<'a>() -> impl Parser<'a, &'a str, u32, extra::Err<Simple<'a, char>>> + Clone
{
    just("ss").ignore_then(text::int(10).from_str().unwrapped())
}

fn opcode_parts_parser<'a>()
-> impl Parser<'a, &'a str, (String, Option<Type>, MemFlags), extra::Err<Simple<'a, char>>> + Clone
{
    identifier()
        .then(
            just('.')
                .ignore_then(identifier())
                .repeated()
                .collect::<Vec<String>>()
                .map(|parts| {
                    let mut ty = None;
                    let mut flags = MemFlags::new();
                    for part in parts {
                        if let Some(t) = parse_type(&part) {
                            ty = Some(t);
                        } else if part == "trusted" {
                            flags = flags.union(MemFlags::TRUSTED);
                        } else if part.starts_with("align") {
                            if let Ok(align) = part["align".len()..].parse::<u32>() {
                                flags = flags.with_alignment(align);
                            }
                        }
                    }
                    (ty, flags)
                }),
        )
        .map(|(opcode, (ty, flags))| (opcode, ty, flags))
}

fn result_names_parser<'a>()
-> impl Parser<'a, &'a str, Vec<String>, extra::Err<Simple<'a, char>>> + Clone {
    let single = identifier().map(|s| vec![s]);
    let multiple = identifier()
        .padded()
        .separated_by(just(','))
        .at_least(1)
        .collect::<Vec<String>>()
        .delimited_by(just('('), just(')'));
    multiple.or(single).then_ignore(just('=').padded())
}

#[derive(Default)]
struct ParsedVMemExt {
    stride: Option<String>,
    index: Option<String>,
    scale: u32,
    mask: Option<String>,
    evl: Option<String>,
}

fn vector_ext_parser<'a>()
-> impl Parser<'a, &'a str, ParsedVMemExt, extra::Err<Simple<'a, char>>> + Clone {
    let stride = just("stride=")
        .ignore_then(value_name_parser())
        .map(|v| (format::STRIDE, (v, 1)));
    let index = just("index=")
        .ignore_then(value_name_parser())
        .then(
            just("*")
                .padded()
                .ignore_then(text::int(10).from_str::<u32>().unwrapped())
                .or_not()
                .map(|v| v.unwrap_or(1)),
        )
        .map(|(v, s)| (format::INDEX, (v, s)));
    let mask = just("mask=")
        .ignore_then(value_name_parser())
        .map(|v| (format::MASK, (v, 1)));
    let evl = just("evl=")
        .ignore_then(value_name_parser())
        .map(|v| (format::EVL, (v, 1)));

    choice((stride, index, mask, evl))
        .padded()
        .separated_by(just(','))
        .allow_trailing()
        .collect::<Vec<(&str, (String, u32))>>()
        .map(|fields| {
            let mut ext = ParsedVMemExt {
                scale: 1,
                ..Default::default()
            };
            for (key, (v, s)) in fields {
                match key {
                    format::STRIDE => ext.stride = Some(v),
                    format::INDEX => {
                        ext.index = Some(v);
                        ext.scale = s;
                    }
                    format::MASK => ext.mask = Some(v),
                    format::EVL => ext.evl = Some(v),
                    _ => unreachable!(),
                }
            }
            ext
        })
}

// ==================== FuncParser Implementation ====================

impl FuncParser {
    pub fn new() -> Self {
        Self {
            ctx: ParseContext {
                value_map: HashMap::new(),
                block_map: HashMap::new(),
                next_value_idx: 0,
            },
        }
    }

    pub fn parse(&mut self, input: &str) -> Result<Function> {
        let mut lines = input.lines();
        let sig_line = lines
            .next()
            .ok_or_else(|| ParseError("Empty input".to_string()))?;
        let (name, linkage, _params, _returns) = self.parse_signature(sig_line)?;

        let mut func = Function::new(name, SigId(0), linkage);
        let mut current_block: Option<Block> = None;

        for (line_idx, line) in lines.enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with("//") {
                continue;
            }

            if line.ends_with(':') && !line.contains('=') && !line.starts_with("ss") {
                let (block_idx, params_info) = self.parse_block_header(line)?;
                let block = Block(block_idx);
                current_block = Some(block);
                self.ctx
                    .block_map
                    .insert(format!("block{}", block_idx), block);

                if func.entry_block.is_none() {
                    func.entry_block = Some(block);
                }
                while func.layout.blocks.len() <= block_idx as usize {
                    func.layout.create_block();
                }
                if !func.layout.block_order.contains(&block) {
                    func.layout.block_order.push(block);
                }

                for (p_name, p_ty) in params_info {
                    let val = get_or_create_value(
                        &p_name,
                        &mut func,
                        &mut self.ctx,
                        p_ty,
                        ValueDef::Param(block),
                    );
                    func.layout.blocks[block].params.push(val);
                }
                continue;
            }

            if line.starts_with("ss") {
                self.parse_stack_slot(line, &mut func)?;
                continue;
            }

            if let Some(block) = current_block {
                self.parse_instruction(line, &mut func, block, line_idx + 2)?;
            } else {
                return Err(ParseError(format!(
                    "Instruction outside of block at line {}: {}",
                    line_idx + 2,
                    line
                ))
                .into());
            }
        }
        Ok(func)
    }

    fn parse_signature(&self, line: &str) -> Result<(String, Linkage, Vec<Type>, Vec<Type>)> {
        let parser = choice((just("local"), just("export"), just("import")))
            .map(|s| parse_linkage(s).unwrap())
            .padded()
            .then_ignore(just("function").padded())
            .then(identifier())
            .then_ignore(just('('))
            .then(
                type_parser()
                    .padded()
                    .separated_by(just(','))
                    .allow_trailing()
                    .collect::<Vec<Type>>(),
            )
            .then_ignore(just(')'))
            .then(
                just("->")
                    .padded()
                    .ignored()
                    .ignore_then(choice((
                        just("void").map(|_| Vec::new()),
                        type_parser()
                            .padded()
                            .separated_by(just(','))
                            .at_least(1)
                            .collect::<Vec<Type>>(),
                    )))
                    .or_not()
                    .map(|v| v.unwrap_or_default()),
            );

        parser
            .parse(line.trim())
            .into_result()
            .map(|(((linkage, name), params), returns)| (name, linkage, params, returns))
            .map_err(|errs| ParseError(format!("Invalid signature: {:?}", errs)).into())
    }

    fn parse_block_header(&self, line: &str) -> Result<(u32, Vec<(String, Type)>)> {
        let parser = block_id_parser()
            .then(
                just('(')
                    .ignore_then(
                        identifier()
                            .then_ignore(just(':').padded())
                            .then(type_parser())
                            .padded()
                            .separated_by(just(','))
                            .allow_trailing()
                            .collect::<Vec<(String, Type)>>(),
                    )
                    .then_ignore(just(')'))
                    .or_not()
                    .map(|v| v.unwrap_or_default()),
            )
            .then_ignore(just(':'));

        parser
            .parse(line.trim())
            .into_result()
            .map_err(|errs| ParseError(format!("Invalid block header: {:?}", errs)).into())
    }

    fn parse_stack_slot(&self, line: &str, func: &mut Function) -> Result<()> {
        let parser = stack_slot_id_parser()
            .then_ignore(just(':').padded())
            .then_ignore(just("size").padded())
            .then(text::int(10).from_str().unwrapped());

        let (idx, size) = parser
            .parse(line.trim())
            .into_result()
            .map_err(|errs| ParseError(format!("Invalid stack slot: {:?}", errs)))?;

        while func.stack_slots.len() <= idx as usize {
            func.stack_slots.push(StackSlotData { size: 0 });
        }
        func.stack_slots[StackSlot(idx)] = StackSlotData { size };
        Ok(())
    }

    fn parse_instruction(
        &mut self,
        line: &str,
        func: &mut Function,
        block: Block,
        _line_no: usize,
    ) -> Result<()> {
        let parser = result_names_parser()
            .or_not()
            .then(opcode_parts_parser())
            .then(any().repeated().collect::<String>());

        let ((result_names, (opcode_str, ty_hint, flags)), rest) = parser
            .parse(line.trim())
            .into_result()
            .map_err(|errs| ParseError(format!("Invalid instruction structure: {:?}", errs)))?;

        let result_names = result_names.unwrap_or_default();
        let opcode = parse_opcode(&opcode_str)
            .ok_or_else(|| ParseError(format!("Unknown opcode: {}", opcode_str)))?;

        let rtypes = {
            let mut inst_parser = InstDataParser {
                func,
                ctx: &mut self.ctx,
            };
            let idata = inst_parser.parse_operands(opcode, ty_hint, flags, &rest)?;
            if let Some(ty) = ty_hint {
                (idata, alloc::vec![ty])
            } else {
                let rtypes = inst_parser.infer_result_types(&idata);
                (idata, rtypes)
            }
        };
        let (idata, rtypes) = rtypes;

        let inst = func.dfg.instructions.push(idata);
        func.layout.append_inst(block, inst);

        if !result_names.is_empty() {
            let values: Vec<Value> = result_names
                .iter()
                .enumerate()
                .map(|(i, name)| {
                    let ty = rtypes.get(i).copied().unwrap_or(Type::VOID);
                    get_or_create_value(name, func, &mut self.ctx, ty, ValueDef::Inst(inst))
                })
                .collect();

            let list = func.dfg.make_value_list(&values);
            func.dfg.inst_results[inst] = list;
        }

        Ok(())
    }
}

struct InstDataParser<'a> {
    func: &'a mut Function,
    ctx: &'a mut ParseContext,
}

/// Helper to parse value ID from string representation (e.g., "v123" -> 123 or "tmp.v123" -> 123)
fn parse_value_idx(name: &str) -> Option<u32> {
    // If it's something like "v123", strip 'v'
    if let Some(s) = name.strip_prefix('v') {
        let digits: String = s.chars().take_while(|c| c.is_ascii_digit()).collect();
        if !digits.is_empty() {
            return digits.parse().ok();
        }
    }
    // Handle "name.v123" - search for ".v" and then take digits
    if let Some(idx) = name.rfind(".v") {
        let s = &name[idx + 2..];
        let digits: String = s.chars().take_while(|c| c.is_ascii_digit()).collect();
        if !digits.is_empty() {
            return digits.parse().ok();
        }
    }
    None
}

impl<'a> InstDataParser<'a> {
    fn parse_operands(
        &mut self,
        opcode: Opcode,
        _ty_hint: Option<Type>,
        flags: MemFlags,
        rest: &str,
    ) -> Result<InstructionData> {
        let rest = rest.trim();
        let inst_format = format::opcode_to_format(opcode);

        match inst_format {
            InstFormat::Iconst => {
                let val: i64 = text::int::<_, extra::Err<Simple<char>>>(10)
                    .from_str::<i64>()
                    .unwrapped()
                    .parse(rest)
                    .into_result()
                    .map_err(|_| ParseError(format!("Expected i64 constant, found: {}", rest)))?;
                Ok(InstructionData::Iconst { value: val as u64 })
            }
            InstFormat::Fconst => {
                let val: u64 = rest.parse().unwrap_or(0);
                Ok(InstructionData::Fconst { value: val })
            }
            InstFormat::Bconst => {
                let b = rest.starts_with("true");
                Ok(InstructionData::Bconst { value: b })
            }
            InstFormat::Unary => {
                let v = self.parse_v(rest)?;
                Ok(InstructionData::Unary { opcode, arg: v })
            }
            InstFormat::Binary => {
                let (v0, v1) = value_name_parser()
                    .padded()
                    .then_ignore(just(',').padded())
                    .then(value_name_parser().padded())
                    .parse(rest)
                    .into_result()
                    .map_err(|e| ParseError(format!("Invalid binary args: {:?}", e)))?;
                Ok(InstructionData::Binary {
                    opcode,
                    args: [self.get_v_by_name(&v0), self.get_v_by_name(&v1)],
                })
            }
            InstFormat::Ternary => {
                let ((v0, v1), v2) = value_name_parser()
                    .padded()
                    .then_ignore(just(',').padded())
                    .then(value_name_parser().padded())
                    .then_ignore(just(',').padded())
                    .then(value_name_parser().padded())
                    .parse(rest)
                    .into_result()
                    .map_err(|e| ParseError(format!("Invalid ternary args: {:?}", e)))?;
                Ok(InstructionData::Ternary {
                    opcode,
                    args: [
                        self.get_v_by_name(&v0),
                        self.get_v_by_name(&v1),
                        self.get_v_by_name(&v2),
                    ],
                })
            }
            InstFormat::IntCompare => {
                let (kind, (v0, v1)) = identifier()
                    .padded()
                    .then(
                        value_name_parser()
                            .padded()
                            .then_ignore(just(',').padded())
                            .then(value_name_parser().padded()),
                    )
                    .parse(rest)
                    .into_result()
                    .map_err(|e| ParseError(format!("Invalid int_compare: {:?}", e)))?;
                let kind = parse_intcc(&kind)?;
                Ok(InstructionData::IntCompare {
                    kind,
                    args: [self.get_v_by_name(&v0), self.get_v_by_name(&v1)],
                })
            }
            InstFormat::FloatCompare => {
                let (kind, (v0, v1)) = identifier()
                    .padded()
                    .then(
                        value_name_parser()
                            .padded()
                            .then_ignore(just(',').padded())
                            .then(value_name_parser().padded()),
                    )
                    .parse(rest)
                    .into_result()
                    .map_err(|e| ParseError(format!("Invalid float_compare: {:?}", e)))?;
                let kind = parse_floatcc(&kind)?;
                Ok(InstructionData::FloatCompare {
                    kind,
                    args: [self.get_v_by_name(&v0), self.get_v_by_name(&v1)],
                })
            }
            InstFormat::Load => {
                let (v, offset) = value_name_parser()
                    .then(
                        just('+')
                            .padded()
                            .ignore_then(text::int(10).from_str::<u32>().unwrapped())
                            .or_not()
                            .map(|v| v.unwrap_or(0)),
                    )
                    .parse(rest)
                    .into_result()
                    .map_err(|e| ParseError(format!("Invalid load args: {:?}", e)))?;
                Ok(InstructionData::Load {
                    ptr: self.get_v_by_name(&v),
                    offset,
                    flags,
                })
            }
            InstFormat::Store => {
                let (val_id, (ptr_id, offset)) = value_name_parser()
                    .padded()
                    .then_ignore(just(',').padded())
                    .then(
                        value_name_parser().then(
                            just('+')
                                .padded()
                                .ignore_then(text::int(10).from_str::<u32>().unwrapped())
                                .or_not()
                                .map(|v| v.unwrap_or(0)),
                        ),
                    )
                    .parse(rest)
                    .into_result()
                    .map_err(|e| ParseError(format!("Invalid store args: {:?}", e)))?;
                Ok(InstructionData::Store {
                    ptr: self.get_v_by_name(&ptr_id),
                    value: self.get_v_by_name(&val_id),
                    offset,
                    flags,
                })
            }
            InstFormat::StackLoad => {
                let (slot, offset) = stack_slot_id_parser()
                    .then(
                        just('+')
                            .padded()
                            .ignore_then(text::int(10).from_str::<u32>().unwrapped())
                            .or_not()
                            .map(|v| v.unwrap_or(0)),
                    )
                    .parse(rest)
                    .into_result()
                    .map_err(|e| ParseError(format!("Invalid stack_load args: {:?}", e)))?;
                Ok(InstructionData::StackLoad {
                    slot: StackSlot(slot),
                    offset,
                })
            }
            InstFormat::StackStore => {
                let (val_id, (slot, offset)) = value_name_parser()
                    .padded()
                    .then_ignore(just(',').padded())
                    .then(
                        stack_slot_id_parser().then(
                            just('+')
                                .padded()
                                .ignore_then(text::int(10).from_str::<u32>().unwrapped())
                                .or_not()
                                .map(|v| v.unwrap_or(0)),
                        ),
                    )
                    .parse(rest)
                    .into_result()
                    .map_err(|e| ParseError(format!("Invalid stack_store args: {:?}", e)))?;
                Ok(InstructionData::StackStore {
                    slot: StackSlot(slot),
                    value: self.get_v_by_name(&val_id),
                    offset,
                })
            }
            InstFormat::StackAddr => {
                let (slot, offset) = stack_slot_id_parser()
                    .then(
                        just('+')
                            .padded()
                            .ignore_then(text::int(10).from_str::<u32>().unwrapped())
                            .or_not()
                            .map(|v| v.unwrap_or(0)),
                    )
                    .parse(rest)
                    .into_result()
                    .map_err(|e| ParseError(format!("Invalid stack_addr args: {:?}", e)))?;
                Ok(InstructionData::StackAddr {
                    slot: StackSlot(slot),
                    offset,
                })
            }
            InstFormat::Jump => {
                let (block_idx, args) = block_call_parser()
                    .parse(rest)
                    .into_result()
                    .map_err(|e| ParseError(format!("Invalid jump args: {:?}", e)))?;
                let dest = self.create_block_call(block_idx, args)?;
                Ok(InstructionData::Jump { dest })
            }
            InstFormat::Br => {
                let ((cond_id, (then_block, then_args)), (else_block, else_args)) =
                    value_name_parser()
                        .padded()
                        .then_ignore(just(',').padded())
                        .then(block_call_parser().padded())
                        .then_ignore(just(',').padded())
                        .then(block_call_parser().padded())
                        .parse(rest)
                        .into_result()
                        .map_err(|e| ParseError(format!("Invalid br args: {:?}", e)))?;

                let then_dest = self.create_block_call(then_block, then_args)?;
                let else_dest = self.create_block_call(else_block, else_args)?;

                Ok(InstructionData::Br {
                    condition: self.get_v_by_name(&cond_id),
                    then_dest,
                    else_dest,
                })
            }
            InstFormat::Return => {
                let args = value_name_parser()
                    .padded()
                    .separated_by(just(','))
                    .collect::<Vec<String>>()
                    .parse(rest)
                    .into_result()
                    .map_err(|e| ParseError(format!("Invalid return args: {:?}", e)))?;
                Ok(InstructionData::Return {
                    values: self.make_value_list_from_names(&args),
                })
            }
            InstFormat::Call => {
                let (_func_name, args) = identifier()
                    .padded()
                    .then(
                        value_name_parser()
                            .padded()
                            .separated_by(just(','))
                            .allow_trailing()
                            .collect::<Vec<String>>(),
                    )
                    .parse(rest)
                    .into_result()
                    .map_err(|e| ParseError(format!("Invalid call args: {:?}", e)))?;
                // TODO: 真正的函数 ID 查找
                let func_id = FuncId(0);
                Ok(InstructionData::Call {
                    func_id,
                    args: self.make_value_list_from_names(&args),
                })
            }
            InstFormat::Shuffle => {
                let ((v0, v1), mask) = value_name_parser()
                    .padded()
                    .then_ignore(just(',').padded())
                    .then(value_name_parser().padded())
                    .then(
                        just(',')
                            .padded()
                            .ignore_then(just("mask="))
                            .ignore_then(text::int(10).from_str::<u32>().unwrapped())
                            .or_not()
                            .map(|v| v.unwrap_or(0)),
                    )
                    .parse(rest)
                    .into_result()
                    .map_err(|e| ParseError(format!("Invalid shuffle args: {:?}", e)))?;
                Ok(InstructionData::Shuffle {
                    args: [self.get_v_by_name(&v0), self.get_v_by_name(&v1)],
                    mask: ConstantPoolId(mask),
                })
            }
            InstFormat::VectorOpWithExt => {
                let (args, ext_info) = value_name_parser()
                    .padded()
                    .repeated()
                    .collect::<Vec<String>>()
                    .then(vector_ext_parser())
                    .parse(rest)
                    .into_result()
                    .map_err(|e| ParseError(format!("Invalid vector_op args: {:?}", e)))?;

                let mask = ext_info.mask.as_ref().map(|id| self.get_v_by_name(id));
                let evl = ext_info.evl.as_ref().map(|id| self.get_v_by_name(id));
                let mask_val = mask.ok_or_else(|| ParseError("Missing mask".to_string()))?;

                let ext = self.func.dfg.vector_ext_pool.push(VectorExtData {
                    mask: mask_val,
                    evl,
                });

                Ok(InstructionData::VectorOpWithExt {
                    opcode,
                    args: self.make_value_list_from_names(&args),
                    ext,
                })
            }
            InstFormat::VectorLoadStrided => {
                let (ptr_id, ext_info) = value_name_parser()
                    .padded()
                    .then(vector_ext_parser())
                    .parse(rest)
                    .into_result()
                    .map_err(|e| ParseError(format!("Invalid vload_strided: {:?}", e)))?;

                let ext = self.build_vector_mem_ext(&ext_info, flags, 1);
                Ok(InstructionData::VectorLoadStrided {
                    ptr: self.get_v_by_name(&ptr_id),
                    stride: self.get_v_by_name(
                        ext_info
                            .stride
                            .as_ref()
                            .ok_or_else(|| ParseError("Missing stride".to_string()))?,
                    ),
                    ext,
                })
            }
            InstFormat::VectorStoreStrided => {
                let ((val_id, ptr_id), ext_info) = value_name_parser()
                    .padded()
                    .then_ignore(just(',').padded())
                    .then(value_name_parser().padded())
                    .then(vector_ext_parser())
                    .parse(rest)
                    .into_result()
                    .map_err(|e| ParseError(format!("Invalid vstore_strided: {:?}", e)))?;

                let ext = self.build_vector_mem_ext(&ext_info, flags, 1);
                let ptr = self.get_v_by_name(&ptr_id);
                let stride = self.get_v_by_name(
                    ext_info
                        .stride
                        .as_ref()
                        .ok_or_else(|| ParseError("Missing stride".to_string()))?,
                );
                let val = self.get_v_by_name(&val_id);
                Ok(InstructionData::VectorStoreStrided {
                    args: self.func.dfg.make_value_list(&[ptr, stride, val]),
                    ext,
                })
            }
            InstFormat::VectorGather => {
                let (ptr_id, ext_info) = value_name_parser()
                    .padded()
                    .then(vector_ext_parser())
                    .parse(rest)
                    .into_result()
                    .map_err(|e| ParseError(format!("Invalid vgather: {:?}", e)))?;

                let ext = self.build_vector_mem_ext(&ext_info, flags, ext_info.scale as u8);
                Ok(InstructionData::VectorGather {
                    ptr: self.get_v_by_name(&ptr_id),
                    index: self.get_v_by_name(
                        ext_info
                            .index
                            .as_ref()
                            .ok_or_else(|| ParseError("Missing index".to_string()))?,
                    ),
                    ext,
                })
            }
            InstFormat::VectorScatter => {
                let ((val_id, ptr_id), ext_info) = value_name_parser()
                    .padded()
                    .then_ignore(just(',').padded())
                    .then(value_name_parser().padded())
                    .then(vector_ext_parser())
                    .parse(rest)
                    .into_result()
                    .map_err(|e| ParseError(format!("Invalid vscatter: {:?}", e)))?;

                let ext = self.build_vector_mem_ext(&ext_info, flags, ext_info.scale as u8);
                let ptr = self.get_v_by_name(&ptr_id);
                let index = self.get_v_by_name(
                    ext_info
                        .index
                        .as_ref()
                        .ok_or_else(|| ParseError("Missing index".to_string()))?,
                );
                let val = self.get_v_by_name(&val_id);
                Ok(InstructionData::VectorScatter {
                    args: self.func.dfg.make_value_list(&[ptr, index, val]),
                    ext,
                })
            }
            _ => Err(ParseError(format!("Unsupported format: {:?}", inst_format)).into()),
        }
    }

    /// Parse a value name and get/create the corresponding Value.
    /// Uses provided type and definition for newly created values.
    fn parse_v(&mut self, s: &str) -> Result<Value> {
        let name = identifier()
            .parse(s.trim())
            .into_result()
            .map_err(|e| ParseError(format!("Invalid value ID: {:?}", e)))?;
        Ok(get_or_create_value(
            &name,
            self.func,
            self.ctx,
            Type::VOID,
            // Placeholder: actual definition will be set when the value is defined by an instruction
            ValueDef::Param(Block(0)),
        ))
    }

    /// Get/create a Value by its name (e.g., "v123", "tmp.v123").
    fn get_v_by_name(&mut self, name: &str) -> Value {
        get_or_create_value(
            name,
            self.func,
            self.ctx,
            Type::VOID,
            // Placeholder: actual definition will be set when the value is defined by an instruction
            ValueDef::Param(Block(0)),
        )
    }

    /// Create a ValueList from a slice of value names.
    fn make_value_list_from_names(&mut self, names: &[String]) -> crate::types::ValueList {
        let values: Vec<Value> = names.iter().map(|n| self.get_v_by_name(n)).collect();
        self.func.dfg.make_value_list(&values)
    }

    /// Create a BlockCall from block ID and argument names.
    fn create_block_call(
        &mut self,
        block_id: u32,
        arg_names: Vec<String>,
    ) -> Result<crate::types::BlockCall> {
        let block = self.get_block_by_id(block_id)?;
        let args = self.make_value_list_from_names(&arg_names);
        Ok(self
            .func
            .dfg
            .block_calls
            .push(BlockCallData { block, args }))
    }

    /// Build VectorMemExtData and push to pool.
    fn build_vector_mem_ext(
        &mut self,
        ext_info: &ParsedVMemExt,
        flags: MemFlags,
        scale: u8,
    ) -> crate::inst::VectorMemExtId {
        let mask = ext_info.mask.as_ref().map(|id| self.get_v_by_name(id));
        let evl = ext_info.evl.as_ref().map(|id| self.get_v_by_name(id));
        self.func.dfg.vector_mem_ext_pool.push(VectorMemExtData {
            offset: 0,
            flags,
            scale,
            mask,
            evl,
        })
    }

    fn get_block_by_id(&self, id: u32) -> Result<Block> {
        let name = format!("block{}", id);
        self.ctx
            .block_map
            .get(&name)
            .copied()
            .ok_or_else(|| ParseError(format!("Unknown block: {}", name)).into())
    }

    fn infer_result_types(&self, data: &InstructionData) -> Vec<Type> {
        use crate::inst::InstructionData as ID;
        match data {
            ID::Unary { arg, .. } => alloc::vec![self.func.dfg.value_type(*arg)],
            ID::Binary { opcode, args, .. } => match opcode {
                Opcode::IAddWithOverflow | Opcode::ISubWithOverflow | Opcode::IMulWithOverflow => {
                    alloc::vec![self.func.dfg.value_type(args[0]), Type::BOOL]
                }
                _ => alloc::vec![self.func.dfg.value_type(args[0])],
            },
            ID::IntCompare { .. } | ID::FloatCompare { .. } => alloc::vec![Type::BOOL],
            _ => alloc::vec![Type::VOID],
        }
    }
}

/// Get an existing Value by name, or create a new one with the given type and definition.
fn get_or_create_value(
    name: &str,
    func: &mut Function,
    ctx: &mut ParseContext,
    ty: Type,
    def: ValueDef,
) -> Value {
    if let Some(&v) = ctx.value_map.get(name) {
        return v;
    }

    // Parse value index from name (e.g., "v123" -> 123), or use next available index
    let idx = parse_value_idx(name).unwrap_or(ctx.next_value_idx);
    let v = Value(idx);

    ctx.value_map.insert(name.to_string(), v);
    if idx >= ctx.next_value_idx {
        ctx.next_value_idx = idx + 1;
    }

    // Ensure values vector has enough capacity
    while func.dfg.values.len() <= idx as usize {
        func.dfg.values.push(ValueData {
            ty: Type::VOID,
            // Placeholder: will be overwritten below for the actual value
            def: ValueDef::Param(Block(0)),
        });
    }
    func.dfg.values[v] = ValueData { ty, def };
    v
}

fn parse_intcc(s: &str) -> Result<IntCC> {
    format::parse_intcc(s).ok_or_else(|| ParseError(format!("Invalid intcc: {}", s)).into())
}

fn parse_floatcc(s: &str) -> Result<FloatCC> {
    format::parse_floatcc(s).ok_or_else(|| ParseError(format!("Invalid floatcc: {}", s)).into())
}
