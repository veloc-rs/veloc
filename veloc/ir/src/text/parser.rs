//! Parser for the canonical textual IR.
//!
//! Instruction dispatch is selected from [`TextCodec`], which is an exhaustive
//! projection of [`crate::OpFormat`]. A small delimiter-aware scanner handles
//! vector types, variadic calls, and nested block calls without duplicating one
//! parser-combinator tree per opcode.

use super::{TextCodec, format};
use crate::{
    Block, BlockCall, CallConv, FuncId, Function, InstructionData, Intrinsic, Linkage, MemFlags,
    Module, ModuleData, Opcode, Result, SigId, Signature, StackSlot, Type, Value, ValueDef,
    function::StackSlotData,
    inst::{ConstantPoolData, Inst, VectorMemOptions},
    opspec::{ResultTypes, TypeList, TypeSchemeError},
    types::{BlockCallData, JumpTableData, ValueData},
};
use alloc::{
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use hashbrown::HashMap;

#[derive(Debug, Clone)]
pub struct ParseError(pub String);

impl core::fmt::Display for ParseError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Debug, Clone)]
struct FunctionHeader {
    name: String,
    linkage: Linkage,
    signature: Signature,
}

#[derive(Debug)]
struct FunctionSource {
    header: FunctionHeader,
    body: Vec<(usize, String)>,
}

/// Complete-module parser. Function symbols are declared before any body is
/// decoded, so direct calls may refer to functions appearing later in a file.
pub struct ModuleParser;

impl Default for ModuleParser {
    fn default() -> Self {
        Self::new()
    }
}

impl ModuleParser {
    pub const fn new() -> Self {
        Self
    }

    pub fn parse(&mut self, input: &str) -> Result<Module> {
        if input.trim().is_empty() {
            return parse_err("empty input");
        }

        let mut sources = Vec::<FunctionSource>::new();
        let mut globals = Vec::<(String, Type, Linkage)>::new();
        let mut current = None;

        for (index, raw) in input.lines().enumerate() {
            let line_no = index + 1;
            let line = strip_comment(raw).trim();
            if line.is_empty() {
                continue;
            }
            if is_function_header(line) {
                let header =
                    parse_function_header(line).map_err(|error| with_line(error, line_no, line))?;
                sources.push(FunctionSource {
                    header,
                    body: Vec::new(),
                });
                current = Some(sources.len() - 1);
            } else if line.starts_with("global ") {
                if current.is_some() {
                    return parse_err(format!(
                        "line {line_no}: global declaration inside function"
                    ));
                }
                globals.push(parse_global(line).map_err(|error| with_line(error, line_no, line))?);
            } else if let Some(source) = current.map(|index| &mut sources[index]) {
                source.body.push((line_no, line.to_string()));
            } else {
                return parse_err(format!(
                    "line {line_no}: expected global or function declaration"
                ));
            }
        }

        if sources.is_empty() && globals.is_empty() {
            return parse_err("module contains no declarations");
        }

        let mut module = ModuleData::default();
        for (name, ty, linkage) in globals {
            module.add_global(name, ty, linkage);
        }

        let mut func_ids = HashMap::new();
        for source in &sources {
            if func_ids.contains_key(&source.header.name) {
                return parse_err(format!("duplicate function `{}`", source.header.name));
            }
            let sig = module.intern_signature(source.header.signature.clone());
            let id =
                module.declare_function(source.header.name.clone(), sig, source.header.linkage);
            func_ids.insert(source.header.name.clone(), id);
        }

        for source in sources {
            let id = func_ids[&source.header.name];
            let sig = module.functions[id].signature;
            let func =
                parse_function_body(source.header, sig, &source.body, &func_ids, &mut module)?;
            module.functions[id] = func;
        }
        Ok(Module::new(module))
    }
}

#[derive(Default)]
struct ParseContext {
    value_map: HashMap<String, Value>,
    block_map: HashMap<String, Block>,
    next_value_idx: u32,
    defined_values: HashMap<Value, String>,
    deferred_types: Vec<(Inst, usize, String)>,
}

fn parse_function_body(
    header: FunctionHeader,
    sig_id: SigId,
    body: &[(usize, String)],
    func_ids: &HashMap<String, FuncId>,
    module: &mut ModuleData,
) -> Result<Function> {
    let mut func = Function::new(header.name, sig_id, header.linkage);
    let mut ctx = ParseContext::default();

    // Predeclare every block and parameter. Forward branches are therefore a
    // normal case instead of depending on textual block order.
    for (line_no, line) in body {
        if is_block_header(line) {
            let (block_id, params) =
                parse_block_header(line).map_err(|error| with_line(error, *line_no, line))?;
            while func.layout.blocks.len() <= block_id as usize {
                func.layout.create_block();
            }
            let block = Block(block_id);
            let key = format!("block{block_id}");
            if ctx.block_map.insert(key, block).is_some() {
                return parse_err(format!("line {line_no}: duplicate block{block_id}"));
            }
            func.layout.append_block(block);
            if func.entry_block.is_none() {
                func.entry_block = Some(block);
            }
            for (name, ty) in params {
                let value = define_value(&name, &mut func, &mut ctx, ty, ValueDef::Param(block))
                    .map_err(|error| with_line(error, *line_no, line))?;
                func.layout.blocks[block].params.push(value);
            }
        } else if line.starts_with("ss") {
            parse_stack_slot(line, &mut func).map_err(|error| with_line(error, *line_no, line))?;
        }
    }

    let mut current = None;
    for (line_no, line) in body {
        if is_block_header(line) {
            let (block_id, _) =
                parse_block_header(line).map_err(|error| with_line(error, *line_no, line))?;
            current = Some(Block(block_id));
            continue;
        }
        if line.starts_with("ss") {
            continue;
        }
        let block = current.ok_or_else(|| {
            ParseError(format!("line {line_no}: instruction outside a basic block"))
        })?;
        parse_instruction(line, *line_no, block, &mut func, &mut ctx, func_ids, module)?;
    }

    for (name, value) in &ctx.value_map {
        if !ctx.defined_values.contains_key(value) {
            return parse_err(format!("undefined SSA value `{name}`"));
        }
    }
    for (inst, line_no, line) in &ctx.deferred_types {
        let data = func.dfg.inst(*inst);
        let mut operands = Vec::new();
        data.visit_type_operands(&func.dfg, |value| operands.push(func.dfg.value_type(value)));
        let results = func
            .dfg
            .inst_results(*inst)
            .iter()
            .map(|&value| func.dfg.value_type(value))
            .collect::<Vec<_>>();
        data.opcode()
            .spec()
            .type_scheme
            .validate(&operands, &results)
            .map_err(|error| {
                with_line(
                    ParseError(format!(
                        "invalid operand types for `{}`: {error:?}",
                        data.opcode().spec().mnemonic
                    )),
                    *line_no,
                    line,
                )
            })?;
    }

    for &block in &func.layout.block_order {
        func.layout.blocks[block].is_sealed = true;
    }
    Ok(func)
}

#[allow(clippy::too_many_arguments)]
fn parse_instruction(
    line: &str,
    line_no: usize,
    block: Block,
    func: &mut Function,
    ctx: &mut ParseContext,
    func_ids: &HashMap<String, FuncId>,
    module: &mut ModuleData,
) -> Result<()> {
    let (result_names, instruction) =
        parse_result_names(line).map_err(|error| with_line(error, line_no, line))?;
    let (opcode, ty_hint, flags, operands) =
        parse_instruction_header(instruction).map_err(|error| with_line(error, line_no, line))?;

    let data = {
        let mut parser = OperandParser {
            func,
            ctx,
            func_ids,
            module,
        };
        parser
            .parse(opcode, ty_hint, flags, operands)
            .map_err(|error| with_line(error, line_no, line))?
    };
    let (result_types, deferred) = resolve_result_types(&data, ty_hint, func, module)
        .map_err(|error| with_line(error, line_no, line))?;
    if result_names.len() != result_types.len() {
        return parse_err(format!(
            "line {line_no}: `{}` defines {} result name(s), but its type scheme produces {}",
            opcode.spec().mnemonic,
            result_names.len(),
            result_types.len()
        ));
    }

    let successors = instruction_successors(&data, func);
    let inst = func.dfg.instructions.push(data);
    func.layout.append_inst(block, inst);
    if deferred {
        ctx.deferred_types.push((inst, line_no, line.to_string()));
    }
    if !result_names.is_empty() {
        let values = result_names
            .iter()
            .zip(result_types)
            .map(|(name, ty)| define_value(name, func, ctx, ty, ValueDef::Inst(inst)))
            .collect::<core::result::Result<Vec<_>, _>>()
            .map_err(|error| with_line(error, line_no, line))?;
        let list = func.dfg.make_value_list(&values);
        func.dfg.inst_results[inst] = list;
    }
    for successor in successors {
        func.layout.add_edge(block, successor);
    }
    Ok(())
}

fn resolve_result_types(
    data: &InstructionData,
    hint: Option<Type>,
    func: &Function,
    module: &ModuleData,
) -> core::result::Result<(Vec<Type>, bool), ParseError> {
    let spec = data.opcode().spec();
    let mut operands = Vec::new();
    data.visit_type_operands(&func.dfg, |value| {
        operands.push(func.dfg.value_type(value));
    });
    let (inferred, deferred) = match spec.type_scheme.infer_results(&operands) {
        Ok(inferred) => (inferred, false),
        Err(error) => {
            // A later textual block may define an SSA value that dominates this
            // instruction. Defer only unresolved variadic prefixes whose result
            // types are independent of that value; check them after all definitions.
            let unresolved_prefix = matches!(spec.type_scheme.operands, TypeList::Variadic(_))
                && matches!(error, TypeSchemeError::Pattern { results: false, got, .. }
                    if got == Type::INVALID);
            let independent = match (unresolved_prefix, spec.type_scheme.results) {
                (true, TypeList::Signature) => Some(ResultTypes::Signature),
                (true, TypeList::Fixed([])) => Some(ResultTypes::Inferred(Default::default())),
                _ => None,
            };
            match independent {
                Some(inferred) => (inferred, true),
                None => {
                    return Err(ParseError(format!(
                        "invalid operand types for `{}`: {error:?}",
                        spec.mnemonic
                    )));
                }
            }
        }
    };
    let mut results = match &inferred {
        ResultTypes::Inferred(types) => types.clone().into_vec(),
        ResultTypes::Explicit => hint.into_iter().collect(),
        ResultTypes::Signature => instruction_signature(data, module)?.returns.clone(),
    };

    if let Some(hint) = hint {
        if let Some(first) = results.first_mut() {
            if first.is_valid() && *first != hint {
                return Err(ParseError(format!(
                    "result annotation `{hint}` conflicts with inferred type `{first}`"
                )));
            }
            *first = hint;
        } else if !matches!(inferred, ResultTypes::Explicit) {
            return Err(ParseError(format!(
                "`{}` does not produce an annotatable result",
                spec.mnemonic
            )));
        }
    }
    Ok((results, deferred))
}

fn instruction_signature<'a>(
    data: &InstructionData,
    module: &'a ModuleData,
) -> core::result::Result<&'a Signature, ParseError> {
    let sig = match data {
        InstructionData::Call { func_id, .. } => module
            .functions
            .get(*func_id)
            .map(|func| func.signature)
            .ok_or_else(|| ParseError(format!("unknown function id {}", func_id.0)))?,
        InstructionData::CallIndirect { sig_id, .. }
        | InstructionData::CallIntrinsic { sig_id, .. } => *sig_id,
        _ => {
            return Err(ParseError(
                "signature result on non-call instruction".into(),
            ));
        }
    };
    module
        .signatures
        .get(sig)
        .ok_or_else(|| ParseError(format!("unknown signature id {}", sig.0)))
}

fn instruction_successors(data: &InstructionData, func: &Function) -> Vec<Block> {
    match data {
        InstructionData::Jump { dest } => vec![func.dfg.block_call_block(*dest)],
        InstructionData::Br {
            then_dest,
            else_dest,
            ..
        } => vec![
            func.dfg.block_call_block(*then_dest),
            func.dfg.block_call_block(*else_dest),
        ],
        InstructionData::BrTable { table, .. } => func
            .dfg
            .jump_table_targets(*table)
            .iter()
            .map(|&call| func.dfg.block_call_block(call))
            .collect(),
        _ => Vec::new(),
    }
}

struct OperandParser<'a> {
    func: &'a mut Function,
    ctx: &'a mut ParseContext,
    func_ids: &'a HashMap<String, FuncId>,
    module: &'a mut ModuleData,
}

impl OperandParser<'_> {
    fn parse(
        &mut self,
        opcode: Opcode,
        ty: Option<Type>,
        flags: MemFlags,
        text: &str,
    ) -> core::result::Result<InstructionData, ParseError> {
        let codec = TextCodec::for_opcode(opcode);
        if !flags.is_empty() && !codec.accepts_memory_flags() {
            return Err(ParseError(format!(
                "memory flags are not valid on `{}`",
                opcode.spec().mnemonic
            )));
        }
        match codec {
            TextCodec::Values { arity } => self.parse_values(opcode, arity as usize, text),
            TextCodec::Nullary => {
                require_empty(text)?;
                InstructionData::from_values(opcode, &[])
                    .ok_or_else(|| codec_mismatch(opcode, codec))
            }
            TextCodec::IntegerConstant => Ok(InstructionData::Iconst {
                value: parse_integer_bits(text)?,
            }),
            TextCodec::FloatConstant => Ok(InstructionData::Fconst {
                value: parse_float_bits(text, ty)?,
            }),
            TextCodec::BoolConstant => Ok(InstructionData::Bconst {
                value: match text.trim() {
                    "true" => true,
                    "false" => false,
                    other => {
                        return Err(ParseError(format!(
                            "expected `true` or `false`, found `{other}`"
                        )));
                    }
                },
            }),
            TextCodec::VectorConstant => {
                let bytes = parse_hex_bytes(text)?;
                let pool_id = self
                    .func
                    .dfg
                    .make_constant_pool_data(ConstantPoolData::Bytes(bytes));
                Ok(InstructionData::Vconst { pool_id })
            }
            TextCodec::Load => {
                let (core, named) = split_core_and_named(text, 1)?;
                reject_unknown_named(&named, &["offset"])?;
                Ok(InstructionData::Load {
                    ptr: self.value(core[0])?,
                    offset: parse_named_u32(&named, "offset", 0)?,
                    flags,
                })
            }
            TextCodec::Store => {
                let (core, named) = split_core_and_named(text, 2)?;
                reject_unknown_named(&named, &["offset"])?;
                Ok(InstructionData::Store {
                    value: self.value(core[0])?,
                    ptr: self.value(core[1])?,
                    offset: parse_named_u32(&named, "offset", 0)?,
                    flags,
                })
            }
            TextCodec::StackLoad => {
                let (core, named) = split_core_and_named(text, 1)?;
                reject_unknown_named(&named, &["offset"])?;
                Ok(InstructionData::StackLoad {
                    slot: parse_stack_slot_ref(core[0])?,
                    offset: parse_named_u32(&named, "offset", 0)?,
                })
            }
            TextCodec::StackStore => {
                let (core, named) = split_core_and_named(text, 2)?;
                reject_unknown_named(&named, &["offset"])?;
                Ok(InstructionData::StackStore {
                    value: self.value(core[0])?,
                    slot: parse_stack_slot_ref(core[1])?,
                    offset: parse_named_u32(&named, "offset", 0)?,
                })
            }
            TextCodec::StackAddr => {
                let (core, named) = split_core_and_named(text, 1)?;
                reject_unknown_named(&named, &["offset"])?;
                Ok(InstructionData::StackAddr {
                    slot: parse_stack_slot_ref(core[0])?,
                    offset: parse_named_u32(&named, "offset", 0)?,
                })
            }
            TextCodec::PtrOffset => {
                let fields = exact_fields(text, 2)?;
                Ok(InstructionData::PtrOffset {
                    ptr: self.value(fields[0])?,
                    offset: parse_i32(fields[1], "pointer offset")?,
                })
            }
            TextCodec::PtrIndex => {
                let (core, named) = split_core_and_named(text, 2)?;
                reject_unknown_named(&named, &["scale", "offset"])?;
                let imm_id = self.func.dfg.make_ptr_imm(
                    parse_named_i32(&named, "offset", 0)?,
                    parse_named_u32(&named, "scale", 1)?,
                );
                Ok(InstructionData::PtrIndex {
                    ptr: self.value(core[0])?,
                    index: self.value(core[1])?,
                    imm_id,
                })
            }
            TextCodec::DirectCall => self.parse_direct_call(text),
            TextCodec::IndirectCall => self.parse_indirect_call(text),
            TextCodec::IntrinsicCall => self.parse_intrinsic_call(text),
            TextCodec::Jump => Ok(InstructionData::Jump {
                dest: self.block_call(text)?,
            }),
            TextCodec::Branch => {
                let fields = exact_fields(text, 3)?;
                Ok(InstructionData::Br {
                    condition: self.value(fields[0])?,
                    then_dest: self.block_call(fields[1])?,
                    else_dest: self.block_call(fields[2])?,
                })
            }
            TextCodec::BranchTable => self.parse_branch_table(text),
            TextCodec::Return => {
                let values = if text.trim().is_empty() {
                    Vec::new()
                } else {
                    split_top_level(text, ',')
                        .into_iter()
                        .map(|field| self.value(field))
                        .collect::<core::result::Result<Vec<_>, _>>()?
                };
                Ok(InstructionData::Return {
                    values: self.func.dfg.make_value_list(&values),
                })
            }
            TextCodec::IntCompare => {
                let (cc, values) = split_condition(text)?;
                Ok(InstructionData::IntCompare {
                    kind: crate::IntCC::from_mnemonic(cc)
                        .ok_or_else(|| ParseError(format!("unknown integer condition `{cc}`")))?,
                    args: [self.value(values[0])?, self.value(values[1])?],
                })
            }
            TextCodec::FloatCompare => {
                let (cc, values) = split_condition(text)?;
                Ok(InstructionData::FloatCompare {
                    kind: crate::FloatCC::from_mnemonic(cc)
                        .ok_or_else(|| ParseError(format!("unknown float condition `{cc}`")))?,
                    args: [self.value(values[0])?, self.value(values[1])?],
                })
            }
            TextCodec::VectorLoadStrided
            | TextCodec::VectorStoreStrided
            | TextCodec::VectorGather
            | TextCodec::VectorScatter => self.parse_vector_memory(codec, flags, text),
            TextCodec::Shuffle => {
                let (core, named) = split_core_and_named(text, 2)?;
                reject_unknown_named(&named, &["mask"])?;
                let bytes = parse_hex_bytes(named_value(&named, "mask")?)?;
                let mask = self
                    .func
                    .dfg
                    .make_constant_pool_data(ConstantPoolData::Bytes(bytes));
                Ok(InstructionData::Shuffle {
                    args: [self.value(core[0])?, self.value(core[1])?],
                    mask,
                })
            }
        }
    }

    fn parse_values(
        &mut self,
        opcode: Opcode,
        arity: usize,
        text: &str,
    ) -> core::result::Result<InstructionData, ParseError> {
        let (core, named) = split_core_and_named(text, arity)?;
        let args = core
            .iter()
            .map(|field| self.value(field))
            .collect::<core::result::Result<Vec<_>, _>>()?;
        if !named.is_empty() {
            reject_unknown_named(&named, &["mask", "evl"])?;
            let mask = self.value(named_value(&named, "mask")?)?;
            let evl = self.optional_named_value(&named, "evl")?;
            let ext = self.func.dfg.make_vector_ext(mask, evl);
            let args = self.func.dfg.make_value_list(&args);
            return Ok(InstructionData::VectorOpWithExt { opcode, args, ext });
        }

        InstructionData::from_values(opcode, &args)
            .ok_or_else(|| codec_mismatch(opcode, TextCodec::Values { arity: arity as u8 }))
    }

    fn parse_direct_call(
        &mut self,
        text: &str,
    ) -> core::result::Result<InstructionData, ParseError> {
        let (callee, args, trailing) = parse_invocation(text)?;
        if !trailing.trim().is_empty() {
            return Err(ParseError(format!(
                "unexpected text after direct call: `{trailing}`"
            )));
        }
        let func_id = self
            .func_ids
            .get(callee)
            .copied()
            .or_else(|| parse_func_ref(callee))
            .ok_or_else(|| ParseError(format!("unknown function `{callee}`")))?;
        let args = self.values(args)?;
        Ok(InstructionData::Call {
            func_id,
            args: self.func.dfg.make_value_list(&args),
        })
    }

    fn parse_indirect_call(
        &mut self,
        text: &str,
    ) -> core::result::Result<InstructionData, ParseError> {
        let (ptr, args, trailing) = parse_invocation(text)?;
        let sig_id = self
            .module
            .intern_signature(parse_signature_suffix(trailing)?);
        let args = self.values(args)?;
        Ok(InstructionData::CallIndirect {
            ptr: self.value(ptr)?,
            args: self.func.dfg.make_value_list(&args),
            sig_id,
        })
    }

    fn parse_intrinsic_call(
        &mut self,
        text: &str,
    ) -> core::result::Result<InstructionData, ParseError> {
        let (name, args, trailing) = parse_invocation(text)?;
        let intrinsic = Intrinsic::from_name(name)
            .ok_or_else(|| ParseError(format!("unknown intrinsic `{name}`")))?;
        let sig_id = self
            .module
            .intern_signature(parse_signature_suffix(trailing)?);
        let args = self.values(args)?;
        Ok(InstructionData::CallIntrinsic {
            intrinsic,
            args: self.func.dfg.make_value_list(&args),
            sig_id,
        })
    }

    fn parse_branch_table(
        &mut self,
        text: &str,
    ) -> core::result::Result<InstructionData, ParseError> {
        let fields = exact_fields(text, 3)?;
        let index = self.value(fields[0])?;
        let cases = fields[1]
            .strip_prefix('[')
            .and_then(|value| value.strip_suffix(']'))
            .ok_or_else(|| ParseError("branch-table cases must be enclosed in `[]`".into()))?;
        let mut targets = if cases.trim().is_empty() {
            Vec::new()
        } else {
            split_top_level(cases, ',')
                .into_iter()
                .map(|field| self.block_call(field))
                .collect::<core::result::Result<Vec<_>, _>>()?
        };
        // Physical order is case zero through case N, followed by default.
        targets.push(self.block_call(fields[2])?);
        let table = self.func.dfg.jump_tables.push(JumpTableData { targets });
        Ok(InstructionData::BrTable { index, table })
    }

    fn parse_vector_memory(
        &mut self,
        codec: TextCodec,
        flags: MemFlags,
        text: &str,
    ) -> core::result::Result<InstructionData, ParseError> {
        let core_count = match codec {
            TextCodec::VectorLoadStrided | TextCodec::VectorGather => 1,
            TextCodec::VectorStoreStrided | TextCodec::VectorScatter => 2,
            _ => unreachable!(),
        };
        let (core, named) = split_core_and_named(text, core_count)?;
        reject_unknown_named(
            &named,
            &["stride", "index", "scale", "offset", "mask", "evl"],
        )?;
        let scale = u8::try_from(parse_named_u32(&named, "scale", 1)?)
            .map_err(|_| ParseError("vector index scale must fit in u8".into()))?;
        let mask = self.optional_named_value(&named, "mask")?;
        let evl = self.optional_named_value(&named, "evl")?;
        let ext = self.func.dfg.make_vector_mem_ext(VectorMemOptions {
            offset: parse_named_i32(&named, "offset", 0)?,
            flags,
            scale,
            mask,
            evl,
        });
        let ptr = self.value(core[core_count - 1])?;

        match codec {
            TextCodec::VectorLoadStrided => Ok(InstructionData::VectorLoadStrided {
                ptr,
                stride: self.value(named_value(&named, "stride")?)?,
                ext,
            }),
            TextCodec::VectorStoreStrided => {
                let value = self.value(core[0])?;
                let stride = self.value(named_value(&named, "stride")?)?;
                let args = self.func.dfg.make_value_list(&[ptr, stride, value]);
                Ok(InstructionData::VectorStoreStrided { args, ext })
            }
            TextCodec::VectorGather => Ok(InstructionData::VectorGather {
                ptr,
                index: self.value(named_value(&named, "index")?)?,
                ext,
            }),
            TextCodec::VectorScatter => {
                let value = self.value(core[0])?;
                let index = self.value(named_value(&named, "index")?)?;
                let args = self.func.dfg.make_value_list(&[ptr, index, value]);
                Ok(InstructionData::VectorScatter { args, ext })
            }
            _ => unreachable!(),
        }
    }

    fn value(&mut self, name: &str) -> core::result::Result<Value, ParseError> {
        let name = name.trim();
        if name.is_empty() || name.chars().any(char::is_whitespace) {
            return Err(ParseError(format!("invalid SSA value `{name}`")));
        }
        Ok(get_or_create_value(name, self.func, self.ctx))
    }

    fn values(&mut self, fields: &str) -> core::result::Result<Vec<Value>, ParseError> {
        if fields.trim().is_empty() {
            return Ok(Vec::new());
        }
        split_top_level(fields, ',')
            .into_iter()
            .map(|field| self.value(field))
            .collect()
    }

    fn block_call(&mut self, text: &str) -> core::result::Result<BlockCall, ParseError> {
        let (name, args, trailing) = parse_invocation(text)?;
        if !trailing.trim().is_empty() {
            return Err(ParseError(format!(
                "unexpected text after block call: `{trailing}`"
            )));
        }
        let block = self
            .ctx
            .block_map
            .get(name)
            .copied()
            .ok_or_else(|| ParseError(format!("unknown block `{name}`")))?;
        let values = self.values(args)?;
        let args = self.func.dfg.make_value_list(&values);
        Ok(self
            .func
            .dfg
            .block_calls
            .push(BlockCallData { block, args }))
    }

    fn optional_named_value(
        &mut self,
        named: &[(&str, &str)],
        key: &str,
    ) -> core::result::Result<Option<Value>, ParseError> {
        named
            .iter()
            .find(|(candidate, _)| *candidate == key)
            .map(|(_, value)| self.value(value))
            .transpose()
    }
}

fn parse_result_names(line: &str) -> core::result::Result<(Vec<String>, &str), ParseError> {
    let Some((lhs, rhs)) = line.split_once(" = ") else {
        return Ok((Vec::new(), line.trim()));
    };
    let lhs = lhs.trim();
    let names = if let Some(inner) = lhs
        .strip_prefix('(')
        .and_then(|value| value.strip_suffix(')'))
    {
        split_top_level(inner, ',')
    } else {
        vec![lhs]
    };
    if names.iter().any(|name| name.is_empty()) {
        return Err(ParseError("empty result name".into()));
    }
    Ok((names.into_iter().map(str::to_string).collect(), rhs.trim()))
}

fn parse_instruction_header(
    text: &str,
) -> core::result::Result<(Opcode, Option<Type>, MemFlags, &str), ParseError> {
    let text = text.trim();
    let opcode = Opcode::ALL
        .iter()
        .copied()
        .filter(|opcode| {
            let name = opcode.spec().mnemonic;
            text.strip_prefix(name).is_some_and(|rest| {
                rest.is_empty()
                    || rest.starts_with('.')
                    || rest.chars().next().is_some_and(char::is_whitespace)
            })
        })
        .max_by_key(|opcode| opcode.spec().mnemonic.len())
        .ok_or_else(|| ParseError(format!("unknown opcode in `{text}`")))?;
    let mut rest = &text[opcode.spec().mnemonic.len()..];
    let mut ty = None;
    let mut flags = MemFlags::new();

    if rest.starts_with('.') {
        let end = token_end_with_angles(rest);
        let suffix = &rest[1..end];
        rest = &rest[end..];
        for part in split_top_level(suffix, '.') {
            if let Some(parsed) = format::parse_type(part) {
                if ty.replace(parsed).is_some() {
                    return Err(ParseError("multiple result type suffixes".into()));
                }
            } else if part == "volatile" {
                flags = flags.union(MemFlags::VOLATILE);
            } else if let Some(value) = part.strip_prefix("align") {
                let alignment = value
                    .parse::<u32>()
                    .map_err(|_| ParseError(format!("invalid alignment `{part}`")))?;
                if alignment == 0 || !alignment.is_power_of_two() {
                    return Err(ParseError(format!(
                        "alignment must be a non-zero power of two: {alignment}"
                    )));
                }
                flags = flags.with_alignment(alignment);
            } else {
                return Err(ParseError(format!("unknown opcode suffix `{part}`")));
            }
        }
    }
    Ok((opcode, ty, flags, rest.trim()))
}

fn token_end_with_angles(text: &str) -> usize {
    let mut angles = 0u32;
    for (index, ch) in text.char_indices() {
        match ch {
            '<' => angles += 1,
            '>' => angles = angles.saturating_sub(1),
            _ if ch.is_whitespace() && angles == 0 => return index,
            _ => {}
        }
    }
    text.len()
}

fn parse_function_header(text: &str) -> core::result::Result<FunctionHeader, ParseError> {
    let split = text
        .find(char::is_whitespace)
        .ok_or_else(|| ParseError("incomplete function header".into()))?;
    let linkage_text = &text[..split];
    let linkage = Linkage::from_mnemonic(linkage_text)
        .ok_or_else(|| ParseError(format!("unknown linkage `{linkage_text}`")))?;
    let rest = text[split..]
        .trim_start()
        .strip_prefix("function ")
        .ok_or_else(|| ParseError("expected `function`".into()))?;
    let (name, params, trailing) = parse_invocation(rest)?;
    let params = parse_type_list(params)?;
    let trailing = trailing.trim();
    let returns = if let Some(ret) = trailing.strip_prefix("->") {
        parse_return_types(ret.trim())?
    } else if trailing.is_empty() {
        Vec::new()
    } else {
        return Err(ParseError(format!(
            "unexpected function header suffix `{trailing}`"
        )));
    };
    Ok(FunctionHeader {
        name: name.to_string(),
        linkage,
        signature: Signature::new(params, returns, CallConv::SystemV),
    })
}

fn parse_signature_suffix(text: &str) -> core::result::Result<Signature, ParseError> {
    let text = text
        .trim()
        .strip_prefix(':')
        .ok_or_else(|| ParseError("call signature must follow `:`".into()))?
        .trim();
    let signature_text = format!("sig{text}");
    let (_, params, trailing) = parse_invocation(&signature_text)?;
    let returns = trailing
        .trim()
        .strip_prefix("->")
        .ok_or_else(|| ParseError("call signature is missing `->`".into()))?;
    Ok(Signature::new(
        parse_type_list(params)?,
        parse_return_types(returns.trim())?,
        CallConv::SystemV,
    ))
}

fn parse_type_list(text: &str) -> core::result::Result<Vec<Type>, ParseError> {
    if text.trim().is_empty() {
        return Ok(Vec::new());
    }
    split_top_level(text, ',')
        .into_iter()
        .map(|field| {
            format::parse_type(field).ok_or_else(|| ParseError(format!("unknown type `{field}`")))
        })
        .collect()
}

fn parse_return_types(text: &str) -> core::result::Result<Vec<Type>, ParseError> {
    let text = text.trim();
    if text == "void" {
        return Ok(Vec::new());
    }
    let inner = text
        .strip_prefix('(')
        .and_then(|value| value.strip_suffix(')'))
        .unwrap_or(text);
    parse_type_list(inner)
}

fn parse_global(text: &str) -> core::result::Result<(String, Type, Linkage), ParseError> {
    let rest = text
        .strip_prefix("global ")
        .ok_or_else(|| ParseError("expected `global`".into()))?;
    let (name, rest) = rest
        .split_once(':')
        .ok_or_else(|| ParseError("global is missing `:`".into()))?;
    let open = rest
        .rfind('(')
        .ok_or_else(|| ParseError("global is missing linkage".into()))?;
    let ty_text = rest[..open].trim();
    let ty = format::parse_type(ty_text)
        .ok_or_else(|| ParseError(format!("unknown global type `{ty_text}`")))?;
    let linkage_text = rest[open + 1..]
        .trim()
        .strip_suffix(')')
        .ok_or_else(|| ParseError("global linkage is missing `)`".into()))?;
    let linkage = Linkage::from_mnemonic(linkage_text)
        .ok_or_else(|| ParseError(format!("unknown linkage `{linkage_text}`")))?;
    Ok((name.trim().to_string(), ty, linkage))
}

fn is_function_header(text: &str) -> bool {
    ["local function ", "export function ", "import function "]
        .iter()
        .any(|prefix| text.starts_with(prefix))
}

fn is_block_header(text: &str) -> bool {
    text.starts_with("block") && text.ends_with(':')
}

fn parse_block_header(text: &str) -> core::result::Result<(u32, Vec<(String, Type)>), ParseError> {
    let text = text
        .strip_suffix(':')
        .ok_or_else(|| ParseError("block header is missing `:`".into()))?
        .trim();
    let (name, params, trailing) = parse_invocation(text)?;
    if !trailing.trim().is_empty() {
        return Err(ParseError(format!(
            "unexpected block metadata `{trailing}`"
        )));
    }
    let id = name
        .strip_prefix("block")
        .ok_or_else(|| ParseError("block name must start with `block`".into()))?
        .parse::<u32>()
        .map_err(|_| ParseError(format!("invalid block name `{name}`")))?;
    let params = if params.trim().is_empty() {
        Vec::new()
    } else {
        split_top_level(params, ',')
            .into_iter()
            .map(|field| {
                let (name, ty) = field.split_once(':').ok_or_else(|| {
                    ParseError(format!("block parameter `{field}` is missing `:`"))
                })?;
                let ty = format::parse_type(ty.trim())
                    .ok_or_else(|| ParseError(format!("unknown type `{}`", ty.trim())))?;
                Ok((name.trim().to_string(), ty))
            })
            .collect::<core::result::Result<Vec<_>, ParseError>>()?
    };
    Ok((id, params))
}

fn parse_stack_slot(text: &str, func: &mut Function) -> core::result::Result<(), ParseError> {
    let (slot, size) = text
        .split_once(':')
        .ok_or_else(|| ParseError("stack slot is missing `:`".into()))?;
    let slot = parse_stack_slot_ref(slot)?;
    let size = size
        .trim()
        .strip_prefix("size ")
        .ok_or_else(|| ParseError("stack slot is missing `size`".into()))?
        .parse::<u32>()
        .map_err(|_| ParseError("invalid stack slot size".into()))?;
    while func.stack_slots.len() <= slot.0 as usize {
        func.stack_slots.push(StackSlotData { size: 0 });
    }
    func.stack_slots[slot] = StackSlotData { size };
    Ok(())
}

fn parse_invocation(text: &str) -> core::result::Result<(&str, &str, &str), ParseError> {
    let open = text
        .find('(')
        .ok_or_else(|| ParseError(format!("expected `(` in `{text}`")))?;
    let close = matching_delimiter(text, open, '(', ')')?;
    let name = text[..open].trim();
    if name.is_empty() {
        return Err(ParseError("missing invocation target".into()));
    }
    Ok((name, &text[open + 1..close], &text[close + 1..]))
}

fn matching_delimiter(
    text: &str,
    open: usize,
    open_char: char,
    close_char: char,
) -> core::result::Result<usize, ParseError> {
    let mut depth = 0u32;
    for (offset, ch) in text[open..].char_indices() {
        if ch == open_char {
            depth += 1;
        } else if ch == close_char {
            depth -= 1;
            if depth == 0 {
                return Ok(open + offset);
            }
        }
    }
    Err(ParseError(format!("unclosed `{open_char}` in `{text}`")))
}

fn split_top_level(text: &str, separator: char) -> Vec<&str> {
    let mut result = Vec::new();
    let mut start = 0;
    let (mut parens, mut brackets, mut angles) = (0u32, 0u32, 0u32);
    for (index, ch) in text.char_indices() {
        match ch {
            '(' => parens += 1,
            ')' => parens = parens.saturating_sub(1),
            '[' => brackets += 1,
            ']' => brackets = brackets.saturating_sub(1),
            '<' => angles += 1,
            '>' => angles = angles.saturating_sub(1),
            _ => {}
        }
        if ch == separator && parens == 0 && brackets == 0 && angles == 0 {
            result.push(text[start..index].trim());
            start = index + ch.len_utf8();
        }
    }
    result.push(text[start..].trim());
    result
}

fn exact_fields(text: &str, count: usize) -> core::result::Result<Vec<&str>, ParseError> {
    let fields = if text.trim().is_empty() {
        Vec::new()
    } else {
        split_top_level(text, ',')
    };
    if fields.len() != count {
        return Err(ParseError(format!(
            "expected {count} operand(s), found {}",
            fields.len()
        )));
    }
    Ok(fields)
}

type NamedFields<'a> = Vec<(&'a str, &'a str)>;

fn split_core_and_named(
    text: &str,
    core_count: usize,
) -> core::result::Result<(Vec<&str>, NamedFields<'_>), ParseError> {
    let fields = if text.trim().is_empty() {
        Vec::new()
    } else {
        split_top_level(text, ',')
    };
    if fields.len() < core_count {
        return Err(ParseError(format!(
            "expected at least {core_count} operand(s), found {}",
            fields.len()
        )));
    }
    let mut named = Vec::new();
    for field in &fields[core_count..] {
        let (key, value) = field
            .split_once('=')
            .ok_or_else(|| ParseError(format!("expected named field, found `{field}`")))?;
        let key = key.trim();
        if named.iter().any(|(existing, _)| *existing == key) {
            return Err(ParseError(format!("duplicate `{key}` field")));
        }
        named.push((key, value.trim()));
    }
    Ok((fields[..core_count].to_vec(), named))
}

fn split_condition(text: &str) -> core::result::Result<(&str, Vec<&str>), ParseError> {
    let split = text
        .find(char::is_whitespace)
        .ok_or_else(|| ParseError("comparison is missing operands".into()))?;
    let cc = text[..split].trim();
    let values = exact_fields(text[split..].trim(), 2)?;
    Ok((cc, values))
}

fn named_value<'a>(
    named: &'a [(&str, &str)],
    key: &str,
) -> core::result::Result<&'a str, ParseError> {
    named
        .iter()
        .find(|(candidate, _)| *candidate == key)
        .map(|(_, value)| *value)
        .ok_or_else(|| ParseError(format!("missing `{key}=` field")))
}

fn reject_unknown_named(
    named: &[(&str, &str)],
    allowed: &[&str],
) -> core::result::Result<(), ParseError> {
    if let Some((key, _)) = named.iter().find(|(key, _)| !allowed.contains(key)) {
        Err(ParseError(format!("unknown named field `{key}`")))
    } else {
        Ok(())
    }
}

fn parse_named_u32(
    named: &[(&str, &str)],
    key: &str,
    default: u32,
) -> core::result::Result<u32, ParseError> {
    match named.iter().find(|(candidate, _)| *candidate == key) {
        Some((_, value)) => value
            .parse()
            .map_err(|_| ParseError(format!("invalid `{key}` value `{value}`"))),
        None => Ok(default),
    }
}

fn parse_named_i32(
    named: &[(&str, &str)],
    key: &str,
    default: i32,
) -> core::result::Result<i32, ParseError> {
    match named.iter().find(|(candidate, _)| *candidate == key) {
        Some((_, value)) => parse_i32(value, key),
        None => Ok(default),
    }
}

fn parse_i32(text: &str, what: &str) -> core::result::Result<i32, ParseError> {
    text.trim()
        .parse()
        .map_err(|_| ParseError(format!("invalid {what} `{text}`")))
}

fn parse_integer_bits(text: &str) -> core::result::Result<u64, ParseError> {
    let text = text.trim();
    if let Some(hex) = text.strip_prefix("0x") {
        u64::from_str_radix(hex, 16)
            .map_err(|_| ParseError(format!("invalid integer constant `{text}`")))
    } else {
        text.parse::<i64>()
            .map(|value| value as u64)
            .map_err(|_| ParseError(format!("invalid integer constant `{text}`")))
    }
}

fn parse_float_bits(text: &str, ty: Option<Type>) -> core::result::Result<u64, ParseError> {
    if !matches!(ty, Some(Type::F32 | Type::F64)) {
        return Err(ParseError(
            "floating constants require an `f32` or `f64` result suffix".into(),
        ));
    }
    let text = text.trim();
    let hex = text.strip_prefix("0x").ok_or_else(|| {
        ParseError("floating constants use an exact hexadecimal bit pattern".into())
    })?;
    let bits = u64::from_str_radix(hex, 16)
        .map_err(|_| ParseError(format!("invalid floating bit pattern `{text}`")))?;
    if ty == Some(Type::F32) && bits > u64::from(u32::MAX) {
        return Err(ParseError(format!(
            "f32 bit pattern does not fit in 32 bits: `{text}`"
        )));
    }
    Ok(bits)
}

fn parse_hex_bytes(text: &str) -> core::result::Result<Vec<u8>, ParseError> {
    let text = text.trim();
    let hex = text
        .strip_prefix("0x")
        .ok_or_else(|| ParseError(format!("expected hexadecimal bytes, found `{text}`")))?;
    if hex.len() % 2 != 0 {
        return Err(ParseError(
            "hex byte strings must contain an even number of digits".into(),
        ));
    }
    (0..hex.len())
        .step_by(2)
        .map(|index| {
            u8::from_str_radix(&hex[index..index + 2], 16)
                .map_err(|_| ParseError(format!("invalid hexadecimal bytes `{text}`")))
        })
        .collect()
}

fn parse_stack_slot_ref(text: &str) -> core::result::Result<StackSlot, ParseError> {
    let id = text
        .trim()
        .strip_prefix("ss")
        .ok_or_else(|| ParseError(format!("expected stack slot, found `{text}`")))?
        .parse::<u32>()
        .map_err(|_| ParseError(format!("invalid stack slot `{text}`")))?;
    Ok(StackSlot(id))
}

fn parse_func_ref(text: &str) -> Option<FuncId> {
    let id = text
        .strip_prefix("func")
        .or_else(|| text.strip_prefix("FuncId(")?.strip_suffix(')'))?;
    id.parse().ok().map(FuncId)
}

fn require_empty(text: &str) -> core::result::Result<(), ParseError> {
    if text.trim().is_empty() {
        Ok(())
    } else {
        Err(ParseError(format!("unexpected operands `{text}`")))
    }
}

fn codec_mismatch(opcode: Opcode, codec: TextCodec) -> ParseError {
    ParseError(format!(
        "internal text codec mismatch for `{}`: {codec:?}",
        opcode.spec().mnemonic
    ))
}

fn parse_value_idx(name: &str) -> Option<u32> {
    name.strip_prefix('v')
        .and_then(|digits| digits.parse().ok())
        .or_else(|| {
            name.rfind(".v")
                .and_then(|index| name[index + 2..].parse().ok())
        })
}

fn get_or_create_value(name: &str, func: &mut Function, ctx: &mut ParseContext) -> Value {
    if let Some(&value) = ctx.value_map.get(name) {
        return value;
    }
    let index = parse_value_idx(name).unwrap_or(ctx.next_value_idx);
    let value = Value(index);
    ensure_value(value, func);
    set_value_name(value, name, func);
    ctx.value_map.insert(name.to_string(), value);
    ctx.next_value_idx = ctx.next_value_idx.max(index + 1);
    value
}

fn define_value(
    name: &str,
    func: &mut Function,
    ctx: &mut ParseContext,
    ty: Type,
    def: ValueDef,
) -> core::result::Result<Value, ParseError> {
    let value = get_or_create_value(name, func, ctx);
    if let Some(previous) = ctx.defined_values.get(&value) {
        return Err(ParseError(format!(
            "SSA value `{name}` aliases already-defined `{previous}`"
        )));
    }
    func.dfg.values[value] = ValueData { ty, def };
    ctx.defined_values.insert(value, name.to_string());
    Ok(value)
}

fn ensure_value(value: Value, func: &mut Function) {
    while func.dfg.values.len() <= value.0 as usize {
        func.dfg.values.push(ValueData {
            ty: Type::INVALID,
            def: ValueDef::Param(Block(0)),
        });
    }
}

fn set_value_name(value: Value, text: &str, func: &mut Function) {
    let name = if text
        .strip_prefix('v')
        .is_some_and(|digits| digits.chars().all(|ch| ch.is_ascii_digit()))
    {
        ""
    } else if let Some(index) = text
        .rfind(".v")
        .filter(|&index| text[index + 2..].chars().all(|ch| ch.is_ascii_digit()))
    {
        &text[..index]
    } else {
        text
    };
    func.dfg.value_names[value] = name.to_string();
}

fn strip_comment(text: &str) -> &str {
    text.split_once("//").map_or(text, |(code, _)| code)
}

fn with_line(error: ParseError, line_no: usize, line: &str) -> ParseError {
    ParseError(format!("line {line_no}: {}\n  {line}", error.0))
}

fn parse_err<T>(message: impl Into<String>) -> Result<T> {
    Err(ParseError(message.into()).into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delimiter_split_ignores_vectors_and_nested_calls() {
        assert_eq!(
            split_top_level("i32<scalable 4>, block1(v0, v1), [block2()]", ','),
            ["i32<scalable 4>", "block1(v0, v1)", "[block2()]"]
        );
    }

    #[test]
    fn instruction_header_accepts_scalable_result_type() {
        let (opcode, ty, _, args) =
            parse_instruction_header("iadd.i32<scalable 4> v0, v1").unwrap();
        assert_eq!(opcode, Opcode::IAdd);
        assert_eq!(ty, Type::new_vector(crate::ScalarType::I32, 4, true));
        assert_eq!(args, "v0, v1");
    }
}
