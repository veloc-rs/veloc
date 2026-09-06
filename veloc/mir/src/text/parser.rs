//! Parser for the canonical textual IR.
//!
//! Instruction syntax and storage construction are generated from OpSpec.
//! A delimiter-aware scanner and typed atoms handle nested lists, signatures,
//! and forward SSA references independently of the instruction set.

use crate::{
    Block, BlockCall, CallConv, FuncId, Function, InstructionData, Linkage, MemFlags, Module,
    ModuleData, Opcode, Result, SigId, Signature, StackSlot, Type, Value, ValueDef,
    function::StackSlotData,
    inst::Inst,
    opspec::ResultTypes,
    types::{ValueData, parse_type},
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
            .validate_types(&operands, &results)
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
    // The generated entry point decides whether unresolved inputs can be
    // deferred. Deferred contracts are rechecked after all value definitions.
    let (inferred, deferred) =
        data.opcode()
            .infer_text_result_types(&operands)
            .map_err(|error| {
                ParseError(format!(
                    "invalid operand types for `{}`: {error:?}",
                    spec.mnemonic
                ))
            })?;
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
    let call = data
        .call_info()
        .ok_or_else(|| ParseError("signature result on non-call instruction".into()))?;
    let sig = call.signature.resolve(module).ok_or_else(|| {
        ParseError(format!(
            "unknown function or signature {:?}",
            call.signature
        ))
    })?;
    module
        .signatures
        .get(sig)
        .ok_or_else(|| ParseError(format!("unknown signature id {}", sig.0)))
}

fn instruction_successors(data: &InstructionData, func: &Function) -> Vec<Block> {
    let mut blocks = Vec::new();
    data.visit_successors(&func.dfg, |call| {
        blocks.push(func.dfg.block_call_block(call));
    });
    blocks
}

pub(super) struct OperandParser<'a> {
    func: &'a mut Function,
    ctx: &'a mut ParseContext,
    func_ids: &'a HashMap<String, FuncId>,
    module: &'a mut ModuleData,
}

impl OperandParser<'_> {
    pub(super) fn value(&mut self, name: &str) -> core::result::Result<Value, ParseError> {
        let name = name.trim();
        if name.is_empty() || name.chars().any(char::is_whitespace) {
            return Err(ParseError(format!("invalid SSA value `{name}`")));
        }
        Ok(get_or_create_value(name, self.func, self.ctx))
    }

    pub(super) fn values(&mut self, fields: &str) -> core::result::Result<Vec<Value>, ParseError> {
        if fields.trim().is_empty() {
            return Ok(Vec::new());
        }
        split_top_level(fields, ',')
            .into_iter()
            .map(|field| self.value(field))
            .collect()
    }

    pub(super) fn block_call(&mut self, text: &str) -> core::result::Result<BlockCall, ParseError> {
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
        Ok(self.func.dfg.make_block_call(block, &values))
    }

    pub(super) fn block_calls(
        &mut self,
        text: &str,
    ) -> core::result::Result<Vec<BlockCall>, ParseError> {
        let inner = text
            .trim()
            .strip_prefix('[')
            .and_then(|text| text.strip_suffix(']'))
            .ok_or_else(|| ParseError("block-call list must be enclosed in `[]`".into()))?;
        if inner.trim().is_empty() {
            return Ok(Vec::new());
        }
        split_top_level(inner, ',')
            .into_iter()
            .map(|text| self.block_call(text))
            .collect()
    }

    pub(super) fn func_ref(&self, text: &str) -> core::result::Result<FuncId, ParseError> {
        let name = text.trim();
        self.func_ids
            .get(name)
            .copied()
            .or_else(|| parse_func_ref(name))
            .ok_or_else(|| ParseError(format!("unknown function `{name}`")))
    }

    pub(super) fn signature(&mut self, text: &str) -> core::result::Result<SigId, ParseError> {
        Ok(self.module.intern_signature(parse_signature(text)?))
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
            if let Some(parsed) = parse_type(part) {
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

fn parse_signature(text: &str) -> core::result::Result<Signature, ParseError> {
    let text = text.trim();
    if !text.starts_with('(') {
        return Err(ParseError(
            "signature parameters must start with `(`".into(),
        ));
    }
    let close = matching_delimiter(text, 0, '(', ')')?;
    let returns = text[close + 1..]
        .trim()
        .strip_prefix("->")
        .ok_or_else(|| ParseError("signature is missing `->`".into()))?;
    if returns.trim().is_empty() {
        return Err(ParseError("signature is missing return types".into()));
    }
    Ok(Signature::new(
        parse_type_list(&text[1..close])?,
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
        .map(|field| parse_type(field).ok_or_else(|| ParseError(format!("unknown type `{field}`"))))
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
    let ty = parse_type(ty_text)
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
                let ty = parse_type(ty.trim())
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

pub(crate) fn parse_invocation(text: &str) -> core::result::Result<(&str, &str, &str), ParseError> {
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

pub(crate) fn split_top_level(text: &str, separator: char) -> Vec<&str> {
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

pub(crate) fn exact_fields(
    text: &str,
    count: usize,
) -> core::result::Result<Vec<&str>, ParseError> {
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

pub(crate) fn split_core_and_named(
    text: &str,
    core_count: Option<usize>,
) -> core::result::Result<(Vec<&str>, NamedFields<'_>), ParseError> {
    let fields = if text.trim().is_empty() {
        Vec::new()
    } else {
        split_top_level(text, ',')
    };
    let core_count = core_count.unwrap_or_else(|| {
        fields
            .iter()
            .position(|field| field.contains('='))
            .unwrap_or(fields.len())
    });
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
        let value = value.trim();
        if key.is_empty() || value.is_empty() {
            return Err(ParseError(format!(
                "empty name or value in named field `{field}`"
            )));
        }
        if named.iter().any(|(existing, _)| *existing == key) {
            return Err(ParseError(format!("duplicate `{key}` field")));
        }
        named.push((key, value));
    }
    Ok((fields[..core_count].to_vec(), named))
}

pub(crate) fn split_space(text: &str) -> core::result::Result<(&str, &str), ParseError> {
    let text = text.trim();
    let split = text
        .find(char::is_whitespace)
        .ok_or_else(|| ParseError("expected whitespace-separated fields".into()))?;
    Ok((&text[..split], text[split..].trim_start()))
}

pub(crate) fn signature_suffix(text: &str) -> core::result::Result<&str, ParseError> {
    let signature = text
        .trim()
        .strip_prefix(':')
        .ok_or_else(|| ParseError("signature must follow `:`".into()))?
        .trim();
    if signature.is_empty() {
        return Err(ParseError("missing signature after `:`".into()));
    }
    Ok(signature)
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

pub(super) fn parse_stack_slot_ref(text: &str) -> core::result::Result<StackSlot, ParseError> {
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

pub(crate) fn require_empty(text: &str) -> core::result::Result<(), ParseError> {
    if text.trim().is_empty() {
        Ok(())
    } else {
        Err(ParseError(format!("unexpected operands `{text}`")))
    }
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

include!(concat!(env!("OUT_DIR"), "/text_parser.rs"));

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::atom::{AtomCodec, Bytes, Decimal, FloatBits, IntegerBits};
    use crate::text::printer::InstPrinter;
    use core::{borrow::Borrow, fmt::Debug};

    fn with_parser(test: impl FnOnce(&mut OperandParser<'_>)) {
        let mut func = Function::new("test".into(), SigId(0), Linkage::Local);
        let mut ctx = ParseContext::default();
        let mut module = ModuleData::default();
        let func_ids = HashMap::new();
        test(&mut OperandParser {
            func: &mut func,
            ctx: &mut ctx,
            func_ids: &func_ids,
            module: &mut module,
        });
    }

    fn round_trip<C: AtomCodec>(cx: &mut OperandParser<'_>, text: &str, ty: Option<Type>) -> String
    where
        C::Owned: Debug + PartialEq,
    {
        let value = C::parse(cx, text, ty).unwrap();
        let mut printed = String::new();
        C::print(
            &InstPrinter::new(&cx.func.dfg, None),
            &mut printed,
            value.borrow(),
            ty,
        )
        .unwrap();
        assert_eq!(C::parse(cx, &printed, ty).unwrap(), value);
        printed
    }

    #[test]
    fn atom_codecs_pair_parsing_and_printing_without_erasing_types() {
        with_parser(|cx| {
            assert_eq!(
                round_trip::<IntegerBits>(cx, "0xffffffffffffffff", None),
                "-1"
            );
            assert_eq!(
                round_trip::<Decimal<u64>>(cx, "18446744073709551615", None),
                "18446744073709551615"
            );
            assert!(Decimal::<u64>::parse(cx, "-1", None).is_err());
            assert_eq!(
                round_trip::<Decimal<i32>>(cx, "-2147483648", None),
                "-2147483648"
            );
            assert_eq!(round_trip::<Decimal<u8>>(cx, "255", None), "255");
            assert!(Decimal::<u8>::parse(cx, "256", None).is_err());
            assert_eq!(round_trip::<bool>(cx, "true", None), "true");
            assert!(bool::parse(cx, "1", None).is_err());
            assert_eq!(round_trip::<crate::IntCC>(cx, "eq", None), "eq");
            assert_eq!(round_trip::<crate::FloatCC>(cx, "eq", None), "eq");
            assert_eq!(round_trip::<StackSlot>(cx, "ss7", None), "ss7");
            assert_eq!(round_trip::<Value>(cx, "v0", None), "v0");
        });
    }

    #[test]
    fn float_codec_preserves_bits_and_checks_width_in_both_directions() {
        with_parser(|cx| {
            for (ty, bits) in [
                (Type::F32, "0x7fc00001"),
                (Type::F32, "0x80000000"),
                (Type::F64, "0x7ff8000000000042"),
            ] {
                assert_eq!(round_trip::<FloatBits>(cx, bits, Some(ty)), bits);
            }
            for ty in [None, Some(Type::I32)] {
                assert!(FloatBits::parse(cx, "0x0", ty).is_err());
                assert!(
                    FloatBits::print(
                        &InstPrinter::new(&cx.func.dfg, None),
                        &mut String::new(),
                        &0,
                        ty
                    )
                    .is_err()
                );
            }
            assert!(FloatBits::parse(cx, "0x100000000", Some(Type::F32)).is_err());
            assert!(
                FloatBits::print(
                    &InstPrinter::new(&cx.func.dfg, None),
                    &mut String::new(),
                    &0x100000000,
                    Some(Type::F32)
                )
                .is_err()
            );
        });
    }

    #[test]
    fn byte_codec_borrows_views_and_rejects_invalid_utf8_boundaries() {
        with_parser(|cx| {
            assert_eq!(round_trip::<Bytes>(cx, "0x00FF", None), "0x00ff");
            assert_eq!(round_trip::<Bytes>(cx, "0x", None), "0x");
            for text in ["0x🦀", "0x界a", "0xé", "0x0", "0xgg"] {
                assert!(Bytes::parse(cx, text, None).is_err(), "{text}");
            }
        });
    }

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
        assert_eq!(
            ty,
            crate::Type::I32
                .as_scalar()
                .unwrap()
                .vector(4, true)
                .map(crate::VectorType::as_type)
        );
        assert_eq!(args, "v0, v1");
    }

    #[test]
    fn generic_fields_preserve_nested_arguments_and_reject_duplicates() {
        let (core, named) = split_core_and_named(
            "block1(v0, v1), [block2(), block3(v2)], mask=v3, evl=v4",
            Some(2),
        )
        .unwrap();
        assert_eq!(core, ["block1(v0, v1)", "[block2(), block3(v2)]"]);
        assert_eq!(named, [("mask", "v3"), ("evl", "v4")]);
        for text in ["v0, mask=v1, mask=v2", "v0, mask=", "v0, =v1"] {
            assert!(split_core_and_named(text, Some(1)).is_err(), "{text}");
        }
        assert_eq!(split_space(" eq  v0, v1 ").unwrap(), ("eq", "v0, v1"));
        assert!(split_space("eq").is_err());
    }

    #[test]
    fn variadic_named_tail_preserves_boundaries_and_rejects_bad_fields() {
        let (core, named) = split_core_and_named("v0, v1, mask=v2, evl=v3", None).unwrap();
        assert_eq!(core, ["v0", "v1"]);
        assert_eq!(named, [("mask", "v2"), ("evl", "v3")]);
        let (core, named) = split_core_and_named("mask=v2", None).unwrap();
        assert!(core.is_empty());
        assert_eq!(named, [("mask", "v2")]);
        let (core, named) = split_core_and_named("v0, v1", None).unwrap();
        assert_eq!(core, ["v0", "v1"]);
        assert!(named.is_empty());
        let (core, named) = split_core_and_named("", None).unwrap();
        assert!(core.is_empty() && named.is_empty());
        let (core, named) = split_core_and_named("block0(v0, v1), mask=v2", None).unwrap();
        assert_eq!(core, ["block0(v0, v1)"]);
        assert_eq!(named, [("mask", "v2")]);
        for text in [
            "v0, mask=v1, mask=v2",
            "v0, mask=",
            "v0, =v1",
            "v0, mask=v1, v2",
        ] {
            assert!(split_core_and_named(text, None).is_err(), "{text}");
        }
    }

    #[test]
    fn invocation_and_signature_atoms_handle_nested_types_and_utf8() {
        assert_eq!(
            parse_invocation("目标(v0, block1(v1)) : (i32) -> i32").unwrap(),
            ("目标", "v0, block1(v1)", " : (i32) -> i32")
        );
        let text = signature_suffix(" : (i32<scalable 4>, ptr) -> (i32, f64) ").unwrap();
        let signature = parse_signature(text).unwrap();
        assert_eq!(signature.params.len(), 2);
        assert_eq!(signature.returns, [Type::I32, Type::F64]);
        for text in ["(i32) -> i32", ":", ""] {
            assert!(signature_suffix(text).is_err(), "{text}");
        }
        for text in ["bad(i32) -> i32", "(i32)", "(i32) ->", "(i32 -> i32"] {
            assert!(parse_signature(text).is_err(), "{text}");
        }
    }

    #[test]
    fn context_atoms_share_ssa_values_and_intern_signatures() {
        let mut func = Function::new("test".into(), SigId(0), Linkage::Local);
        let block = func.layout.create_block();
        let mut ctx = ParseContext::default();
        ctx.block_map.insert("block0".into(), block);
        let mut module = ModuleData::default();
        let func_ids = HashMap::new();
        let mut parser = OperandParser {
            func: &mut func,
            ctx: &mut ctx,
            func_ids: &func_ids,
            module: &mut module,
        };
        let value = parser.value("v0").unwrap();
        let calls = parser.block_calls("[block0(v0), block0()]").unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(parser.func.dfg.block_call_args(calls[0]), &[value]);
        assert!(parser.func.dfg.block_call_args(calls[1]).is_empty());
        assert!(parser.block_calls("[]").unwrap().is_empty());
        assert!(parser.block_calls("block0()").is_err());
        assert!(parser.block_calls("[block0(),]").is_err());
        assert!(parser.block_call("block0() extra").is_err());
        let sig = parser.signature("(i32) -> i32").unwrap();
        assert_eq!(parser.signature(" (i32)->i32 ").unwrap(), sig);
        assert_eq!(parser.module.signatures[sig].params, [Type::I32]);
        assert_eq!(parser.module.signatures[sig].returns, [Type::I32]);
        assert!(parser.signature(": (i32) -> i32").is_err());
    }
}
