//! Recursive-descent parser for canonical MIR text.
//!
//! OpSpec generates instruction grammar and construction; shared token parsers
//! handle types, nested operands and forward references. Validation is explicit.

use super::lexer::{Cursor, Kind, Location};
use crate::{
    Block, BlockCall, CallConv, FuncId, Function, Linkage, MemFlags, Module, ModuleData, Opcode,
    Result, SigId, Signature, StackSlot, Type, Value, ValueDef, function::StackSlotData,
    types::ValueData,
};
use alloc::{
    format,
    string::{String, ToString},
    vec::Vec,
};
use hashbrown::HashMap;

#[derive(Debug, Clone)]
pub struct ParseError {
    pub location: Location,
    pub message: String,
}

type ParseResult<T> = core::result::Result<T, ParseError>;

impl ParseError {
    /// Add grammatical context without embedding or replacing the source position.
    pub(super) fn context(mut self, context: &str) -> Self {
        self.message = format!("{context}: {}", self.message);
        self
    }
}

impl core::fmt::Display for ParseError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "line {}, column {}: {}",
            self.location.line, self.location.column, self.message
        )
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ParseError {}

struct FunctionHeader {
    name: String,
    linkage: Linkage,
    signature: Signature,
}

/// Parser-local function slots retain identity until all declarations are read.
/// The final remap preserves source declaration order, including numeric refs.
#[derive(Default)]
struct Functions {
    names: HashMap<String, FuncId>,
    entries: Vec<FunctionSymbol>,
    order: Vec<FuncId>,
}

struct FunctionSymbol {
    defined: bool,
    location: Location,
}

impl Functions {
    fn reference(
        &mut self,
        name: &str,
        signature: SigId,
        location: Location,
        module: &mut ModuleData,
    ) -> ParseResult<FuncId> {
        if let Some(&id) = self.names.get(name) {
            if module.functions[id].signature != signature {
                return Err(location.error(format!(
                    "call signature does not match declaration of `{name}`"
                )));
            }
            return Ok(id);
        }
        let id = module.declare_function(name.into(), signature, Linkage::Local);
        self.names.insert(name.into(), id);
        self.entries.push(FunctionSymbol {
            defined: false,
            location,
        });
        Ok(id)
    }

    fn declare(
        &mut self,
        header: &FunctionHeader,
        location: Location,
        module: &mut ModuleData,
    ) -> ParseResult<FuncId> {
        if self
            .names
            .get(&header.name)
            .is_some_and(|id| self.entries[id.0 as usize].defined)
        {
            return Err(location.error(format!("duplicate function `{}`", header.name)));
        }
        let signature = module.intern_signature(header.signature.clone());
        let id = self.reference(&header.name, signature, location, module)?;
        self.entries[id.0 as usize].defined = true;
        module.functions[id].linkage = header.linkage;
        self.order.push(id);
        Ok(id)
    }

    fn finish(self, module: &mut ModuleData) -> ParseResult<()> {
        let mut map = vec![FuncId(u32::MAX); self.entries.len()];
        for (index, &id) in self.order.iter().enumerate() {
            map[id.0 as usize] = FuncId(index as u32);
        }
        for (index, entry) in self.entries.iter().enumerate() {
            if entry.defined {
                continue;
            }
            // Names take precedence over the legacy numeric spellings.
            let function = &module.functions[FuncId(index as u32)];
            let name = &function.name;
            let number = name
                .strip_prefix("func")
                .or_else(|| {
                    name.strip_prefix("FuncId(")
                        .and_then(|s| s.strip_suffix(')'))
                })
                .and_then(|s| s.parse::<usize>().ok());
            let Some(target) = number.and_then(|n| self.order.get(n)).copied() else {
                return Err(entry.location.error(format!("unknown function `{name}`")));
            };
            if function.signature != module.functions[target].signature {
                return Err(entry.location.error(format!(
                    "call signature does not match declaration of `{}`",
                    module.functions[target].name
                )));
            }
            map[index] = map[target.0 as usize];
        }
        if map
            .iter()
            .enumerate()
            .all(|(index, id)| id.0 as usize == index)
        {
            return Ok(());
        }
        let old = core::mem::take(&mut module.functions);
        let mut functions: Vec<_> = old.into_iter().map(|(_, func)| Some(func)).collect();
        for id in self.order {
            let mut func = functions[id.0 as usize]
                .take()
                .expect("unique function declaration");
            func.dfg.remap_functions(&map);
            module.functions.push(func);
        }
        Ok(())
    }
}

/// Source is consumed once; result types are explicit and forward symbols are filled in IR.
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
    /// Resolve names/result types, without validating IR contracts.
    /// Call `Module::validate` explicitly when validation is required.
    pub fn parse(&mut self, source: &str) -> Result<Module> {
        parse_module(source).map(Module::new).map_err(Into::into)
    }
}

fn parse_module(source: &str) -> ParseResult<ModuleData> {
    let mut input = Cursor::new(source);
    let mut module = ModuleData::default();
    let mut functions = Functions::default();
    let mut current: Option<FunctionParser> = None;
    input.skip_newlines();
    if input.kind() == Kind::Eof {
        return Err(input.error("empty input"));
    }
    while input.kind() != Kind::Eof {
        if is_function_header(&mut input) {
            if let Some(previous) = current.take() {
                previous.finish(&mut module)?;
            }
            let location = input.location();
            let header = parse_function_header(&mut input, &mut module)?;
            let id = functions.declare(&header, location, &mut module)?;
            current = Some(FunctionParser {
                id,
                func: Function::new(header.name, module.functions[id].signature, header.linkage),
                symbols: Symbols::default(),
                block: None,
            });
        } else if input.is("global") && input.peek_kind(1) == Kind::Word {
            if current.is_some() {
                return Err(input.error("global declaration inside function"));
            }
            let (name, ty, linkage) = parse_global(&mut input, &mut module)?;
            module.add_global(name, ty, linkage);
        } else {
            current
                .as_mut()
                .ok_or_else(|| input.error("expected global or function declaration"))?
                .statement(&mut input, &mut functions, &mut module)?;
        }
        // Every declaration and instruction has exactly one boundary check.
        input.finish()?;
        input.skip_newlines();
    }
    if let Some(current) = current {
        current.finish(&mut module)?;
    }
    functions.finish(&mut module)?;
    Ok(module)
}

struct FunctionParser {
    id: FuncId,
    func: Function,
    symbols: Symbols,
    block: Option<Block>,
}

impl FunctionParser {
    fn statement(
        &mut self,
        input: &mut Cursor<'_>,
        functions: &mut Functions,
        module: &mut ModuleData,
    ) -> ParseResult<()> {
        let name = input.text();
        if name.starts_with("block") && input.peek_kind(1) == Kind::LParen {
            self.block = Some(declare_block(
                input,
                &mut self.func,
                &mut self.symbols,
                module,
            )?);
        } else if name.starts_with("ss")
            && input.peek_kind(1) == Kind::Colon
            && input.peek_is(2, "size")
        {
            parse_stack_slot(input, &mut self.func)?;
        } else {
            let block = self
                .block
                .ok_or_else(|| input.error("instruction outside a basic block"))?;
            OperandParser {
                func: &mut self.func,
                symbols: &mut self.symbols,
                functions,
                module,
            }
            .instruction(input, block)?;
        }
        Ok(())
    }

    fn finish(self, module: &mut ModuleData) -> ParseResult<()> {
        self.symbols.finish()?;
        module.functions[self.id] = self.func;
        Ok(())
    }
}

#[derive(Default)]
struct Symbols {
    values: HashMap<String, Value>,
    numbered: HashMap<u32, Value>,
    blocks: HashMap<String, (Block, Location)>,
    block_defs: hashbrown::HashSet<Block>,
    next_value: u32,
    definitions: HashMap<Value, Definition>,
}

struct Definition {
    name: Option<String>,
    location: Location,
}

impl Symbols {
    fn block(&mut self, name: &str, func: &mut Function, location: Location) -> ParseResult<Block> {
        if let Some(&(block, _)) = self.blocks.get(name) {
            return Ok(block);
        }
        let id = name
            .strip_prefix("block")
            .and_then(|s| s.parse::<u32>().ok())
            .ok_or_else(|| location.error(format!("unknown block `{name}`")))?;
        while func.layout.blocks.len() <= id as usize {
            func.layout.create_block();
        }
        let block = Block(id);
        self.blocks.insert(name.into(), (block, location));
        Ok(block)
    }

    // References reserve the final Value ID. Definitions fill that same slot,
    // so resolving a forward reference never rewrites its uses.
    fn reference(&mut self, name: &str, func: &mut Function, location: Location) -> Value {
        if let Some(&value) = self.values.get(name) {
            return value;
        }
        let value = if let Some(index) = parse_value_idx(name) {
            if let Some(&value) = self.numbered.get(&index) {
                value
            } else {
                // A symbolic spelling may already occupy the preferred number.
                // The spelling identifies a value, not a preallocated DFG slot.
                let value = if self.definitions.contains_key(&Value(index)) {
                    Value(self.next_value)
                } else {
                    Value(index)
                };
                self.numbered.insert(index, value);
                value
            }
        } else {
            Value(self.next_value)
        };
        // Reserved slots are not definitions. Their placeholder def must not be
        // interpreted until parsing succeeds and all symbols are resolved.
        while func.dfg.values.len() <= value.0 as usize {
            func.dfg.values.push(ValueData {
                ty: Type::INVALID,
                def: ValueDef::Param(Block(0)),
            });
        }
        set_value_name(value, name, func);
        self.values.insert(name.to_string(), value);
        self.definitions.entry(value).or_insert(Definition {
            name: None,
            location,
        });
        self.next_value = self.next_value.max(value.0 + 1);
        value
    }

    /// Claim a definition's stable ID before attaching its owner in the DFG.
    fn define(
        &mut self,
        name: &str,
        func: &mut Function,
        location: Location,
    ) -> ParseResult<Value> {
        let value = self.reference(name, func, location);
        let definition = self.definitions.get_mut(&value).expect("reserved value");
        if let Some(previous) = &definition.name {
            return Err(location.error(format!(
                "SSA value `{name}` aliases already-defined `{previous}`"
            )));
        }
        definition.name = Some(name.to_string());
        Ok(value)
    }

    fn finish(&self) -> ParseResult<()> {
        for (name, (block, location)) in &self.blocks {
            if !self.block_defs.contains(block) {
                return Err(location.error(format!("unknown block `{name}`")));
            }
        }
        for (name, value) in &self.values {
            if self.definitions[value].name.is_none() {
                return Err(self.definitions[value]
                    .location
                    .error(format!("undefined SSA value `{name}`")));
            }
        }
        Ok(())
    }
}

fn declare_block(
    input: &mut Cursor<'_>,
    func: &mut Function,
    symbols: &mut Symbols,
    module: &mut ModuleData,
) -> ParseResult<Block> {
    let location = input.location();
    let name = input.word()?;
    let block_id = name
        .strip_prefix("block")
        .and_then(|s| s.parse::<u32>().ok())
        .ok_or_else(|| location.error(format!("invalid block name `{name}`")))?;
    input.expect(Kind::LParen)?;
    let block = symbols.block(&format!("block{block_id}"), func, location)?;
    if !symbols.block_defs.insert(block) {
        return Err(location.error(format!("duplicate block{block_id}")));
    }
    func.layout.append_block(block);
    if func.entry_block.is_none() {
        func.entry_block = Some(block);
    }
    if !input.eat(Kind::RParen) {
        loop {
            let param = parse_typed_name(input, module)?;
            let value = symbols.define(param.name, func, param.location)?;
            func.dfg.values[value] = ValueData {
                ty: param.ty,
                def: ValueDef::Param(block),
            };
            func.layout.blocks[block].params.push(value);
            if !input.eat(Kind::Comma) {
                break;
            }
        }
        input.expect(Kind::RParen)?;
    }
    input.expect(Kind::Colon)?;
    Ok(block)
}

pub(super) struct OperandParser<'a> {
    func: &'a mut Function,
    symbols: &'a mut Symbols,
    functions: &'a mut Functions,
    module: &'a mut ModuleData,
}

impl OperandParser<'_> {
    fn instruction(&mut self, input: &mut Cursor<'_>, block: Block) -> ParseResult<()> {
        let results = self.parse_results(input)?;
        let (opcode, flags) = parse_instruction_header(input)?;
        let data = self.parse(opcode, flags, input)?;
        let inst = self.func.edit().append_inst(block, data, &[]);
        self.func.dfg.bind_results(inst, &results);
        Ok(())
    }

    fn parse_results(&mut self, input: &mut Cursor<'_>) -> ParseResult<Vec<(Value, Type)>> {
        let mut results = Vec::new();
        let multiple = input.eat(Kind::LParen);
        if !multiple
            && !(input.kind() == Kind::Word
                && matches!(input.peek_kind(1), Kind::Colon | Kind::Equal))
        {
            return Ok(results);
        }
        loop {
            let result = parse_typed_name(input, self.module)?;
            let value = self
                .symbols
                .define(result.name, self.func, result.location)?;
            results.push((value, result.ty));
            if !multiple || !input.eat(Kind::Comma) {
                break;
            }
        }
        if multiple {
            input.expect(Kind::RParen)?;
        }
        input.expect(Kind::Equal)?;
        Ok(results)
    }

    pub(super) fn value(&mut self, input: &mut Cursor<'_>) -> ParseResult<Value> {
        let location = input.location();
        let name = input.word().map_err(|e| e.context("invalid SSA value"))?;
        Ok(self.symbols.reference(name, self.func, location))
    }

    pub(super) fn values(&mut self, input: &mut Cursor<'_>) -> ParseResult<Vec<Value>> {
        let mut values = Vec::new();
        if matches!(input.kind(), Kind::Eof | Kind::Newline | Kind::RParen) || input.named() {
            return Ok(values);
        }
        loop {
            values.push(self.value(input)?);
            if input.kind() != Kind::Comma {
                break;
            }
            if input.named_at(1) {
                break;
            }
            input.advance();
        }
        Ok(values)
    }

    pub(super) fn block_call(&mut self, input: &mut Cursor<'_>) -> ParseResult<BlockCall> {
        let location = input.location();
        let name = input.word()?;
        let block = self.symbols.block(name, self.func, location)?;
        input.expect(Kind::LParen)?;
        let values = self.values(input)?;
        input.expect(Kind::RParen)?;
        Ok(BlockCall::new(block, &values))
    }

    pub(super) fn block_calls(&mut self, input: &mut Cursor<'_>) -> ParseResult<Vec<BlockCall>> {
        input.expect(Kind::LBracket)?;
        let mut calls = Vec::new();
        if !input.eat(Kind::RBracket) {
            loop {
                calls.push(self.block_call(input)?);
                if !input.eat(Kind::Comma) {
                    break;
                }
            }
            input.expect(Kind::RBracket)?;
        }
        Ok(calls)
    }

    pub(super) fn func_ref(&mut self, input: &mut Cursor<'_>) -> ParseResult<FuncId> {
        let name = parse_function_name(input, false)?;
        input.expect(Kind::Colon)?;
        let signature = self.signature(input)?;
        self.function_reference(name, signature)
    }

    /// Register only after the complete signature has been parsed.
    pub(super) fn function_reference(
        &mut self,
        name: FunctionName,
        signature: SigId,
    ) -> ParseResult<FuncId> {
        self.functions
            .reference(&name.name, signature, name.location, self.module)
    }

    pub(super) fn signature(&mut self, input: &mut Cursor<'_>) -> ParseResult<SigId> {
        let sig = parse_signature(input, self.module)?;
        Ok(self.module.intern_signature(sig))
    }
}

pub(super) struct FunctionName {
    name: String,
    location: Location,
}

pub(super) fn parse_function_name(
    input: &mut Cursor<'_>,
    invoke: bool,
) -> ParseResult<FunctionName> {
    let location = input.location();
    let name = input.word()?;
    let name = if name == "FuncId"
        && input.kind() == Kind::LParen
        && input.peek_kind(2) == Kind::RParen
        && (!invoke || input.peek_kind(3) != Kind::Colon)
    {
        input.advance();
        let id = input.atom(|text| {
            text.parse::<u32>()
                .map_err(|_| "invalid function ID".into())
        })?;
        input.expect(Kind::RParen)?;
        format!("FuncId({id})")
    } else {
        name.into()
    };
    Ok(FunctionName { name, location })
}

struct TypedName<'a> {
    name: &'a str,
    ty: Type,
    location: Location,
}

fn parse_typed_name<'a>(
    input: &mut Cursor<'a>,
    module: &mut ModuleData,
) -> ParseResult<TypedName<'a>> {
    let location = input.location();
    let name = input.word()?;
    input.expect(Kind::Colon)?;
    let ty = parse_type(input, module)?;
    Ok(TypedName { name, ty, location })
}

fn parse_instruction_header(input: &mut Cursor<'_>) -> ParseResult<(Opcode, MemFlags)> {
    let location = input.location();
    let (opcode, suffix) = input.atom(parse_opcode)?;
    let mut flags = MemFlags::new();
    for part in suffix.strip_prefix('.').unwrap_or(suffix).split('.') {
        if part.is_empty() {
            if !suffix.is_empty() {
                return Err(location.error("empty opcode suffix"));
            }
        } else if part == "volatile" {
            flags = flags.with_volatile(true);
        } else if let Some(value) = part.strip_prefix("align") {
            flags = flags.with_alignment(parse_alignment(value, part, location)?);
        } else {
            return Err(location.error(format!("unknown opcode suffix `{part}`")));
        }
    }
    Ok((opcode, flags))
}

fn parse_alignment(value: &str, suffix: &str, location: Location) -> ParseResult<u32> {
    let alignment = value
        .parse::<u32>()
        .map_err(|_| location.error(format!("invalid alignment `{suffix}`")))?;
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(location.error(format!(
            "alignment must be a non-zero power of two: {alignment}"
        )));
    }
    Ok(alignment)
}

fn parse_opcode(word: &str) -> core::result::Result<(Opcode, &str), String> {
    // Mnemonics and suffixes are dotted words. Prefer the longest mnemonic,
    // including any dots belonging to the opcode itself.
    let mut end = word.len();
    let opcode = loop {
        if let Some(opcode) = Opcode::from_mnemonic(&word[..end]) {
            break opcode;
        }
        end = word[..end]
            .rfind('.')
            .ok_or_else(|| format!("unknown opcode `{word}`"))?;
    };
    Ok((opcode, &word[end..]))
}

fn parse_type(input: &mut Cursor<'_>, module: &mut ModuleData) -> ParseResult<Type> {
    let location = input.location();
    let name = input.word()?;
    if !input.eat(Kind::Less) {
        return Type::from_name(name)
            .ok_or_else(|| location.error(format!("unknown type `{name}`")));
    }
    if matches!(name, "owned" | "local" | "shared") {
        let signature = parse_signature(input, module)?;
        input.expect(Kind::Greater)?;
        let id = module.intern_signature(signature);
        let kind = match name {
            "owned" => crate::CallableKind::Owned,
            "local" => crate::CallableKind::Local,
            _ => crate::CallableKind::Shared,
        };
        return Ok(Type::callable(id, kind));
    }
    let base = if name == "mask" {
        Some(Type::BOOL)
    } else {
        Type::from_name(name)
    };
    let scalable = input.is("scalable");
    if scalable {
        input.advance();
    }
    let lanes = input.atom(|text| {
        text.parse::<u16>()
            .map_err(|_| "invalid vector lane count".into())
    })?;
    input.expect(Kind::Greater)?;
    base.and_then(Type::as_scalar)
        .and_then(|s| s.vector(lanes, scalable))
        .map(crate::VectorType::as_type)
        .ok_or_else(|| location.error(format!("invalid vector type `{name}`")))
}

fn parse_types(input: &mut Cursor<'_>, module: &mut ModuleData) -> ParseResult<Vec<Type>> {
    input.expect(Kind::LParen)?;
    let mut types = Vec::new();
    if !input.eat(Kind::RParen) {
        loop {
            types.push(parse_type(input, module)?);
            if !input.eat(Kind::Comma) {
                break;
            }
        }
        input.expect(Kind::RParen)?;
    }
    Ok(types)
}

fn parse_function_returns(
    input: &mut Cursor<'_>,
    module: &mut ModuleData,
) -> ParseResult<Vec<Type>> {
    if input.is("void") {
        input.advance();
        return Ok(Vec::new());
    }
    if input.kind() == Kind::LParen {
        return parse_types(input, module);
    }
    let mut types = Vec::new();
    loop {
        types.push(parse_type(input, module)?);
        if !input.eat(Kind::Comma) {
            break;
        }
    }
    Ok(types)
}

fn parse_signature(input: &mut Cursor<'_>, module: &mut ModuleData) -> ParseResult<Signature> {
    let params = parse_types(input, module)?;
    input.expect(Kind::Arrow)?;
    let returns = if input.is("void") {
        input.advance();
        Vec::new()
    } else if input.kind() == Kind::LParen {
        parse_types(input, module)?
    } else {
        alloc::vec![parse_type(input, module)?]
    };
    Ok(Signature::new(params, returns, CallConv::SystemV))
}

fn parse_linkage(input: &mut Cursor<'_>) -> ParseResult<Linkage> {
    input.atom(|name| {
        Linkage::from_mnemonic(name).ok_or_else(|| format!("unknown linkage `{name}`"))
    })
}

fn is_function_header(input: &mut Cursor<'_>) -> bool {
    input.kind() == Kind::Word
        && Linkage::from_mnemonic(input.text()).is_some()
        && input.peek_is(1, "function")
}

fn parse_function_header(
    input: &mut Cursor<'_>,
    module: &mut ModuleData,
) -> ParseResult<FunctionHeader> {
    let linkage = parse_linkage(input)?;
    input.keyword("function")?;
    let name = input.word()?.to_string();
    let params = parse_types(input, module)?;
    let returns = if input.eat(Kind::Arrow) {
        parse_function_returns(input, module)?
    } else {
        Vec::new()
    };
    Ok(FunctionHeader {
        name,
        linkage,
        signature: Signature::new(params, returns, CallConv::SystemV),
    })
}

fn parse_global(
    input: &mut Cursor<'_>,
    module: &mut ModuleData,
) -> ParseResult<(String, Type, Linkage)> {
    input.keyword("global")?;
    let name = input.word()?.to_string();
    input.expect(Kind::Colon)?;
    let ty = parse_type(input, module)?;
    input.expect(Kind::LParen)?;
    let linkage = parse_linkage(input)?;
    input.expect(Kind::RParen)?;
    Ok((name, ty, linkage))
}

fn parse_stack_slot(input: &mut Cursor<'_>, func: &mut Function) -> ParseResult<()> {
    let slot = input.atom(parse_stack_slot_ref)?;
    input.expect(Kind::Colon)?;
    input.keyword("size")?;
    let size = input.atom(|text| {
        text.parse::<u32>()
            .map_err(|_| "invalid stack slot size".into())
    })?;
    while func.stack_slots.len() <= slot.0 as usize {
        func.stack_slots.push(StackSlotData { size: 0 });
    }
    func.stack_slots[slot] = StackSlotData { size };
    Ok(())
}

pub(super) fn parse_stack_slot_ref(text: &str) -> core::result::Result<StackSlot, String> {
    let id = text
        .strip_prefix("ss")
        .ok_or_else(|| format!("expected stack slot, found `{text}`"))?
        .parse::<u32>()
        .map_err(|_| format!("invalid stack slot `{text}`"))?;
    Ok(StackSlot(id))
}

fn parse_value_idx(name: &str) -> Option<u32> {
    name.strip_prefix('v')
        .and_then(|digits| digits.parse().ok())
        .or_else(|| {
            name.rfind(".v")
                .and_then(|index| name[index + 2..].parse().ok())
        })
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

include!(concat!(env!("OUT_DIR"), "/text_parser.rs"));

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::atom::{AtomCodec, Bytes, Decimal, FloatBits, IntegerBits};
    use crate::text::printer::InstPrinter;
    use core::{borrow::Borrow, fmt::Debug};

    fn with_parser(test: impl FnOnce(&mut OperandParser<'_>)) {
        let mut func = Function::new("test".into(), SigId(0), Linkage::Local);
        let mut symbols = Symbols::default();
        let mut module = ModuleData::default();
        let mut functions = Functions::default();
        test(&mut OperandParser {
            func: &mut func,
            symbols: &mut symbols,
            functions: &mut functions,
            module: &mut module,
        });
    }

    fn parse<C: AtomCodec>(cx: &mut OperandParser<'_>, text: &str) -> ParseResult<C::Owned> {
        let mut input = Cursor::new(text);
        let value = C::parse(cx, &mut input)?;
        input.finish()?;
        Ok(value)
    }

    fn round_trip<C: AtomCodec>(cx: &mut OperandParser<'_>, text: &str, ty: Option<Type>) -> String
    where
        C::Owned: Debug + PartialEq + for<'a> Borrow<C::View<'a>>,
    {
        let value = parse::<C>(cx, text).unwrap();
        let mut printed = String::new();
        C::print(
            &InstPrinter::new(&cx.func.dfg, None),
            &mut printed,
            value.borrow(),
            ty,
        )
        .unwrap();
        assert_eq!(parse::<C>(cx, &printed).unwrap(), value);
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
            assert!(parse::<Decimal<u64>>(cx, "-1").is_err());
            assert_eq!(
                round_trip::<Decimal<i32>>(cx, "-2147483648", None),
                "-2147483648"
            );
            assert_eq!(round_trip::<Decimal<u8>>(cx, "255", None), "255");
            assert!(parse::<Decimal<u8>>(cx, "256").is_err());
            assert_eq!(round_trip::<bool>(cx, "true", None), "true");
            assert!(parse::<bool>(cx, "1").is_err());
            assert_eq!(round_trip::<crate::IntCC>(cx, "eq", None), "eq");
            assert_eq!(round_trip::<crate::FloatCC>(cx, "eq", None), "eq");
            assert_eq!(round_trip::<StackSlot>(cx, "ss7", None), "ss7");
            assert_eq!(round_trip::<Value>(cx, "v0", None), "v0");
        });
    }

    #[test]
    fn float_codec_decodes_raw_bits_and_formats_by_result_type() {
        with_parser(|cx| {
            for (ty, bits) in [
                (Type::F32, "0x7fc00001"),
                (Type::F32, "0x80000000"),
                (Type::F64, "0x7ff8000000000042"),
            ] {
                assert_eq!(round_trip::<FloatBits>(cx, bits, Some(ty)), bits);
            }
            for ty in [None, Some(Type::I32)] {
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
            assert_eq!(parse::<FloatBits>(cx, "0x100000000").unwrap(), 0x100000000);
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
                assert!(parse::<Bytes>(cx, text).is_err(), "{text}");
            }
        });
    }

    #[test]
    fn function_reference_is_registered_only_after_its_signature() {
        with_parser(|cx| {
            for text in ["later()", "later() : () ->"] {
                let mut input = Cursor::new(text);
                assert!(
                    cx.parse(Opcode::Call, MemFlags::empty(), &mut input)
                        .is_err()
                );
                assert!(cx.module.functions.is_empty());
                assert!(cx.functions.entries.is_empty());
            }
            let mut input = Cursor::new("later() : () -> i32");
            cx.parse(Opcode::Call, MemFlags::empty(), &mut input)
                .unwrap();
            assert_eq!(cx.module.functions.len(), 1);
            let function = &cx.module.functions[FuncId(0)];
            assert_eq!(
                cx.module.signatures[function.signature].returns,
                [Type::I32]
            );
        });
    }

    #[test]
    fn parse_errors_preserve_locations_separately_from_context() {
        for (source, line, column, message) in [
            (
                "local function bad()->i32\nblock0():\n  v0:i32=iconst nope",
                3,
                17,
                "operand `value`: invalid integer constant",
            ),
            (
                "local function bad()->void\nblock0():\n  jump block7()",
                3,
                8,
                "unknown block",
            ),
            (
                "local function bad()->i32\nblock0():\n  return missing",
                3,
                10,
                "undefined SSA value",
            ),
        ] {
            let crate::Error::Parse(error) = ModuleParser::new().parse(source).unwrap_err() else {
                panic!("expected structured parse error");
            };
            assert_eq!(error.location, crate::text::Location { line, column });
            assert!(error.message.contains(message), "{error}");
            assert!(!error.message.contains("line "), "{error}");
            assert_eq!(error.to_string().matches("line ").count(), 1);
        }
    }

    #[test]
    fn line_endings_and_final_eof_preserve_source_ranges() {
        // File tests use LF and a final newline; exercise the other physical
        // encodings through the complete parser, printer and validator.
        let source = "local function first()->void\nblock0():\n  call second() : () -> void// comment\n  return\nimport function second()->void";
        let mut expected = None;
        for newline in ["\n", "\r\n"] {
            for ending in ["", newline] {
                let text = format!("{}{ending}", source.replace('\n', newline));
                let module = ModuleParser::new().parse(&text).unwrap();
                module.validate().unwrap();
                let printed = module.to_string();
                if let Some(expected) = &expected {
                    assert_eq!(&printed, expected);
                } else {
                    expected = Some(printed);
                }
            }
        }
    }

    #[test]
    fn result_declaration_accepts_scalable_type() {
        with_parser(|cx| {
            let mut input = Cursor::new("sum: i32<scalable 4> = iadd v0, v1");
            let results = cx.parse_results(&mut input).unwrap();
            let (opcode, _) = parse_instruction_header(&mut input).unwrap();
            assert_eq!(opcode, Opcode::IAdd);
            assert_eq!(
                Some(results[0].1),
                crate::Type::I32
                    .as_scalar()
                    .unwrap()
                    .vector(4, true)
                    .map(crate::VectorType::as_type)
            );
            assert_eq!(input.word().unwrap(), "v0");
            input.expect(Kind::Comma).unwrap();
            assert_eq!(input.word().unwrap(), "v1");
            input.finish().unwrap();
        });
    }

    #[test]
    fn context_atoms_share_ssa_values_and_intern_signatures() {
        let mut func = Function::new("test".into(), SigId(0), Linkage::Local);
        let block = func.layout.create_block();
        let mut symbols = Symbols::default();
        symbols
            .blocks
            .insert("block0".into(), (block, Location { line: 1, column: 1 }));
        let mut module = ModuleData::default();
        let mut functions = Functions::default();
        let mut parser = OperandParser {
            func: &mut func,
            symbols: &mut symbols,
            functions: &mut functions,
            module: &mut module,
        };
        let value = parser.value(&mut Cursor::new("v0")).unwrap();
        let calls = parser
            .block_calls(&mut Cursor::new("[block0(v0), block0()]"))
            .unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].args.as_slice(), &[value]);
        assert!(calls[1].args.is_empty());
        assert!(
            parser
                .block_calls(&mut Cursor::new("[]"))
                .unwrap()
                .is_empty()
        );
        assert!(parser.block_calls(&mut Cursor::new("block0()")).is_err());
        assert!(parser.block_calls(&mut Cursor::new("[block0(),]")).is_err());
        let mut input = Cursor::new("block0() extra");
        parser.block_call(&mut input).unwrap();
        assert!(input.finish().is_err());
        let sig = parser.signature(&mut Cursor::new("(i32) -> i32")).unwrap();
        assert_eq!(
            parser.signature(&mut Cursor::new(" (i32)->i32 ")).unwrap(),
            sig
        );
        assert_eq!(parser.module.signatures[sig].params, [Type::I32]);
        assert_eq!(parser.module.signatures[sig].returns, [Type::I32]);
        assert!(
            parser
                .signature(&mut Cursor::new(": (i32) -> i32"))
                .is_err()
        );
    }
}
