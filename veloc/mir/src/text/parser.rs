//! Recursive-descent parser for canonical MIR text.
//!
//! OpSpec generates instruction grammar and construction; shared token parsers
//! handle types, nested operands and forward references. Validation is explicit.

use super::lexer::{Cursor, Kind};
use crate::{
    Block, BlockCall, CallConv, FuncId, Function, InstDraft, Linkage, MemFlags, Module, ModuleData,
    Opcode, Result, SigId, Signature, StackSlot, Type, Value, ValueDef, function::StackSlotData,
    types::ValueData,
};
use alloc::{
    format,
    string::{String, ToString},
    vec::Vec,
};
use core::ops::Range;
use hashbrown::HashMap;

#[derive(Debug, Clone)]
pub struct ParseError(pub String);

type ParseResult<T> = core::result::Result<T, ParseError>;

impl core::fmt::Display for ParseError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.0)
    }
}

struct FunctionHeader {
    name: String,
    linkage: Linkage,
    signature: Signature,
}

struct FunctionSource {
    header: FunctionHeader,
    body: Range<usize>,
}

/// Function symbols are declared before bodies, so calls may refer forwards.
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
    if source.trim().is_empty() {
        return parse_err("empty input");
    }
    let mut module = ModuleData::default();
    let sources = parse_declarations(source, &mut module)?;

    // All function IDs must exist before resolving any call operands.
    let mut func_ids = HashMap::new();
    for source in &sources {
        if func_ids.contains_key(&source.header.name) {
            return parse_err(format!("duplicate function `{}`", source.header.name));
        }
        let sig = module.intern_signature(source.header.signature.clone());
        let id = module.declare_function(source.header.name.clone(), sig, source.header.linkage);
        func_ids.insert(source.header.name.clone(), id);
    }
    for function in sources {
        let id = func_ids[&function.header.name];
        let sig = module.functions[id].signature;
        let func = parse_function_body(
            function.header,
            sig,
            Cursor::range(source, function.body),
            &func_ids,
            &mut module,
        )?;
        module.functions[id] = func;
    }
    Ok(module)
}

fn parse_declarations(source: &str, module: &mut ModuleData) -> ParseResult<Vec<FunctionSource>> {
    let mut input = Cursor::new(source);
    let mut sources = Vec::<FunctionSource>::new();
    while let Some(mut line) = input.statement() {
        if is_function_header(&line) {
            if let Some(previous) = sources.last_mut() {
                previous.body.end = line.offset();
            }
            let header = parse_line(&mut line, parse_function_header)?;
            sources.push(FunctionSource {
                header,
                body: input.offset()..source.len(),
            });
        } else if line.is("global") {
            if !sources.is_empty() {
                return Err(line.locate(ParseError("global declaration inside function".into())));
            }
            let (name, ty, linkage) = parse_line(&mut line, parse_global)?;
            module.add_global(name, ty, linkage);
        } else if sources.is_empty() {
            return Err(line.locate(ParseError("expected global or function declaration".into())));
        }
    }
    if sources.is_empty() && module.globals.is_empty() {
        return parse_err("module contains no declarations");
    }

    Ok(sources)
}

/// Complete one declaration before attaching its source location to an error.
fn parse_line<T>(
    input: &mut Cursor<'_>,
    parse: impl FnOnce(&mut Cursor<'_>) -> ParseResult<T>,
) -> ParseResult<T> {
    let result = parse(input).and_then(|value| {
        input.finish()?;
        Ok(value)
    });
    result.map_err(|error| input.locate(error))
}

#[derive(Default)]
struct Symbols {
    values: HashMap<String, Value>,
    blocks: HashMap<String, Block>,
    next_value: u32,
    definitions: HashMap<Value, String>,
}

impl Symbols {
    // References reserve the final Value ID. Definitions fill that same slot,
    // so resolving a forward reference never rewrites its uses.
    fn reference(&mut self, name: &str, func: &mut Function) -> Value {
        if let Some(&value) = self.values.get(name) {
            return value;
        }
        let index = parse_value_idx(name).unwrap_or(self.next_value);
        let value = Value(index);
        ensure_value(value, func);
        set_value_name(value, name, func);
        self.values.insert(name.to_string(), value);
        self.next_value = self.next_value.max(index + 1);
        value
    }

    fn define(
        &mut self,
        name: &str,
        func: &mut Function,
        ty: Type,
        def: ValueDef,
    ) -> ParseResult<Value> {
        let value = self.reference(name, func);
        if let Some(previous) = self.definitions.get(&value) {
            return Err(ParseError(format!(
                "SSA value `{name}` aliases already-defined `{previous}`"
            )));
        }
        func.dfg.values[value] = ValueData { ty, def };
        self.definitions.insert(value, name.to_string());
        Ok(value)
    }

    fn finish(&self) -> ParseResult<()> {
        for (name, value) in &self.values {
            if !self.definitions.contains_key(value) {
                return parse_err(format!("undefined SSA value `{name}`"));
            }
        }
        Ok(())
    }
}

fn parse_function_body(
    header: FunctionHeader,
    sig_id: SigId,
    mut body: Cursor<'_>,
    func_ids: &HashMap<String, FuncId>,
    module: &mut ModuleData,
) -> ParseResult<Function> {
    let mut func = Function::new(header.name, sig_id, header.linkage);
    let mut symbols = Symbols::default();
    // Predeclare blocks/parameters and remember only borrowed instruction
    // source ranges. No copied lines or second parse of block headers.
    let mut instructions = Vec::new();
    let mut current = None;
    while let Some(mut line) = body.statement() {
        let mut look = line.clone();
        let name = look.text();
        look.advance();
        if name.starts_with("block") && look.kind() == Kind::LParen {
            current = Some(parse_line(&mut line, |input| {
                declare_block(input, &mut func, &mut symbols)
            })?);
        } else if name.starts_with("ss") && look.kind() == Kind::Colon {
            parse_line(&mut line, |input| parse_stack_slot(input, &mut func))?;
        } else {
            let block = current.ok_or_else(|| {
                line.locate(ParseError("instruction outside a basic block".into()))
            })?;
            instructions.push((block, line.remaining()));
        }
    }
    let mut parser = OperandParser {
        func: &mut func,
        symbols: &mut symbols,
        func_ids,
        module,
    };
    for (block, range) in instructions {
        let mut input = body.slice(range);
        parser
            .instruction(&mut input, block)
            .map_err(|e| input.locate(e))?;
    }
    symbols.finish()?;
    for &block in &func.layout.block_order {
        func.layout.blocks[block].is_sealed = true;
    }
    Ok(func)
}

fn declare_block(
    input: &mut Cursor<'_>,
    func: &mut Function,
    symbols: &mut Symbols,
) -> ParseResult<Block> {
    let (block_id, params) = parse_block_header(input)?;
    while func.layout.blocks.len() <= block_id as usize {
        func.layout.create_block();
    }
    let block = Block(block_id);
    if symbols
        .blocks
        .insert(format!("block{block_id}"), block)
        .is_some()
    {
        return Err(ParseError(format!("duplicate block{block_id}")));
    }
    func.layout.append_block(block);
    if func.entry_block.is_none() {
        func.entry_block = Some(block);
    }
    for (name, ty) in params {
        let value = symbols.define(&name, func, ty, ValueDef::Param(block))?;
        func.layout.blocks[block].params.push(value);
    }
    Ok(block)
}

fn resolve_result_types(
    data: &InstDraft,
    hint: Option<Type>,
    func: &Function,
    module: &ModuleData,
) -> ParseResult<smallvec::SmallVec<[Type; 2]>> {
    let spec = data.opcode().spec();
    // Resolve only information needed to create result values. Type contracts
    // (including forward-referenced operands) belong to the validator.
    let results = data
        .result_types(&func.dfg, module, hint.as_slice())
        .map_err(|error| ParseError(format!("`{}`: {error}", spec.mnemonic)))?;

    if let Some(hint) = hint {
        if let Some(first) = results.first() {
            if first.is_valid() && *first != hint {
                return Err(ParseError(format!(
                    "result annotation `{hint}` conflicts with inferred type `{first}`"
                )));
            }
        } else {
            return Err(ParseError(format!(
                "`{}` does not produce an annotatable result",
                spec.mnemonic
            )));
        }
    }
    Ok(results)
}

pub(super) struct OperandParser<'a> {
    func: &'a mut Function,
    symbols: &'a mut Symbols,
    func_ids: &'a HashMap<String, FuncId>,
    module: &'a mut ModuleData,
}

impl OperandParser<'_> {
    fn instruction(&mut self, input: &mut Cursor<'_>, block: Block) -> ParseResult<()> {
        let result_names = parse_result_names(input)?;
        let (opcode, ty_hint, flags) = parse_instruction_header(input)?;
        let data = self.parse(opcode, ty_hint, flags, input)?;
        let result_types = resolve_result_types(&data, ty_hint, self.func, self.module)?;
        if result_names.len() != result_types.len() {
            return Err(ParseError(format!(
                "`{}` defines {} result name(s), but its type scheme produces {}",
                opcode.spec().mnemonic,
                result_names.len(),
                result_types.len()
            )));
        }
        let inst = self.func.edit().append_inst(block, data, &[]);
        if !result_names.is_empty() {
            let values = result_names
                .iter()
                .zip(result_types)
                .map(|(name, ty)| {
                    self.symbols
                        .define(name, self.func, ty, ValueDef::Inst(inst))
                })
                .collect::<ParseResult<Vec<_>>>()?;
            let list = self.func.dfg.make_value_list(&values);
            self.func.dfg.inst_results[inst] = list;
        }
        Ok(())
    }

    pub(super) fn value(&mut self, input: &mut Cursor<'_>) -> ParseResult<Value> {
        let name = input
            .word()
            .map_err(|e| ParseError(format!("invalid SSA value: {e}")))?;
        Ok(self.symbols.reference(name, self.func))
    }

    pub(super) fn values(&mut self, input: &mut Cursor<'_>) -> ParseResult<Vec<Value>> {
        let mut values = Vec::new();
        if matches!(input.kind(), Kind::Eof | Kind::RParen) || input.named() {
            return Ok(values);
        }
        loop {
            values.push(self.value(input)?);
            if input.kind() != Kind::Comma {
                break;
            }
            let mut next = input.clone();
            next.advance();
            if next.named() {
                break;
            }
            input.advance();
        }
        Ok(values)
    }

    pub(super) fn block_call(&mut self, input: &mut Cursor<'_>) -> ParseResult<BlockCall> {
        let name = input.word()?;
        let block = self
            .symbols
            .blocks
            .get(name)
            .copied()
            .ok_or_else(|| ParseError(format!("unknown block `{name}`")))?;
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

    pub(super) fn func_ref(&self, input: &mut Cursor<'_>) -> ParseResult<FuncId> {
        let name = input.word()?;
        if let Some(&id) = self.func_ids.get(name) {
            return Ok(id);
        }
        if name == "FuncId" {
            input.expect(Kind::LParen)?;
            let id = input
                .word()?
                .parse()
                .map_err(|_| ParseError("invalid function ID".into()))?;
            input.expect(Kind::RParen)?;
            return Ok(FuncId(id));
        }
        name.strip_prefix("func")
            .and_then(|s| s.parse().ok())
            .map(FuncId)
            .ok_or_else(|| ParseError(format!("unknown function `{name}`")))
    }

    /// A textual function reference declares a signature, not a second
    /// per-instruction signature. Check symbol consistency here; argument and
    /// result value types are still checked by the explicit validator.
    fn function_signature(&self, callee: FuncId, input: &mut Cursor<'_>) -> ParseResult<()> {
        let signature = parse_signature(input)?;
        let function = self
            .module
            .functions
            .get(callee)
            .ok_or_else(|| ParseError("unknown function".into()))?;
        let declared = self
            .module
            .signatures
            .get(function.signature)
            .ok_or_else(|| ParseError("unknown function signature".into()))?;
        if signature != *declared {
            return parse_err(format!(
                "call signature does not match declaration of `{}`",
                function.name
            ));
        }
        Ok(())
    }

    pub(super) fn signature(&mut self, input: &mut Cursor<'_>) -> ParseResult<SigId> {
        Ok(self.module.intern_signature(parse_signature(input)?))
    }
}

fn parse_result_names(input: &mut Cursor<'_>) -> ParseResult<Vec<String>> {
    let mut names = Vec::new();
    if input.eat(Kind::LParen) {
        loop {
            names.push(input.word()?.to_string());
            if !input.eat(Kind::Comma) {
                break;
            }
        }
        input.expect(Kind::RParen)?;
        input.expect(Kind::Equal)?;
    } else if input.named() {
        names.push(input.word()?.to_string());
        input.expect(Kind::Equal)?;
    }
    Ok(names)
}

fn parse_instruction_header(
    input: &mut Cursor<'_>,
) -> ParseResult<(Opcode, Option<Type>, MemFlags)> {
    let word = input.word()?;
    let (opcode, mut suffix) = parse_opcode(word)?;
    let mut ty = None;
    let mut flags = MemFlags::new();
    loop {
        let mut parts = suffix
            .strip_prefix('.')
            .unwrap_or(suffix)
            .split('.')
            .peekable();
        while let Some(part) = parts.next() {
            if part.is_empty() {
                if !suffix.is_empty() {
                    return Err(ParseError("empty opcode suffix".into()));
                }
                continue;
            }
            if Type::from_name(part).is_some() || part == "mask" {
                let parsed = if parts.peek().is_none() {
                    parse_type_suffix(part, input)?
                } else {
                    Type::from_name(part)
                        .ok_or_else(|| ParseError(format!("unknown type `{part}`")))?
                };
                if ty.replace(parsed).is_some() {
                    return Err(ParseError("multiple result type suffixes".into()));
                }
            } else if part == "volatile" {
                flags = flags.with_volatile(true);
            } else if let Some(value) = part.strip_prefix("align") {
                let alignment = parse_alignment(value, part)?;
                flags = flags.with_alignment(alignment);
            } else {
                return Err(ParseError(format!("unknown opcode suffix `{part}`")));
            }
        }
        if !input.joined() || input.kind() != Kind::Word || !input.text().starts_with('.') {
            break;
        }
        suffix = input.word()?;
    }
    Ok((opcode, ty, flags))
}

fn parse_alignment(value: &str, suffix: &str) -> ParseResult<u32> {
    let alignment = value
        .parse::<u32>()
        .map_err(|_| ParseError(format!("invalid alignment `{suffix}`")))?;
    if alignment == 0 || !alignment.is_power_of_two() {
        return parse_err(format!(
            "alignment must be a non-zero power of two: {alignment}"
        ));
    }
    Ok(alignment)
}

fn parse_opcode(word: &str) -> ParseResult<(Opcode, &str)> {
    // Mnemonics and suffixes are dotted words. Prefer the longest mnemonic,
    // including any dots belonging to the opcode itself.
    let mut end = word.len();
    let opcode = loop {
        if let Some(opcode) = Opcode::from_mnemonic(&word[..end]) {
            break opcode;
        }
        end = word[..end]
            .rfind('.')
            .ok_or_else(|| ParseError(format!("unknown opcode `{word}`")))?;
    };
    Ok((opcode, &word[end..]))
}

fn parse_type(input: &mut Cursor<'_>) -> ParseResult<Type> {
    let name = input.word()?;
    parse_type_suffix(name, input)
}

fn parse_type_suffix(name: &str, input: &mut Cursor<'_>) -> ParseResult<Type> {
    if !input.eat(Kind::Less) {
        return Type::from_name(name).ok_or_else(|| ParseError(format!("unknown type `{name}`")));
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
    let lanes = input
        .word()?
        .parse::<u16>()
        .map_err(|_| ParseError("invalid vector lane count".into()))?;
    input.expect(Kind::Greater)?;
    base.and_then(Type::as_scalar)
        .and_then(|s| s.vector(lanes, scalable))
        .map(crate::VectorType::as_type)
        .ok_or_else(|| ParseError(format!("invalid vector type `{name}`")))
}

fn parse_types(input: &mut Cursor<'_>) -> ParseResult<Vec<Type>> {
    input.expect(Kind::LParen)?;
    let mut types = Vec::new();
    if !input.eat(Kind::RParen) {
        loop {
            types.push(parse_type(input)?);
            if !input.eat(Kind::Comma) {
                break;
            }
        }
        input.expect(Kind::RParen)?;
    }
    Ok(types)
}

fn parse_function_returns(input: &mut Cursor<'_>) -> ParseResult<Vec<Type>> {
    if input.is("void") {
        input.advance();
        return Ok(Vec::new());
    }
    if input.kind() == Kind::LParen {
        return parse_types(input);
    }
    let mut types = Vec::new();
    loop {
        types.push(parse_type(input)?);
        if !input.eat(Kind::Comma) {
            break;
        }
    }
    Ok(types)
}

fn parse_signature(input: &mut Cursor<'_>) -> ParseResult<Signature> {
    let params = parse_types(input)?;
    input.expect(Kind::Arrow)?;
    let returns = if input.is("void") {
        input.advance();
        Vec::new()
    } else if input.kind() == Kind::LParen {
        parse_types(input)?
    } else {
        alloc::vec![parse_type(input)?]
    };
    Ok(Signature::new(params, returns, CallConv::SystemV))
}

fn parse_linkage(input: &mut Cursor<'_>) -> ParseResult<Linkage> {
    let name = input.word()?;
    Linkage::from_mnemonic(name).ok_or_else(|| ParseError(format!("unknown linkage `{name}`")))
}

fn is_function_header(input: &Cursor<'_>) -> bool {
    let mut next = input.clone();
    if next.kind() != Kind::Word || Linkage::from_mnemonic(next.text()).is_none() {
        return false;
    }
    next.advance();
    next.is("function")
}

fn parse_function_header(input: &mut Cursor<'_>) -> ParseResult<FunctionHeader> {
    let linkage = parse_linkage(input)?;
    input.keyword("function")?;
    let name = input.word()?.to_string();
    let params = parse_types(input)?;
    let returns = if input.eat(Kind::Arrow) {
        parse_function_returns(input)?
    } else {
        Vec::new()
    };
    Ok(FunctionHeader {
        name,
        linkage,
        signature: Signature::new(params, returns, CallConv::SystemV),
    })
}

fn parse_global(input: &mut Cursor<'_>) -> ParseResult<(String, Type, Linkage)> {
    input.keyword("global")?;
    let name = input.word()?.to_string();
    input.expect(Kind::Colon)?;
    let ty = parse_type(input)?;
    input.expect(Kind::LParen)?;
    let linkage = parse_linkage(input)?;
    input.expect(Kind::RParen)?;
    Ok((name, ty, linkage))
}

fn parse_block_header(input: &mut Cursor<'_>) -> ParseResult<(u32, Vec<(String, Type)>)> {
    let name = input.word()?;
    let id = name
        .strip_prefix("block")
        .and_then(|s| s.parse::<u32>().ok())
        .ok_or_else(|| ParseError(format!("invalid block name `{name}`")))?;
    input.expect(Kind::LParen)?;
    let mut params = Vec::new();
    if !input.eat(Kind::RParen) {
        loop {
            let name = input.word()?.to_string();
            input.expect(Kind::Colon)?;
            params.push((name, parse_type(input)?));
            if !input.eat(Kind::Comma) {
                break;
            }
        }
        input.expect(Kind::RParen)?;
    }
    input.expect(Kind::Colon)?;
    Ok((id, params))
}

fn parse_stack_slot(input: &mut Cursor<'_>, func: &mut Function) -> ParseResult<()> {
    let slot = parse_stack_slot_ref(input.word()?)?;
    input.expect(Kind::Colon)?;
    input.keyword("size")?;
    let size = input
        .word()?
        .parse::<u32>()
        .map_err(|_| ParseError("invalid stack slot size".into()))?;
    while func.stack_slots.len() <= slot.0 as usize {
        func.stack_slots.push(StackSlotData { size: 0 });
    }
    func.stack_slots[slot] = StackSlotData { size };
    Ok(())
}

pub(super) fn parse_stack_slot_ref(text: &str) -> ParseResult<StackSlot> {
    let id = text
        .strip_prefix("ss")
        .ok_or_else(|| ParseError(format!("expected stack slot, found `{text}`")))?
        .parse::<u32>()
        .map_err(|_| ParseError(format!("invalid stack slot `{text}`")))?;
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

fn ensure_value(value: Value, func: &mut Function) {
    // These slots are not definitions. Symbols::definitions tracks resolution;
    // the placeholder def must not be interpreted before parsing succeeds.
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

fn parse_err<T>(message: impl Into<String>) -> ParseResult<T> {
    Err(ParseError(message.into()))
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
        let func_ids = HashMap::new();
        test(&mut OperandParser {
            func: &mut func,
            symbols: &mut symbols,
            func_ids: &func_ids,
            module: &mut module,
        });
    }

    fn parse<C: AtomCodec>(
        cx: &mut OperandParser<'_>,
        text: &str,
        ty: Option<Type>,
    ) -> ParseResult<C::Owned> {
        let mut input = Cursor::new(text);
        let value = C::parse(cx, &mut input, ty)?;
        input.finish()?;
        Ok(value)
    }

    fn round_trip<C: AtomCodec>(cx: &mut OperandParser<'_>, text: &str, ty: Option<Type>) -> String
    where
        C::Owned: Debug + PartialEq + for<'a> Borrow<C::View<'a>>,
    {
        let value = parse::<C>(cx, text, ty).unwrap();
        let mut printed = String::new();
        C::print(
            &InstPrinter::new(&cx.func.dfg, None),
            &mut printed,
            value.borrow(),
            ty,
        )
        .unwrap();
        assert_eq!(parse::<C>(cx, &printed, ty).unwrap(), value);
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
            assert!(parse::<Decimal<u64>>(cx, "-1", None).is_err());
            assert_eq!(
                round_trip::<Decimal<i32>>(cx, "-2147483648", None),
                "-2147483648"
            );
            assert_eq!(round_trip::<Decimal<u8>>(cx, "255", None), "255");
            assert!(parse::<Decimal<u8>>(cx, "256", None).is_err());
            assert_eq!(round_trip::<bool>(cx, "true", None), "true");
            assert!(parse::<bool>(cx, "1", None).is_err());
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
                assert!(parse::<FloatBits>(cx, "0x0", ty).is_err());
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
            assert!(parse::<FloatBits>(cx, "0x100000000", Some(Type::F32)).is_err());
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
                assert!(parse::<Bytes>(cx, text, None).is_err(), "{text}");
            }
        });
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
    fn instruction_header_accepts_scalable_result_type() {
        let mut input = Cursor::new("iadd.i32<scalable 4> v0, v1");
        let (opcode, ty, _) = parse_instruction_header(&mut input).unwrap();
        assert_eq!(opcode, Opcode::IAdd);
        assert_eq!(
            ty,
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
    }

    #[test]
    fn context_atoms_share_ssa_values_and_intern_signatures() {
        let mut func = Function::new("test".into(), SigId(0), Linkage::Local);
        let block = func.layout.create_block();
        let mut symbols = Symbols::default();
        symbols.blocks.insert("block0".into(), block);
        let mut module = ModuleData::default();
        let func_ids = HashMap::new();
        let mut parser = OperandParser {
            func: &mut func,
            symbols: &mut symbols,
            func_ids: &func_ids,
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
