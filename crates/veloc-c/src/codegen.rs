//! C declarations and initializers are resolved before constructing function SSA.
use crate::ast::*;
use crate::error::{Error, Result};
use crate::types::{CTargetModel, CType, FunctionType, Types, fail};
use std::collections::HashMap;
use veloc_mir::{
    DataRelocation, FuncId, GlobalData, GlobalId, Linkage, Module, ModuleBuilder, SigId, Signature,
    Type,
};

mod function;
use function::Function;

#[derive(Clone)]
struct FunctionSymbol {
    id: FuncId,
    ty: FunctionType,
}
#[derive(Clone)]
struct GlobalSymbol {
    id: GlobalId,
    ty: CType,
}
struct StringSymbol {
    id: GlobalId,
    name: String,
}

pub struct CodeGenContext {
    module: ModuleBuilder,
    types: Types,
    functions: HashMap<String, FunctionSymbol>,
    globals: HashMap<String, GlobalSymbol>,
    strings: HashMap<String, StringSymbol>,
    signatures: HashMap<Signature, SigId>,
}

impl CodeGenContext {
    pub fn new(target: CTargetModel) -> Self {
        Self {
            module: ModuleBuilder::new(),
            types: Types::new(target),
            functions: HashMap::new(),
            globals: HashMap::new(),
            strings: HashMap::new(),
            signatures: HashMap::new(),
        }
    }

    pub fn generate(mut self, tu: &TranslationUnit) -> Result<Module> {
        let mut functions = Vec::new();
        let mut globals = std::collections::BTreeMap::<GlobalId, Option<Initializer>>::new();
        let mut defined_functions = std::collections::HashSet::new();
        // Resolve types and names before emitting bodies, including forward calls.
        for declaration in &tu.declarations {
            let (specs, items) = match declaration {
                ExternalDeclaration::Declaration(d) => (&d.specifiers, d.init_declarators.clone()),
                ExternalDeclaration::FunctionDefinition(f) => {
                    if !defined_functions.insert(f.declarator.name()) {
                        return fail("duplicate function definition");
                    }
                    functions.push(f);
                    (
                        &f.specifiers,
                        vec![InitDeclarator {
                            declarator: f.declarator.clone(),
                            initializer: None,
                        }],
                    )
                }
            };
            let base = self.types.specifiers(specs)?;
            let typedef = specs.contains(&DeclarationSpecifier::StorageClass(
                StorageClassSpecifier::Typedef,
            ));
            let local = specs.contains(&DeclarationSpecifier::StorageClass(
                StorageClassSpecifier::Static,
            ));
            let external = specs.contains(&DeclarationSpecifier::StorageClass(
                StorageClassSpecifier::Extern,
            ));
            for item in items {
                let name = item.declarator.name().to_owned();
                let mut ty = self.types.declarator(base.clone(), &item.declarator)?;
                if typedef {
                    self.types.typedefs.insert(name, ty);
                    continue;
                }
                if let CType::Function(ty) = ty {
                    if !self.functions.contains_key(&name) {
                        let sig = self.signature(&ty)?;
                        let definition = tu.declarations.iter().any(|d| matches!(d, ExternalDeclaration::FunctionDefinition(f) if f.declarator.name() == name));
                        let linkage = if local {
                            Linkage::Local
                        } else if definition {
                            Linkage::Export
                        } else {
                            Linkage::Import
                        };
                        let id = self.module.declare_function(name.clone(), sig, linkage);
                        self.functions.insert(name, FunctionSymbol { id, ty: *ty });
                    } else {
                        let old = self.functions.get_mut(&name).unwrap();
                        if !old.ty.compatible(&ty) {
                            return fail(format!("conflicting declaration of {name}"));
                        }
                        if matches!(declaration, ExternalDeclaration::FunctionDefinition(_)) {
                            old.ty = *ty;
                        }
                    }
                } else {
                    complete_array(&mut ty, item.initializer.as_ref())?;
                    let id = if let Some(global) = self.globals.get_mut(&name) {
                        if !global.ty.compatible(&ty) {
                            return fail(format!("conflicting declaration of {name}"));
                        }
                        if matches!(global.ty, CType::Array(_, 0)) {
                            global.ty = ty.clone();
                        }
                        global.id
                    } else {
                        let id = self.module.add_global(
                            name.clone(),
                            Type::PTR,
                            if local {
                                Linkage::Local
                            } else if external {
                                Linkage::Import
                            } else {
                                Linkage::Export
                            },
                        );
                        self.globals
                            .insert(name.clone(), GlobalSymbol { id, ty: ty.clone() });
                        id
                    };
                    if !external || item.initializer.is_some() {
                        let previous = globals.entry(id).or_default();
                        if item.initializer.is_some() {
                            if previous.is_some() {
                                return fail(format!("duplicate definition of {name}"));
                            }
                            *previous = item.initializer;
                        }
                    }
                }
            }
        }
        // Address literals and indirect-call signatures must exist before a builder
        // borrows module declarations for the lifetime of a function.
        let mut strings = Vec::new();
        let mut collect = |expr: &Expression| {
            if let Expression::String(s) = expr {
                strings.push(s.clone());
            }
        };
        for declaration in &tu.declarations {
            match declaration {
                ExternalDeclaration::FunctionDefinition(f) => {
                    crate::visit::compound(&f.body, &mut collect)
                }
                ExternalDeclaration::Declaration(d) => crate::visit::declaration(d, &mut collect),
            }
        }
        for string in strings {
            self.string(&string)?;
        }
        for ty in self.types.typedefs.values().cloned().collect::<Vec<_>>() {
            self.collect_signatures(&ty)?;
        }
        for record in self.types.records.clone() {
            for member in record.members {
                self.collect_signatures(&member.ty)?;
            }
        }
        for func in self.functions.values().cloned().collect::<Vec<_>>() {
            for (_, ty) in &func.ty.params {
                self.collect_signatures(ty)?;
            }
        }
        for (id, init) in globals {
            let global = self.globals.values_mut().find(|g| g.id == id).unwrap();
            // An incomplete tentative array definition has one zeroed element.
            if let CType::Array(_, n) = &mut global.ty {
                if *n == 0 {
                    *n = 1;
                }
            }
            let ty = global.ty.clone();
            let (size, align) = self.types.layout(&ty)?;
            let mut data = GlobalData {
                bytes: vec![0; size],
                align: align as u64,
                writable: true,
                relocations: Vec::new(),
            };
            if let Some(init) = &init {
                self.initialize_data(&mut data, 0, &ty, init)?;
            }
            self.module.define_global(id, data);
        }
        for definition in functions {
            let symbol = self.functions[definition.declarator.name()].clone();
            let builder = self.module.define(symbol.id);
            Function::new(
                builder,
                &mut self.types,
                &self.functions,
                &self.globals,
                &self.strings,
                &self.signatures,
                &symbol.ty,
                definition,
            )
            .generate(definition)
            .map_err(|e| Error::semantic(format!("{}: {e}", definition.declarator.name()), 0, 0))?;
        }
        Ok(self.module.build())
    }

    fn signature(&mut self, ty: &FunctionType) -> Result<SigId> {
        let signature = ty.abi_signature()?;
        if let Some(id) = self.signatures.get(&signature) {
            return Ok(*id);
        }
        let id = self.module.intern_signature(signature.clone());
        self.signatures.insert(signature, id);
        Ok(id)
    }
    fn collect_signatures(&mut self, ty: &CType) -> Result<()> {
        match ty.plain() {
            CType::Function(f) => {
                self.signature(f)?;
                for (_, t) in &f.params {
                    self.collect_signatures(t)?;
                }
            }
            CType::Pointer(t) | CType::Array(t, _) => self.collect_signatures(t)?,
            _ => {}
        }
        Ok(())
    }
    fn string(&mut self, text: &str) -> Result<GlobalId> {
        if let Some(symbol) = self.strings.get(text) {
            return Ok(symbol.id);
        }
        let name = format!(".L.str.{}", self.strings.len());
        let id = self
            .module
            .add_global(name.clone(), Type::PTR, Linkage::Local);
        let mut bytes = string_bytes(text)?;
        bytes.push(0);
        self.module.define_global(
            id,
            GlobalData {
                bytes,
                align: 1,
                writable: false,
                relocations: vec![],
            },
        );
        self.strings.insert(text.into(), StringSymbol { id, name });
        Ok(id)
    }
    fn initialize_data(
        &mut self,
        data: &mut GlobalData,
        offset: usize,
        ty: &CType,
        init: &Initializer,
    ) -> Result<()> {
        match (ty.plain(), init) {
            (CType::Array(element, count), Initializer::List(items)) => {
                if items.len() > *count {
                    return fail("too many array initializers");
                }
                let stride = self.types.layout(element)?.0;
                for (index, item) in items.iter().enumerate() {
                    self.initialize_data(data, offset + index * stride, element, item)?;
                }
            }
            (CType::Record(id), Initializer::List(items)) => {
                let record = &self.types.records[*id];
                let fields: Vec<_> = record
                    .members
                    .iter()
                    .take(if record.is_union {
                        1
                    } else {
                        record.members.len()
                    })
                    .cloned()
                    .collect();
                if items.len() > fields.len() {
                    return fail("too many record initializers");
                }
                for (item, field) in items.iter().zip(fields) {
                    self.initialize_data(data, offset + field.offset, &field.ty, item)?;
                }
            }
            (CType::Array(element, count), Initializer::Expression(Expression::String(text)))
                if matches!(element.plain(), CType::Int(8, _)) =>
            {
                let mut bytes = string_bytes(text)?;
                bytes.push(0);
                let len = bytes.len().min(*count);
                data.bytes[offset..offset + len].copy_from_slice(&bytes[..len]);
            }
            (_, Initializer::List(items)) if items.len() == 1 => {
                self.initialize_data(data, offset, ty, &items[0])?
            }
            (_, Initializer::Expression(expr)) => {
                let original = expr;
                let mut expr = expr;
                let mut address = false;
                loop {
                    match expr {
                        Expression::Cast(_, e) | Expression::Parenthesized(e) => expr = e,
                        Expression::AddressOf(e) if !address => {
                            address = true;
                            expr = e;
                        }
                        _ => break,
                    }
                }
                let symbol = match expr {
                    Expression::String(text) => Some(self.strings[text].name.clone()),
                    Expression::Identifier(name)
                        if self.functions.contains_key(name)
                            || self.globals.get(name).is_some_and(|global| {
                                address || matches!(global.ty.plain(), CType::Array(_, _))
                            }) =>
                    {
                        Some(name.clone())
                    }
                    _ => None,
                };
                if let Some(symbol) = symbol {
                    if !matches!(ty.plain(), CType::Pointer(_)) {
                        return fail("address initializer requires a pointer");
                    }
                    data.relocations.push(DataRelocation {
                        offset: offset as u64,
                        symbol,
                        addend: 0,
                    });
                } else {
                    let bits = match (ty.plain(), original) {
                        (CType::Float(64), Expression::Float(v)) => (if v.bits == 32 {
                            v.value as f32 as f64
                        } else {
                            v.value
                        })
                        .to_bits(),
                        (CType::Float(32), Expression::Float(v)) => {
                            (v.value as f32).to_bits() as u64
                        }
                        (CType::Float(bits), _) => {
                            let value = self.types.integer_constant(original)?;
                            let value = if value.signed {
                                value.signed_value() as f64
                            } else {
                                value.value as f64
                            };
                            if *bits == 32 {
                                (value as f32).to_bits() as u64
                            } else {
                                value.to_bits()
                            }
                        }
                        (CType::Bool, _) => {
                            u64::from(self.types.integer_constant(original)?.value != 0)
                        }
                        _ => self.types.constant(original)? as u64,
                    };
                    let size = self.types.layout(ty)?.0;
                    if size > 8 {
                        return fail("unsupported aggregate initializer");
                    }
                    data.bytes[offset..offset + size].copy_from_slice(&bits.to_le_bytes()[..size]);
                }
            }
            _ => return fail("unsupported initializer"),
        }
        Ok(())
    }
}

fn complete_array(ty: &mut CType, init: Option<&Initializer>) -> Result<()> {
    if let CType::Array(_, n) = ty {
        if *n == 0 {
            *n = match init {
                Some(Initializer::List(items)) => items.len(),
                Some(Initializer::Expression(Expression::String(s))) => string_bytes(s)?.len() + 1,
                _ => 0,
            };
        }
    }
    Ok(())
}

fn string_bytes(text: &str) -> Result<Vec<u8>> {
    let mut chars = text.bytes().peekable();
    let mut out = Vec::new();
    while let Some(c) = chars.next() {
        if c != b'\\' {
            out.push(c);
            continue;
        }
        let c = chars
            .next()
            .ok_or_else(|| Error::semantic("incomplete escape", 0, 0))?;
        out.push(match c {
            b'n' => b'\n',
            b'r' => b'\r',
            b't' => b'\t',
            b'a' => 7,
            b'b' => 8,
            b'f' => 12,
            b'v' => 11,
            b'0'..=b'7' => {
                let mut n = c - b'0';
                for _ in 0..2 {
                    if let Some(b'0'..=b'7') = chars.peek() {
                        n = n.wrapping_mul(8).wrapping_add(chars.next().unwrap() - b'0');
                    } else {
                        break;
                    }
                }
                n
            }
            b'x' => {
                let mut n = 0u8;
                let mut any = false;
                while let Some(c) = chars.peek().and_then(|c| (*c as char).to_digit(16)) {
                    any = true;
                    n = n.wrapping_mul(16).wrapping_add(c as u8);
                    chars.next();
                }
                if !any {
                    return fail("invalid hexadecimal escape");
                }
                n
            }
            c => c,
        });
    }
    Ok(out)
}

pub fn compile_to_ir(source: &str, target: CTargetModel) -> Result<Module> {
    CodeGenContext::new(target).generate(&crate::parse(source)?)
}
