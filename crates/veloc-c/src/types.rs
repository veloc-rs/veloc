//! C scalar semantics and LP64 aggregate layout, separate from MIR value types.
use crate::{
    ast::*,
    error::{Error, Result},
};
use std::collections::HashMap;
use veloc_mir::Type;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CType {
    Void,
    Bool,
    Int(u16, bool),
    Float(u16),
    Pointer(Box<CType>),
    Array(Box<CType>, usize),
    Record(usize),
    Function(Box<FunctionType>),
    Volatile(Box<CType>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FunctionType {
    pub result: CType,
    pub params: Vec<(String, CType)>,
    pub variadic: bool,
}

impl CType {
    pub fn compatible(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Array(a, n), Self::Array(b, m)) => {
                a.compatible(b) && (n == m || *n == 0 || *m == 0)
            }
            (Self::Pointer(a), Self::Pointer(b)) | (Self::Volatile(a), Self::Volatile(b)) => {
                a.compatible(b)
            }
            (Self::Function(a), Self::Function(b)) => a.compatible(b),
            _ => self == other,
        }
    }
    pub const INT: Self = Self::Int(32, true);
    pub const SIZE: Self = Self::Int(64, false);
    pub fn plain(&self) -> &Self {
        if let Self::Volatile(t) = self {
            t.plain()
        } else {
            self
        }
    }
    pub fn pointer(self) -> Self {
        Self::Pointer(Box::new(self))
    }
    pub fn decay(&self) -> Self {
        match self.plain() {
            Self::Array(t, _) => Self::Pointer(t.clone()),
            Self::Function(_) => self.clone().pointer(),
            t => t.clone(),
        }
    }
    pub fn promoted(&self) -> Self {
        match self.plain() {
            Self::Bool => Self::INT,
            Self::Int(bits, _) if *bits < 32 => Self::INT,
            _ => self.decay(),
        }
    }
    /// LP64D transports narrow C integers widened to 32 bits, with the source
    /// signedness determining extension. RV64 then sign-extends that word to
    /// XLEN. Keep C object types narrow and express this at call boundaries.
    pub fn abi_type(&self) -> Self {
        self.promoted()
    }
    pub fn signed(&self) -> bool {
        matches!(self.plain(), Self::Int(_, true))
    }
    pub fn pointee(&self) -> Result<CType> {
        match self.plain() {
            Self::Pointer(t) | Self::Array(t, _) => Ok(*t.clone()),
            _ => fail("expected pointer"),
        }
    }
    pub fn mir(&self) -> Result<Type> {
        Ok(match self.plain() {
            Self::Bool => Type::I8,
            Self::Int(8, _) => Type::I8,
            Self::Int(16, _) => Type::I16,
            Self::Int(32, _) => Type::I32,
            Self::Int(64, _) => Type::I64,
            Self::Float(32) => Type::F32,
            Self::Float(64) => Type::F64,
            Self::Pointer(_) => Type::PTR,
            _ => return fail("aggregate or void cannot be used as a scalar value"),
        })
    }
    pub fn common(a: &Self, b: &Self) -> Self {
        let a = a.promoted();
        let b = b.promoted();
        match (&a, &b) {
            (Self::Float(x), Self::Float(y)) => Self::Float((*x).max(*y)),
            (Self::Float(_), _) => a,
            (_, Self::Float(_)) => b,
            (Self::Int(x, sx), Self::Int(y, sy)) => Self::Int(
                (*x).max(*y),
                if x == y {
                    *sx && *sy
                } else if x > y {
                    *sx
                } else {
                    *sy
                },
            ),
            (Self::Pointer(_), _) => a,
            (_, Self::Pointer(_)) => b,
            _ => a,
        }
    }
}

impl FunctionType {
    pub fn abi_signature(&self) -> Result<veloc_mir::Signature> {
        let params = self
            .params
            .iter()
            .map(|(_, ty)| ty.abi_type().mir())
            .collect::<Result<Vec<_>>>()?;
        let returns = if self.result == CType::Void {
            vec![]
        } else {
            vec![self.result.abi_type().mir()?]
        };
        Ok(
            veloc_mir::Signature::new(params, returns, veloc_mir::CallConv::SystemV)
                .with_variadic(self.variadic),
        )
    }

    pub fn compatible(&self, other: &Self) -> bool {
        self.result.compatible(&other.result)
            && self.variadic == other.variadic
            && self.params.len() == other.params.len()
            && self
                .params
                .iter()
                .zip(&other.params)
                .all(|((_, a), (_, b))| a.compatible(b))
    }
}

#[derive(Clone, Debug)]
pub struct Member {
    pub name: String,
    pub ty: CType,
    pub offset: usize,
}
#[derive(Clone, Debug, Default)]
pub struct Record {
    pub members: Vec<Member>,
    pub size: usize,
    pub align: usize,
    pub is_union: bool,
}

#[derive(Default)]
pub struct Types {
    pub typedefs: HashMap<String, CType>,
    pub constants: HashMap<String, i64>,
    tags: HashMap<String, usize>,
    pub records: Vec<Record>,
}

pub fn fail<T>(message: impl Into<String>) -> Result<T> {
    Err(Error::semantic(message, 0, 0))
}
fn align_up(size: usize, align: usize) -> usize {
    size.div_ceil(align) * align
}

impl Types {
    pub fn layout(&self, ty: &CType) -> Result<(usize, usize)> {
        Ok(match ty.plain() {
            CType::Bool => (1, 1),
            CType::Int(bits, _) | CType::Float(bits) => {
                ((*bits / 8) as usize, (*bits / 8) as usize)
            }
            CType::Pointer(_) => (8, 8),
            CType::Array(t, n) => {
                let (size, align) = self.layout(t)?;
                (
                    size.checked_mul(*n)
                        .ok_or_else(|| Error::semantic("object too large", 0, 0))?,
                    align,
                )
            }
            CType::Record(id) => {
                let r = &self.records[*id];
                if r.align == 0 {
                    return fail("incomplete record type");
                }
                (r.size, r.align)
            }
            _ => return fail("type has no object layout"),
        })
    }
    pub fn member(&self, ty: &CType, name: &str) -> Result<Member> {
        let CType::Record(id) = ty.plain() else {
            return fail("member access requires a record");
        };
        self.records[*id]
            .members
            .iter()
            .find(|m| m.name == name)
            .cloned()
            .ok_or_else(|| Error::semantic(format!("unknown member {name}"), 0, 0))
    }
    pub fn specifiers(&mut self, specs: &[DeclarationSpecifier]) -> Result<CType> {
        let mut unsigned = false;
        let mut explicitly_signed = false;
        let mut short = false;
        let mut longs = 0;
        let mut ty = CType::INT;
        let mut volatile = false;
        for spec in specs {
            match spec {
                DeclarationSpecifier::TypeQualifier(TypeQualifier::Volatile) => volatile = true,
                DeclarationSpecifier::TypeSpecifier(t) => match t {
                    TypeSpecifier::Void => ty = CType::Void,
                    TypeSpecifier::Bool => ty = CType::Bool,
                    TypeSpecifier::Char => ty = CType::Int(8, false),
                    TypeSpecifier::Short => short = true,
                    TypeSpecifier::Long => longs += 1,
                    TypeSpecifier::Unsigned => unsigned = true,
                    TypeSpecifier::Signed => explicitly_signed = true,
                    TypeSpecifier::Int => {}
                    TypeSpecifier::Float => ty = CType::Float(32),
                    TypeSpecifier::Double => ty = CType::Float(64),
                    TypeSpecifier::TypedefName(name) => {
                        ty = self.typedefs.get(name).cloned().ok_or_else(|| {
                            Error::semantic(format!("unknown typedef {name}"), 0, 0)
                        })?
                    }
                    TypeSpecifier::Struct(record) => ty = self.record(record)?,
                    TypeSpecifier::Enum(en) => {
                        if let Some(enumerators) = &en.enumerators {
                            let mut next = 0;
                            for entry in enumerators {
                                let value = if let Some(expr) = &entry.value {
                                    self.constant(expr)?
                                } else {
                                    next
                                };
                                self.constants.insert(entry.name.clone(), value);
                                next = value + 1;
                            }
                        }
                        ty = CType::INT;
                    }
                    _ => return fail("unsupported C type specifier"),
                },
                _ => {}
            }
        }
        if let CType::Int(bits, signed) = &mut ty {
            if short {
                *bits = 16;
            } else if longs > 0 {
                *bits = 64;
            }
            if unsigned {
                *signed = false;
            } else if explicitly_signed {
                *signed = true;
            }
        } else if longs != 0 {
            return fail("long double is not supported");
        }
        Ok(if volatile {
            CType::Volatile(Box::new(ty))
        } else {
            ty
        })
    }
    fn record(&mut self, spec: &StructSpecifier) -> Result<CType> {
        let id = if let Some(id) = spec.name.as_ref().and_then(|n| self.tags.get(n)) {
            *id
        } else {
            let id = self.records.len();
            self.records.push(Record::default());
            if let Some(n) = &spec.name {
                self.tags.insert(n.clone(), id);
            }
            id
        };
        if let Some(fields) = &spec.members {
            let mut record = Record {
                align: 1,
                is_union: spec.is_union,
                ..Default::default()
            };
            for field in fields {
                let specs = qualifiers(&field.specifiers);
                let base = self.specifiers(&specs)?;
                for decl in &field.declarators {
                    if decl.bit_width.is_some() {
                        return fail("bit fields are not supported");
                    }
                    let Some(decl) = &decl.declarator else {
                        return fail("anonymous fields are not supported");
                    };
                    let ty = self.declarator(base.clone(), decl)?;
                    let (size, align) = self.layout(&ty)?;
                    let offset = if spec.is_union {
                        0
                    } else {
                        align_up(record.size, align)
                    };
                    record.size = record.size.max(offset + size);
                    record.align = record.align.max(align);
                    record.members.push(Member {
                        name: decl.name().into(),
                        ty,
                        offset,
                    });
                }
            }
            record.size = align_up(record.size, record.align);
            self.records[id] = record;
        }
        Ok(CType::Record(id))
    }
    pub fn declarator(&mut self, mut base: CType, decl: &Declarator) -> Result<CType> {
        let mut ptr = decl.pointer.as_ref();
        while let Some(p) = ptr {
            base = base.pointer();
            if p.qualifiers.contains(&TypeQualifier::Volatile) {
                base = CType::Volatile(Box::new(base));
            }
            ptr = p.inner.as_deref();
        }
        self.direct(base, &decl.direct)
    }
    fn direct(&mut self, base: CType, decl: &DirectDeclarator) -> Result<CType> {
        match decl {
            DirectDeclarator::Identifier(_) => Ok(base),
            DirectDeclarator::Parenthesized(decl) => self.declarator(base, decl),
            DirectDeclarator::Array(inner, size) => {
                let n = match size {
                    Some(expr) => usize::try_from(self.constant(expr)?)
                        .map_err(|_| Error::semantic("invalid array bound", 0, 0))?,
                    None => 0,
                };
                self.direct(CType::Array(Box::new(base), n), inner)
            }
            DirectDeclarator::Function(inner, params) => {
                let mut types = Vec::new();
                if let Some(params) = params {
                    for param in &params.parameters {
                        let base = self.specifiers(&param.specifiers)?;
                        let ty = if let Some(decl) = &param.declarator {
                            self.declarator(base, decl)?
                        } else if let Some(decl) = &param.abstract_declarator {
                            self.abstract_type(base, decl)?
                        } else {
                            base
                        };
                        if ty == CType::Void && params.parameters.len() == 1 {
                            continue;
                        }
                        types.push((
                            param.declarator.as_ref().map_or("", |d| d.name()).into(),
                            ty.decay(),
                        ));
                    }
                }
                self.direct(
                    CType::Function(Box::new(FunctionType {
                        result: base,
                        params: types,
                        variadic: params.as_ref().is_some_and(|p| p.variadic),
                    })),
                    inner,
                )
            }
            _ => fail("old-style function definitions are not supported"),
        }
    }
    pub fn type_name(&mut self, name: &TypeName) -> Result<CType> {
        let base = self.specifiers(&qualifiers(&name.specifiers))?;
        if let Some(decl) = &name.abstract_declarator {
            self.abstract_type(base, decl)
        } else {
            Ok(base)
        }
    }
    pub fn abstract_type(&mut self, mut base: CType, decl: &AbstractDeclarator) -> Result<CType> {
        let mut ptr = decl.pointer.as_ref();
        while let Some(p) = ptr {
            base = base.pointer();
            ptr = p.inner.as_deref();
        }
        match decl.direct.as_deref() {
            None => Ok(base),
            Some(DirectAbstractDeclarator::Parenthesized(inner)) => self.abstract_type(base, inner),
            _ => fail("unsupported abstract array/function declarator"),
        }
    }
}

pub fn qualifiers(specs: &[SpecifierQualifier]) -> Vec<DeclarationSpecifier> {
    specs
        .iter()
        .map(|s| match s {
            SpecifierQualifier::TypeSpecifier(t) => DeclarationSpecifier::TypeSpecifier(t.clone()),
            SpecifierQualifier::TypeQualifier(t) => DeclarationSpecifier::TypeQualifier(*t),
        })
        .collect()
}
