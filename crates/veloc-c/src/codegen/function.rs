use super::*;
use crate::types::qualifiers;
use std::collections::{HashSet, VecDeque};
use veloc_mir::{Block, FloatCC, Int, IntCC, MemFlags, SsaBuilder, Value, Variable};

mod statements;

#[derive(Clone)]
struct TypedValue {
    value: Value,
    ty: CType,
}
#[derive(Clone, Copy)]
enum Location {
    Variable(Variable),
    Memory(Value),
}
#[derive(Clone)]
struct Place {
    location: Location,
    ty: CType,
}

pub(super) struct Function<'a, 'b> {
    builder: SsaBuilder<'a>,
    types: &'b mut Types,
    functions: &'b HashMap<String, FunctionSymbol>,
    globals: &'b HashMap<String, GlobalSymbol>,
    strings: &'b HashMap<String, StringSymbol>,
    signatures: &'b HashMap<Signature, SigId>,
    signature: &'b FunctionType,
    scopes: Vec<HashMap<String, Place>>,
    addressed: HashSet<String>,
    next_variable: u32,
    breaks: Vec<Block>,
    continues: Vec<Block>,
    cases: Vec<VecDeque<Block>>,
}

impl<'a, 'b> Function<'a, 'b> {
    pub(super) fn new(
        builder: SsaBuilder<'a>,
        types: &'b mut Types,
        functions: &'b HashMap<String, FunctionSymbol>,
        globals: &'b HashMap<String, GlobalSymbol>,
        strings: &'b HashMap<String, StringSymbol>,
        signatures: &'b HashMap<Signature, SigId>,
        signature: &'b FunctionType,
        definition: &FunctionDefinition,
    ) -> Self {
        let mut addressed = HashSet::new();
        crate::visit::compound(&definition.body, &mut |expr| {
            if let Expression::AddressOf(expr) = expr {
                let mut inner = &**expr;
                while let Expression::Parenthesized(next) = inner {
                    inner = next;
                }
                if let Expression::Identifier(name) = inner {
                    addressed.insert(name.clone());
                }
            }
        });
        Self {
            builder,
            types,
            functions,
            globals,
            strings,
            signatures,
            signature,
            scopes: vec![HashMap::new()],
            addressed,
            next_variable: 0,
            breaks: vec![],
            continues: vec![],
            cases: vec![],
        }
    }
    pub(super) fn generate(mut self, definition: &FunctionDefinition) -> Result<()> {
        if self.signature.variadic {
            return fail("variadic function definitions are not implemented");
        }
        let params = self.builder.func().params().to_vec();
        for ((name, ty), value) in self.signature.params.iter().zip(params) {
            let place = self.local(name, ty.clone())?;
            self.write(
                &place,
                TypedValue {
                    value,
                    ty: ty.abi_type(),
                },
            )?;
        }
        self.compound(&definition.body)?;
        if !self.builder.is_current_block_terminated() {
            if self.signature.result == CType::Void {
                self.builder.ins().ret(&[]);
            } else if definition.declarator.name() == "main" {
                let zero = self.integer(0, CType::INT)?;
                self.builder.ins().ret(&[zero.value]);
            } else {
                self.builder.ins().unreachable();
            }
        }
        self.builder.seal_all_blocks();
        Ok(())
    }
    fn integer(&mut self, bits: u64, ty: CType) -> Result<TypedValue> {
        let value = self.builder.ins().iconst(
            Int::from_bits(ty.mir()?, bits)
                .ok_or_else(|| Error::semantic("invalid integer type", 0, 0))?,
        );
        Ok(TypedValue { value, ty })
    }
    fn variable(&mut self, ty: &CType) -> Result<Variable> {
        let var = Variable(self.next_variable);
        self.next_variable += 1;
        self.builder.declare_var(var, ty.mir()?);
        Ok(var)
    }
    fn local(&mut self, name: &str, ty: CType) -> Result<Place> {
        let location = if self.addressed.contains(name)
            || matches!(ty.plain(), CType::Array(..) | CType::Record(_))
            || matches!(ty, CType::Volatile(_))
        {
            let (size, align) = self.types.layout(&ty)?;
            let entry = self.builder.func().entry_block();
            Location::Memory(
                self.builder.at_start(entry).alloca(
                    u32::try_from(size.max(1))
                        .map_err(|_| Error::semantic("object too large", 0, 0))?,
                    align as u32,
                ),
            )
        } else {
            Location::Variable(self.variable(&ty)?)
        };
        let place = Place { location, ty };
        self.scopes
            .last_mut()
            .unwrap()
            .insert(name.into(), place.clone());
        Ok(place)
    }
    fn lookup(&self, name: &str) -> Option<Place> {
        self.scopes.iter().rev().find_map(|s| s.get(name)).cloned()
    }
    fn flags(ty: &CType) -> MemFlags {
        MemFlags::new()
            .with_notrap(true)
            .with_volatile(matches!(ty, CType::Volatile(_)))
    }
    fn read(&mut self, place: Place) -> Result<TypedValue> {
        if matches!(place.ty.plain(), CType::Array(..)) {
            let Location::Memory(value) = place.location else {
                return fail("array requires storage");
            };
            return Ok(TypedValue {
                value,
                ty: place.ty.decay(),
            });
        }
        let ty = place.ty.plain().clone();
        let value = match place.location {
            Location::Variable(v) => self.builder.use_var(v),
            Location::Memory(p) => self
                .builder
                .ins()
                .load(p, 0, Self::flags(&place.ty), ty.mir()?),
        };
        Ok(TypedValue { value, ty })
    }
    fn write(&mut self, place: &Place, value: TypedValue) -> Result<TypedValue> {
        let value = self.cast(value, place.ty.plain())?;
        match place.location {
            Location::Variable(var) => self.builder.def_var(var, value.value),
            Location::Memory(ptr) => {
                self.builder
                    .ins()
                    .store(ptr, value.value, 0, Self::flags(&place.ty))
            }
        }
        Ok(value)
    }
    fn cast(&mut self, v: TypedValue, to: &CType) -> Result<TypedValue> {
        let to = to.plain().clone();
        let from = v.ty.plain();
        if to == CType::Void {
            return Ok(TypedValue {
                value: v.value,
                ty: to,
            });
        }
        if to == CType::Bool {
            let condition = self.truth(v)?;
            return Ok(TypedValue {
                value: self.builder.ins().extendu(condition, Type::I8),
                ty: to,
            });
        }
        if from == &CType::Bool {
            return self.cast(
                TypedValue {
                    value: v.value,
                    ty: CType::Int(8, false),
                },
                &to,
            );
        }
        let value = match (from, &to) {
            (a, b) if a == b => v.value,
            (CType::Pointer(_), CType::Pointer(_)) => v.value,
            (CType::Pointer(_), CType::Int(bits, _)) => {
                let n = self.builder.ins().ptrtoint(v.value, Type::I64);
                if *bits == 64 {
                    n
                } else {
                    self.builder.ins().wrap(n, to.mir()?)
                }
            }
            (CType::Int(_, _), CType::Pointer(_)) => {
                let n = self.cast(v.clone(), &CType::Int(64, v.ty.signed()))?;
                self.builder.ins().inttoptr(n.value)
            }
            (CType::Int(a, signed), CType::Int(b, _)) => {
                if a == b {
                    v.value
                } else if a > b {
                    self.builder.ins().wrap(v.value, to.mir()?)
                } else if *signed {
                    self.builder.ins().extends(v.value, to.mir()?)
                } else {
                    self.builder.ins().extendu(v.value, to.mir()?)
                }
            }
            (CType::Float(32), CType::Float(64)) => self.builder.ins().float_promote(v.value),
            (CType::Float(64), CType::Float(32)) => self.builder.ins().float_demote(v.value),
            (CType::Int(_, true), CType::Float(_)) => {
                self.builder.ins().int_to_float_s(v.value, to.mir()?)
            }
            (CType::Int(_, false), CType::Float(_)) => {
                self.builder.ins().int_to_float_u(v.value, to.mir()?)
            }
            (CType::Float(_), CType::Int(_, true)) => {
                self.builder.ins().float_to_int_s(v.value, to.mir()?)
            }
            (CType::Float(_), CType::Int(_, false)) => {
                self.builder.ins().float_to_int_u(v.value, to.mir()?)
            }
            _ => return fail(format!("unsupported conversion from {from:?} to {to:?}")),
        };
        Ok(TypedValue { value, ty: to })
    }
    fn truth(&mut self, v: TypedValue) -> Result<Value> {
        if matches!(v.ty.plain(), CType::Float(_)) {
            let zero = if v.ty == CType::Float(32) {
                self.builder.ins().f32const(0.)
            } else {
                self.builder.ins().f64const(0.)
            };
            Ok(self.builder.ins().fcmp(FloatCC::Ne, v.value, zero))
        } else {
            let v = if matches!(v.ty.plain(), CType::Pointer(_)) {
                self.cast(v, &CType::SIZE)?
            } else {
                v
            };
            let zero = self.integer(0, v.ty.clone())?;
            Ok(self.builder.ins().icmp(IntCC::Ne, v.value, zero.value))
        }
    }
    fn boolean_int(&mut self, value: Value) -> TypedValue {
        TypedValue {
            value: self.builder.ins().extendu(value, Type::I32),
            ty: CType::INT,
        }
    }
    fn offset(
        &mut self,
        pointer: TypedValue,
        index: TypedValue,
        subtract: bool,
    ) -> Result<TypedValue> {
        let stride = self.types.layout(&pointer.ty.pointee()?)?.0;
        let mut index = self.cast(index, &CType::Int(64, true))?;
        if subtract {
            index.value = self.builder.ins().ineg(index.value);
        }
        let value = self.builder.ins().ptr_index(
            pointer.value,
            index.value,
            veloc_mir::inst::PtrIndexImm {
                offset: 0,
                scale: stride
                    .try_into()
                    .map_err(|_| Error::semantic("element too large", 0, 0))?,
            },
        );
        Ok(TypedValue {
            value,
            ty: pointer.ty,
        })
    }
    fn place(&mut self, expr: &Expression) -> Result<Place> {
        use Expression::*;
        match expr {
            Parenthesized(e) => self.place(e),
            Identifier(name) => {
                if let Some(p) = self.lookup(name) {
                    return Ok(p);
                }
                let g = self
                    .globals
                    .get(name)
                    .ok_or_else(|| Error::semantic(format!("unknown object {name}"), 0, 0))?;
                Ok(Place {
                    location: Location::Memory(self.builder.ins().global_addr(g.id)),
                    ty: g.ty.clone(),
                })
            }
            Dereference(e) => {
                let v = self.expr(e)?;
                Ok(Place {
                    location: Location::Memory(v.value),
                    ty: v.ty.pointee()?,
                })
            }
            ArrayAccess(a, b) => {
                let a = self.expr(a)?;
                let b = self.expr(b)?;
                let p = if matches!(a.ty.plain(), CType::Pointer(_)) {
                    self.offset(a, b, false)?
                } else {
                    self.offset(b, a, false)?
                };
                Ok(Place {
                    location: Location::Memory(p.value),
                    ty: p.ty.pointee()?,
                })
            }
            MemberAccess(e, name) | PointerMemberAccess(e, name) => {
                let (ptr, ty) = if matches!(expr, PointerMemberAccess(..)) {
                    let v = self.expr(e)?;
                    (v.value, v.ty.pointee()?)
                } else {
                    let p = self.place(e)?;
                    let Location::Memory(ptr) = p.location else {
                        return fail("record requires storage");
                    };
                    (ptr, p.ty)
                };
                let member = self.types.member(&ty, name)?;
                Ok(Place {
                    location: Location::Memory(
                        self.builder.ins().ptr_offset(
                            ptr,
                            member
                                .offset
                                .try_into()
                                .map_err(|_| Error::semantic("member offset too large", 0, 0))?,
                        ),
                    ),
                    ty: member.ty,
                })
            }
            _ => fail("expression is not an lvalue"),
        }
    }
    fn expr(&mut self, expr: &Expression) -> Result<TypedValue> {
        use Expression::*;
        match expr {
            Integer(n) => self.integer(n.value, CType::Int(n.bits, n.signed)),
            Char(c) => self.integer(*c as u64, CType::INT),
            Float(v) => Ok(TypedValue {
                value: if v.bits == 32 {
                    self.builder.ins().f32const(v.value as f32)
                } else {
                    self.builder.ins().f64const(v.value)
                },
                ty: CType::Float(v.bits),
            }),
            String(s) => Ok(TypedValue {
                value: self.builder.ins().global_addr(self.strings[s].id),
                ty: CType::Int(8, false).pointer(),
            }),
            Identifier(n) if self.types.constants.contains_key(n) => {
                self.integer(self.types.constants[n] as u64, CType::INT)
            }
            Identifier(n) if self.lookup(n).is_none() && self.functions.contains_key(n) => {
                let f = &self.functions[n];
                Ok(TypedValue {
                    value: self.builder.ins().func_addr(f.id),
                    ty: CType::Function(Box::new(f.ty.clone())).pointer(),
                })
            }
            Identifier(_)
            | ArrayAccess(..)
            | MemberAccess(..)
            | PointerMemberAccess(..)
            | Dereference(_) => {
                let p = self.place(expr)?;
                self.read(p)
            }
            Parenthesized(e) => self.expr(e),
            Cast(ty, e) => {
                let ty = self.types.type_name(ty)?;
                let v = self.expr(e)?;
                self.cast(v, &ty)
            }
            AddressOf(e) => {
                if let Identifier(n) = &**e {
                    if self.functions.contains_key(n) && self.lookup(n).is_none() {
                        return self.expr(e);
                    }
                }
                let p = self.place(e)?;
                let Location::Memory(value) = p.location else {
                    return fail("address-taken variable was not assigned storage");
                };
                Ok(TypedValue {
                    value,
                    ty: p.ty.pointer(),
                })
            }
            SizeofType(specs, decl) => {
                let base = self.types.specifiers(&qualifiers(specs))?;
                let ty = if let Some(d) = decl {
                    self.types.abstract_type(base, d)?
                } else {
                    base
                };
                let size = self.types.layout(&ty)?.0;
                self.integer(size as u64, CType::SIZE)
            }
            SizeofExpression(e) => {
                let ty = self.expr_type(e)?;
                let size = self.types.layout(&ty)?.0;
                self.integer(size as u64, CType::SIZE)
            }
            Assign(a, b) => {
                let p = self.place(a)?;
                let v = self.expr(b)?;
                self.write(&p, v)
            }
            AddAssign(a, b)
            | SubtractAssign(a, b)
            | MultiplyAssign(a, b)
            | DivideAssign(a, b)
            | ModuloAssign(a, b)
            | ShiftLeftAssign(a, b)
            | ShiftRightAssign(a, b)
            | BitwiseAndAssign(a, b)
            | BitwiseOrAssign(a, b)
            | BitwiseXorAssign(a, b) => {
                let p = self.place(a)?;
                let a = self.read(p.clone())?;
                let b = self.expr(b)?;
                let v = self.binary(expr, a, b)?;
                self.write(&p, v)
            }
            PreIncrement(e) | PreDecrement(e) | PostIncrement(e) | PostDecrement(e) => {
                let p = self.place(e)?;
                let old = self.read(p.clone())?;
                let one = self.integer(1, CType::INT)?;
                let subtract = matches!(expr, PreDecrement(_) | PostDecrement(_));
                let op = if subtract {
                    Subtract(e.clone(), e.clone())
                } else {
                    Add(e.clone(), e.clone())
                };
                let new = self.binary(&op, old.clone(), one)?;
                let new = self.write(&p, new)?;
                Ok(if matches!(expr, PostIncrement(_) | PostDecrement(_)) {
                    old
                } else {
                    new
                })
            }
            UnaryPlus(e) | UnaryMinus(e) | BitwiseNot(e) => {
                let v = self.expr(e)?;
                let ty = v.ty.promoted();
                let mut v = self.cast(v, &ty)?;
                v.value = match expr {
                    UnaryMinus(_) => {
                        if matches!(ty, CType::Float(_)) {
                            self.builder.ins().fneg(v.value)
                        } else {
                            self.builder.ins().ineg(v.value)
                        }
                    }
                    BitwiseNot(_) => {
                        let mask = self.integer(u64::MAX, ty)?;
                        self.builder.ins().ixor(v.value, mask.value)
                    }
                    _ => v.value,
                };
                Ok(v)
            }
            LogicalNot(e) => {
                let v = self.expr(e)?;
                let condition = self.truth(v)?;
                let condition = self.boolean_int(condition);
                let result = self.builder.ins().ieqz(condition.value);
                Ok(self.boolean_int(result))
            }
            LogicalAnd(a, b) | LogicalOr(a, b) => self.logical(a, b, matches!(expr, LogicalOr(..))),
            Conditional(c, a, b) => self.conditional(c, a, b),
            Comma(a, b) => {
                self.expr(a)?;
                self.expr(b)
            }
            FunctionCall(c, args) => self.call(c, args),
            Add(a, b)
            | Subtract(a, b)
            | Multiply(a, b)
            | Divide(a, b)
            | Modulo(a, b)
            | ShiftLeft(a, b)
            | ShiftRight(a, b)
            | BitwiseAnd(a, b)
            | BitwiseOr(a, b)
            | BitwiseXor(a, b)
            | Equal(a, b)
            | NotEqual(a, b)
            | LessThan(a, b)
            | LessThanOrEqual(a, b)
            | GreaterThan(a, b)
            | GreaterThanOrEqual(a, b) => {
                let a = self.expr(a)?;
                let b = self.expr(b)?;
                self.binary(expr, a, b)
            }
            _ => fail(format!("unsupported expression: {expr:?}")),
        }
    }
    fn binary(&mut self, op: &Expression, a: TypedValue, b: TypedValue) -> Result<TypedValue> {
        use Expression::*;
        if matches!(
            op,
            Add(..) | AddAssign(..) | Subtract(..) | SubtractAssign(..)
        ) {
            let sub = matches!(op, Subtract(..) | SubtractAssign(..));
            if matches!(a.ty.plain(), CType::Pointer(_)) {
                if matches!(b.ty.plain(), CType::Pointer(_)) && sub {
                    let size = self.types.layout(&a.ty.pointee()?)?.0;
                    let a = self.cast(a, &CType::Int(64, true))?;
                    let b = self.cast(b, &CType::Int(64, true))?;
                    let diff = self.builder.ins().isub(a.value, b.value);
                    let stride = self.integer(size as u64, CType::Int(64, true))?;
                    return Ok(TypedValue {
                        value: self.builder.ins().idiv_s(diff, stride.value),
                        ty: CType::Int(64, true),
                    });
                }
                return self.offset(a, b, sub);
            }
            if !sub && matches!(b.ty.plain(), CType::Pointer(_)) {
                return self.offset(b, a, false);
            }
        }
        let shift = matches!(
            op,
            ShiftLeft(..) | ShiftRight(..) | ShiftLeftAssign(..) | ShiftRightAssign(..)
        );
        let ty = if shift {
            a.ty.promoted()
        } else {
            CType::common(&a.ty, &b.ty)
        };
        let a = self.cast(a, &ty)?;
        let b = self.cast(b, &ty)?;
        let (a, b) = if matches!(ty, CType::Pointer(_)) {
            (
                self.builder.ins().ptrtoint(a.value, Type::I64),
                self.builder.ins().ptrtoint(b.value, Type::I64),
            )
        } else {
            (a.value, b.value)
        };
        let float = matches!(ty, CType::Float(_));
        let signed = ty.signed();
        let value = match op {
            Add(..) | AddAssign(..) => {
                if float {
                    self.builder.ins().fadd(a, b)
                } else {
                    self.builder.ins().iadd(a, b)
                }
            }
            Subtract(..) | SubtractAssign(..) => {
                if float {
                    self.builder.ins().fsub(a, b)
                } else {
                    self.builder.ins().isub(a, b)
                }
            }
            Multiply(..) | MultiplyAssign(..) => {
                if float {
                    self.builder.ins().fmul(a, b)
                } else {
                    self.builder.ins().imul(a, b)
                }
            }
            Divide(..) | DivideAssign(..) => {
                if float {
                    self.builder.ins().fdiv(a, b)
                } else if signed {
                    self.builder.ins().idiv_s(a, b)
                } else {
                    self.builder.ins().idiv_u(a, b)
                }
            }
            Modulo(..) | ModuloAssign(..) => {
                if signed {
                    self.builder.ins().irem_s(a, b)
                } else {
                    self.builder.ins().irem_u(a, b)
                }
            }
            ShiftLeft(..) | ShiftLeftAssign(..) => self.builder.ins().ishl(a, b),
            ShiftRight(..) | ShiftRightAssign(..) => {
                if signed {
                    self.builder.ins().ishr_s(a, b)
                } else {
                    self.builder.ins().ishr_u(a, b)
                }
            }
            BitwiseAnd(..) | BitwiseAndAssign(..) => self.builder.ins().iand(a, b),
            BitwiseOr(..) | BitwiseOrAssign(..) => self.builder.ins().ior(a, b),
            BitwiseXor(..) | BitwiseXorAssign(..) => self.builder.ins().ixor(a, b),
            _ => {
                let value = if float {
                    let cc = match op {
                        Equal(..) => FloatCC::Eq,
                        NotEqual(..) => FloatCC::Ne,
                        LessThan(..) => FloatCC::Lt,
                        LessThanOrEqual(..) => FloatCC::Le,
                        GreaterThan(..) => FloatCC::Gt,
                        GreaterThanOrEqual(..) => FloatCC::Ge,
                        _ => return fail("invalid comparison"),
                    };
                    self.builder.ins().fcmp(cc, a, b)
                } else {
                    let cc = match op {
                        Equal(..) => IntCC::Eq,
                        NotEqual(..) => IntCC::Ne,
                        LessThan(..) => {
                            if signed {
                                IntCC::LtS
                            } else {
                                IntCC::LtU
                            }
                        }
                        LessThanOrEqual(..) => {
                            if signed {
                                IntCC::LeS
                            } else {
                                IntCC::LeU
                            }
                        }
                        GreaterThan(..) => {
                            if signed {
                                IntCC::GtS
                            } else {
                                IntCC::GtU
                            }
                        }
                        GreaterThanOrEqual(..) => {
                            if signed {
                                IntCC::GeS
                            } else {
                                IntCC::GeU
                            }
                        }
                        _ => return fail("invalid comparison"),
                    };
                    self.builder.ins().icmp(cc, a, b)
                };
                return Ok(self.boolean_int(value));
            }
        };
        Ok(TypedValue { value, ty })
    }
    fn call(&mut self, callee: &Expression, args: &[Expression]) -> Result<TypedValue> {
        let direct = if let Expression::Identifier(name) = callee {
            if self.lookup(name).is_none() {
                self.functions.get(name).cloned()
            } else {
                None
            }
        } else {
            None
        };
        let (ty, ptr) = if let Some(f) = &direct {
            (f.ty.clone(), None)
        } else {
            let v = self.expr(callee)?;
            let CType::Function(f) = v.ty.pointee()? else {
                return fail("callee is not a function");
            };
            (*f, Some(v.value))
        };
        if args.len() < ty.params.len() || (!ty.variadic && args.len() != ty.params.len()) {
            return fail("argument count mismatch");
        }
        let mut values = Vec::with_capacity(args.len());
        for (index, arg) in args.iter().enumerate() {
            let v = self.expr(arg)?;
            let target = if let Some((_, target)) = ty.params.get(index) {
                target.clone()
            } else if v.ty == CType::Float(32) {
                CType::Float(64)
            } else {
                v.ty.promoted()
            };
            let v = self.cast(v, &target)?;
            values.push(self.cast(v, &target.abi_type())?.value);
        }
        let inst = if let Some(f) = direct {
            self.builder.ins().call(f.id, &values)
        } else {
            let sig = self
                .signatures
                .get(&ty.abi_signature()?)
                .copied()
                .ok_or_else(|| Error::semantic("indirect-call signature not prepared", 0, 0))?;
            self.builder.ins().call_indirect(sig, ptr.unwrap(), &values)
        };
        let value = if let Some(v) = self.builder.func().dfg().first_result(inst) {
            v
        } else {
            self.integer(0, CType::INT)?.value
        };
        self.cast(
            TypedValue {
                value,
                ty: ty.result.abi_type(),
            },
            &ty.result,
        )
    }
    fn expr_type(&mut self, expr: &Expression) -> Result<CType> {
        use Expression::*;
        Ok(match expr {
            Identifier(n) => {
                if let Some(p) = self.lookup(n) {
                    p.ty
                } else if let Some(g) = self.globals.get(n) {
                    g.ty.clone()
                } else if let Some(f) = self.functions.get(n) {
                    CType::Function(Box::new(f.ty.clone()))
                } else {
                    CType::INT
                }
            }
            String(s) => CType::Array(Box::new(CType::Int(8, false)), string_bytes(s)?.len() + 1),
            Integer(n) => CType::Int(n.bits, n.signed),
            Float(v) => CType::Float(v.bits),
            Char(_) => CType::INT,
            Parenthesized(e) | PostIncrement(e) | PostDecrement(e) | PreIncrement(e)
            | PreDecrement(e) => self.expr_type(e)?,
            UnaryPlus(e) | UnaryMinus(e) | BitwiseNot(e) => self.expr_type(e)?.promoted(),
            Cast(t, _) => self.types.type_name(t)?,
            AddressOf(e) => self.expr_type(e)?.pointer(),
            Dereference(e) => self.expr_type(e)?.pointee()?,
            ArrayAccess(a, b) => {
                let a = self.expr_type(a)?;
                if matches!(a.plain(), CType::Array(..) | CType::Pointer(_)) {
                    a.pointee()?
                } else {
                    self.expr_type(b)?.pointee()?
                }
            }
            MemberAccess(e, n) => {
                let ty = self.expr_type(e)?;
                self.types.member(&ty, n)?.ty
            }
            PointerMemberAccess(e, n) => {
                let ty = self.expr_type(e)?.pointee()?;
                self.types.member(&ty, n)?.ty
            }
            SizeofExpression(_) | SizeofType(..) | AlignofType(..) => CType::SIZE,
            Equal(..)
            | NotEqual(..)
            | LessThan(..)
            | LessThanOrEqual(..)
            | GreaterThan(..)
            | GreaterThanOrEqual(..)
            | LogicalNot(_)
            | LogicalAnd(..)
            | LogicalOr(..) => CType::INT,
            Subtract(a, b) => {
                let a = self.expr_type(a)?.decay();
                let b = self.expr_type(b)?.decay();
                if matches!((&a, &b), (CType::Pointer(_), CType::Pointer(_))) {
                    CType::Int(64, true)
                } else {
                    CType::common(&a, &b)
                }
            }
            Conditional(_, a, b)
            | Add(a, b)
            | Multiply(a, b)
            | Divide(a, b)
            | Modulo(a, b)
            | BitwiseAnd(a, b)
            | BitwiseOr(a, b)
            | BitwiseXor(a, b) => CType::common(&self.expr_type(a)?, &self.expr_type(b)?),
            ShiftLeft(a, _) | ShiftRight(a, _) => self.expr_type(a)?.promoted(),
            Assign(a, _)
            | AddAssign(a, _)
            | SubtractAssign(a, _)
            | MultiplyAssign(a, _)
            | DivideAssign(a, _)
            | ModuloAssign(a, _)
            | ShiftLeftAssign(a, _)
            | ShiftRightAssign(a, _)
            | BitwiseAndAssign(a, _)
            | BitwiseOrAssign(a, _)
            | BitwiseXorAssign(a, _) => self.expr_type(a)?,
            Comma(_, b) => self.expr_type(b)?,
            FunctionCall(c, _) => {
                let ty = self.expr_type(c)?.decay().pointee()?;
                let CType::Function(f) = ty else {
                    return fail("callee is not a function");
                };
                f.result
            }
            _ => return fail("unsupported expression type"),
        })
    }
}
