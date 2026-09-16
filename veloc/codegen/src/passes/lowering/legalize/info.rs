extern crate alloc;

use crate::error::{Error, Result};
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use veloc_lir::{InstField, MachineFunction, Reg};
use veloc_mir::{Type, TypeInfo};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LegalizeAction {
    Legal,
    Lower,
    WidenScalar { to: Type },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LegalizeResult {
    Replace(Vec<veloc_lir::InstId>),
}

#[derive(Debug, Clone, Copy)]
pub enum TypePattern {
    Exact(Type),
    OneOf(&'static [Type]),
    Family(TypeFamilyPattern),
    Vector(VectorPattern),
    Predicate(TypePredicate),
    Any,
}

impl PartialEq for TypePattern {
    fn eq(&self, other: &Self) -> bool {
        match (*self, *other) {
            (Self::Exact(lhs), Self::Exact(rhs)) => lhs == rhs,
            (Self::OneOf(lhs), Self::OneOf(rhs)) => lhs == rhs,
            (Self::Family(lhs), Self::Family(rhs)) => lhs == rhs,
            (Self::Vector(lhs), Self::Vector(rhs)) => lhs == rhs,
            (Self::Predicate(lhs), Self::Predicate(rhs)) => core::ptr::fn_addr_eq(lhs, rhs),
            (Self::Any, Self::Any) => true,
            _ => false,
        }
    }
}

impl Eq for TypePattern {}

impl TypePattern {
    pub fn matches(self, ty: Type) -> bool {
        match self {
            Self::Exact(expected) => ty == expected,
            Self::OneOf(options) => options.contains(&ty),
            Self::Family(family) => family.matches(ty),
            Self::Vector(vector) => vector.matches(ty),
            Self::Predicate(predicate) => predicate(ty),
            Self::Any => true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeFamilyPattern {
    ScalarInt(&'static [u16]),
    ScalarFloat(&'static [u16]),
    ScalarNumeric(&'static [u16]),
    IntOrPtrScalar(&'static [u16]),
    ScalarValue(&'static [u16]),
}

impl TypeFamilyPattern {
    pub fn matches(self, ty: Type) -> bool {
        match self {
            Self::ScalarInt(widths) => is_scalar_int_type(ty) && type_width_is_one_of(ty, widths),
            Self::ScalarFloat(widths) => {
                is_scalar_float_type(ty) && type_width_is_one_of(ty, widths)
            }
            Self::ScalarNumeric(widths) => {
                is_scalar_numeric_type(ty) && type_width_is_one_of(ty, widths)
            }
            Self::IntOrPtrScalar(widths) => {
                is_int_or_ptr_scalar_type(ty) && type_width_is_one_of(ty, widths)
            }
            Self::ScalarValue(widths) => {
                is_scalar_value_type(ty) && type_width_is_one_of(ty, widths)
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VectorPattern {
    pub element: Type,
    pub lanes: Option<u16>,
    pub scalable: Option<bool>,
}

impl VectorPattern {
    pub const fn new(element: Type, lanes: Option<u16>, scalable: Option<bool>) -> Self {
        Self {
            element,
            lanes,
            scalable,
        }
    }

    pub fn matches(self, ty: Type) -> bool {
        let Some(ty) = ty.as_vector() else {
            return false;
        };

        if ty.element_type().as_type() != self.element {
            return false;
        }

        if let Some(lanes) = self.lanes {
            if ty.lane_count() != lanes {
                return false;
            }
        }

        if let Some(scalable) = self.scalable {
            if ty.is_scalable() != scalable {
                return false;
            }
        }

        true
    }
}

pub type TypePredicate = fn(Type) -> bool;

pub fn is_int_type(ty: Type) -> bool {
    ty.is_integer()
}

pub fn is_scalar_int_type(ty: Type) -> bool {
    ty.is_scalar() && ty.is_integer()
}

pub fn is_float_type(ty: Type) -> bool {
    ty.is_float()
}

pub fn is_scalar_float_type(ty: Type) -> bool {
    ty.is_scalar() && ty.is_float()
}

pub fn is_numeric_type(ty: Type) -> bool {
    ty.is_integer() || ty.is_float()
}

pub fn is_scalar_numeric_type(ty: Type) -> bool {
    ty.is_scalar() && is_numeric_type(ty)
}

pub fn is_int_or_ptr_type(ty: Type) -> bool {
    ty.is_integer() || ty.is_ptr()
}

pub fn is_int_or_ptr_scalar_type(ty: Type) -> bool {
    ty.is_scalar() && is_int_or_ptr_type(ty)
}

pub fn is_ptr_type(ty: Type) -> bool {
    ty.is_ptr()
}

pub fn is_ptr_sized_type(ty: Type) -> bool {
    ty.is_ptr() || (is_scalar_int_type(ty) && ty.min_bit_width() == Some(64))
}

pub fn is_ptr_sized_int_type(ty: Type) -> bool {
    is_scalar_int_type(ty) && ty.min_bit_width() == Some(64)
}

pub fn is_scalar_type(ty: Type) -> bool {
    ty.is_scalar()
}

pub fn is_scalar_value_type(ty: Type) -> bool {
    ty.is_scalar() && (ty.is_integer() || ty.is_float() || ty.is_ptr())
}

pub fn is_vector_type(ty: Type) -> bool {
    ty.is_vector()
}

pub fn is_fixed_vector_type(ty: Type) -> bool {
    ty.is_vector() && !ty.is_scalable()
}

pub fn is_scalable_vector_type(ty: Type) -> bool {
    ty.is_vector() && ty.is_scalable()
}

pub fn is_predicate_type(ty: Type) -> bool {
    ty.is_predicate()
}

fn legalization_bit_width(ty: Type) -> Option<u32> {
    // The only code-generation target currently wired to this generic matcher
    // is 64-bit. Move this fallback into target-owned matching metadata when a
    // target with a different pointer width is added.
    ty.min_bit_width().or_else(|| ty.is_ptr().then_some(64))
}

pub fn type_width_is_one_of(ty: Type, widths: &[u16]) -> bool {
    widths.is_empty()
        || widths
            .iter()
            .copied()
            .any(|width| legalization_bit_width(ty) == Some(u32::from(width)))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperandPattern {
    Def(TypePattern),
    Use(TypePattern),
    Imm,
    FImm,
    Block,
    StackSlot,
    CondCode,
    Global,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperandSeqPattern {
    Fixed(OperandPattern),
    Rest(OperandPattern),
}

/// Matching consumes each storage domain independently. A pattern's role selects
/// the domain; attributes never occupy register positions.
#[derive(Clone, Copy)]
struct Operands<'a> {
    results: &'a [Reg],
    inputs: &'a [Reg],
    fields: &'a [InstField],
}
impl<'a> Operands<'a> {
    fn new(inst: &veloc_lir::InstRef<'a>) -> Self {
        Self {
            results: inst.results(),
            inputs: inst.inputs(),
            fields: inst.fields(),
        }
    }
    fn is_empty(self) -> bool {
        self.results.is_empty() && self.inputs.is_empty() && self.fields.is_empty()
    }
    fn take(&mut self, mfunc: &MachineFunction, pattern: OperandPattern) -> Result<bool> {
        match pattern {
            OperandPattern::Def(ty) | OperandPattern::Use(ty) => {
                let regs = if matches!(pattern, OperandPattern::Def(_)) {
                    &mut self.results
                } else {
                    &mut self.inputs
                };
                let Some((&reg, rest)) = regs.split_first() else {
                    return Ok(false);
                };
                *regs = rest;
                Ok(ty.matches(reg_type(mfunc, reg)?))
            }
            _ => {
                let Some((field, rest)) = self.fields.split_first() else {
                    return Ok(false);
                };
                self.fields = rest;
                Ok(matches!(
                    (field, pattern),
                    (InstField::Imm(_), OperandPattern::Imm)
                        | (InstField::FImm(_), OperandPattern::FImm)
                        | (InstField::Block(_), OperandPattern::Block)
                        | (InstField::StackSlot(_), OperandPattern::StackSlot)
                        | (
                            InstField::IntCC(_) | InstField::FloatCC(_),
                            OperandPattern::CondCode
                        )
                        | (InstField::Global(_), OperandPattern::Global)
                ))
            }
        }
    }
}
pub fn format_inst_operands(
    inst: &veloc_lir::InstRef<'_>,
    mfunc: &MachineFunction,
) -> Result<String> {
    let mut parts = Vec::new();
    for &reg in inst.results() {
        parts.push(format!("def({:?})", reg_type(mfunc, reg)?));
    }
    for &reg in inst.inputs() {
        parts.push(format!("use({:?})", reg_type(mfunc, reg)?));
    }
    for field in inst.fields() {
        parts.push(
            match field {
                InstField::Imm(_) => "imm",
                InstField::FImm(_) => "fimm",
                InstField::Block(_) => "block",
                InstField::StackSlot(_) => "stackslot",
                InstField::IntCC(_) | InstField::FloatCC(_) => "condcode",
                InstField::Global(_) => "global",
            }
            .to_string(),
        );
    }
    Ok(format!("[{}]", parts.join(", ")))
}
pub fn inst_matches_operands(
    inst: &veloc_lir::InstRef<'_>,
    mfunc: &MachineFunction,
    patterns: &[OperandPattern],
) -> Result<bool> {
    let mut operands = Operands::new(inst);
    for &pattern in patterns {
        if !operands.take(mfunc, pattern)? {
            return Ok(false);
        }
    }
    Ok(operands.is_empty())
}
pub fn inst_matches_operand_sequence(
    inst: &veloc_lir::InstRef<'_>,
    mfunc: &MachineFunction,
    patterns: &[OperandSeqPattern],
) -> Result<bool> {
    match_operand_sequence_impl(Operands::new(inst), mfunc, patterns)
}

pub fn operand_type_at(
    inst: &veloc_lir::InstRef<'_>,
    mfunc: &MachineFunction,
    index: usize,
) -> Result<Option<Type>> {
    inst.results()
        .iter()
        .chain(inst.inputs())
        .nth(index)
        .map(|&reg| reg_type(mfunc, reg))
        .transpose()
}

pub fn operand_bit_width_at(
    inst: &veloc_lir::InstRef<'_>,
    mfunc: &MachineFunction,
    index: usize,
) -> Result<Option<usize>> {
    Ok(operand_type_at(inst, mfunc, index)?
        .and_then(legalization_bit_width)
        .map(|width| width as usize))
}

pub fn same_operand_types(
    inst: &veloc_lir::InstRef<'_>,
    mfunc: &MachineFunction,
    indices: &[usize],
) -> Result<bool> {
    same_operand_property(inst, mfunc, indices, operand_type_at)
}

pub fn same_operand_widths(
    inst: &veloc_lir::InstRef<'_>,
    mfunc: &MachineFunction,
    indices: &[usize],
) -> Result<bool> {
    same_operand_property(inst, mfunc, indices, operand_bit_width_at)
}

fn same_operand_property<T, F>(
    inst: &veloc_lir::InstRef<'_>,
    mfunc: &MachineFunction,
    indices: &[usize],
    mut property_at: F,
) -> Result<bool>
where
    T: Copy + PartialEq,
    F: FnMut(&veloc_lir::InstRef<'_>, &MachineFunction, usize) -> Result<Option<T>>,
{
    let Some((&first_index, rest)) = indices.split_first() else {
        return Ok(true);
    };

    let Some(first_value) = property_at(inst, mfunc, first_index)? else {
        return Ok(false);
    };

    for &index in rest {
        if property_at(inst, mfunc, index)? != Some(first_value) {
            return Ok(false);
        }
    }

    Ok(true)
}

fn match_operand_sequence_impl(
    mut operands: Operands<'_>,
    mfunc: &MachineFunction,
    patterns: &[OperandSeqPattern],
) -> Result<bool> {
    let Some((first, rest)) = patterns.split_first() else {
        return Ok(operands.is_empty());
    };
    match *first {
        OperandSeqPattern::Fixed(pattern) => {
            if !operands.take(mfunc, pattern)? {
                return Ok(false);
            }
            match_operand_sequence_impl(operands, mfunc, rest)
        }
        OperandSeqPattern::Rest(pattern) => loop {
            if match_operand_sequence_impl(operands, mfunc, rest)? {
                return Ok(true);
            }
            if !operands.take(mfunc, pattern)? {
                return Ok(false);
            }
        },
    }
}

fn reg_type(mfunc: &MachineFunction, reg: Reg) -> Result<Type> {
    if reg.is_vreg() {
        Ok(mfunc.vreg_data(reg).ty)
    } else if reg.is_preg() {
        Err(Error::codegen(alloc::format!(
            "legalization expected virtual registers, found physical register {:?}",
            reg
        )))
    } else {
        Err(Error::codegen("legalization encountered invalid register"))
    }
}

#[macro_export]
macro_rules! legalize_matcher {
    ($inst:expr, $mfunc:expr, { $($body:tt)* }) => {{
        (|| -> $crate::error::Result<Option<$crate::passes::lowering::LegalizeAction>> {
            $crate::legalize_matcher!(@dispatch $inst, $mfunc, { $($body)* })
        })()
    }};
    (@dispatch $inst:expr, $mfunc:expr, { $($opcode:ident)|+ => { $($rules:tt)* }; $($rest:tt)* }) => {{
        match $inst.generic_opcode() {
            $(Some($crate::veloc_lir::GenericOpcode::$opcode))|+ => {
                $crate::legalize_matcher!(@rules $inst, $mfunc, $($rules)*)
            }
            _ => $crate::legalize_matcher!(@dispatch $inst, $mfunc, { $($rest)* }),
        }
    }};
    (@dispatch $inst:expr, $mfunc:expr, {}) => {
        Ok(None)
    };
    (@rules $inst:expr, $mfunc:expr,) => {
        Ok(None)
    };
    (@emit_action (legal)) => {
        Ok(Some($crate::passes::lowering::LegalizeAction::Legal))
    };
    (@emit_action (lower)) => {
        Ok(Some($crate::passes::lowering::LegalizeAction::Lower))
    };
    (@emit_action (widen_scalar $($ty:tt)+)) => {
        Ok(Some($crate::passes::lowering::LegalizeAction::WidenScalar {
            to: $crate::legalize_matcher!(@type $($ty)+),
        }))
    };
    (@emit_action (expr $action:expr)) => {
        Ok(Some($action))
    };
    (@match_fixed $inst:expr, $mfunc:expr,
        $patterns:expr,
        $guard:expr,
        $emit:tt,
        $($rest:tt)*
    ) => {{
        if $crate::passes::lowering::legalize::info::inst_matches_operands(
            $inst,
            $mfunc,
            $patterns,
        )? && ($guard) {
            $crate::legalize_matcher!(@emit_action $emit)
        } else {
            $crate::legalize_matcher!(@rules $inst, $mfunc, $($rest)*)
        }
    }};
    (@match_sequence $inst:expr, $mfunc:expr,
        $patterns:expr,
        $guard:expr,
        $emit:tt,
        $($rest:tt)*
    ) => {{
        if $crate::passes::lowering::legalize::info::inst_matches_operand_sequence(
            $inst,
            $mfunc,
            &($patterns),
        )? && ($guard) {
            $crate::legalize_matcher!(@emit_action $emit)
        } else {
            $crate::legalize_matcher!(@rules $inst, $mfunc, $($rest)*)
        }
    }};
    (@guard_expr $inst:expr, $mfunc:expr, same_widths($($index:expr),+ $(,)?)) => {
        $crate::passes::lowering::legalize::info::same_operand_widths(
            $inst,
            $mfunc,
            &[$($index),+],
        )?
    };
    (@guard_expr $inst:expr, $mfunc:expr, same_types($($index:expr),+ $(,)?) ) => {
        $crate::passes::lowering::legalize::info::same_operand_types(
            $inst,
            $mfunc,
            &[$($index),+],
        )?
    };
    (@guard_expr $inst:expr, $mfunc:expr, $guard:expr) => {
        $guard
    };
    (@match_fixed_action
        $inst:expr,
        $mfunc:expr,
        $patterns:expr,
        $guard:expr,
        legal,
        $($rest:tt)*
    ) => {{
        $crate::legalize_matcher!(@match_fixed
            $inst,
            $mfunc,
            $patterns,
            $guard,
            (legal),
            $($rest)*
        )
    }};
    (@match_fixed_action
        $inst:expr,
        $mfunc:expr,
        $patterns:expr,
        $guard:expr,
        lower,
        $($rest:tt)*
    ) => {{
        $crate::legalize_matcher!(@match_fixed
            $inst,
            $mfunc,
            $patterns,
            $guard,
            (lower),
            $($rest)*
        )
    }};
    (@match_fixed_action
        $inst:expr,
        $mfunc:expr,
        $patterns:expr,
        $guard:expr,
        widen_scalar($($ty:tt)+),
        $($rest:tt)*
    ) => {{
        $crate::legalize_matcher!(@match_fixed
            $inst,
            $mfunc,
            $patterns,
            $guard,
            (widen_scalar $($ty)+),
            $($rest)*
        )
    }};
    (@match_fixed_action
        $inst:expr,
        $mfunc:expr,
        $patterns:expr,
        $guard:expr,
        $action:expr,
        $($rest:tt)*
    ) => {{
        $crate::legalize_matcher!(@match_fixed
            $inst,
            $mfunc,
            $patterns,
            $guard,
            (expr $action),
            $($rest)*
        )
    }};
    (@match_sequence_action
        $inst:expr,
        $mfunc:expr,
        $patterns:expr,
        $guard:expr,
        legal,
        $($rest:tt)*
    ) => {{
        $crate::legalize_matcher!(@match_sequence
            $inst,
            $mfunc,
            $patterns,
            $guard,
            (legal),
            $($rest)*
        )
    }};
    (@match_sequence_action
        $inst:expr,
        $mfunc:expr,
        $patterns:expr,
        $guard:expr,
        lower,
        $($rest:tt)*
    ) => {{
        $crate::legalize_matcher!(@match_sequence
            $inst,
            $mfunc,
            $patterns,
            $guard,
            (lower),
            $($rest)*
        )
    }};
    (@match_sequence_action
        $inst:expr,
        $mfunc:expr,
        $patterns:expr,
        $guard:expr,
        widen_scalar($($ty:tt)+),
        $($rest:tt)*
    ) => {{
        $crate::legalize_matcher!(@match_sequence
            $inst,
            $mfunc,
            $patterns,
            $guard,
            (widen_scalar $($ty)+),
            $($rest)*
        )
    }};
    (@match_sequence_action
        $inst:expr,
        $mfunc:expr,
        $patterns:expr,
        $guard:expr,
        $action:expr,
        $($rest:tt)*
    ) => {{
        $crate::legalize_matcher!(@match_sequence
            $inst,
            $mfunc,
            $patterns,
            $guard,
            (expr $action),
            $($rest)*
        )
    }};
    (@rules $inst:expr, $mfunc:expr,
        [$($kind:ident $(($($ty:tt)+))?),* $(,)?] if same_widths($($index:expr),+ $(,)?) => $action:ident,
        $($rest:tt)*
    ) => {{
        $crate::legalize_matcher!(@match_fixed_action
            $inst,
            $mfunc,
            &[$($crate::legalize_matcher!(@pattern $kind $(($($ty)+))?)),*],
            ($crate::legalize_matcher!(@guard_expr $inst, $mfunc, same_widths($($index),+))),
            $action,
            $($rest)*
        )
    }};
    (@rules $inst:expr, $mfunc:expr,
        [$($kind:ident $(($($ty:tt)+))?),* $(,)?] if same_types($($index:expr),+ $(,)?) => $action:ident,
        $($rest:tt)*
    ) => {{
        $crate::legalize_matcher!(@match_fixed_action
            $inst,
            $mfunc,
            &[$($crate::legalize_matcher!(@pattern $kind $(($($ty)+))?)),*],
            ($crate::legalize_matcher!(@guard_expr $inst, $mfunc, same_types($($index),+))),
            $action,
            $($rest)*
        )
    }};
    (@rules $inst:expr, $mfunc:expr,
        [$($kind:ident $(($($ty:tt)+))?),* $(,)?] if $guard:expr => $action:ident,
        $($rest:tt)*
    ) => {{
        $crate::legalize_matcher!(@match_fixed_action
            $inst,
            $mfunc,
            &[$($crate::legalize_matcher!(@pattern $kind $(($($ty)+))?)),*],
            ($crate::legalize_matcher!(@guard_expr $inst, $mfunc, $guard)),
            $action,
            $($rest)*
        )
    }};
    (@rules $inst:expr, $mfunc:expr,
        [$($kind:ident $(($($ty:tt)+))?),* $(,)?] => $action:ident,
        $($rest:tt)*
    ) => {{
        $crate::legalize_matcher!(@match_fixed_action
            $inst,
            $mfunc,
            &[$($crate::legalize_matcher!(@pattern $kind $(($($ty)+))?)),*],
            (true),
            $action,
            $($rest)*
        )
    }};
    (@rules $inst:expr, $mfunc:expr,
        [$($kind:ident $(($($ty:tt)+))?),* $(,)?] if same_widths($($index:expr),+ $(,)?) => widen_scalar($($widen_ty:tt)+),
        $($rest:tt)*
    ) => {{
        $crate::legalize_matcher!(@match_fixed_action
            $inst,
            $mfunc,
            &[$($crate::legalize_matcher!(@pattern $kind $(($($ty)+))?)),*],
            ($crate::legalize_matcher!(@guard_expr $inst, $mfunc, same_widths($($index),+))),
            widen_scalar($($widen_ty)+),
            $($rest)*
        )
    }};
    (@rules $inst:expr, $mfunc:expr,
        [$($kind:ident $(($($ty:tt)+))?),* $(,)?] if same_types($($index:expr),+ $(,)?) => widen_scalar($($widen_ty:tt)+),
        $($rest:tt)*
    ) => {{
        $crate::legalize_matcher!(@match_fixed_action
            $inst,
            $mfunc,
            &[$($crate::legalize_matcher!(@pattern $kind $(($($ty)+))?)),*],
            ($crate::legalize_matcher!(@guard_expr $inst, $mfunc, same_types($($index),+))),
            widen_scalar($($widen_ty)+),
            $($rest)*
        )
    }};
    (@rules $inst:expr, $mfunc:expr,
        [$($kind:ident $(($($ty:tt)+))?),* $(,)?] if $guard:expr => widen_scalar($($widen_ty:tt)+),
        $($rest:tt)*
    ) => {{
        $crate::legalize_matcher!(@match_fixed_action
            $inst,
            $mfunc,
            &[$($crate::legalize_matcher!(@pattern $kind $(($($ty)+))?)),*],
            ($crate::legalize_matcher!(@guard_expr $inst, $mfunc, $guard)),
            widen_scalar($($widen_ty)+),
            $($rest)*
        )
    }};
    (@rules $inst:expr, $mfunc:expr,
        [$($kind:ident $(($($ty:tt)+))?),* $(,)?] => widen_scalar($($widen_ty:tt)+),
        $($rest:tt)*
    ) => {{
        $crate::legalize_matcher!(@match_fixed_action
            $inst,
            $mfunc,
            &[$($crate::legalize_matcher!(@pattern $kind $(($($ty)+))?)),*],
            (true),
            widen_scalar($($widen_ty)+),
            $($rest)*
        )
    }};
    (@rules $inst:expr, $mfunc:expr,
        [$($kind:ident $(($($ty:tt)+))?),* $(,)?] if same_widths($($index:expr),+ $(,)?) => $action:expr,
        $($rest:tt)*
    ) => {{
        $crate::legalize_matcher!(@match_fixed_action
            $inst,
            $mfunc,
            &[$($crate::legalize_matcher!(@pattern $kind $(($($ty)+))?)),*],
            ($crate::legalize_matcher!(@guard_expr $inst, $mfunc, same_widths($($index),+))),
            $action,
            $($rest)*
        )
    }};
    (@rules $inst:expr, $mfunc:expr,
        [$($kind:ident $(($($ty:tt)+))?),* $(,)?] if same_types($($index:expr),+ $(,)?) => $action:expr,
        $($rest:tt)*
    ) => {{
        $crate::legalize_matcher!(@match_fixed_action
            $inst,
            $mfunc,
            &[$($crate::legalize_matcher!(@pattern $kind $(($($ty)+))?)),*],
            ($crate::legalize_matcher!(@guard_expr $inst, $mfunc, same_types($($index),+))),
            $action,
            $($rest)*
        )
    }};
    (@rules $inst:expr, $mfunc:expr,
        [$($kind:ident $(($($ty:tt)+))?),* $(,)?] if $guard:expr => $action:expr,
        $($rest:tt)*
    ) => {{
        $crate::legalize_matcher!(@match_fixed_action
            $inst,
            $mfunc,
            &[$($crate::legalize_matcher!(@pattern $kind $(($($ty)+))?)),*],
            ($crate::legalize_matcher!(@guard_expr $inst, $mfunc, $guard)),
            $action,
            $($rest)*
        )
    }};
    (@rules $inst:expr, $mfunc:expr,
        [$($kind:ident $(($($ty:tt)+))?),* $(,)?] => $action:expr,
        $($rest:tt)*
    ) => {{
        $crate::legalize_matcher!(@match_fixed_action
            $inst,
            $mfunc,
            &[$($crate::legalize_matcher!(@pattern $kind $(($($ty)+))?)),*],
            (true),
            $action,
            $($rest)*
        )
    }};
    (@rules $inst:expr, $mfunc:expr,
        seq[$($seq:tt)*] if same_widths($($index:expr),+ $(,)?) => $action:ident,
        $($rest:tt)*
    ) => {{
        $crate::legalize_matcher!(@match_sequence_action
            $inst,
            $mfunc,
            $crate::legalize_matcher!(@seq_array [] $($seq)*),
            ($crate::legalize_matcher!(@guard_expr $inst, $mfunc, same_widths($($index),+))),
            $action,
            $($rest)*
        )
    }};
    (@rules $inst:expr, $mfunc:expr,
        seq[$($seq:tt)*] if same_types($($index:expr),+ $(,)?) => $action:ident,
        $($rest:tt)*
    ) => {{
        $crate::legalize_matcher!(@match_sequence_action
            $inst,
            $mfunc,
            $crate::legalize_matcher!(@seq_array [] $($seq)*),
            ($crate::legalize_matcher!(@guard_expr $inst, $mfunc, same_types($($index),+))),
            $action,
            $($rest)*
        )
    }};
    (@rules $inst:expr, $mfunc:expr,
        seq[$($seq:tt)*] if $guard:expr => $action:ident,
        $($rest:tt)*
    ) => {{
        $crate::legalize_matcher!(@match_sequence_action
            $inst,
            $mfunc,
            $crate::legalize_matcher!(@seq_array [] $($seq)*),
            ($crate::legalize_matcher!(@guard_expr $inst, $mfunc, $guard)),
            $action,
            $($rest)*
        )
    }};
    (@rules $inst:expr, $mfunc:expr,
        seq[$($seq:tt)*] => $action:ident,
        $($rest:tt)*
    ) => {{
        $crate::legalize_matcher!(@match_sequence_action
            $inst,
            $mfunc,
            $crate::legalize_matcher!(@seq_array [] $($seq)*),
            (true),
            $action,
            $($rest)*
        )
    }};
    (@rules $inst:expr, $mfunc:expr,
        seq[$($seq:tt)*] if same_widths($($index:expr),+ $(,)?) => widen_scalar($($widen_ty:tt)+),
        $($rest:tt)*
    ) => {{
        $crate::legalize_matcher!(@match_sequence_action
            $inst,
            $mfunc,
            $crate::legalize_matcher!(@seq_array [] $($seq)*),
            ($crate::legalize_matcher!(@guard_expr $inst, $mfunc, same_widths($($index),+))),
            widen_scalar($($widen_ty)+),
            $($rest)*
        )
    }};
    (@rules $inst:expr, $mfunc:expr,
        seq[$($seq:tt)*] if same_types($($index:expr),+ $(,)?) => widen_scalar($($widen_ty:tt)+),
        $($rest:tt)*
    ) => {{
        $crate::legalize_matcher!(@match_sequence_action
            $inst,
            $mfunc,
            $crate::legalize_matcher!(@seq_array [] $($seq)*),
            ($crate::legalize_matcher!(@guard_expr $inst, $mfunc, same_types($($index),+))),
            widen_scalar($($widen_ty)+),
            $($rest)*
        )
    }};
    (@rules $inst:expr, $mfunc:expr,
        seq[$($seq:tt)*] if $guard:expr => widen_scalar($($widen_ty:tt)+),
        $($rest:tt)*
    ) => {{
        $crate::legalize_matcher!(@match_sequence_action
            $inst,
            $mfunc,
            $crate::legalize_matcher!(@seq_array [] $($seq)*),
            ($crate::legalize_matcher!(@guard_expr $inst, $mfunc, $guard)),
            widen_scalar($($widen_ty)+),
            $($rest)*
        )
    }};
    (@rules $inst:expr, $mfunc:expr,
        seq[$($seq:tt)*] => widen_scalar($($widen_ty:tt)+),
        $($rest:tt)*
    ) => {{
        $crate::legalize_matcher!(@match_sequence_action
            $inst,
            $mfunc,
            $crate::legalize_matcher!(@seq_array [] $($seq)*),
            (true),
            widen_scalar($($widen_ty)+),
            $($rest)*
        )
    }};
    (@rules $inst:expr, $mfunc:expr,
        seq[$($seq:tt)*] if same_widths($($index:expr),+ $(,)?) => $action:expr,
        $($rest:tt)*
    ) => {{
        $crate::legalize_matcher!(@match_sequence_action
            $inst,
            $mfunc,
            $crate::legalize_matcher!(@seq_array [] $($seq)*),
            ($crate::legalize_matcher!(@guard_expr $inst, $mfunc, same_widths($($index),+))),
            $action,
            $($rest)*
        )
    }};
    (@rules $inst:expr, $mfunc:expr,
        seq[$($seq:tt)*] if same_types($($index:expr),+ $(,)?) => $action:expr,
        $($rest:tt)*
    ) => {{
        $crate::legalize_matcher!(@match_sequence_action
            $inst,
            $mfunc,
            $crate::legalize_matcher!(@seq_array [] $($seq)*),
            ($crate::legalize_matcher!(@guard_expr $inst, $mfunc, same_types($($index),+))),
            $action,
            $($rest)*
        )
    }};
    (@rules $inst:expr, $mfunc:expr,
        seq[$($seq:tt)*] if $guard:expr => $action:expr,
        $($rest:tt)*
    ) => {{
        $crate::legalize_matcher!(@match_sequence_action
            $inst,
            $mfunc,
            $crate::legalize_matcher!(@seq_array [] $($seq)*),
            ($crate::legalize_matcher!(@guard_expr $inst, $mfunc, $guard)),
            $action,
            $($rest)*
        )
    }};
    (@rules $inst:expr, $mfunc:expr,
        seq[$($seq:tt)*] => $action:expr,
        $($rest:tt)*
    ) => {{
        $crate::legalize_matcher!(@match_sequence_action
            $inst,
            $mfunc,
            $crate::legalize_matcher!(@seq_array [] $($seq)*),
            (true),
            $action,
            $($rest)*
        )
    }};
    (@seq_array [$($out:expr,)*]) => {
        [$($out),*]
    };
    (@seq_array [$($out:expr,)*] , $($rest:tt)*) => {
        $crate::legalize_matcher!(@seq_array [$($out,)*] $($rest)*)
    };
    (@seq_array [$($out:expr,)*] .. $kind:ident $(($($ty:tt)+))? , $($rest:tt)*) => {
        $crate::legalize_matcher!(
            @seq_array
            [
                $($out,)*
                $crate::legalize_matcher!(@seq_pattern .. $kind $(($($ty)+))?),
            ]
            $($rest)*
        )
    };
    (@seq_array [$($out:expr,)*] $kind:ident $(($($ty:tt)+))? , $($rest:tt)*) => {
        $crate::legalize_matcher!(
            @seq_array
            [
                $($out,)*
                $crate::legalize_matcher!(@seq_pattern $kind $(($($ty)+))?),
            ]
            $($rest)*
        )
    };
    (@seq_array [$($out:expr,)*] .. $kind:ident $(($($ty:tt)+))?) => {
        [
            $($out,)*
            $crate::legalize_matcher!(@seq_pattern .. $kind $(($($ty)+))?)
        ]
    };
    (@seq_array [$($out:expr,)*] $kind:ident $(($($ty:tt)+))?) => {
        [
            $($out,)*
            $crate::legalize_matcher!(@seq_pattern $kind $(($($ty)+))?)
        ]
    };
    (@seq_pattern .. $kind:ident $(($($ty:tt)+))?) => {
        $crate::passes::lowering::legalize::info::OperandSeqPattern::Rest(
            $crate::legalize_matcher!(@pattern $kind $(($($ty)+))?)
        )
    };
    (@seq_pattern $kind:ident $(($($ty:tt)+))?) => {
        $crate::passes::lowering::legalize::info::OperandSeqPattern::Fixed(
            $crate::legalize_matcher!(@pattern $kind $(($($ty)+))?)
        )
    };
    (@pattern def($($ty:tt)+)) => {
        $crate::passes::lowering::legalize::info::OperandPattern::Def(
            $crate::legalize_matcher!(@type_pattern $($ty)+)
        )
    };
    (@pattern use($($ty:tt)+)) => {
        $crate::passes::lowering::legalize::info::OperandPattern::Use(
            $crate::legalize_matcher!(@type_pattern $($ty)+)
        )
    };
    (@pattern imm) => {
        $crate::passes::lowering::legalize::info::OperandPattern::Imm
    };
    (@pattern fimm) => {
        $crate::passes::lowering::legalize::info::OperandPattern::FImm
    };
    (@pattern block) => {
        $crate::passes::lowering::legalize::info::OperandPattern::Block
    };
    (@pattern stackslot) => {
        $crate::passes::lowering::legalize::info::OperandPattern::StackSlot
    };
    (@pattern condcode) => {
        $crate::passes::lowering::legalize::info::OperandPattern::CondCode
    };
    (@pattern global) => {
        $crate::passes::lowering::legalize::info::OperandPattern::Global
    };
    (@type_pattern _) => {
        $crate::passes::lowering::legalize::info::TypePattern::Any
    };
    (@type_pattern any) => {
        $crate::passes::lowering::legalize::info::TypePattern::Any
    };
    (@type_pattern vector_of($($elem:tt)+)) => {
        $crate::passes::lowering::legalize::info::TypePattern::Vector(
            $crate::passes::lowering::legalize::info::VectorPattern::new(
                $crate::legalize_matcher!(@type $($elem)+),
                None,
                None,
            )
        )
    };
    (@type_pattern fixed_vector_of($elem:ident ; $lanes:expr)) => {
        $crate::passes::lowering::legalize::info::TypePattern::Vector(
            $crate::passes::lowering::legalize::info::VectorPattern::new(
                $crate::legalize_matcher!(@type $elem),
                Some($lanes),
                Some(false),
            )
        )
    };
    (@type_pattern scalable_vector_of($elem:ident ; $lanes:expr)) => {
        $crate::passes::lowering::legalize::info::TypePattern::Vector(
            $crate::passes::lowering::legalize::info::VectorPattern::new(
                $crate::legalize_matcher!(@type $elem),
                Some($lanes),
                Some(true),
            )
        )
    };
    (@type_pattern scalar_int($($width:expr),+ $(,)?)) => {
        $crate::passes::lowering::legalize::info::TypePattern::Family(
            $crate::passes::lowering::legalize::info::TypeFamilyPattern::ScalarInt(&[
                $($width),+
            ])
        )
    };
    (@type_pattern scalar_float($($width:expr),+ $(,)?)) => {
        $crate::passes::lowering::legalize::info::TypePattern::Family(
            $crate::passes::lowering::legalize::info::TypeFamilyPattern::ScalarFloat(&[
                $($width),+
            ])
        )
    };
    (@type_pattern scalar_numeric($($width:expr),+ $(,)?)) => {
        $crate::passes::lowering::legalize::info::TypePattern::Family(
            $crate::passes::lowering::legalize::info::TypeFamilyPattern::ScalarNumeric(&[
                $($width),+
            ])
        )
    };
    (@type_pattern int_or_ptr_scalar($($width:expr),+ $(,)?)) => {
        $crate::passes::lowering::legalize::info::TypePattern::Family(
            $crate::passes::lowering::legalize::info::TypeFamilyPattern::IntOrPtrScalar(&[
                $($width),+
            ])
        )
    };
    (@type_pattern scalar_value($($width:expr),+ $(,)?)) => {
        $crate::passes::lowering::legalize::info::TypePattern::Family(
            $crate::passes::lowering::legalize::info::TypeFamilyPattern::ScalarValue(&[
                $($width),+
            ])
        )
    };
    (@type_pattern matches($pred:path)) => {
        $crate::passes::lowering::legalize::info::TypePattern::Predicate($pred)
    };
    (@type_pattern int) => {
        $crate::passes::lowering::legalize::info::TypePattern::Predicate(
            $crate::passes::lowering::legalize::info::is_int_type
        )
    };
    (@type_pattern scalar_int) => {
        $crate::passes::lowering::legalize::info::TypePattern::Predicate(
            $crate::passes::lowering::legalize::info::is_scalar_int_type
        )
    };
    (@type_pattern float) => {
        $crate::passes::lowering::legalize::info::TypePattern::Predicate(
            $crate::passes::lowering::legalize::info::is_float_type
        )
    };
    (@type_pattern scalar_float) => {
        $crate::passes::lowering::legalize::info::TypePattern::Predicate(
            $crate::passes::lowering::legalize::info::is_scalar_float_type
        )
    };
    (@type_pattern numeric) => {
        $crate::passes::lowering::legalize::info::TypePattern::Predicate(
            $crate::passes::lowering::legalize::info::is_numeric_type
        )
    };
    (@type_pattern scalar_numeric) => {
        $crate::passes::lowering::legalize::info::TypePattern::Predicate(
            $crate::passes::lowering::legalize::info::is_scalar_numeric_type
        )
    };
    (@type_pattern int_or_ptr) => {
        $crate::passes::lowering::legalize::info::TypePattern::Predicate(
            $crate::passes::lowering::legalize::info::is_int_or_ptr_type
        )
    };
    (@type_pattern int_or_ptr_scalar) => {
        $crate::passes::lowering::legalize::info::TypePattern::Predicate(
            $crate::passes::lowering::legalize::info::is_int_or_ptr_scalar_type
        )
    };
    (@type_pattern ptr) => {
        $crate::passes::lowering::legalize::info::TypePattern::Predicate(
            $crate::passes::lowering::legalize::info::is_ptr_type
        )
    };
    (@type_pattern ptr_sized) => {
        $crate::passes::lowering::legalize::info::TypePattern::Predicate(
            $crate::passes::lowering::legalize::info::is_ptr_sized_type
        )
    };
    (@type_pattern ptr_sized_int) => {
        $crate::passes::lowering::legalize::info::TypePattern::Predicate(
            $crate::passes::lowering::legalize::info::is_ptr_sized_int_type
        )
    };
    (@type_pattern scalar) => {
        $crate::passes::lowering::legalize::info::TypePattern::Predicate(
            $crate::passes::lowering::legalize::info::is_scalar_type
        )
    };
    (@type_pattern scalar_value) => {
        $crate::passes::lowering::legalize::info::TypePattern::Predicate(
            $crate::passes::lowering::legalize::info::is_scalar_value_type
        )
    };
    (@type_pattern vector) => {
        $crate::passes::lowering::legalize::info::TypePattern::Predicate(
            $crate::passes::lowering::legalize::info::is_vector_type
        )
    };
    (@type_pattern fixed_vector) => {
        $crate::passes::lowering::legalize::info::TypePattern::Predicate(
            $crate::passes::lowering::legalize::info::is_fixed_vector_type
        )
    };
    (@type_pattern scalable_vector) => {
        $crate::passes::lowering::legalize::info::TypePattern::Predicate(
            $crate::passes::lowering::legalize::info::is_scalable_vector_type
        )
    };
    (@type_pattern predicate) => {
        $crate::passes::lowering::legalize::info::TypePattern::Predicate(
            $crate::passes::lowering::legalize::info::is_predicate_type
        )
    };
    (@type_pattern one_of($($ty:tt),+ $(,)?)) => {
        $crate::passes::lowering::legalize::info::TypePattern::OneOf(&[
            $($crate::legalize_matcher!(@type $ty)),+
        ])
    };
    (@type_pattern $($ty:tt)+) => {
        $crate::passes::lowering::legalize::info::TypePattern::Exact(
            $crate::legalize_matcher!(@type $($ty)+)
        )
    };
    (@type I8) => {
        veloc_mir::Type::I8
    };
    (@type I16) => {
        veloc_mir::Type::I16
    };
    (@type I32) => {
        veloc_mir::Type::I32
    };
    (@type I64) => {
        veloc_mir::Type::I64
    };
    (@type F32) => {
        veloc_mir::Type::F32
    };
    (@type F64) => {
        veloc_mir::Type::F64
    };
    (@type BOOL) => {
        veloc_mir::Type::BOOL
    };
    (@type PTR) => {
        veloc_mir::Type::PTR
    };
    (@type $($ty:tt)+) => {
        $($ty)+
    };
}

#[cfg(test)]
mod tests {
    use super::{
        LegalizeAction, operand_bit_width_at, operand_type_at, same_operand_types,
        same_operand_widths, type_width_is_one_of,
    };
    use alloc::string::ToString;
    use veloc_lir::InstBuild;
    use veloc_lir::{
        GenericOpcode, MachineBlock, MachineFunction, MachineOpcode, SymbolId, Writable,
    };
    use veloc_mir::{Block, Type};

    fn make_function() -> MachineFunction {
        let mut mfunc = MachineFunction::new("test".to_string());
        mfunc.blocks.push(MachineBlock::new(Block::from_u32(0)));
        mfunc
    }

    #[test]
    fn matcher_supports_one_of_and_type_equality_guards() {
        let mut mfunc = make_function();
        let dst = mfunc.alloc_vreg(Type::I64);
        let lhs = mfunc.alloc_vreg(Type::I64);
        let rhs = mfunc.alloc_vreg(Type::I64);
        let inst = mfunc.writer().binary(
            MachineOpcode::Generic(GenericOpcode::Add),
            Writable(dst),
            lhs,
            rhs,
        );
        let inst_ref = &mfunc.inst(inst);

        let action = crate::legalize_matcher!(inst_ref, &mfunc, {
            Add => {
                [def(one_of(I32, I64)), use(one_of(I32, I64)), use(one_of(I32, I64))]
                    if same_operand_types(inst_ref, &mfunc, &[0, 1, 2])? => legal,
            };
        })
        .unwrap();

        assert_eq!(action, Some(LegalizeAction::Legal));
    }

    #[test]
    fn matcher_supports_same_types_guard_sugar() {
        let mut mfunc = make_function();
        let dst = mfunc.alloc_vreg(Type::I32);
        let lhs = mfunc.alloc_vreg(Type::I32);
        let rhs = mfunc.alloc_vreg(Type::I32);
        let inst = mfunc.writer().binary(
            MachineOpcode::Generic(GenericOpcode::Add),
            Writable(dst),
            lhs,
            rhs,
        );
        let inst_ref = &mfunc.inst(inst);

        let action = crate::legalize_matcher!(inst_ref, &mfunc, {
            Add => {
                [def(scalar_int(32, 64)), use(scalar_int(32, 64)), use(scalar_int(32, 64))]
                    if same_types(0, 1, 2) => legal,
            };
        })
        .unwrap();

        assert_eq!(action, Some(LegalizeAction::Legal));
    }

    #[test]
    fn matcher_supports_same_widths_guard_sugar() {
        let mut mfunc = make_function();
        let dst = mfunc.alloc_vreg(Type::F32);
        let src = mfunc.alloc_vreg(Type::I32);
        let inst = mfunc.writer().unary(
            MachineOpcode::Generic(GenericOpcode::Bitcast),
            Writable(dst),
            src,
        );
        let inst_ref = &mfunc.inst(inst);

        let action = crate::legalize_matcher!(inst_ref, &mfunc, {
            Bitcast => {
                [def(scalar_value(32, 64)), use(scalar_value(32, 64))]
                    if same_widths(0, 1) => legal,
            };
        })
        .unwrap();

        assert_eq!(action, Some(LegalizeAction::Legal));
    }

    #[test]
    fn matcher_supports_any_type_pattern() {
        let mut mfunc = make_function();
        let dst = mfunc.alloc_vreg(Type::PTR);
        let inst = mfunc.writer().arg(Writable(dst), 3);
        let inst_ref = &mfunc.inst(inst);

        let action = crate::legalize_matcher!(inst_ref, &mfunc, {
            Arg => {
                [def(any), imm] => legal,
            };
        })
        .unwrap();

        assert_eq!(action, Some(LegalizeAction::Legal));
    }

    #[test]
    fn matcher_supports_ptr_sized_pattern() {
        let mut mfunc = make_function();
        let dst = mfunc.alloc_vreg(Type::PTR);
        let inst = mfunc.writer().constant(Writable(dst), 42);
        let inst_ref = &mfunc.inst(inst);

        let action = crate::legalize_matcher!(inst_ref, &mfunc, {
            Constant => {
                [def(ptr_sized), imm] => legal,
            };
        })
        .unwrap();

        assert_eq!(action, Some(LegalizeAction::Legal));
    }

    #[test]
    fn matcher_supports_fixed_vector_element_pattern() {
        let mut mfunc = make_function();
        let vector_ty = Type::I32
            .as_scalar()
            .unwrap()
            .vector(4, false)
            .unwrap()
            .as_type();
        let dst = mfunc.alloc_vreg(vector_ty);
        let lhs = mfunc.alloc_vreg(vector_ty);
        let rhs = mfunc.alloc_vreg(vector_ty);
        let inst = mfunc.writer().binary(
            MachineOpcode::Generic(GenericOpcode::Add),
            Writable(dst),
            lhs,
            rhs,
        );
        let inst_ref = &mfunc.inst(inst);

        let action = crate::legalize_matcher!(inst_ref, &mfunc, {
            Add => {
                [
                    def(fixed_vector_of(I32 ; 4)),
                    use(fixed_vector_of(I32 ; 4)),
                    use(fixed_vector_of(I32 ; 4)),
                ] if same_types(0, 1, 2) => legal,
            };
        })
        .unwrap();

        assert_eq!(action, Some(LegalizeAction::Legal));
    }

    #[test]
    fn matcher_supports_scalable_vector_element_pattern() {
        let mut mfunc = make_function();
        let vector_ty = Type::F32
            .as_scalar()
            .unwrap()
            .vector(4, true)
            .unwrap()
            .as_type();
        let dst = mfunc.alloc_vreg(vector_ty);
        let lhs = mfunc.alloc_vreg(vector_ty);
        let rhs = mfunc.alloc_vreg(vector_ty);
        let inst = mfunc.writer().binary(
            MachineOpcode::Generic(GenericOpcode::Fadd),
            Writable(dst),
            lhs,
            rhs,
        );
        let inst_ref = &mfunc.inst(inst);

        let action = crate::legalize_matcher!(inst_ref, &mfunc, {
            Fadd => {
                [
                    def(scalable_vector_of(F32 ; 4)),
                    use(scalable_vector_of(F32 ; 4)),
                    use(scalable_vector_of(F32 ; 4)),
                ] if same_types(0, 1, 2) => legal,
            };
        })
        .unwrap();

        assert_eq!(action, Some(LegalizeAction::Legal));
    }

    #[test]
    fn matcher_supports_width_qualified_scalar_value_patterns() {
        let mut mfunc = make_function();
        let dst = mfunc.alloc_vreg(Type::PTR);
        let src = mfunc.alloc_vreg(Type::PTR);
        let inst = mfunc.writer().copy(Writable(dst), src);
        let inst_ref = &mfunc.inst(inst);

        let action = crate::legalize_matcher!(inst_ref, &mfunc, {
            Copy => {
                [def(scalar_value(32, 64)), use(scalar_value(32, 64))]
                    if same_types(0, 1) => legal,
            };
        })
        .unwrap();

        assert_eq!(action, Some(LegalizeAction::Legal));
    }

    #[test]
    fn operand_type_helpers_ignore_non_register_operands() {
        let mut mfunc = make_function();
        let dst = mfunc.alloc_vreg(Type::I32);
        let inst = mfunc.writer().arg(Writable(dst), 7);

        assert_eq!(
            operand_type_at(&mfunc.inst(inst), &mfunc, 0).unwrap(),
            Some(Type::I32)
        );
        assert_eq!(
            operand_bit_width_at(&mfunc.inst(inst), &mfunc, 0).unwrap(),
            Some(32)
        );
        assert_eq!(operand_type_at(&mfunc.inst(inst), &mfunc, 1).unwrap(), None);
        assert_eq!(
            operand_bit_width_at(&mfunc.inst(inst), &mfunc, 1).unwrap(),
            None
        );
        assert!(!same_operand_types(&mfunc.inst(inst), &mfunc, &[0, 1]).unwrap());
        assert!(!same_operand_widths(&mfunc.inst(inst), &mfunc, &[0, 1]).unwrap());
    }

    #[test]
    fn matcher_returns_none_when_no_opcode_branch_matches() {
        let _mfunc = make_function();
        let mut mfunc = make_function();
        let inst = mfunc.writer().ret(&[]);
        let _inst_ref = &inst;

        let action = crate::legalize_matcher!(_inst_ref, &_mfunc, {}).unwrap();

        assert_eq!(action, None);
    }

    #[test]
    fn matcher_supports_variadic_return_pattern() {
        let mut mfunc = make_function();
        let r0 = mfunc.alloc_vreg(Type::I32);
        let r1 = mfunc.alloc_vreg(Type::F64);
        let inst = mfunc.writer().ret(&[r0, r1]);
        let inst_ref = &mfunc.inst(inst);

        let action = crate::legalize_matcher!(inst_ref, &mfunc, {
            Ret => {
                seq[..use(any)] => legal,
            };
        })
        .unwrap();

        assert_eq!(action, Some(LegalizeAction::Legal));
    }

    #[test]
    fn matcher_supports_variadic_direct_call_pattern() {
        let mut mfunc = make_function();
        let ret0 = mfunc.alloc_vreg(Type::I32);
        let ret1 = mfunc.alloc_vreg(Type::PTR);
        let arg0 = mfunc.alloc_vreg(Type::I64);
        let arg1 = mfunc.alloc_vreg(Type::F32);
        let inst = mfunc
            .writer()
            .call(&[ret0, ret1], SymbolId::from_u32(4), &[arg0, arg1]);
        let inst_ref = &mfunc.inst(inst);

        let action = crate::legalize_matcher!(inst_ref, &mfunc, {
            Call => {
                seq[..def(any), global, ..use(any)] => legal,
            };
        })
        .unwrap();

        assert_eq!(action, Some(LegalizeAction::Legal));
    }

    #[test]
    fn matcher_supports_variadic_indirect_call_pattern() {
        let mut mfunc = make_function();
        let ret = mfunc.alloc_vreg(Type::I64);
        let callee = mfunc.alloc_vreg(Type::PTR);
        let arg0 = mfunc.alloc_vreg(Type::I32);
        let arg1 = mfunc.alloc_vreg(Type::I32);
        let inst = mfunc.writer().callind(&[ret], callee, &[arg0, arg1]);
        let inst_ref = &mfunc.inst(inst);

        let action = crate::legalize_matcher!(inst_ref, &mfunc, {
            Callind => {
                seq[..def(any), use(any), ..use(any)] => legal,
            };
        })
        .unwrap();

        assert_eq!(action, Some(LegalizeAction::Legal));
    }

    #[test]
    fn matcher_supports_builtin_scalar_float_predicate() {
        let mut mfunc = make_function();
        let dst = mfunc.alloc_vreg(Type::F32);
        let lhs = mfunc.alloc_vreg(Type::F32);
        let rhs = mfunc.alloc_vreg(Type::F32);
        let inst = mfunc.writer().binary(
            MachineOpcode::Generic(GenericOpcode::Fadd),
            Writable(dst),
            lhs,
            rhs,
        );
        let inst_ref = &mfunc.inst(inst);

        let action = crate::legalize_matcher!(inst_ref, &mfunc, {
            Fadd => {
                [def(scalar_float), use(scalar_float), use(scalar_float)]
                    if same_operand_types(inst_ref, &mfunc, &[0, 1, 2])? => legal,
            };
        })
        .unwrap();

        assert_eq!(action, Some(LegalizeAction::Legal));
    }

    #[test]
    fn matcher_supports_custom_type_predicate_paths() {
        fn is_legal_copy_ty(ty: Type) -> bool {
            matches!(ty, Type::I32 | Type::PTR | Type::F64)
        }

        let mut mfunc = make_function();
        let dst = mfunc.alloc_vreg(Type::PTR);
        let src = mfunc.alloc_vreg(Type::PTR);
        let inst = mfunc.writer().copy(Writable(dst), src);
        let inst_ref = &mfunc.inst(inst);

        let action = crate::legalize_matcher!(inst_ref, &mfunc, {
            Copy => {
                [def(matches(is_legal_copy_ty)), use(matches(is_legal_copy_ty))]
                    if same_operand_types(inst_ref, &mfunc, &[0, 1])? => legal,
            };
        })
        .unwrap();

        assert_eq!(action, Some(LegalizeAction::Legal));
    }

    #[test]
    fn type_width_helper_accepts_empty_set_as_any_width() {
        assert!(type_width_is_one_of(Type::I16, &[]));
        assert!(type_width_is_one_of(Type::PTR, &[64]));
        assert!(!type_width_is_one_of(Type::F32, &[64]));
    }
}
