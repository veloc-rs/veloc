//! Compact search expressions, with MIR occurrences kept as emission witnesses.
use crate::evaluate::Properties;
use cranelift_entity::{PrimaryMap, SecondaryMap, entity_impl, packed_option::PackedOption};
use hashbrown::HashMap;
use smallvec::SmallVec;
use veloc_mir::{Constant, FuncBody, Opcode, ScalarConst, Type};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct Value(pub u32);
entity_impl!(Value, "expr");

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct Inst(pub u32);
entity_impl!(Inst, "expr_inst");

#[derive(Clone, Copy, PartialEq, Eq)]
struct Literal(u32);
entity_impl!(Literal, "literal");

/// Every search value starts a typed class. Unions preserve its type and choose
/// the interned literal as root when the class acquires a constant fact.
struct ValueData {
    ty: Type,
    inst: PackedOption<Inst>,
    literal: PackedOption<Literal>,
}

/// Imported and generated operations have exactly the same representation.
/// Inputs followed by results occupy one range in the shared edge buffer.
struct Operation {
    opcode: Opcode,
    properties: Properties,
    start: u32,
    inputs: u32,
    outputs: u32,
    movable: bool,
}

#[derive(Default)]
pub(super) struct Storage {
    values: PrimaryMap<Value, ValueData>,
    operations: PrimaryMap<Inst, Operation>,
    edges: Vec<Value>,
    constants: PrimaryMap<Literal, Constant>,
    literals: HashMap<Constant, Value>,
    // Cold occurrence metadata, never consulted by matching or congruence.
    original_values: SecondaryMap<Value, PackedOption<veloc_mir::Value>>,
    original_insts: SecondaryMap<Inst, PackedOption<veloc_mir::Inst>>,
}

/// MIR is borrowed only at the import and extraction boundaries. Hot search
/// queries read Storage, without decoding MIR or distinguishing new candidates.
pub(super) struct Expressions<'a> {
    body: &'a FuncBody,
    storage: Storage,
    imported: SecondaryMap<veloc_mir::Value, PackedOption<Value>>,
    replacements: HashMap<Value, Value>,
}

impl core::ops::Deref for Expressions<'_> {
    type Target = Storage;
    fn deref(&self) -> &Storage {
        &self.storage
    }
}

impl<'a> Expressions<'a> {
    pub(super) fn new(body: &'a FuncBody) -> Self {
        Self {
            body,
            storage: Storage::default(),
            imported: SecondaryMap::new(),
            replacements: HashMap::new(),
        }
    }

    pub(super) fn body(&self) -> &'a FuncBody {
        self.body
    }

    pub(super) fn into_storage(self) -> Storage {
        self.storage
    }

    pub(super) fn import_value(&mut self, original: veloc_mir::Value) -> Value {
        if let Some(value) = self.imported[original].expand() {
            return value;
        }
        let dfg = self.body.dfg();
        let value = if let Some(constant) = dfg.as_const(original) {
            self.storage.constant(constant.clone())
        } else {
            self.storage.values.push(ValueData {
                ty: dfg.value_type(original),
                inst: None.into(),
                literal: None.into(),
            })
        };
        self.imported[original] = value.into();
        self.storage.original_values[value] = original.into();
        value
    }

    /// Called once per executable instruction. Unsupported operations stay in
    /// MIR; their inputs and outputs become opaque search leaves as needed.
    pub(super) fn import_inst(&mut self, original: veloc_mir::Inst) -> Option<Inst> {
        let dfg = self.body.dfg();
        let view = dfg.inst(original);
        let supported = crate::evaluate::can_reduce(dfg, original)
            || (crate::evaluate::can_rewrite(view.opcode()) && view.can_speculate())
            || matches!(
                view,
                veloc_mir::InstView::PtrOffset { .. } | veloc_mir::InstView::PtrIndex { .. }
            );
        // Anchor operands must also be registered, including values produced by
        // a later RPO block. Importing that definition will fill in its Inst.
        if !supported {
            for &arg in dfg.operands(original) {
                self.import_value(arg);
            }
            return None;
        }
        let args: SmallVec<[Value; 3]> = dfg
            .operands(original)
            .iter()
            .map(|&v| self.import_value(v))
            .collect();
        let results: SmallVec<[Value; 2]> = dfg
            .inst_results(original)
            .iter()
            .map(|&v| self.import_value(v))
            .collect();
        let inst = self.storage.push_operation(
            view.opcode(),
            &args,
            &results,
            Properties::read(view),
            view.can_speculate(),
        );
        self.storage.original_insts[inst] = original.into();
        Some(inst)
    }

    pub(super) fn imported_inst(&self, original: veloc_mir::Inst) -> Option<Inst> {
        let result = *self.body.dfg().inst_results(original).first()?;
        self.value_inst(self.imported[result].expand()?)
    }

    pub(super) fn constant(&mut self, constant: Constant) -> Value {
        self.storage.constant(constant)
    }

    pub(super) fn create(
        &mut self,
        opcode: Opcode,
        args: &[Value],
        ty: Type,
        properties: Properties,
    ) -> Inst {
        let types: SmallVec<[Type; 3]> = args.iter().map(|&v| self.value_type(v)).collect();
        properties
            .validate(opcode, &types, &[ty])
            .expect("checked expression recipe");
        let result = self.storage.values.push(ValueData {
            ty,
            inst: None.into(),
            literal: None.into(),
        });
        self.storage
            .push_operation(opcode, args, &[result], properties, true)
    }

    pub(super) fn set_replacements(&mut self, replacements: HashMap<Value, Value>) {
        self.replacements = replacements;
    }

    fn replaced(&self, value: Value) -> Value {
        self.replacements.get(&value).copied().unwrap_or(value)
    }

    /// Read executable uses through proven folds without copying their owner.
    pub(super) fn anchor_operands(
        &self,
        inst: veloc_mir::Inst,
    ) -> impl Iterator<Item = Value> + '_ {
        self.body
            .dfg()
            .operands(inst)
            .iter()
            .map(|&v| self.replaced(self.imported[v].expect("imported anchor operand")))
    }

    /// Price the original computation as it will look after publishing folds.
    pub(super) fn folded_operands(&self, inst: Inst) -> impl Iterator<Item = Value> + '_ {
        let original = self.original_inst(inst).is_some();
        self.operands(inst)
            .iter()
            .map(move |&v| if original { self.replaced(v) } else { v })
    }
}

impl Storage {
    fn push_operation(
        &mut self,
        opcode: Opcode,
        args: &[Value],
        results: &[Value],
        properties: Properties,
        movable: bool,
    ) -> Inst {
        let inst = self.operations.push(Operation {
            opcode,
            properties,
            start: u32::try_from(self.edges.len()).expect("expression edge count"),
            inputs: u32::try_from(args.len()).expect("expression arity"),
            outputs: u32::try_from(results.len()).expect("expression results"),
            movable,
        });
        self.edges.extend_from_slice(args);
        self.edges.extend_from_slice(results);
        for &result in results {
            self.values[result].inst = inst.into();
        }
        inst
    }

    pub(super) fn value_type(&self, value: Value) -> Type {
        self.values[value].ty
    }

    pub(super) fn value_inst(&self, value: Value) -> Option<Inst> {
        self.values[value].inst.expand()
    }

    pub(super) fn original_value(&self, value: Value) -> Option<veloc_mir::Value> {
        self.original_values[value].expand()
    }

    pub(super) fn original_inst(&self, inst: Inst) -> Option<veloc_mir::Inst> {
        self.original_insts[inst].expand()
    }

    pub(super) fn values(&self) -> impl Iterator<Item = Value> + '_ {
        self.values.keys()
    }

    pub(super) fn insts(&self) -> impl Iterator<Item = Inst> + '_ {
        self.operations.keys()
    }

    #[cfg(test)]
    pub(super) fn inst_count(&self) -> usize {
        self.operations.len()
    }

    pub(super) fn as_const(&self, value: Value) -> Option<&Constant> {
        Some(&self.constants[self.values[value].literal.expand()?])
    }

    pub(super) fn as_scalar_const(&self, value: Value) -> Option<ScalarConst> {
        self.as_const(value)?.as_scalar()
    }

    pub(super) fn opcode(&self, inst: Inst) -> Opcode {
        self.operations[inst].opcode
    }

    pub(super) fn properties(&self, inst: Inst) -> Properties {
        self.operations[inst].properties
    }

    pub(super) fn operands(&self, inst: Inst) -> &[Value] {
        let op = &self.operations[inst];
        let start = op.start as usize;
        &self.edges[start..start + op.inputs as usize]
    }

    pub(super) fn inst_results(&self, inst: Inst) -> &[Value] {
        let op = &self.operations[inst];
        let start = op.start as usize + op.inputs as usize;
        &self.edges[start..start + op.outputs as usize]
    }

    pub(super) fn first_result(&self, inst: Inst) -> Option<Value> {
        self.inst_results(inst).first().copied()
    }

    pub(super) fn can_speculate(&self, inst: Inst) -> bool {
        self.operations[inst].movable
    }

    fn constant(&mut self, constant: Constant) -> Value {
        if let Some(&value) = self.literals.get(&constant) {
            return value;
        }
        let ty = constant.ty();
        let literal = self.constants.push(constant.clone());
        let value = self.values.push(ValueData {
            ty,
            inst: None.into(),
            literal: literal.into(),
        });
        self.literals.insert(constant, value);
        value
    }

    /// Original templates remain alive until every planned operation is emitted.
    pub(super) fn emit(
        &self,
        body: &mut FuncBody,
        before: veloc_mir::Inst,
        source: Inst,
        args: &[veloc_mir::Value],
    ) -> veloc_mir::Inst {
        assert!(
            self.can_speculate(source),
            "cannot duplicate an effectful operation"
        );
        let types: SmallVec<[Type; 2]> = self
            .inst_results(source)
            .iter()
            .map(|&v| self.value_type(v))
            .collect();
        if let Some(original) = self.original_inst(source) {
            return body.edit().insert_before(
                before,
                |w| w.copy_with_operands(original, args),
                &types,
            );
        }
        let operation = &self.operations[source];
        body.edit().insert_before(
            before,
            |w| operation.properties.write(operation.opcode, args, w),
            &types,
        )
    }

    pub(super) fn materialize(&self, body: &mut FuncBody, value: Value) -> veloc_mir::Value {
        if let Some(original) = self.original_value(value) {
            return original;
        }
        let constant = self
            .as_const(value)
            .expect("candidate result needs a planned occurrence");
        body.edit().constant(constant.clone())
    }
}

impl crate::rewrite::View for Expressions<'_> {
    type Value = Value;
    fn ty(&self, value: Value) -> Type {
        self.value_type(value)
    }
    fn constant(&self, value: Value) -> Option<ScalarConst> {
        self.as_scalar_const(value)
    }
    fn properties(&self, value: Value) -> Option<Properties> {
        Some(self.storage.properties(self.value_inst(value)?))
    }
}
