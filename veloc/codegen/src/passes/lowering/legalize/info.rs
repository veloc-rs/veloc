extern crate alloc;

use crate::mir::GenericOpcode;
use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;
use veloc_ir::Type;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LegalizeAction {
    Legal,
    Unsupported,
    NarrowScalar,
    WidenScalar,
    Lower,
    Libcall,
    FewerElements,
    MoreElements,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LegalityKey {
    pub opcode: GenericOpcode,
    pub types: Vec<Type>,
}

pub struct LegalizerInfo {
    rules: BTreeMap<LegalityKey, LegalizeAction>,
}

pub struct LegalizeRuleSet<'a> {
    info: &'a mut LegalizerInfo,
    opcode: GenericOpcode,
}

impl<'a> LegalizeRuleSet<'a> {
    pub fn new(info: &'a mut LegalizerInfo, opcode: GenericOpcode) -> Self {
        Self { info, opcode }
    }

    pub fn legal_for(self, types: Vec<Type>) -> Self {
        self.info
            .set_action(self.opcode.clone(), types, LegalizeAction::Legal);
        self
    }

    pub fn legal_for_many(self, types_list: Vec<Vec<Type>>) -> Self {
        for types in types_list {
            self.info
                .set_action(self.opcode.clone(), types, LegalizeAction::Legal);
        }
        self
    }

    pub fn widen_scalar_for(self, types: Vec<Type>) -> Self {
        self.info
            .set_action(self.opcode.clone(), types, LegalizeAction::WidenScalar);
        self
    }

    pub fn lower_for(self, types: Vec<Type>) -> Self {
        self.info
            .set_action(self.opcode.clone(), types, LegalizeAction::Lower);
        self
    }

    pub fn unsupported_for(self, types: Vec<Type>) -> Self {
        self.info
            .set_action(self.opcode.clone(), types, LegalizeAction::Unsupported);
        self
    }

    pub fn legal_for_types(self, types: &[Type]) -> Self {
        for ty in types {
            self.info
                .set_action(self.opcode.clone(), vec![*ty; 3], LegalizeAction::Legal);
        }
        self
    }

    pub fn widen_scalar_for_type(self, ty: Type) -> Self {
        self.info.set_action(
            self.opcode.clone(),
            vec![ty; 3],
            LegalizeAction::WidenScalar,
        );
        self
    }

    pub fn lower_for_type(self, ty: Type) -> Self {
        self.info
            .set_action(self.opcode.clone(), vec![ty; 3], LegalizeAction::Lower);
        self
    }
}

impl LegalizerInfo {
    pub fn new() -> Self {
        Self {
            rules: BTreeMap::new(),
        }
    }

    pub fn get_action_definitions_builder(&mut self, opcode: GenericOpcode) -> LegalizeRuleSet<'_> {
        LegalizeRuleSet::new(self, opcode)
    }

    pub fn set_action(&mut self, opcode: GenericOpcode, types: Vec<Type>, action: LegalizeAction) {
        self.rules.insert(LegalityKey { opcode, types }, action);
    }

    pub fn get_action(&self, opcode: &GenericOpcode, types: &[Type]) -> LegalizeAction {
        self.rules
            .get(&LegalityKey {
                opcode: opcode.clone(),
                types: types.to_vec(),
            })
            .copied()
            .unwrap_or(LegalizeAction::Unsupported)
    }
}
