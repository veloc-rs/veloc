//! Shared build-time constant pool handling for rule bytecode backends.
pub(crate) fn intern<T: PartialEq>(items: &mut Vec<T>, item: T) -> usize {
    if let Some(id) = items.iter().position(|old| *old == item) {
        id
    } else {
        let id = items.len();
        items.push(item);
        id
    }
}

/// Shared assembler: labels identify instruction boundaries, branches carry
/// explicit fixed-width relocations. No target or IR knowledge lives here.
#[derive(Default)]
pub(crate) struct Assembler {
    pub instructions: Vec<Vec<u8>>,
    labels: Vec<usize>,
    relocations: Vec<(usize, usize, usize)>,
}
pub(crate) struct Encoded {
    pub instructions: Vec<Vec<u8>>,
    pub offsets: Vec<usize>,
    pub labels: Vec<usize>,
}
impl Assembler {
    pub fn label(&mut self) -> usize {
        let id = self.labels.len();
        self.labels.push(self.instructions.len());
        id
    }
    pub fn emit(&mut self, op: impl veloc_bytecode::Encode) {
        let mut bytes = Vec::new();
        op.encode(&mut bytes);
        self.instructions.push(bytes);
    }
    pub fn branch(&mut self, op: impl veloc_bytecode::Encode, field: &str, label: usize) {
        let offset = op.field_offset(field).expect("branch operand");
        self.relocations
            .push((self.instructions.len(), offset, label));
        self.emit(op);
    }
    pub fn finish(&self) -> Encoded {
        let mut offset = 0;
        let offsets: Vec<_> = self
            .instructions
            .iter()
            .map(|bytes| {
                let start = offset;
                offset += bytes.len();
                start
            })
            .collect();
        assert!(offset <= u32::MAX as usize, "bytecode program too large");
        let labels: Vec<_> = self.labels.iter().map(|&i| offsets[i]).collect();
        let mut instructions = self.instructions.clone();
        for &(inst, field, label) in &self.relocations {
            instructions[inst][field..field + 4]
                .copy_from_slice(&veloc_bytecode::encode_u32(labels[label]));
        }
        Encoded {
            instructions,
            offsets,
            labels,
        }
    }
}
