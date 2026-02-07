use alloc::string::String;
use alloc::vec::Vec;
use hashbrown::HashMap;
use veloc::ir::{FuncId, Type as VelocType};
use wasmparser::{ExternalKind, RefType, ValType};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WasmSignature {
    pub params: Box<[ValType]>,
    pub results: Box<[ValType]>,
    pub hash: u32,
}

impl WasmSignature {
    pub fn new(params: Vec<ValType>, results: Vec<ValType>) -> Self {
        let mut sig = Self {
            params: params.into_boxed_slice(),
            results: results.into_boxed_slice(),
            hash: 0,
        };
        sig.hash = sig.calc_hash();
        sig
    }

    fn calc_hash(&self) -> u32 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.params.hash(&mut hasher);
        self.results.hash(&mut hasher);
        (hasher.finish() & 0xFFFFFFFF) as u32
    }

    pub fn hash_u64(&self) -> u64 {
        self.hash as u64
    }
}

pub struct WasmFunction {
    pub name: String,
    pub type_index: u32,
    pub func_id: FuncId,
}

pub struct WasmTable {
    pub element_type: RefType,
    pub initial: u32,
    pub maximum: Option<u32>,
    pub init: Option<Box<[GlobalInit]>>,
}

pub struct WasmMemory {
    pub initial: u64,
    pub maximum: Option<u64>,
}

pub struct WasmGlobal {
    pub ty: ValType,
    pub mutable: bool,
    pub init: Box<[GlobalInit]>,
}

pub struct WasmElement {
    pub offset: Box<[GlobalInit]>,
    pub items: Box<[Box<[GlobalInit]>]>,
    pub table_index: u32,
    pub is_active: bool,
}

pub struct WasmData {
    pub offset: Box<[GlobalInit]>,
    pub data: Box<[u8]>,
    pub memory_index: u32,
    pub is_active: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GlobalInit {
    I32Const(i32),
    I64Const(i64),
    F32Const(u32),
    F64Const(u64),
    RefNull,
    RefFunc(u32),
    GlobalGet(u32),
    I32Add,
    I32Sub,
    I32Mul,
    I64Add,
    I64Sub,
    I64Mul,
}

pub struct WasmImport {
    pub module: String,
    pub field: String,
    pub kind: ExternalKind,
    pub index: u32,
}

pub struct ModuleMetadata {
    pub exports: HashMap<String, (ExternalKind, u32)>,
    pub functions: Box<[WasmFunction]>,
    pub signatures: Box<[WasmSignature]>,
    pub ir_sig_ids: Box<[veloc::ir::SigId]>,
    pub tables: Box<[WasmTable]>,
    pub memories: Box<[WasmMemory]>,
    pub elements: Box<[WasmElement]>,
    pub data: Box<[WasmData]>,
    pub imports: Box<[WasmImport]>,
    pub globals: Box<[WasmGlobal]>,
    pub num_imported_funcs: usize,
    pub num_imported_tables: usize,
    pub num_imported_memories: usize,
    pub num_imported_globals: usize,
}

pub(crate) fn valtype_to_veloc(ty: ValType) -> VelocType {
    match ty {
        ValType::I32 => VelocType::I32,
        ValType::I64 => VelocType::I64,
        ValType::F32 => VelocType::F32,
        ValType::F64 => VelocType::F64,
        ValType::Ref(_) => VelocType::Ptr,
        _ => VelocType::I64,
    }
}
