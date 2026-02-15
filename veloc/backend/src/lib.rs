#![no_std]
extern crate alloc;

pub mod error;
pub(crate) mod x86_64;

pub use error::{Error, Result};

use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use hashbrown::HashMap;
use object::write::{Object, Relocation as ObjectReloc, Symbol, SymbolSection};
use object::{
    Architecture, BinaryFormat, Endianness, RelocationEncoding, RelocationFlags, RelocationKind,
    SymbolFlags, SymbolKind, SymbolScope,
};
use veloc_ir::{Function, Linkage};

// Remove local CallingConvention since we use veloc_ir's

/// 重定位类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]#[allow(dead_code)]pub(crate) enum RelocKind {
    /// 32位相对偏移 (x86_64 PC32)
    X86_64Pc32,
    /// 32位 PLT 相对偏移
    X86_64Plt32,
    /// 64位绝对地址
    X86_64_64Reloc,
    /// 8位相对偏移 (用于短跳转)
    X86_64Rel8,
}

/// 重定位目标
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum RelocTarget {
    Block(veloc_ir::Block),
    Symbol(String),
}

/// 重定位信息
#[derive(Debug, Clone)]
pub(crate) struct Relocation {
    /// 指向指令中需要修改的偏移量
    pub offset: u32,
    pub kind: RelocKind,
    pub target: RelocTarget,
    pub addend: i64,
}

/// 编译后的节（Section）
#[derive(Debug, Clone)]
pub(crate) struct Section {
    #[allow(dead_code)]
    pub name: String,
    pub data: Vec<u8>,
    pub relocs: Vec<Relocation>,
}

/// 寄存器在指令中的编码位置描述
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RegisterHole {
    /// 指令字节流中的偏移
    pub offset: usize,
    /// 在该字节中的位移
    pub shift: u8,
    /// 掩码（在该字节中占用的位）
    pub mask: u8,
    /// 扩展位描述，例如 x86 的 REX/VEX 扩展位
    /// (偏移, 位掩码)，仅在寄存器索引 >= 8 时生效
    pub extra_bit: Option<(usize, u8)>,
}

/// 指令模板，用于抽象带参数的机器码片段
#[derive(Debug, Clone, Copy)]
struct InstructionTemplate {
    pub bytes: &'static [u8],
    /// 立即数偏移和大小
    pub hole_offset: Option<usize>,
    pub hole_size: usize,
    /// 寄存器空缺
    pub reg1: Option<RegisterHole>,
    pub reg2: Option<RegisterHole>,
    pub reg3: Option<RegisterHole>,
}

/// Represents a machine instruction with metadata.
trait TargetInstruction {
    fn template(&self) -> InstructionTemplate;
    #[allow(dead_code)]
    fn name(&self) -> &'static str;
}

/// 通用的指令定义宏，可以被不同的架构后端使用
#[macro_export]
macro_rules! define_backend_insts {
    ($enum_name:ident {
        $($variant:ident ($name:expr, $bytes:expr $(; $($key:ident = $val:expr),*)?);)*
    }) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]        #[allow(dead_code)]        pub enum $enum_name {
            $($variant),*
        }

        impl $crate::TargetInstruction for $enum_name {
            fn name(&self) -> &'static str {
                match self {
                    $(Self::$variant => $name),*
                }
            }

            fn template(&self) -> $crate::InstructionTemplate {
                match self {
                    $(
                        Self::$variant => {
                            #[allow(unused_mut)]
                            let mut t = $crate::InstructionTemplate::new($bytes);
                            $($(
                                $crate::backend_apply_attr!(t, $key, $val);
                            )*)?
                            t
                        }
                    )*
                }
            }
        }
    };
}

#[macro_export]
macro_rules! backend_apply_attr {
    ($t:ident, imm_off, $val:expr) => {
        $t.hole_offset = Some($val);
    };
    ($t:ident, h, $val:expr) => {
        $t.hole_offset = Some($val);
    };
    ($t:ident, imm_size, $val:expr) => {
        $t.hole_size = $val;
    };
    ($t:ident, s, $val:expr) => {
        $t.hole_size = $val;
    };
    ($t:ident, reg1, $val:expr) => {
        $t.reg1 = Some($val);
    };
    ($t:ident, r1, $val:expr) => {
        $t.reg1 = Some($val);
    };
    ($t:ident, reg2, $val:expr) => {
        $t.reg2 = Some($val);
    };
    ($t:ident, r2, $val:expr) => {
        $t.reg2 = Some($val);
    };
    ($t:ident, reg3, $val:expr) => {
        $t.reg3 = Some($val);
    };
    ($t:ident, r3, $val:expr) => {
        $t.reg3 = Some($val);
    };
}

struct Emitter<'a> {
    pub section: &'a mut Section,
}

impl<'a> Emitter<'a> {
    fn new(section: &'a mut Section) -> Self {
        Self { section }
    }

    fn current_offset(&self) -> u32 {
        self.section.data.len() as u32
    }

    fn write_bytes(&mut self, bytes: &[u8]) {
        self.section.data.extend_from_slice(bytes);
    }

    fn inst<I: TargetInstruction>(&mut self, inst: I) -> InstBuilder<'_, 'a, I> {
        InstBuilder::new(self, inst)
    }

    /// Emits a rich instruction and potentially logs its name.
    fn emit_inst<I: TargetInstruction>(&mut self, inst: I) {
        self.inst(inst).emit();
    }
}

impl InstructionTemplate {
    pub const fn new(bytes: &'static [u8]) -> Self {
        Self {
            bytes,
            hole_offset: None,
            hole_size: 0,
            reg1: None,
            reg2: None,
            reg3: None,
        }
    }

    pub fn patch_imm(&self, code: &mut [u8], offset: usize, val: u64) {
        let hole_off = self.hole_offset.expect("Template has no hole");
        match self.hole_size {
            1 => code[offset + hole_off] = val as u8,
            4 => {
                let bytes = (val as u32).to_le_bytes();
                code[offset + hole_off..offset + hole_off + 4].copy_from_slice(&bytes);
            }
            8 => {
                let bytes = val.to_le_bytes();
                code[offset + hole_off..offset + hole_off + 8].copy_from_slice(&bytes);
            }
            _ => panic!("Unsupported hole size"),
        }
    }

    pub fn patch_reg(&self, code: &mut [u8], offset: usize, reg_index: u8, hole_idx: usize) {
        let hole = match hole_idx {
            0 => self.reg1,
            1 => self.reg2,
            2 => self.reg3,
            _ => None,
        }
        .expect("No such reg hole");
        let reg_low = reg_index & 0x07;

        // 修改目标字节
        let byte_pos = offset + hole.offset;
        code[byte_pos] = (code[byte_pos] & !hole.mask) | ((reg_low << hole.shift) & hole.mask);

        // 如果寄存器索引 >= 8 且定义了额外位（如 x86 REX.R/B/X）
        if reg_index >= 8 {
            if let Some((extra_off, extra_mask)) = hole.extra_bit {
                code[offset + extra_off] |= extra_mask;
            }
        }
    }

    /// 获取模板预期的寄存器空缺数量
    pub fn expected_reg_count(&self) -> usize {
        (self.reg1.is_some() as usize)
            + (self.reg2.is_some() as usize)
            + (self.reg3.is_some() as usize)
    }

    /// 模板是否预期一个立即数
    pub fn expects_imm(&self) -> bool {
        self.hole_offset.is_some()
    }

    /// 安全地应用所有 patch 并校验
    pub fn apply_all(&self, code: &mut [u8], offset: usize, regs: &[u8], imm: Option<u64>) {
        // 运行时校验，防止漏打或多打
        debug_assert_eq!(
            self.expected_reg_count(),
            regs.len(),
            "register count mismatch"
        );
        debug_assert_eq!(
            self.expects_imm(),
            imm.is_some(),
            "immediate expectation mismatch"
        );

        for (i, &reg) in regs.iter().enumerate() {
            self.patch_reg(code, offset, reg, i);
        }
        if let Some(val) = imm {
            self.patch_imm(code, offset, val);
        }
    }
}

/// Represents a physical register index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Reg(pub u8);

/// 指令构建器，用于安全地发射带参数的指令
struct InstBuilder<'a, 'b, I: TargetInstruction> {
    emitter: &'a mut Emitter<'b>,
    inst: I,
    regs: [Reg; 3],
    reg_count: usize,
    imm: Option<u64>,
}

impl<'a, 'b, I: TargetInstruction> InstBuilder<'a, 'b, I> {
    fn new(emitter: &'a mut Emitter<'b>, inst: I) -> Self {
        Self {
            emitter,
            inst,
            regs: [Reg(0); 3],
            reg_count: 0,
            imm: None,
        }
    }

    fn reg(mut self, r: impl Into<Reg>) -> Self {
        assert!(self.reg_count < 3, "超过指令模板最大寄存器限制 (3)");
        self.regs[self.reg_count] = r.into();
        self.reg_count += 1;
        self
    }

    fn imm(mut self, i: u64) -> Self {
        self.imm = Some(i);
        self
    }

    fn emit(self) {
        let template = self.inst.template();
        let offset = self.emitter.current_offset() as usize;

        // 1. 写入原始字节
        self.emitter.write_bytes(template.bytes);

        // 2. 将 Reg 转换为 u8 数组进行 patch
        let mut reg_indices = [0u8; 3];
        for i in 0..self.reg_count {
            reg_indices[i] = self.regs[i].0;
        }

        // 3. 统一 Patch 并校验
        template.apply_all(
            &mut self.emitter.section.data,
            offset,
            &reg_indices[..self.reg_count],
            self.imm,
        );
    }
}

/// 这个 Trait 定义了不同硬件架构后端需要实现的最基本能力。
trait TargetBackend {
    /// 生成函数开始的机器码
    fn emit_prologue(&self, emitter: &mut Emitter, module: &veloc_ir::Module, func: &Function);

    /// 生成函数结束的机器码
    fn emit_epilogue(&self, emitter: &mut Emitter, module: &veloc_ir::Module, func: &Function);

    /// 将 SSA IR 的单个指令发射为目标架构的机器码
    fn emit_inst(
        &self,
        emitter: &mut Emitter,
        module: &veloc_ir::Module,
        func: &Function,
        inst: veloc_ir::Inst,
    );

    /// 为非 Entry Block 生成接收 Block Parameters 的代码
    fn emit_block_params(
        &self,
        emitter: &mut Emitter,
        module: &veloc_ir::Module,
        func: &Function,
        block: veloc_ir::Block,
    );

    /// 获取 unwind 信息 (通常是 .eh_frame)
    fn emit_unwind_info(&self, _func: &Function) -> Option<Section> {
        None
    }

    /// 编译整个模块
    fn compile_module(&self, module: &veloc_ir::Module) -> crate::Result<Vec<u8>> {
        let mut obj = Object::new(BinaryFormat::Elf, Architecture::X86_64, Endianness::Little);

        let text_section_id = obj.add_section(vec![], b".text".to_vec(), object::SectionKind::Text);
        let mut current_offset = 0;
        let mut symbols = HashMap::new();

        for (_func_id, func) in &module.functions {
            if func.linkage == Linkage::Import {
                if !symbols.contains_key(&func.name) {
                    let symbol_id = obj.add_symbol(Symbol {
                        name: func.name.as_bytes().to_vec(),
                        value: 0,
                        size: 0,
                        kind: SymbolKind::Text,
                        scope: SymbolScope::Dynamic,
                        weak: false,
                        section: SymbolSection::Undefined,
                        flags: SymbolFlags::None,
                    });
                    symbols.insert(func.name.clone(), symbol_id);
                }
                continue;
            }

            let mut text = Section {
                name: ".text".into(),
                data: Vec::new(),
                relocs: Vec::new(),
            };

            // 第一次遍历记录所有 Block 的偏移
            let mut block_offsets = HashMap::new();

            {
                let mut emitter = Emitter::new(&mut text);
                self.emit_prologue(&mut emitter, module, func);

                for &block in &func.layout.block_order {
                    block_offsets.insert(block, emitter.current_offset());

                    // 处理非 entry 块的参数 (Entry Block 的参数已经在 prologue 处理)
                    if Some(block) != func.entry_block {
                        self.emit_block_params(&mut emitter, module, func, block);
                    }

                    for &inst in &func.layout.blocks[block].insts {
                        self.emit_inst(&mut emitter, module, func, inst);
                    }
                }

                self.emit_epilogue(&mut emitter, module, func);
            }

            // 本地重定位处理 (比如跳转到已经生成的 Block)
            for reloc in &text.relocs {
                if let RelocTarget::Block(block) = reloc.target {
                    if let Some(&target_off) = block_offsets.get(&block) {
                        let patch_pos = reloc.offset as usize;
                        match reloc.kind {
                            RelocKind::X86_64Rel8 => {
                                let next_inst_off = (patch_pos + 1) as isize;
                                let diff = (target_off as isize) - next_inst_off;
                                text.data[patch_pos] = (diff + reloc.addend as isize) as i8 as u8;
                            }
                            RelocKind::X86_64Pc32 => {
                                let next_inst_off = (patch_pos + 4) as isize;
                                let diff = (target_off as isize) - next_inst_off;
                                let bytes = ((diff + reloc.addend as isize) as i32).to_le_bytes();
                                text.data[patch_pos..patch_pos + 4].copy_from_slice(&bytes);
                            }
                            _ => {}
                        }
                    }
                }
            }

            let scope = match func.linkage {
                Linkage::Export => SymbolScope::Dynamic,
                Linkage::Local => SymbolScope::Compilation,
                Linkage::Import => unreachable!(),
            };

            // 如果符号已经存在（可能是在其定义之前被引用），则更新它
            if let Some(symbol_id) = symbols.get(&func.name) {
                let symbol = obj.symbol_mut(*symbol_id);
                symbol.section = SymbolSection::Section(text_section_id);
                symbol.value = current_offset as u64;
                symbol.size = text.data.len() as u64;
                symbol.kind = SymbolKind::Text;
                symbol.scope = scope;
            } else {
                let symbol_id = obj.add_symbol(Symbol {
                    name: func.name.as_bytes().to_vec(),
                    value: current_offset as u64,
                    size: text.data.len() as u64,
                    kind: SymbolKind::Text,
                    scope,
                    weak: false,
                    section: SymbolSection::Section(text_section_id),
                    flags: SymbolFlags::None,
                });
                symbols.insert(func.name.clone(), symbol_id);
            }

            for reloc in &text.relocs {
                if let RelocTarget::Symbol(name) = &reloc.target {
                    let symbol_id = *symbols.entry(name.clone()).or_insert_with(|| {
                        obj.add_symbol(Symbol {
                            name: name.as_bytes().to_vec(),
                            value: 0,
                            size: 0,
                            kind: SymbolKind::Unknown,
                            scope: SymbolScope::Unknown,
                            weak: false,
                            section: SymbolSection::Undefined,
                            flags: SymbolFlags::None,
                        })
                    });

                    // 尝试就地解析：如果符号在当前模块内已定义，则直接 Patch 并不生成 ELF 重定位
                    let symbol = obj.symbol(symbol_id);
                    if symbol.section != SymbolSection::Undefined
                        && reloc.kind == RelocKind::X86_64Pc32
                    {
                        let target_addr = symbol.value as i64;
                        let patch_offset = reloc.offset as usize;
                        let patch_addr = (current_offset + reloc.offset as u32) as i64;
                        let next_inst_addr = patch_addr + 4;
                        let diff = target_addr - next_inst_addr;
                        let bytes = ((diff + reloc.addend as i64) as i32).to_le_bytes();
                        text.data[patch_offset..patch_offset + 4].copy_from_slice(&bytes);
                    } else {
                        let (kind, encoding, size) = match reloc.kind {
                            RelocKind::X86_64Pc32 => {
                                (RelocationKind::Relative, RelocationEncoding::Generic, 32)
                            }
                            RelocKind::X86_64Plt32 => {
                                (RelocationKind::Relative, RelocationEncoding::X86Branch, 32)
                            }
                            RelocKind::X86_64_64Reloc => {
                                (RelocationKind::Absolute, RelocationEncoding::Generic, 64)
                            }
                            RelocKind::X86_64Rel8 => {
                                (RelocationKind::Relative, RelocationEncoding::Generic, 8)
                            }
                        };

                        obj.add_relocation(
                            text_section_id,
                            ObjectReloc {
                                offset: (current_offset + reloc.offset as u32) as u64,
                                symbol: symbol_id,
                                addend: reloc.addend,
                                flags: RelocationFlags::Generic {
                                    kind,
                                    encoding,
                                    size,
                                },
                            },
                        )
                        .map_err(|e| {
                            crate::Error::Message(format!("Failed to add relocation: {}", e))
                        })?;
                    }
                }
            }

            obj.append_section_data(text_section_id, &text.data, 1);
            current_offset += text.data.len() as u32;

            if let Some(unwind) = self.emit_unwind_info(func) {
                let unwind_section_id = obj.add_section(
                    vec![],
                    b".eh_frame".to_vec(),
                    object::SectionKind::ReadOnlyData,
                );
                obj.append_section_data(unwind_section_id, &unwind.data, 1);
            }
        }

        obj.write()
            .map_err(|e| crate::Error::Message(format!("Failed to write object file: {}", e)))
    }
}

pub enum Backend {
    X86_64(x86_64::X86_64Backend),
}

impl Backend {
    pub fn new() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Backend::X86_64(x86_64::X86_64Backend::new())
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            unimplemented!("Unsupported architecture")
        }
    }

    pub fn compile_module(&self, module: &veloc_ir::Module) -> crate::Result<Vec<u8>> {
        match self {
            Backend::X86_64(b) => b.compile_module(module),
        }
    }
}
