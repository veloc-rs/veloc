use super::Reg;
use super::snippets::*;
use crate::{Emitter, RelocKind, RelocTarget, Relocation, Section, TargetBackend};
use cranelift_entity::EntityRef;
use veloc_ir::CallConv;
use veloc_ir::{FloatCC, Function, Inst, InstructionData, IntCC, Linkage, Opcode, Value};

pub struct X86_64Backend;

impl X86_64Backend {
    pub fn new() -> Self {
        Self
    }

    fn get_param_regs(&self, cc: CallConv) -> (&'static [Reg], &'static [Reg]) {
        match cc {
            CallConv::SystemV => (
                &[Reg::RDI, Reg::RSI, Reg::RDX, Reg::RCX, Reg::R8, Reg::R9],
                &[
                    Reg::XMM0,
                    Reg::XMM1,
                    Reg::XMM2,
                    Reg::XMM3,
                    Reg::XMM4,
                    Reg::XMM5,
                    Reg::XMM6,
                    Reg::XMM7,
                ],
            ),
        }
    }

    fn rex(&self, emitter: &mut Emitter, is_64: bool) {
        if is_64 {
            emitter.write_bytes(&[0x48]);
        }
    }

    fn bin_op(&self, emitter: &mut Emitter, is_64: bool, inst: X86_64Inst) {
        self.rex(emitter, is_64);
        emitter.inst(inst).reg(Reg::RAX).reg(Reg::RCX).emit();
    }

    fn val_offset(&self, _func: &Function, v: Value) -> i32 {
        // 每个 Value 在栈上占用 8 字节
        -(8 * (v.index() as i32 + 1))
    }

    fn ss_offset(&self, func: &Function, ss: veloc_ir::StackSlot) -> i32 {
        // 栈格式: [Value 0..N] [StackSlot 0..M]
        // SSA values 占据的总大小
        let ssa_size = (func.dfg.values.len() * 8) as i32;
        let mut offset = ssa_size;
        for (idx, (_, data)) in func.stack_slots.iter().enumerate() {
            if idx == ss.index() {
                return -(offset + data.size as i32);
            }
            offset += data.size as i32;
        }
        0
    }

    fn emit_store_val(&self, emitter: &mut Emitter, func: &Function, v: Value, src_reg: Reg) {
        let offset = self.val_offset(func, v);
        let ty = func.dfg.values[v].ty;
        if (src_reg as u8) >= 16 {
            let inst = if ty == veloc_ir::Type::F32 {
                X86_64Inst::MovssRbpOffX
            } else {
                X86_64Inst::MovsdRbpOffX
            };
            emitter.inst(inst).reg(src_reg).imm(offset as u64).emit();
        } else {
            let inst = if self.is_64(ty) {
                X86_64Inst::MovRbpOffR
            } else {
                X86_64Inst::MovRbpOffR32
            };
            emitter.inst(inst).reg(src_reg).imm(offset as u64).emit();
        }
    }

    fn emit_load_val(&self, emitter: &mut Emitter, func: &Function, v: Value, dst_reg: Reg) {
        let offset = self.val_offset(func, v);
        let ty = func.dfg.values[v].ty;
        if (dst_reg as u8) >= 16 {
            let inst = if ty == veloc_ir::Type::F32 {
                X86_64Inst::MovssXRbpOff
            } else {
                X86_64Inst::MovsdXRbpOff
            };
            emitter.inst(inst).reg(dst_reg).imm(offset as u64).emit();
        } else {
            let inst = if self.is_64(ty) {
                X86_64Inst::MovRRbpOff
            } else {
                X86_64Inst::MovR32RbpOff
            };
            emitter.inst(inst).reg(dst_reg).imm(offset as u64).emit();
        }
    }

    fn emit_load_xmm(
        &self,
        emitter: &mut Emitter,
        func: &Function,
        v: Value,
        xmm_idx: Reg,
        is_f32: bool,
    ) {
        let offset = self.val_offset(func, v);
        let inst = if is_f32 {
            X86_64Inst::MovssXRbpOff
        } else {
            X86_64Inst::MovsdXRbpOff
        };
        emitter.inst(inst).reg(xmm_idx).imm(offset as u64).emit();
    }

    fn emit_store_xmm(
        &self,
        emitter: &mut Emitter,
        func: &Function,
        v: Value,
        xmm_idx: Reg,
        is_f32: bool,
    ) {
        let offset = self.val_offset(func, v);
        let inst = if is_f32 {
            X86_64Inst::MovssRbpOffX
        } else {
            X86_64Inst::MovsdRbpOffX
        };
        emitter.inst(inst).reg(xmm_idx).imm(offset as u64).emit();
    }

    fn is_64(&self, ty: veloc_ir::Type) -> bool {
        ty == veloc_ir::Type::I64 || ty == veloc_ir::Type::F64
    }

    fn is_float(&self, ty: veloc_ir::Type) -> bool {
        ty == veloc_ir::Type::F32 || ty == veloc_ir::Type::F64
    }

    fn emit_unary(
        &self,
        emitter: &mut Emitter,
        func: &Function,
        opcode: Opcode,
        arg: Value,
        res_ty: veloc_ir::Type,
        res: Option<Value>,
    ) {
        let src_ty = func.dfg.values[arg].ty;
        let is_64_src = self.is_64(src_ty);
        let is_64_dest = self.is_64(res_ty);
        let is_float_dest = self.is_float(res_ty);

        self.emit_load_val(emitter, func, arg, Reg::RAX);

        match opcode {
            Opcode::Eqz => {
                emitter.emit_inst(if is_64_src {
                    X86_64Inst::TestRaxRax
                } else {
                    X86_64Inst::TestEaxEax
                });
                emitter.emit_inst(X86_64Inst::SeteAl);
                emitter.emit_inst(X86_64Inst::MovzxEaxAl);
            }
            Opcode::Clz | Opcode::Ctz | Opcode::Popcnt => {
                let inst = if is_64_src {
                    match opcode {
                        Opcode::Clz => X86_64Inst::LzcntR64R64,
                        Opcode::Ctz => X86_64Inst::TzcntR64R64,
                        _ => X86_64Inst::PopcntR64R64,
                    }
                } else {
                    match opcode {
                        Opcode::Clz => X86_64Inst::LzcntRR,
                        Opcode::Ctz => X86_64Inst::TzcntRR,
                        _ => X86_64Inst::PopcntRR,
                    }
                };
                emitter.inst(inst).reg(Reg::RAX).reg(Reg::RAX).emit();
            }
            Opcode::Ineg
            | Opcode::Fneg
            | Opcode::Abs
            | Opcode::Sqrt
            | Opcode::Ceil
            | Opcode::Floor
            | Opcode::Trunc
            | Opcode::Nearest => {
                if is_float_dest {
                    let is_f32 = src_ty == veloc_ir::Type::F32;
                    self.emit_load_xmm(emitter, func, arg, Reg::XMM0, is_f32);
                    match opcode {
                        Opcode::Fneg => {
                            if is_f32 {
                                emitter
                                    .inst(X86_64Inst::MovR32Imm32)
                                    .reg(Reg::RAX)
                                    .imm(0x80000000u32 as u64)
                                    .emit();
                                emitter
                                    .inst(X86_64Inst::MovdXR)
                                    .reg(Reg::RAX)
                                    .reg(Reg::XMM1)
                                    .emit();
                                emitter
                                    .inst(X86_64Inst::XorpsXX)
                                    .reg(Reg::XMM0)
                                    .reg(Reg::XMM1)
                                    .emit();
                            } else {
                                emitter
                                    .inst(X86_64Inst::MovR64Imm64)
                                    .reg(Reg::RAX)
                                    .imm(0x8000000000000000u64)
                                    .emit();
                                emitter
                                    .inst(X86_64Inst::MovqXR64)
                                    .reg(Reg::RAX)
                                    .reg(Reg::XMM1)
                                    .emit();
                                emitter
                                    .inst(X86_64Inst::XorpdXX)
                                    .reg(Reg::XMM0)
                                    .reg(Reg::XMM1)
                                    .emit();
                            }
                        }
                        Opcode::Abs => {
                            if is_f32 {
                                emitter
                                    .inst(X86_64Inst::MovR32Imm32)
                                    .reg(Reg::RAX)
                                    .imm(0x7FFFFFFFu32 as u64)
                                    .emit();
                                emitter
                                    .inst(X86_64Inst::MovdXR)
                                    .reg(Reg::RAX)
                                    .reg(Reg::XMM1)
                                    .emit();
                                emitter
                                    .inst(X86_64Inst::AndpsXX)
                                    .reg(Reg::XMM0)
                                    .reg(Reg::XMM1)
                                    .emit();
                            } else {
                                emitter
                                    .inst(X86_64Inst::MovR64Imm64)
                                    .reg(Reg::RAX)
                                    .imm(0x7FFFFFFFFFFFFFFFu64)
                                    .emit();
                                emitter
                                    .inst(X86_64Inst::MovqXR64)
                                    .reg(Reg::RAX)
                                    .reg(Reg::XMM1)
                                    .emit();
                                emitter
                                    .inst(X86_64Inst::AndpdXX)
                                    .reg(Reg::XMM0)
                                    .reg(Reg::XMM1)
                                    .emit();
                            }
                        }
                        Opcode::Sqrt => {
                            let inst = if is_f32 {
                                X86_64Inst::SqrtssXX
                            } else {
                                X86_64Inst::SqrtsdXX
                            };
                            emitter.inst(inst).reg(Reg::XMM0).reg(Reg::XMM0).emit();
                        }
                        Opcode::Ceil | Opcode::Floor | Opcode::Trunc | Opcode::Nearest => {
                            let imm = match opcode {
                                Opcode::Ceil => 0x02,
                                Opcode::Floor => 0x01,
                                Opcode::Trunc => 0x03,
                                _ => 0x00, // Nearest
                            };
                            let inst = if is_f32 {
                                X86_64Inst::RoundssXXI
                            } else {
                                X86_64Inst::RoundsdXXI
                            };
                            emitter
                                .inst(inst)
                                .reg(Reg::XMM0)
                                .reg(Reg::XMM0)
                                .imm(imm as u64)
                                .emit();
                        }
                        _ => unreachable!(),
                    }
                    if let Some(v) = res {
                        self.emit_store_xmm(emitter, func, v, Reg::XMM0, is_f32);
                    }
                    return;
                }
            }
            Opcode::Wrap if res_ty == veloc_ir::Type::I32 => {
                emitter.emit_inst(X86_64Inst::MovEaxEax);
            }
            Opcode::ExtendS | Opcode::ExtendU => {
                let is_signed = opcode == Opcode::ExtendS;
                match src_ty {
                    veloc_ir::Type::I8 => {
                        self.rex(emitter, is_64_dest);
                        emitter.emit_inst(if is_signed {
                            X86_64Inst::Movsx8RaxRax
                        } else {
                            X86_64Inst::Movzx8RaxRax
                        });
                    }
                    veloc_ir::Type::I16 => {
                        self.rex(emitter, is_64_dest);
                        emitter.emit_inst(if is_signed {
                            X86_64Inst::Movsx16RaxRax
                        } else {
                            X86_64Inst::Movzx16RaxRax
                        });
                    }
                    veloc_ir::Type::I32 if is_64_dest => {
                        emitter.emit_inst(if is_signed {
                            X86_64Inst::MovsxdRaxEax
                        } else {
                            X86_64Inst::MovEaxEax
                        });
                    }
                    _ => {}
                }
            }
            Opcode::Promote => {
                self.emit_load_xmm(emitter, func, arg, Reg::XMM0, true);
                emitter
                    .inst(X86_64Inst::Cvtss2sdXX)
                    .reg(Reg::XMM0)
                    .reg(Reg::XMM0)
                    .emit();
            }
            Opcode::Demote => {
                self.emit_load_xmm(emitter, func, arg, Reg::XMM0, false);
                emitter
                    .inst(X86_64Inst::Cvtsd2ssXX)
                    .reg(Reg::XMM0)
                    .reg(Reg::XMM0)
                    .emit();
            }
            Opcode::ConvertS | Opcode::ConvertU => {
                let inst = match (res_ty == veloc_ir::Type::F64, is_64_src) {
                    (true, true) => X86_64Inst::Cvtsi2sdXR64,
                    (true, false) => X86_64Inst::Cvtsi2sdXR,
                    (false, true) => X86_64Inst::Cvtsi2ssXR64,
                    (false, false) => X86_64Inst::Cvtsi2ssXR,
                };
                emitter.inst(inst).reg(Reg::RAX).reg(Reg::XMM0).emit();
            }
            Opcode::TruncS | Opcode::TruncU => {
                self.emit_load_xmm(emitter, func, arg, Reg::XMM0, src_ty == veloc_ir::Type::F32);
                let inst = match (src_ty == veloc_ir::Type::F64, is_64_dest) {
                    (true, true) => X86_64Inst::Cvttsd2siRX64,
                    (true, false) => X86_64Inst::Cvttsd2siRX,
                    (false, true) => X86_64Inst::Cvttss2siRX64,
                    (false, false) => X86_64Inst::Cvttss2siRX,
                };
                emitter.inst(inst).reg(Reg::XMM0).reg(Reg::RAX).emit();
            }
            Opcode::Reinterpret => {
                if is_float_dest {
                    let inst = if is_64_dest {
                        X86_64Inst::MovqXR64
                    } else {
                        X86_64Inst::MovdXR
                    };
                    emitter.inst(inst).reg(Reg::RAX).reg(Reg::XMM0).emit();
                } else {
                    self.emit_load_xmm(
                        emitter,
                        func,
                        arg,
                        Reg::XMM0,
                        src_ty == veloc_ir::Type::F32,
                    );
                    let inst = if is_64_dest {
                        X86_64Inst::MovqRX64
                    } else {
                        X86_64Inst::MovdRX
                    };
                    emitter.inst(inst).reg(Reg::XMM0).reg(Reg::RAX).emit();
                }
            }
            _ => {}
        }

        if let Some(v) = res {
            if is_float_dest {
                self.emit_store_xmm(emitter, func, v, Reg::XMM0, res_ty == veloc_ir::Type::F32);
            } else {
                self.emit_store_val(emitter, func, v, Reg::RAX);
            }
        }
    }

    fn emit_binary(
        &self,
        emitter: &mut Emitter,
        func: &Function,
        opcode: Opcode,
        args: &[Value],
        res: Option<Value>,
    ) {
        let ty = func.dfg.values[args[0]].ty;
        let is_64 = self.is_64(ty);
        let is_float = self.is_float(ty);

        if is_float {
            let is_f32 = ty == veloc_ir::Type::F32;
            self.emit_load_xmm(emitter, func, args[0], Reg::XMM0, is_f32);
            self.emit_load_xmm(emitter, func, args[1], Reg::XMM1, is_f32);

            match opcode {
                Opcode::Fadd => {
                    let inst = if is_f32 {
                        X86_64Inst::AddssXX
                    } else {
                        X86_64Inst::AddsdXX
                    };
                    emitter.inst(inst).reg(Reg::XMM1).reg(Reg::XMM0).emit();
                }
                Opcode::Fsub => {
                    let inst = if is_f32 {
                        X86_64Inst::SubssXX
                    } else {
                        X86_64Inst::SubsdXX
                    };
                    emitter.inst(inst).reg(Reg::XMM1).reg(Reg::XMM0).emit();
                }
                Opcode::Fmul => {
                    let inst = if is_f32 {
                        X86_64Inst::MulssXX
                    } else {
                        X86_64Inst::MulsdXX
                    };
                    emitter.inst(inst).reg(Reg::XMM1).reg(Reg::XMM0).emit();
                }
                Opcode::Fdiv => {
                    let inst = if is_f32 {
                        X86_64Inst::DivssXX
                    } else {
                        X86_64Inst::DivsdXX
                    };
                    emitter.inst(inst).reg(Reg::XMM1).reg(Reg::XMM0).emit();
                }
                Opcode::Min => {
                    let inst = if is_f32 {
                        X86_64Inst::MinssXX
                    } else {
                        X86_64Inst::MinsdXX
                    };
                    emitter.inst(inst).reg(Reg::XMM1).reg(Reg::XMM0).emit();
                }
                Opcode::Max => {
                    let inst = if is_f32 {
                        X86_64Inst::MaxssXX
                    } else {
                        X86_64Inst::MaxsdXX
                    };
                    emitter.inst(inst).reg(Reg::XMM1).reg(Reg::XMM0).emit();
                }
                Opcode::Copysign => {
                    if is_f32 {
                        emitter
                            .inst(X86_64Inst::MovR32Imm32)
                            .reg(Reg::RAX)
                            .imm(0x80000000u32 as u64)
                            .emit();
                        emitter
                            .inst(X86_64Inst::MovdXR)
                            .reg(Reg::RAX)
                            .reg(Reg::XMM2)
                            .emit();
                        emitter
                            .inst(X86_64Inst::AndpsXX)
                            .reg(Reg::XMM2)
                            .reg(Reg::XMM1)
                            .emit();
                        emitter
                            .inst(X86_64Inst::AndnpsXX)
                            .reg(Reg::XMM2)
                            .reg(Reg::XMM0)
                            .emit();
                        emitter
                            .inst(X86_64Inst::OrpsXX)
                            .reg(Reg::XMM1)
                            .reg(Reg::XMM0)
                            .emit();
                    } else {
                        emitter
                            .inst(X86_64Inst::MovR64Imm64)
                            .reg(Reg::RAX)
                            .imm(0x8000000000000000u64)
                            .emit();
                        emitter
                            .inst(X86_64Inst::MovqXR64)
                            .reg(Reg::RAX)
                            .reg(Reg::XMM2)
                            .emit();
                        emitter
                            .inst(X86_64Inst::AndpdXX)
                            .reg(Reg::XMM2)
                            .reg(Reg::XMM1)
                            .emit();
                        emitter
                            .inst(X86_64Inst::AndnpdXX)
                            .reg(Reg::XMM2)
                            .reg(Reg::XMM0)
                            .emit();
                        emitter
                            .inst(X86_64Inst::OrpdXX)
                            .reg(Reg::XMM1)
                            .reg(Reg::XMM0)
                            .emit();
                    }
                }
                _ => return,
            }

            if let Some(v) = res {
                self.emit_store_xmm(emitter, func, v, Reg::XMM0, is_f32);
            }
        } else {
            self.emit_load_val(emitter, func, args[0], Reg::RAX);
            self.emit_load_val(emitter, func, args[1], Reg::RCX);
            match opcode {
                Opcode::Iadd => self.bin_op(emitter, is_64, X86_64Inst::AddR32R32),
                Opcode::Isub => self.bin_op(emitter, is_64, X86_64Inst::SubR32R32),
                Opcode::Imul => self.bin_op(emitter, is_64, X86_64Inst::MulR32R32),
                Opcode::And => self.bin_op(emitter, is_64, X86_64Inst::AndR32R32),
                Opcode::Or => self.bin_op(emitter, is_64, X86_64Inst::OrR32R32),
                Opcode::Xor => self.bin_op(emitter, is_64, X86_64Inst::XorR32R32),
                Opcode::Shl | Opcode::ShrU | Opcode::ShrS | Opcode::Rotl | Opcode::Rotr => {
                    emitter.emit_inst(X86_64Inst::MovEcxEcx);
                    let op = match opcode {
                        Opcode::Shl => X86_64Inst::ShlEaxCl,
                        Opcode::ShrU => X86_64Inst::ShrEaxCl,
                        Opcode::ShrS => X86_64Inst::SarEaxCl,
                        Opcode::Rotl => X86_64Inst::RolEaxCl,
                        _ => X86_64Inst::RorEaxCl,
                    };
                    self.rex(emitter, is_64);
                    emitter.emit_inst(op);
                }
                Opcode::Idiv | Opcode::Udiv | Opcode::Irem | Opcode::Urem => {
                    if opcode == Opcode::Udiv || opcode == Opcode::Urem {
                        if is_64 {
                            emitter.emit_inst(X86_64Inst::XorRdxRdx64);
                        } else {
                            emitter.emit_inst(X86_64Inst::XorEdxEdx);
                        }
                    } else {
                        if is_64 {
                            emitter.emit_inst(X86_64Inst::Cqo);
                        } else {
                            emitter.emit_inst(X86_64Inst::Cdq);
                        }
                    }

                    let inst = if is_64 {
                        if opcode == Opcode::Udiv || opcode == Opcode::Urem {
                            X86_64Inst::DivRcx64
                        } else {
                            X86_64Inst::IdivRcx64
                        }
                    } else {
                        if opcode == Opcode::Udiv || opcode == Opcode::Urem {
                            X86_64Inst::DivRcx
                        } else {
                            X86_64Inst::IdivRcx
                        }
                    };
                    emitter.emit_inst(inst);

                    if opcode == Opcode::Irem || opcode == Opcode::Urem {
                        emitter.emit_inst(if is_64 {
                            X86_64Inst::MovRaxRdx64
                        } else {
                            X86_64Inst::MovEaxEdx
                        });
                    }
                }
                _ => {}
            }

            if let Some(v) = res {
                self.emit_store_val(emitter, func, v, Reg::RAX);
            }
        }
    }

    fn emit_load(
        &self,
        emitter: &mut Emitter,
        func: &Function,
        ptr: Value,
        offset: i32,
        ty: veloc_ir::Type,
        res: Option<Value>,
    ) {
        self.emit_load_val(emitter, func, ptr, Reg::RAX);
        if self.is_float(ty) {
            let inst = if ty == veloc_ir::Type::F32 {
                X86_64Inst::MovssXR64Off
            } else {
                X86_64Inst::MovsdXR64Off
            };
            emitter
                .inst(inst)
                .reg(Reg::XMM0)
                .reg(Reg::RAX)
                .imm(offset as u64)
                .emit();
            if let Some(v) = res {
                self.emit_store_val(emitter, func, v, Reg::XMM0);
            }
        } else {
            let inst = match ty {
                veloc_ir::Type::I64 => X86_64Inst::MovR64R64Off,
                veloc_ir::Type::I32 => X86_64Inst::MovR32R64Off,
                veloc_ir::Type::I16 => X86_64Inst::Movzx16R64R64Off,
                veloc_ir::Type::I8 => X86_64Inst::Movzx8R64R64Off,
                _ => return,
            };
            emitter
                .inst(inst)
                .reg(Reg::RAX)
                .reg(Reg::RAX)
                .imm(offset as u64)
                .emit();
            if let Some(v) = res {
                self.emit_store_val(emitter, func, v, Reg::RAX);
            }
        }
    }

    fn emit_store(
        &self,
        emitter: &mut Emitter,
        func: &Function,
        ptr: Value,
        value: Value,
        offset: i32,
    ) {
        let ty = func.dfg.values[value].ty;
        self.emit_load_val(emitter, func, ptr, Reg::RAX);
        if self.is_float(ty) {
            self.emit_load_val(emitter, func, value, Reg::XMM0);
            let inst = if ty == veloc_ir::Type::F32 {
                X86_64Inst::MovssR64OffX
            } else {
                X86_64Inst::MovsdR64OffX
            };
            emitter
                .inst(inst)
                .reg(Reg::RAX)
                .reg(Reg::XMM0)
                .imm(offset as u64)
                .emit();
        } else {
            self.emit_load_val(emitter, func, value, Reg::RBX);
            let inst = match ty {
                veloc_ir::Type::I64 => X86_64Inst::MovR64OffR64,
                veloc_ir::Type::I32 => X86_64Inst::MovR64OffR32,
                veloc_ir::Type::I16 => X86_64Inst::MovR64OffR16,
                veloc_ir::Type::I8 => X86_64Inst::MovR64OffR8,
                _ => return,
            };
            emitter
                .inst(inst)
                .reg(Reg::RBX)
                .reg(Reg::RAX)
                .imm(offset as u64)
                .emit();
        }
    }
}

impl TargetBackend for X86_64Backend {
    fn emit_prologue(&self, emitter: &mut Emitter, module: &veloc_ir::Module, func: &Function) {
        emitter.emit_inst(X86_64Inst::Prologue);
        // 计算总栈大小: SSA Values + Stack Slots
        let ssa_size = func.dfg.values.len() * 8;
        let ss_size: u32 = func.stack_slots.values().map(|s| s.size).sum();
        let mut stack_size = (ssa_size as i32) + (ss_size as i32);

        // 16 字节对齐
        if stack_size % 16 != 0 {
            stack_size += 16 - (stack_size % 16);
        }

        emitter
            .inst(X86_64Inst::SubRspImm32)
            .imm(stack_size as u64)
            .emit();

        // 处理 Entry Block 的参数
        if let Some(entry) = func.entry_block {
            let params = &func.layout.blocks[entry].params;
            let sig = &module.signatures[func.signature];
            let (int_regs, float_regs) = self.get_param_regs(sig.call_conv);
            let mut int_idx = 0;
            let mut float_idx = 0;

            for &param in params {
                let ty = func.dfg.values[param].ty;
                if self.is_float(ty) {
                    if float_idx < float_regs.len() {
                        self.emit_store_val(emitter, func, param, float_regs[float_idx]);
                        float_idx += 1;
                    }
                } else {
                    if int_idx < int_regs.len() {
                        self.emit_store_val(emitter, func, param, int_regs[int_idx]);
                        int_idx += 1;
                    }
                }
            }
        }
    }

    fn emit_epilogue(&self, emitter: &mut Emitter, _module: &veloc_ir::Module, func: &Function) {
        let ssa_size = func.dfg.values.len() * 8;
        let ss_size: u32 = func.stack_slots.values().map(|s| s.size).sum();
        let mut stack_size = (ssa_size as i32) + (ss_size as i32);
        if stack_size % 16 != 0 {
            stack_size += 16 - (stack_size % 16);
        }

        emitter
            .inst(X86_64Inst::AddRspImm32)
            .imm(stack_size as u64)
            .emit();
        emitter.emit_inst(X86_64Inst::Ret);
    }

    fn emit_block_params(
        &self,
        emitter: &mut Emitter,
        module: &veloc_ir::Module,
        func: &Function,
        block: veloc_ir::Block,
    ) {
        let params = &func.layout.blocks[block].params;
        let sig = &module.signatures[func.signature];
        let (int_regs, float_regs) = self.get_param_regs(sig.call_conv);
        let mut int_idx = 0;
        let mut float_idx = 0;

        for &param in params {
            let ty = func.dfg.values[param].ty;
            if self.is_float(ty) {
                if float_idx < float_regs.len() {
                    self.emit_store_val(emitter, func, param, float_regs[float_idx]);
                    float_idx += 1;
                }
            } else {
                if int_idx < int_regs.len() {
                    self.emit_store_val(emitter, func, param, int_regs[int_idx]);
                    int_idx += 1;
                }
            }
        }
    }

    fn emit_inst(
        &self,
        emitter: &mut Emitter,
        module: &veloc_ir::Module,
        func: &Function,
        inst: Inst,
    ) {
        let idata = &func.dfg.instructions[inst];
        let res = func.dfg.inst_results(inst);

        match idata {
            InstructionData::Iconst { value, ty } => {
                if ty.is_integer() {
                    if *ty == veloc_ir::Type::I64 {
                        emitter
                            .inst(X86_64Inst::MovR64Imm64)
                            .reg(Reg::RAX)
                            .imm(*value as u64)
                            .emit();
                    } else {
                        emitter
                            .inst(X86_64Inst::MovR32Imm32)
                            .reg(Reg::RAX)
                            .imm(*value as u64)
                            .emit();
                    }

                    if let Some(v) = res {
                        self.emit_store_val(emitter, func, v, Reg::RAX);
                    }
                }
            }
            InstructionData::Fconst { value, ty } => {
                if *ty == veloc_ir::Type::F64 {
                    emitter
                        .inst(X86_64Inst::MovR64Imm64)
                        .reg(Reg::RAX)
                        .imm(*value)
                        .emit();
                } else {
                    emitter
                        .inst(X86_64Inst::MovR32Imm32)
                        .reg(Reg::RAX)
                        .imm(*value as u64)
                        .emit();
                }

                if let Some(v) = res {
                    self.emit_store_val(emitter, func, v, Reg::RAX);
                }
            }
            InstructionData::Bconst { value } => {
                emitter
                    .inst(X86_64Inst::MovR32Imm32)
                    .reg(Reg::RAX)
                    .imm(if *value { 1 } else { 0 })
                    .emit();

                if let Some(v) = res {
                    self.emit_store_val(emitter, func, v, Reg::RAX);
                }
            }
            InstructionData::Binary { opcode, args, .. } => {
                self.emit_binary(emitter, func, *opcode, args, res);
            }
            InstructionData::Load { ptr, offset, ty } => {
                self.emit_load(emitter, func, *ptr, *offset as i32, *ty, res);
            }
            InstructionData::Store { ptr, value, offset } => {
                self.emit_store(emitter, func, *ptr, *value, *offset as i32);
            }
            InstructionData::IntCompare { kind, args, .. } => {
                let ty = func.dfg.values[args[0]].ty;
                let is_64 = ty == veloc_ir::Type::I64;

                self.emit_load_val(emitter, func, args[0], Reg::RAX); // RAX = args[0]
                self.emit_load_val(emitter, func, args[1], Reg::RCX); // RCX = args[1]

                self.bin_op(emitter, is_64, X86_64Inst::CmpR32R32);

                let setcc = match kind {
                    IntCC::Eq => X86_64Inst::SeteAl,
                    IntCC::Ne => X86_64Inst::SetneAl,
                    IntCC::LtS => X86_64Inst::SetlAl,
                    IntCC::LtU => X86_64Inst::SetbAl,
                    IntCC::GtS => X86_64Inst::SetgAl,
                    IntCC::GtU => X86_64Inst::SetaAl,
                    IntCC::LeS => X86_64Inst::SetleAl,
                    IntCC::LeU => X86_64Inst::SetbeAl,
                    IntCC::GeS => X86_64Inst::SetgeAl,
                    IntCC::GeU => X86_64Inst::SetaeAl,
                };

                emitter.emit_inst(setcc);
                emitter.emit_inst(X86_64Inst::MovzxEaxAl);

                if let Some(v) = res {
                    self.emit_store_val(emitter, func, v, Reg::RAX);
                }
            }
            InstructionData::Unary {
                opcode,
                arg,
                ty: res_ty,
            } => {
                self.emit_unary(emitter, func, *opcode, *arg, *res_ty, res);
            }
            InstructionData::StackLoad { slot, offset, ty } => {
                let sso = self.ss_offset(func, *slot) + (*offset as i32);
                if self.is_float(*ty) {
                    let inst = if *ty == veloc_ir::Type::F32 {
                        X86_64Inst::MovssXRbpOff
                    } else {
                        X86_64Inst::MovsdXRbpOff
                    };
                    emitter.inst(inst).reg(Reg::XMM0).imm(sso as u64).emit();
                    if let Some(v) = res {
                        self.emit_store_val(emitter, func, v, Reg::XMM0);
                    }
                } else {
                    emitter
                        .inst(X86_64Inst::MovRaxRbpOff)
                        .imm(sso as u64)
                        .emit();
                    if let Some(v) = res {
                        self.emit_store_val(emitter, func, v, Reg::RAX);
                    }
                }
            }
            InstructionData::StackStore {
                slot,
                value,
                offset,
            } => {
                let ty = func.dfg.values[*value].ty;
                let sso = self.ss_offset(func, *slot) + (*offset as i32);
                if self.is_float(ty) {
                    self.emit_load_val(emitter, func, *value, Reg::XMM0);
                    let inst = if ty == veloc_ir::Type::F32 {
                        X86_64Inst::MovssRbpOffX
                    } else {
                        X86_64Inst::MovsdRbpOffX
                    };
                    emitter.inst(inst).reg(Reg::XMM0).imm(sso as u64).emit();
                } else {
                    self.emit_load_val(emitter, func, *value, Reg::RAX); // RAX = value
                    emitter
                        .inst(X86_64Inst::MovRbpOffRax)
                        .imm(sso as u64)
                        .emit();
                }
            }
            InstructionData::StackAddr { slot, offset } => {
                let sso = self.ss_offset(func, *slot) + (*offset as i32);
                emitter
                    .inst(X86_64Inst::LeaRaxRbpOff)
                    .imm(sso as u64)
                    .emit();

                if let Some(v) = res {
                    self.emit_store_val(emitter, func, v, Reg::RAX);
                }
            }
            InstructionData::Return { value } => {
                if let Some(v) = value {
                    let ty = func.dfg.values[*v].ty;
                    if self.is_float(ty) {
                        self.emit_load_val(emitter, func, *v, Reg::XMM0);
                        if ty == veloc_ir::Type::F32 {
                            emitter
                                .inst(X86_64Inst::MovdRX)
                                .reg(Reg::RAX)
                                .reg(Reg::XMM0)
                                .emit();
                        } else {
                            emitter
                                .inst(X86_64Inst::MovqRX64)
                                .reg(Reg::RAX)
                                .reg(Reg::XMM0)
                                .emit();
                        }
                    } else {
                        self.emit_load_val(emitter, func, *v, Reg::RAX);
                    }
                }

                self.emit_epilogue(emitter, module, func);
            }
            InstructionData::Call { func_id, args, .. } => {
                let mut int_idx = 0;
                let mut float_idx = 0;
                let (int_regs, float_regs) =
                    self.get_param_regs(module.signatures[func.signature].call_conv);

                for &arg in func.dfg.get_value_list(*args) {
                    let arg_ty = func.dfg.values[arg].ty;
                    if self.is_float(arg_ty) {
                        if float_idx < float_regs.len() {
                            self.emit_load_val(emitter, func, arg, float_regs[float_idx]);
                            float_idx += 1;
                        }
                    } else {
                        if int_idx < int_regs.len() {
                            self.emit_load_val(emitter, func, arg, int_regs[int_idx]);
                            int_idx += 1;
                        }
                    }
                }

                // CALL
                let target_func = &module.functions[*func_id];
                let kind = if target_func.linkage == Linkage::Import {
                    RelocKind::X86_64Plt32
                } else {
                    RelocKind::X86_64Pc32
                };

                let patch_offset = emitter.current_offset() + 1;
                emitter.inst(X86_64Inst::CallRel32).imm(0).emit();
                emitter.section.relocs.push(Relocation {
                    offset: patch_offset,
                    kind,
                    target: RelocTarget::Symbol(target_func.name.clone()),
                    addend: -4,
                });

                if let Some(v) = res {
                    let ty = func.dfg.values[v].ty;
                    if self.is_float(ty) {
                        self.emit_store_val(emitter, func, v, Reg::XMM0);
                    } else {
                        self.emit_store_val(emitter, func, v, Reg::RAX);
                    }
                }
            }
            InstructionData::Jump { dest } => {
                let (int_regs, float_regs) =
                    self.get_param_regs(module.signatures[func.signature].call_conv);
                let mut int_idx = 0;
                let mut float_idx = 0;

                let dest_data = func.dfg.block_calls[*dest];
                for &arg in func.dfg.get_value_list(dest_data.args) {
                    let ty = func.dfg.values[arg].ty;
                    if self.is_float(ty) {
                        if float_idx < float_regs.len() {
                            self.emit_load_val(emitter, func, arg, float_regs[float_idx]);
                            float_idx += 1;
                        }
                    } else {
                        if int_idx < int_regs.len() {
                            self.emit_load_val(emitter, func, arg, int_regs[int_idx]);
                            int_idx += 1;
                        }
                    }
                }

                let patch_offset = emitter.current_offset() + 1;
                emitter.inst(X86_64Inst::JmpRel32).imm(0).emit();
                emitter.section.relocs.push(Relocation {
                    offset: patch_offset,
                    kind: RelocKind::X86_64Pc32,
                    target: RelocTarget::Block(dest_data.block),
                    addend: 0,
                });
            }
            InstructionData::Br {
                condition,
                then_dest,
                else_dest,
            } => {
                let then_data = func.dfg.block_calls[*then_dest];
                let else_data = func.dfg.block_calls[*else_dest];
                self.emit_load_val(emitter, func, *condition, Reg::RAX);
                emitter.emit_inst(X86_64Inst::TestRaxRax);

                // JZ else_start
                let jz_patch_offset = emitter.current_offset() + 2;
                emitter.inst(X86_64Inst::JzRel32).imm(0).emit();

                // Then Case
                let (int_regs, float_regs) =
                    self.get_param_regs(module.signatures[func.signature].call_conv);
                let mut int_idx = 0;
                let mut float_idx = 0;
                for &arg in func.dfg.get_value_list(then_data.args) {
                    let ty = func.dfg.values[arg].ty;
                    if self.is_float(ty) {
                        if float_idx < float_regs.len() {
                            self.emit_load_val(emitter, func, arg, float_regs[float_idx]);
                            float_idx += 1;
                        }
                    } else {
                        if int_idx < int_regs.len() {
                            self.emit_load_val(emitter, func, arg, int_regs[int_idx]);
                            int_idx += 1;
                        }
                    }
                }

                let then_patch_offset = emitter.current_offset() + 1;
                emitter.inst(X86_64Inst::JmpRel32).imm(0).emit();
                emitter.section.relocs.push(Relocation {
                    offset: then_patch_offset,
                    kind: RelocKind::X86_64Pc32,
                    target: RelocTarget::Block(then_data.block),
                    addend: 0,
                });

                // Else Case (Patch JZ address)
                let else_start = emitter.current_offset();
                let jump_dist = (else_start as i32) - (jz_patch_offset as i32 + 4);
                let dist_bytes = jump_dist.to_le_bytes();
                emitter.section.data[jz_patch_offset as usize..jz_patch_offset as usize + 4]
                    .copy_from_slice(&dist_bytes);

                let mut int_idx = 0;
                let mut float_idx = 0;
                for &arg in func.dfg.get_value_list(else_data.args) {
                    let ty = func.dfg.values[arg].ty;
                    if self.is_float(ty) {
                        if float_idx < float_regs.len() {
                            self.emit_load_val(emitter, func, arg, float_regs[float_idx]);
                            float_idx += 1;
                        }
                    } else {
                        if int_idx < int_regs.len() {
                            self.emit_load_val(emitter, func, arg, int_regs[int_idx]);
                            int_idx += 1;
                        }
                    }
                }

                let else_patch_offset = emitter.current_offset() + 1;
                emitter.inst(X86_64Inst::JmpRel32).imm(0).emit();
                emitter.section.relocs.push(Relocation {
                    offset: else_patch_offset,
                    kind: RelocKind::X86_64Pc32,
                    target: RelocTarget::Block(else_data.block),
                    addend: 0,
                });
            }
            InstructionData::BrTable { index, table } => {
                self.emit_load_val(emitter, func, *index, Reg::RAX);
                let (int_regs, float_regs) =
                    self.get_param_regs(module.signatures[func.signature].call_conv);

                let table_data = &func.dfg.jump_tables[*table];
                let default_bc = table_data.targets[0];
                let default_bc_data = func.dfg.block_calls[default_bc];
                let targets = &table_data.targets[1..];

                for (i, &bc) in targets.iter().enumerate() {
                    let bc_data = func.dfg.block_calls[bc];
                    // CMP RAX, i
                    if i <= 127 {
                        emitter.inst(X86_64Inst::CmpRaxImm8).imm(i as u64).emit();
                    } else {
                        emitter.inst(X86_64Inst::CmpRaxImm32).imm(i as u64).emit();
                    }

                    // JNE next
                    let jne_offset = emitter.current_offset() + 2;
                    emitter.inst(X86_64Inst::JnzRel32).imm(0).emit();

                    // Pass arguments
                    let mut int_idx = 0;
                    let mut float_idx = 0;
                    for &arg in func.dfg.get_value_list(bc_data.args) {
                        let ty = func.dfg.values[arg].ty;
                        if self.is_float(ty) {
                            if float_idx < float_regs.len() {
                                self.emit_load_val(emitter, func, arg, float_regs[float_idx]);
                                float_idx += 1;
                            }
                        } else {
                            if int_idx < int_regs.len() {
                                self.emit_load_val(emitter, func, arg, int_regs[int_idx]);
                                int_idx += 1;
                            }
                        }
                    }

                    // JMP target
                    let jmp_offset = emitter.current_offset() + 1;
                    emitter.inst(X86_64Inst::JmpRel32).imm(0).emit();
                    emitter.section.relocs.push(Relocation {
                        offset: jmp_offset,
                        kind: RelocKind::X86_64Pc32,
                        target: RelocTarget::Block(bc_data.block),
                        addend: 0,
                    });

                    // Patch JNE
                    let next_start = emitter.current_offset();
                    let dist = (next_start as i32) - (jne_offset as i32 + 4);
                    let dist_bytes = dist.to_le_bytes();
                    emitter.section.data[jne_offset as usize..jne_offset as usize + 4]
                        .copy_from_slice(&dist_bytes);
                }

                // Default
                let mut int_idx = 0;
                let mut float_idx = 0;
                for &arg in func.dfg.get_value_list(default_bc_data.args) {
                    let ty = func.dfg.values[arg].ty;
                    if self.is_float(ty) {
                        if float_idx < float_regs.len() {
                            self.emit_load_val(emitter, func, arg, float_regs[float_idx]);
                            float_idx += 1;
                        }
                    } else {
                        if int_idx < int_regs.len() {
                            self.emit_load_val(emitter, func, arg, int_regs[int_idx]);
                            int_idx += 1;
                        }
                    }
                }
                let jmp_offset = emitter.current_offset() + 1;
                emitter.inst(X86_64Inst::JmpRel32).imm(0).emit();
                emitter.section.relocs.push(Relocation {
                    offset: jmp_offset,
                    kind: RelocKind::X86_64Pc32,
                    target: RelocTarget::Block(default_bc_data.block),
                    addend: 0,
                });
            }
            InstructionData::Unreachable => {
                emitter.emit_inst(X86_64Inst::Ud2);
            }
            InstructionData::CallIndirect {
                ptr, args, sig_id, ..
            } => {
                let sig = &module.signatures[*sig_id];
                // Pass arguments
                let mut int_idx = 0;
                let mut float_idx = 0;
                let (int_regs, float_regs) = self.get_param_regs(sig.call_conv);

                for &arg in func.dfg.get_value_list(*args) {
                    let arg_ty = func.dfg.values[arg].ty;
                    if self.is_float(arg_ty) {
                        if float_idx < float_regs.len() {
                            self.emit_load_val(emitter, func, arg, float_regs[float_idx]);
                            float_idx += 1;
                        }
                    } else {
                        if int_idx < int_regs.len() {
                            self.emit_load_val(emitter, func, arg, int_regs[int_idx]);
                            int_idx += 1;
                        }
                    }
                }

                // Now 'ptr' is the actual function address (loaded in translator.rs)
                self.emit_load_val(emitter, func, *ptr, Reg::RAX);

                // Indirect call
                emitter.inst(X86_64Inst::CallReg).reg(Reg::RAX).emit();

                if let Some(v) = res {
                    let res_ty = func.dfg.values[v].ty;
                    if self.is_float(res_ty) {
                        self.emit_store_val(emitter, func, v, Reg::XMM0);
                    } else {
                        self.emit_store_val(emitter, func, v, Reg::RAX);
                    }
                }
            }
            InstructionData::Select {
                condition,
                then_val,
                else_val,
                ty,
            } => {
                let is_64 = self.is_64(*ty);
                self.emit_load_val(emitter, func, *condition, Reg::RAX);
                emitter.emit_inst(X86_64Inst::TestRaxRax);

                let offset_else = self.val_offset(func, *else_val);
                let offset_then = self.val_offset(func, *then_val);

                if is_64 {
                    emitter
                        .inst(X86_64Inst::MovRRbpOff)
                        .reg(Reg::RAX)
                        .imm(offset_else as u64)
                        .emit();
                    emitter
                        .inst(X86_64Inst::MovRRbpOff)
                        .reg(Reg::RBX)
                        .imm(offset_then as u64)
                        .emit();
                    emitter.emit_inst(X86_64Inst::CmovnzRaxRbx64);
                } else {
                    emitter
                        .inst(X86_64Inst::MovR32RbpOff)
                        .reg(Reg::RAX)
                        .imm(offset_else as u64)
                        .emit();
                    emitter
                        .inst(X86_64Inst::MovR32RbpOff)
                        .reg(Reg::RBX)
                        .imm(offset_then as u64)
                        .emit();
                    emitter.emit_inst(X86_64Inst::CmovnzRaxRbx);
                }

                if let Some(v) = res {
                    let offset_res = self.val_offset(func, v);
                    if is_64 {
                        emitter
                            .inst(X86_64Inst::MovRbpOffR)
                            .reg(Reg::RAX)
                            .imm(offset_res as u64)
                            .emit();
                    } else {
                        emitter
                            .inst(X86_64Inst::MovRbpOffR32)
                            .reg(Reg::RAX)
                            .imm(offset_res as u64)
                            .emit();
                    }
                }
            }
            InstructionData::FloatCompare { kind, args, ty: _ } => {
                let lhs_ty = func.dfg.values[args[0]].ty;
                let is_f64 = lhs_ty == veloc_ir::Type::F64;
                self.emit_load_val(emitter, func, args[0], Reg::XMM0);
                self.emit_load_val(emitter, func, args[1], Reg::XMM1);

                if is_f64 {
                    emitter
                        .inst(X86_64Inst::UcomisdXX)
                        .reg(Reg::XMM0)
                        .reg(Reg::XMM1)
                        .emit();
                } else {
                    emitter
                        .inst(X86_64Inst::UcomissXX)
                        .reg(Reg::XMM0)
                        .reg(Reg::XMM1)
                        .emit();
                }

                emitter.emit_inst(X86_64Inst::XorRaxRax);

                match kind {
                    FloatCC::Eq => {
                        emitter.emit_inst(X86_64Inst::XorRbxRbx);
                        emitter.emit_inst(X86_64Inst::SeteAl);
                        emitter.emit_inst(X86_64Inst::SetnpBl);
                        emitter.emit_inst(X86_64Inst::AndAlBl);
                    }
                    FloatCC::Ne => {
                        emitter.emit_inst(X86_64Inst::XorRbxRbx);
                        emitter.emit_inst(X86_64Inst::SetneAl);
                        emitter.emit_inst(X86_64Inst::SetpBl);
                        emitter.emit_inst(X86_64Inst::OrAlBl);
                    }
                    FloatCC::Lt => emitter.emit_inst(X86_64Inst::SetbAl),
                    FloatCC::Le => emitter.emit_inst(X86_64Inst::SetbeAl),
                    FloatCC::Gt => emitter.emit_inst(X86_64Inst::SetaAl),
                    FloatCC::Ge => emitter.emit_inst(X86_64Inst::SetaeAl),
                }

                if let Some(v) = res {
                    self.emit_store_val(emitter, func, v, Reg::RAX);
                }
            }
            InstructionData::IntToPtr { .. }
            | InstructionData::PtrToInt { .. }
            | InstructionData::Gep { .. } => {
                todo!("Implement codegen for pointer instructions")
            }
        }
    }

    fn emit_unwind_info(&self, _func: &Function) -> Option<Section> {
        // 在实际项目中，这里会根据栈帧布局生成 DWARF FDE。
        // 目前返回 None 作为一个起点，用户可以通过注册自定义的 .eh_frame 来解决 unwind 崩溃。
        None
    }
}
