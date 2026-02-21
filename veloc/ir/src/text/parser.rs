//! IR 文本解析器
//!
//! 将 IR 文本格式解析回 IR 数据结构。
//! 这是 printer.rs 的逆操作。

use crate::{
    function::{Function, StackSlotData},
    inst::InstructionData,
    module::{Linkage, Module},
    opcode::{FloatCC, IntCC, MemFlags, Opcode},
    types::{
        Block, BlockCall, BlockCallData, FuncId, ScalarType, SigId,
        Signature, StackSlot, Type, Value, ValueData, ValueDef,
    },
    CallConv, Result,
};
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use hashbrown::HashMap;

/// 解析错误
#[derive(Debug, Clone)]
pub struct ParseError(pub String);

/// 从文本解析 IR Module
pub fn parse_module(input: &str) -> Result<Module> {
    let input = input.trim();
    if input.is_empty() {
        return Err(ParseError("Empty input".to_string()).into());
    }
    
    // 检查是否是函数定义
    if input.starts_with("local") || input.starts_with("export") || input.starts_with("import") {
        let func = parse_function(input)?;
        // 创建一个包含单个函数的模块
        let mut module_data = crate::module::ModuleData::default();
        let sig = Signature::new(vec![], vec![], CallConv::SystemV);
        let sig_id = module_data.intern_signature(sig);
        module_data.declare_function(func.name.clone(), sig_id, func.linkage);
        module_data.functions[FuncId(0)].dfg = func.dfg;
        module_data.functions[FuncId(0)].layout = func.layout;
        module_data.functions[FuncId(0)].stack_slots = func.stack_slots;
        module_data.functions[FuncId(0)].entry_block = func.entry_block;
        return Ok(Module::new(module_data));
    }
    
    // TODO: 解析完整的模块格式
    Err(ParseError("Module parsing not fully implemented".to_string()).into())
}

/// 从文本解析单个函数
pub fn parse_function(input: &str) -> Result<Function> {
    let lines_vec: Vec<&str> = input.lines().collect();
    let mut lines = alloc::collections::VecDeque::from(lines_vec);
    
    // 解析函数签名行
    let sig_line = lines.pop_front().ok_or_else(|| ParseError("Empty input".to_string()))?;
    let (name, linkage, params, returns) = parse_signature(sig_line)?;
    
    let _sig = Signature::new(params, returns, CallConv::SystemV);
    let sig_id = SigId(0);
    let mut func = Function::new(name, sig_id, linkage);
    
    // 解析函数体
    let mut ctx = ParseContext::new();
    
    while let Some(line) = lines.pop_front() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        
        // 解析栈槽定义
        if line.starts_with("ss") {
            parse_stack_slot(line, &mut func)?;
            continue;
        }
        
        // 解析基本块
        if line.contains(':') && !line.contains('=') {
            let mut block_lines = alloc::collections::VecDeque::new();
            block_lines.push_back(line);
            
            // 收集块的所有行
            while let Some(next_line) = lines.pop_front() {
                let trimmed = next_line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                // 检查是否是新块的开始
                if trimmed.ends_with(':') && !trimmed.contains('=') && !trimmed.starts_with("ss") {
                    // 这是下一个块的开始，把它放回去
                    lines.push_front(next_line);
                    break;
                }
                // 检查是否是下一个函数
                if trimmed.starts_with("local") || trimmed.starts_with("export") || 
                   trimmed.starts_with("import") || trimmed.starts_with("global") {
                    lines.push_front(next_line);
                    break;
                }
                block_lines.push_back(trimmed);
            }
            
            parse_block_lines(block_lines, &mut func, &mut ctx)?;
            continue;
        }
    }
    
    Ok(func)
}

/// 解析上下文
struct ParseContext {
    value_map: HashMap<String, Value>,
    block_map: HashMap<String, Block>,
    next_value_idx: u32,
}

impl ParseContext {
    fn new() -> Self {
        Self {
            value_map: HashMap::new(),
            block_map: HashMap::new(),
            next_value_idx: 0,
        }
    }
    
    fn get_or_create_value(&mut self, name: &str, func: &mut Function, ty: Type, def: ValueDef) -> Value {
        if let Some(v) = self.value_map.get(name) {
            *v
        } else {
            let idx = if let Some(idx_str) = name.strip_prefix('v') {
                idx_str.parse::<u32>().unwrap_or(self.next_value_idx)
            } else {
                self.next_value_idx
            };
            
            let v = Value(idx);
            self.value_map.insert(name.to_string(), v);
            
            if idx >= self.next_value_idx {
                self.next_value_idx = idx + 1;
            }
            
            // 确保值存在
            while func.dfg.values.len() <= idx as usize {
                func.dfg.values.push(ValueData { ty: Type::VOID, def: ValueDef::Param(Block(0)) });
            }
            func.dfg.values[v] = ValueData { ty, def };
            
            v
        }
    }
}

fn parse_signature(line: &str) -> Result<(String, Linkage, Vec<Type>, Vec<Type>)> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() < 4 {
        return Err(ParseError(format!("Invalid signature line: {}", line)).into());
    }
    
    // 解析 linkage
    let linkage = match parts[0] {
        "local" => Linkage::Local,
        "export" => Linkage::Export,
        "import" => Linkage::Import,
        _ => return Err(ParseError(format!("Invalid linkage: {}", parts[0])).into()),
    };
    
    // 检查 "function" 关键字
    if parts[1] != "function" {
        return Err(ParseError(format!("Expected 'function', got: {}", parts[1])).into());
    }
    
    // 解析函数名和参数
    let name_and_params = parts[2];
    let Some((name, params_str)) = name_and_params.split_once('(') else {
        return Err(ParseError(format!("Invalid function name/params: {}", name_and_params)).into());
    };
    
    let params_str = params_str.trim_end_matches(')');
    let params = if params_str.is_empty() {
        Vec::new()
    } else {
        parse_type_list(params_str)?
    };
    
    // 解析返回类型 (格式: "-> ret_type")
    let returns = if parts.len() >= 4 && parts[3] == "->" {
        if parts.len() >= 5 {
            if parts[4] == "void" {
                Vec::new()
            } else {
                parse_type_list(parts[4])?
            }
        } else {
            Vec::new()
        }
    } else {
        Vec::new()
    };
    
    Ok((name.to_string(), linkage, params, returns))
}

fn parse_type_list(s: &str) -> Result<Vec<Type>> {
    let mut types = Vec::new();
    for ty_str in s.split(',') {
        let ty_str = ty_str.trim();
        types.push(parse_type(ty_str)?);
    }
    Ok(types)
}

fn parse_type(s: &str) -> Result<Type> {
    match s {
        "i8" => Ok(Type::I8),
        "i16" => Ok(Type::I16),
        "i32" => Ok(Type::I32),
        "i64" => Ok(Type::I64),
        "f32" => Ok(Type::F32),
        "f64" => Ok(Type::F64),
        "bool" => Ok(Type::BOOL),
        "ptr" => Ok(Type::PTR),
        "void" => Ok(Type::VOID),
        _ => {
            // 尝试解析向量类型, 如 "i32<4>"
            if let Some(lt_pos) = s.find('<') {
                let base = &s[..lt_pos];
                let rest = &s[lt_pos + 1..s.len() - 1]; // 去掉 >
                
                if let Some(scalar) = parse_scalar_type(base) {
                    let (is_scalable, lanes_str) = if rest.starts_with("scalable ") {
                        (true, &rest[9..])
                    } else {
                        (false, rest)
                    };
                    
                    let lanes: u16 = lanes_str.parse()
                        .map_err(|_| ParseError(format!("Invalid lane count: {}", lanes_str)))?;
                    
                    return Ok(Type::new_vector(scalar, lanes, is_scalable));
                }
            }
            Err(ParseError(format!("Unknown type: {}", s)).into())
        }
    }
}

fn parse_scalar_type(s: &str) -> Option<ScalarType> {
    match s {
        "i8" => Some(ScalarType::I8),
        "i16" => Some(ScalarType::I16),
        "i32" => Some(ScalarType::I32),
        "i64" => Some(ScalarType::I64),
        "f32" => Some(ScalarType::F32),
        "f64" => Some(ScalarType::F64),
        _ => None,
    }
}

fn parse_stack_slot(line: &str, func: &mut Function) -> Result<()> {
    // 格式: ssN: size M
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() < 3 {
        return Err(ParseError(format!("Invalid stack slot: {}", line)).into());
    }
    
    let slot_name = parts[0].trim_end_matches(':');
    let idx_str = slot_name.strip_prefix("ss")
        .ok_or_else(|| ParseError(format!("Invalid stack slot name: {}", slot_name)))?;
    let idx: u32 = idx_str.parse()
        .map_err(|_| ParseError(format!("Invalid stack slot index: {}", idx_str)))?;
    
    let size: u32 = parts[2].parse()
        .map_err(|_| ParseError(format!("Invalid stack slot size: {}", parts[2])))?;
    
    while func.stack_slots.len() <= idx as usize {
        func.stack_slots.push(StackSlotData { size: 0 });
    }
    func.stack_slots[StackSlot(idx)] = StackSlotData { size };
    
    Ok(())
}

fn parse_block_lines(
    lines: alloc::collections::VecDeque<&str>,
    func: &mut Function,
    ctx: &mut ParseContext,
) -> Result<()> {
    let mut iter = lines.into_iter();
    let header = iter.next().ok_or_else(|| ParseError("Empty block".to_string()))?;
    
    // 解析块头
    let colon_pos = header.find(':').unwrap();
    let header_part = &header[..colon_pos];
    
    let (block_name, params_str) = if let Some(lparen) = header_part.find('(') {
        (&header_part[..lparen], &header_part[lparen..])
    } else {
        (header_part, "()")
    };
    
    let block_idx: u32 = block_name.strip_prefix("block")
        .ok_or_else(|| ParseError(format!("Invalid block name: {}", block_name)))?
        .parse()
        .map_err(|_| ParseError(format!("Invalid block index: {}", block_name)))?;
    
    let block = Block(block_idx);
    ctx.block_map.insert(block_name.to_string(), block);
    
    // 确保块存在
    while func.layout.blocks.len() <= block_idx as usize {
        func.layout.create_block();
    }
    
    if func.entry_block.is_none() {
        func.entry_block = Some(block);
    }
    
    if !func.layout.block_order.contains(&block) {
        func.layout.block_order.push(block);
    }
    
    // 解析参数
    let params_str = params_str.trim_start_matches('(').trim_end_matches(')');
    if !params_str.is_empty() {
        for param_def in params_str.split(',') {
            let param_def = param_def.trim();
            let parts: Vec<&str> = param_def.split(':').collect();
            if parts.len() != 2 {
                return Err(ParseError(format!("Invalid param definition: {}", param_def)).into());
            }
            
            let param_name = parts[0].trim();
            let ty = parse_type(parts[1].trim())?;
            
            let val = ctx.get_or_create_value(param_name, func, ty, ValueDef::Param(block));
            func.layout.blocks[block].params.push(val);
        }
    }
    
    // 解析块中的指令
    for line in iter {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        parse_instruction(line, func, ctx, block)?;
    }
    
    Ok(())
}

fn parse_instruction(
    line: &str,
    func: &mut Function,
    ctx: &mut ParseContext,
    block: Block,
) -> Result<()> {
    // 解析结果值 (如果有)
    let (result_names, rest) = if let Some(eq_pos) = line.find('=') {
        let result_part = line[..eq_pos].trim();
        let rest = line[eq_pos + 1..].trim();
        
        let names = if result_part.starts_with('(') {
            // 多返回值: (v0, v1)
            result_part[1..result_part.len() - 1]
                .split(',')
                .map(|s| s.trim().to_string())
                .collect()
        } else {
            alloc::vec![result_part.to_string()]
        };
        
        (names, rest)
    } else {
        (Vec::new(), line)
    };
    
    // 解析指令
    let inst_data = parse_inst_data(rest, func, ctx)?;
    
    let inst = func.dfg.instructions.push(inst_data.clone());
    func.layout.append_inst(block, inst);
    
    // 创建结果值
    if !result_names.is_empty() {
        let result_types = infer_result_types(&func.dfg, &inst_data);
        
        let values: Vec<Value> = result_names
            .iter()
            .enumerate()
            .map(|(i, name)| {
                let ty = result_types.get(i).copied().unwrap_or(Type::VOID);
                ctx.get_or_create_value(name, func, ty, ValueDef::Inst(inst))
            })
            .collect();
        
        let list = func.dfg.make_value_list(&values);
        func.dfg.inst_results[inst] = list;
    }
    
    Ok(())
}

fn parse_inst_data(
    s: &str,
    func: &mut Function,
    ctx: &mut ParseContext,
) -> Result<InstructionData> {
    let parts: Vec<&str> = s.split_whitespace().collect();
    if parts.is_empty() {
        return Err(ParseError("Empty instruction".to_string()).into());
    }
    
    let opcode_full = parts[0];
    let args = &parts[1..];
    
    // 解析 opcode 和类型
    let (opcode_str, _ty_str) = if let Some(dot_pos) = opcode_full.find('.') {
        (&opcode_full[..dot_pos], Some(&opcode_full[dot_pos + 1..]))
    } else {
        (opcode_full, None)
    };
    
    match opcode_str {
        "iconst" => {
            let value: i64 = args[0].parse()
                .map_err(|_| ParseError(format!("Invalid iconst value: {}", args[0])))?;
            Ok(InstructionData::Iconst { value: value as u64 })
        }
        "fconst" => {
            let value: f64 = args[0].parse()
                .map_err(|_| ParseError(format!("Invalid fconst value: {}", args[0])))?;
            Ok(InstructionData::Fconst { value: value.to_bits() })
        }
        "bconst" => {
            let value = args[0] == "true";
            Ok(InstructionData::Bconst { value })
        }
        "iadd" | "isub" | "imul" => {
            let lhs = parse_value_ref(args[0], func, ctx)?;
            let rhs = parse_value_ref(args[1].trim_end_matches(','), func, ctx)?;
            let opcode = match opcode_str {
                "iadd" => Opcode::IAdd,
                "isub" => Opcode::ISub,
                "imul" => Opcode::IMul,
                _ => unreachable!(),
            };
            Ok(InstructionData::Binary { opcode, args: [lhs, rhs] })
        }
        "icmp" => {
            let cond = parse_intcc(args[0].trim_end_matches(','))?;
            let lhs = parse_value_ref(args[1].trim_end_matches(','), func, ctx)?;
            let rhs = parse_value_ref(args[2], func, ctx)?;
            Ok(InstructionData::IntCompare { kind: cond, args: [lhs, rhs] })
        }
        "load" => {
            let ptr = parse_value_ref(args[0], func, ctx)?;
            // 简化处理，假设偏移量为0
            Ok(InstructionData::Load { ptr, offset: 0, flags: MemFlags::new() })
        }
        "store" => {
            let value = parse_value_ref(args[0].trim_end_matches(','), func, ctx)?;
            let ptr = parse_value_ref(args[1], func, ctx)?;
            Ok(InstructionData::Store { ptr, value, offset: 0, flags: MemFlags::new() })
        }
        "jump" => {
            parse_jump(args, func, ctx)
        }
        "br" => {
            parse_br(args, func, ctx)
        }
        "return" => {
            let values: Result<Vec<Value>> = args.iter()
                .map(|&arg| parse_value_ref(arg.trim_end_matches(','), func, ctx))
                .collect();
            let values = values?;
            let list = func.dfg.make_value_list(&values);
            Ok(InstructionData::Return { values: list })
        }
        "call" => {
            // 简化处理
            let _func_name = args[0];
            let func_id = FuncId(0);
            let call_args: Result<Vec<Value>> = args[1..].iter()
                .map(|&arg| parse_value_ref(arg.trim_end_matches(',').trim_end_matches(')'), func, ctx))
                .collect();
            let call_args = call_args?;
            let list = func.dfg.make_value_list(&call_args);
            Ok(InstructionData::Call { func_id, args: list })
        }
        _ => Err(ParseError(format!("Unknown opcode: {}", opcode_str)).into()),
    }
}

fn parse_value_ref(s: &str, _func: &mut Function, _ctx: &mut ParseContext) -> Result<Value> {
    // 移除可能的逗号
    let s = s.trim_end_matches(',');
    
    if let Some(idx_str) = s.strip_prefix('v') {
        let idx: u32 = idx_str.parse()
            .map_err(|_| ParseError(format!("Invalid value reference: {}", s)))?;
        Ok(Value(idx))
    } else {
        Err(ParseError(format!("Invalid value reference: {}", s)).into())
    }
}

fn parse_intcc(s: &str) -> Result<IntCC> {
    format::parse_intcc_str(s)
        .ok_or_else(|| ParseError(format!("Invalid intcc: {}", s)).into())
}

fn parse_jump(
    args: &[&str],
    func: &mut Function,
    ctx: &mut ParseContext,
) -> Result<InstructionData> {
    // 格式: jump blockN(args...)
    let target_str = args[0];
    let (block_name, args_str) = if let Some(lparen) = target_str.find('(') {
        (&target_str[..lparen], &target_str[lparen..])
    } else {
        (target_str, "()")
    };
    
    let block = *ctx.block_map.get(block_name).unwrap_or(&Block(0));
    
    let args_str = args_str.trim_start_matches('(').trim_end_matches(')');
    let args_list = if args_str.is_empty() {
        func.dfg.make_value_list(&[])
    } else {
        let values: Result<Vec<Value>> = args_str.split(',')
            .map(|s| parse_value_ref(s.trim(), func, ctx))
            .collect();
        func.dfg.make_value_list(&values?)
    };
    
    let dest = func.dfg.block_calls.push(BlockCallData { block, args: args_list });
    Ok(InstructionData::Jump { dest })
}

fn parse_br(
    args: &[&str],
    func: &mut Function,
    ctx: &mut ParseContext,
) -> Result<InstructionData> {
    // 格式: br cond, then_block(args...), else_block(args...)
    let cond = parse_value_ref(args[0].trim_end_matches(','), func, ctx)?;
    
    let then_str = args[1].trim_end_matches(',');
    let else_str = args[2];
    
    let then_dest = parse_block_call(then_str, func, ctx)?;
    let else_dest = parse_block_call(else_str, func, ctx)?;
    
    Ok(InstructionData::Br { condition: cond, then_dest, else_dest })
}

fn parse_block_call(s: &str, func: &mut Function, ctx: &mut ParseContext) -> Result<BlockCall> {
    let (block_name, args_str) = if let Some(lparen) = s.find('(') {
        (&s[..lparen], &s[lparen..])
    } else {
        (s, "()")
    };
    
    let block = *ctx.block_map.get(block_name).unwrap_or(&Block(0));
    
    let args_str = args_str.trim_start_matches('(').trim_end_matches(')');
    let args_list = if args_str.is_empty() {
        func.dfg.make_value_list(&[])
    } else {
        let values: Result<Vec<Value>> = args_str.split(',')
            .map(|s| parse_value_ref(s.trim(), func, ctx))
            .collect();
        func.dfg.make_value_list(&values?)
    };
    
    Ok(func.dfg.block_calls.push(BlockCallData { block, args: args_list }))
}

fn infer_result_types(dfg: &crate::DataFlowGraph, data: &InstructionData) -> Vec<Type> {
    use crate::inst::InstructionData as ID;
    match data {
        ID::Unary { arg, .. } => alloc::vec![dfg.value_type(*arg)],
        ID::Binary { opcode, args, .. } => {
            match opcode {
                Opcode::IAddWithOverflow | Opcode::ISubWithOverflow | Opcode::IMulWithOverflow => {
                    alloc::vec![dfg.value_type(args[0]), Type::BOOL]
                }
                _ => alloc::vec![dfg.value_type(args[0])],
            }
        }
        ID::IntCompare { .. } | ID::FloatCompare { .. } => {
            alloc::vec![Type::BOOL]
        }
        _ => alloc::vec![Type::VOID],
    }
}

// =============================================================================
// IR Format Documentation and Utilities
// =============================================================================

/// IR 格式化 Trait
/// 
/// 实现此 trait 的类型可以被格式化为 IR 文本或从 IR 文本解析。
/// 这确保了 Printer 和 Parser 使用一致的格式。
pub trait IRFormat: Sized {
    /// 将值格式化为 IR 文本
    fn fmt_ir(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result;
    
    /// 从 IR 文本解析值
    fn parse_ir(s: &str) -> Result<Self>;
}

/// IR 格式化辅助函数
pub fn fmt_type(ty: Type) -> alloc::string::String {
    format::type_to_string(ty)
}

/// IR 解析辅助函数
pub fn parse_type_ir(s: &str) -> Result<Type> {
    format::parse_type_str(s).ok_or_else(|| ParseError(format!("Invalid type: {}", s)).into())
}

/// IR 文本格式规范
///
/// # 函数格式
/// ```text
/// [local|export|import] function name(param1, param2, ...) -> ret_type
/// ss0: size N          // 栈槽定义 (可选)
/// block0(v0: i32, v1: i32):  // 基本块定义
///     v2 = iadd.i32 v0, v1   // 指令
///     return v2
/// ```
///
/// # 类型格式
/// - 标量: i8, i16, i32, i64, f32, f64, bool, ptr
/// - 向量: i32<4>, f64<scalable 2>
/// - 谓词: mask<4>, mask<scalable 8>
pub mod format {
    use super::*;
    
    // ==================== 类型格式 ====================
    
    /// 获取类型的文本表示
    pub fn type_to_string(ty: Type) -> String {
        ty.to_string()
    }
    
    /// 解析类型字符串
    pub fn parse_type_str(s: &str) -> Option<Type> {
        parse_type(s).ok()
    }
    
    // ==================== 条件码格式 ====================
    
    /// 获取 IntCC 的文本表示
    pub fn intcc_to_string(cc: IntCC) -> &'static str {
        match cc {
            IntCC::Eq => "eq",
            IntCC::Ne => "ne",
            IntCC::LtS => "lts",
            IntCC::LtU => "ltu",
            IntCC::GtS => "gts",
            IntCC::GtU => "gtu",
            IntCC::LeS => "les",
            IntCC::LeU => "leu",
            IntCC::GeS => "ges",
            IntCC::GeU => "geu",
        }
    }
    
    /// 从字符串解析 IntCC
    pub fn parse_intcc_str(s: &str) -> Option<IntCC> {
        match s {
            "eq" => Some(IntCC::Eq),
            "ne" => Some(IntCC::Ne),
            "lts" => Some(IntCC::LtS),
            "ltu" => Some(IntCC::LtU),
            "gts" => Some(IntCC::GtS),
            "gtu" => Some(IntCC::GtU),
            "les" => Some(IntCC::LeS),
            "leu" => Some(IntCC::LeU),
            "ges" => Some(IntCC::GeS),
            "geu" => Some(IntCC::GeU),
            _ => None,
        }
    }
    
    /// 获取 FloatCC 的文本表示
    pub fn floatcc_to_string(cc: FloatCC) -> &'static str {
        match cc {
            FloatCC::Eq => "eq",
            FloatCC::Ne => "ne",
            FloatCC::Lt => "lt",
            FloatCC::Gt => "gt",
            FloatCC::Le => "le",
            FloatCC::Ge => "ge",
        }
    }
    
    /// 从字符串解析 FloatCC
    pub fn parse_floatcc_str(s: &str) -> Option<FloatCC> {
        match s {
            "eq" => Some(FloatCC::Eq),
            "ne" => Some(FloatCC::Ne),
            "lt" => Some(FloatCC::Lt),
            "gt" => Some(FloatCC::Gt),
            "le" => Some(FloatCC::Le),
            "ge" => Some(FloatCC::Ge),
            _ => None,
        }
    }
    
    // ==================== Opcode 格式 ====================
    
    /// 将 Opcode 转换为字符串表示
    /// 
    /// 注意：这与 printer.rs 中的格式保持一致
    pub fn opcode_to_string(op: Opcode) -> &'static str {
        match op {
            Opcode::Iconst => "iconst",
            Opcode::Fconst => "fconst",
            Opcode::Bconst => "bconst",
            Opcode::Vconst => "vconst",
            Opcode::IAdd => "iadd",
            Opcode::ISub => "isub",
            Opcode::IMul => "imul",
            Opcode::INeg => "ineg",
            Opcode::IAddSat => "iadd-sat",
            Opcode::ISubSat => "isub-sat",
            Opcode::IAddWithOverflow => "iadd-with-overflow",
            Opcode::ISubWithOverflow => "isub-with-overflow",
            Opcode::IMulWithOverflow => "imul-with-overflow",
            Opcode::IDivS => "idiv-s",
            Opcode::IDivU => "idiv-u",
            Opcode::IRemS => "irem-s",
            Opcode::IRemU => "irem-u",
            Opcode::FAdd => "fadd",
            Opcode::FSub => "fsub",
            Opcode::FMul => "fmul",
            Opcode::FNeg => "fneg",
            Opcode::FDiv => "fdiv",
            Opcode::FMin => "fmin",
            Opcode::FMax => "fmax",
            Opcode::FCopysign => "fcopysign",
            Opcode::FAbs => "fabs",
            Opcode::FSqrt => "fsqrt",
            Opcode::FCeil => "fceil",
            Opcode::FFloor => "ffloor",
            Opcode::FTrunc => "ftrunc",
            Opcode::FNearest => "fnearest",
            Opcode::IAnd => "iand",
            Opcode::IOr => "ior",
            Opcode::IXor => "ixor",
            Opcode::IShl => "ishl",
            Opcode::IShrS => "ishr-s",
            Opcode::IShrU => "ishr-u",
            Opcode::IRotl => "irotl",
            Opcode::IRotr => "irotr",
            Opcode::IClz => "iclz",
            Opcode::ICtz => "ictz",
            Opcode::IPopcnt => "ipopcnt",
            Opcode::IEqz => "ieqz",
            Opcode::Icmp => "icmp",
            Opcode::Fcmp => "fcmp",
            Opcode::ExtendS => "extends",
            Opcode::ExtendU => "extendu",
            Opcode::Wrap => "wrap",
            Opcode::FloatToIntSatS => "float-to-int-sat-s",
            Opcode::FloatToIntSatU => "float-to-int-sat-u",
            Opcode::FloatToIntS => "float-to-int-s",
            Opcode::FloatToIntU => "float-to-int-u",
            Opcode::IntToFloatS => "int-to-float-s",
            Opcode::IntToFloatU => "int-to-float-u",
            Opcode::FloatPromote => "float-promote",
            Opcode::FloatDemote => "float-demote",
            Opcode::Reinterpret => "reinterpret",
            Opcode::IntToPtr => "inttoptr",
            Opcode::PtrToInt => "ptrtoint",
            Opcode::Load => "load",
            Opcode::Store => "store",
            Opcode::StackLoad => "stack-load",
            Opcode::StackStore => "stack-store",
            Opcode::StackAddr => "stack-addr",
            Opcode::PtrOffset => "ptr-offset",
            Opcode::PtrIndex => "ptr-index",
            Opcode::Call => "call",
            Opcode::CallIndirect => "call-indirect",
            Opcode::CallIntrinsic => "call-intrinsic",
            Opcode::Jump => "jump",
            Opcode::Br => "br",
            Opcode::BrTable => "br-table",
            Opcode::Return => "return",
            Opcode::Select => "select",
            Opcode::Unreachable => "unreachable",
            Opcode::Nop => "nop",
            Opcode::Splat => "splat",
            Opcode::Shuffle => "shuffle",
            Opcode::InsertElement => "insertelement",
            Opcode::ExtractElement => "extractelement",
            Opcode::ReduceSum => "reduce-sum",
            Opcode::ReduceAdd => "reduce-add",
            Opcode::ReduceMin => "reduce-min",
            Opcode::ReduceMax => "reduce-max",
            Opcode::ReduceAnd => "reduce-and",
            Opcode::ReduceOr => "reduce-or",
            Opcode::ReduceXor => "reduce-xor",
            Opcode::LoadStride => "load-stride",
            Opcode::StoreStride => "store-stride",
            Opcode::Gather => "gather",
            Opcode::Scatter => "scatter",
            Opcode::SetVL => "setvl",
        }
    }
    
    /// 从字符串解析 Opcode
    pub fn parse_opcode_str(s: &str) -> Option<Opcode> {
        match s {
            "iconst" => Some(Opcode::Iconst),
            "fconst" => Some(Opcode::Fconst),
            "bconst" => Some(Opcode::Bconst),
            "vconst" => Some(Opcode::Vconst),
            "iadd" => Some(Opcode::IAdd),
            "isub" => Some(Opcode::ISub),
            "imul" => Some(Opcode::IMul),
            "ineg" => Some(Opcode::INeg),
            "iadd-sat" => Some(Opcode::IAddSat),
            "isub-sat" => Some(Opcode::ISubSat),
            "iadd-with-overflow" => Some(Opcode::IAddWithOverflow),
            "isub-with-overflow" => Some(Opcode::ISubWithOverflow),
            "imul-with-overflow" => Some(Opcode::IMulWithOverflow),
            "idiv-s" => Some(Opcode::IDivS),
            "idiv-u" => Some(Opcode::IDivU),
            "irem-s" => Some(Opcode::IRemS),
            "irem-u" => Some(Opcode::IRemU),
            "fadd" => Some(Opcode::FAdd),
            "fsub" => Some(Opcode::FSub),
            "fmul" => Some(Opcode::FMul),
            "fneg" => Some(Opcode::FNeg),
            "fdiv" => Some(Opcode::FDiv),
            "fmin" => Some(Opcode::FMin),
            "fmax" => Some(Opcode::FMax),
            "fcopysign" => Some(Opcode::FCopysign),
            "fabs" => Some(Opcode::FAbs),
            "fsqrt" => Some(Opcode::FSqrt),
            "fceil" => Some(Opcode::FCeil),
            "ffloor" => Some(Opcode::FFloor),
            "ftrunc" => Some(Opcode::FTrunc),
            "fnearest" => Some(Opcode::FNearest),
            "iand" => Some(Opcode::IAnd),
            "ior" => Some(Opcode::IOr),
            "ixor" => Some(Opcode::IXor),
            "ishl" => Some(Opcode::IShl),
            "ishr-s" => Some(Opcode::IShrS),
            "ishr-u" => Some(Opcode::IShrU),
            "irotl" => Some(Opcode::IRotl),
            "irotr" => Some(Opcode::IRotr),
            "iclz" => Some(Opcode::IClz),
            "ictz" => Some(Opcode::ICtz),
            "ipopcnt" => Some(Opcode::IPopcnt),
            "ieqz" => Some(Opcode::IEqz),
            "icmp" => Some(Opcode::Icmp),
            "fcmp" => Some(Opcode::Fcmp),
            "extends" => Some(Opcode::ExtendS),
            "extendu" => Some(Opcode::ExtendU),
            "wrap" => Some(Opcode::Wrap),
            "float-to-int-sat-s" => Some(Opcode::FloatToIntSatS),
            "float-to-int-sat-u" => Some(Opcode::FloatToIntSatU),
            "float-to-int-s" => Some(Opcode::FloatToIntS),
            "float-to-int-u" => Some(Opcode::FloatToIntU),
            "int-to-float-s" => Some(Opcode::IntToFloatS),
            "int-to-float-u" => Some(Opcode::IntToFloatU),
            "float-promote" => Some(Opcode::FloatPromote),
            "float-demote" => Some(Opcode::FloatDemote),
            "reinterpret" => Some(Opcode::Reinterpret),
            "inttoptr" => Some(Opcode::IntToPtr),
            "ptrtoint" => Some(Opcode::PtrToInt),
            "load" => Some(Opcode::Load),
            "store" => Some(Opcode::Store),
            "stack-load" => Some(Opcode::StackLoad),
            "stack-store" => Some(Opcode::StackStore),
            "stack-addr" => Some(Opcode::StackAddr),
            "ptr-offset" => Some(Opcode::PtrOffset),
            "ptr-index" => Some(Opcode::PtrIndex),
            "call" => Some(Opcode::Call),
            "call-indirect" => Some(Opcode::CallIndirect),
            "call-intrinsic" => Some(Opcode::CallIntrinsic),
            "jump" => Some(Opcode::Jump),
            "br" => Some(Opcode::Br),
            "br-table" => Some(Opcode::BrTable),
            "return" => Some(Opcode::Return),
            "select" => Some(Opcode::Select),
            "unreachable" => Some(Opcode::Unreachable),
            "nop" => Some(Opcode::Nop),
            "splat" => Some(Opcode::Splat),
            "shuffle" => Some(Opcode::Shuffle),
            "insertelement" => Some(Opcode::InsertElement),
            "extractelement" => Some(Opcode::ExtractElement),
            "reduce-sum" => Some(Opcode::ReduceSum),
            "reduce-add" => Some(Opcode::ReduceAdd),
            "reduce-min" => Some(Opcode::ReduceMin),
            "reduce-max" => Some(Opcode::ReduceMax),
            "reduce-and" => Some(Opcode::ReduceAnd),
            "reduce-or" => Some(Opcode::ReduceOr),
            "reduce-xor" => Some(Opcode::ReduceXor),
            "load-stride" => Some(Opcode::LoadStride),
            "store-stride" => Some(Opcode::StoreStride),
            "gather" => Some(Opcode::Gather),
            "scatter" => Some(Opcode::Scatter),
            "setvl" => Some(Opcode::SetVL),
            _ => None,
        }
    }
    
    // ==================== Linkage 格式 ====================
    
    /// 将 Linkage 转换为字符串
    pub fn linkage_to_string(linkage: Linkage) -> &'static str {
        match linkage {
            Linkage::Local => "local",
            Linkage::Export => "export",
            Linkage::Import => "import",
        }
    }
    
    /// 从字符串解析 Linkage
    pub fn parse_linkage_str(s: &str) -> Option<Linkage> {
        match s {
            "local" => Some(Linkage::Local),
            "export" => Some(Linkage::Export),
            "import" => Some(Linkage::Import),
            _ => None,
        }
    }
    
    // ==================== 指令格式常量 ====================
    
    /// 指令格式模板常量
    pub mod templates {
        /// 二元运算指令格式: "opcode.type lhs, rhs"
        pub const BINARY: &str = "{opcode}.{type} {lhs}, {rhs}";
        
        /// 一元运算指令格式: "opcode.type arg"
        pub const UNARY: &str = "{opcode}.{type} {arg}";
        
        /// 常量指令格式: "iconst.type value"
        pub const CONST: &str = "{opcode}.{type} {value}";
        
        /// 加载指令格式: "load.type ptr + offset"
        pub const LOAD: &str = "load.{type} {ptr} + {offset}";
        
        /// 存储指令格式: "store value, ptr + offset"
        pub const STORE: &str = "store {value}, {ptr} + {offset}";
        
        /// 跳转指令格式: "jump block(args...)"
        pub const JUMP: &str = "jump {block}({args})";
        
        /// 条件分支指令格式: "br cond, then_block(args...), else_block(args...)"
        pub const BR: &str = "br {cond}, {then_block}({then_args}), {else_block}({else_args})";
        
        /// 返回指令格式: "return values..."
        pub const RETURN: &str = "return {values}";
        
        /// 调用指令格式: "call func(args...)"
        pub const CALL: &str = "call {func}({args})";
        
        /// 整数比较指令格式: "icmp.type cond, lhs, rhs"
        pub const ICMP: &str = "icmp.{type} {cond}, {lhs}, {rhs}";
        
        /// 浮点比较指令格式: "fcmp.type cond, lhs, rhs"
        pub const FCMP: &str = "fcmp.{type} {cond}, {lhs}, {rhs}";
    }
    
    // ==================== 格式验证 ====================
    
    /// 验证标识符是否为有效的值名 (vN)
    pub fn is_valid_value_name(s: &str) -> bool {
        s.starts_with('v') && s[1..].parse::<u32>().is_ok()
    }
    
    /// 验证标识符是否为有效的块名 (blockN)
    pub fn is_valid_block_name(s: &str) -> bool {
        s.starts_with("block") && s[5..].parse::<u32>().is_ok()
    }
    
    /// 验证标识符是否为有效的栈槽名 (ssN)
    pub fn is_valid_stackslot_name(s: &str) -> bool {
        s.starts_with("ss") && s[2..].parse::<u32>().is_ok()
    }
    
    /// 提取值名中的索引
    pub fn extract_value_index(s: &str) -> Option<u32> {
        s.strip_prefix('v')?.parse().ok()
    }
    
    /// 提取块名中的索引
    pub fn extract_block_index(s: &str) -> Option<u32> {
        s.strip_prefix("block")?.parse().ok()
    }
    
    /// 提取栈槽名中的索引
    pub fn extract_stackslot_index(s: &str) -> Option<u32> {
        s.strip_prefix("ss")?.parse().ok()
    }
    
    /// 构造值名
    pub fn make_value_name(index: u32) -> alloc::string::String {
        alloc::format!("v{}", index)
    }
    
    /// 构造块名
    pub fn make_block_name(index: u32) -> alloc::string::String {
        alloc::format!("block{}", index)
    }
    
    /// 构造栈槽名
    pub fn make_stackslot_name(index: u32) -> alloc::string::String {
        alloc::format!("ss{}", index)
    }
}
