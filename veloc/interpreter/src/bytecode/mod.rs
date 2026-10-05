mod compile;
mod inst;
pub mod printer;

pub use compile::CompiledFunction;
pub(crate) use compile::{ControlSite, DataSection, JumpTarget};

pub(crate) use compile::compile_function;
pub(crate) use inst::{CodeWord, Opcode, OpcodeHandlers, Reg, decode};

mod stack;
pub(crate) use stack::stack_layout;

/// Compile and format a module without linking imports or executing it.
pub fn format_module(module: &veloc_mir::Module) -> crate::Result<alloc::string::String> {
    use core::fmt::Write;
    let mut output = alloc::string::String::new();
    for (id, function) in module.functions() {
        if let Some(body) = function.body {
            stack_layout(body).map_err(crate::Error::Message)?;
            let compiled = compile_function(veloc_mir::ModuleId::from_u32(0), id, body)?;
            writeln!(output, "; function {id:?}\n{compiled}").expect("writing to String");
        }
    }
    Ok(output)
}
