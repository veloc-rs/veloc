//! C Language to IR Code Generator
//!
//! This module converts C language AST into veloc_mir intermediate representation.

use crate::ast::*;
use crate::error::{Error, Result};
use veloc_mir::{CallConv, Linkage, Module, ModuleBuilder, Signature, Type};

/// Code generator context
pub struct CodeGenContext {
    /// Module builder
    module_builder: Option<ModuleBuilder>,
}

impl CodeGenContext {
    /// Create a new code generation context
    pub fn new() -> Self {
        CodeGenContext {
            module_builder: Some(ModuleBuilder::new()),
        }
    }

    /// Generate IR from a translation unit
    pub fn generate(&mut self, tu: &TranslationUnit) -> Result<Module> {
        for decl in &tu.declarations {
            self.generate_external_decl(decl)?;
        }
        Ok(self.module_builder.take().unwrap().build())
    }

    /// Generate code for external declaration
    fn generate_external_decl(&mut self, decl: &ExternalDeclaration) -> Result<()> {
        match decl {
            ExternalDeclaration::FunctionDefinition(func) => {
                self.generate_function(func)?;
            }
            ExternalDeclaration::Declaration(_) => {}
        }
        Ok(())
    }

    /// Generate code for a function
    fn generate_function(&mut self, func: &FunctionDefinition) -> Result<()> {
        let name = func.declarator.name().to_string();
        let sig = self.signature_from_declarator(&func.declarator, &func.specifiers)?;

        let builder = self.module_builder.as_mut().unwrap();
        let sig_id =
            builder.make_signature(sig.params.to_vec(), sig.returns.to_vec(), CallConv::SystemV);
        let func_id = builder.declare_function(name, sig_id, Linkage::Export);

        let return_type = sig.returns.first().copied();
        let mut func_builder = builder.builder(func_id);
        func_builder.init_entry_block();

        // Generate a zero value for the temporary function body.
        if let Some(ty) = return_type {
            let zero = match ty {
                Type::I8 | Type::I16 | Type::I32 | Type::I64 => func_builder.ins().iconst(0, ty),
                Type::F32 => func_builder.ins().f32const(0.0),
                Type::F64 => func_builder.ins().f64const(0.0),
                _ => return Err(Error::semantic("unsupported return type", 0, 0)),
            };
            func_builder.ins().ret(&[zero]);
        } else {
            func_builder.ins().ret(&[]);
        }
        func_builder.seal_all_blocks();

        Ok(())
    }

    /// Build signature from function declarator
    fn signature_from_declarator(
        &self,
        _declarator: &Declarator,
        specifiers: &[DeclarationSpecifier],
    ) -> Result<Signature> {
        let ret_type = type_from_specifiers(specifiers)?;
        let params = Vec::new();
        Ok(Signature::new(
            params,
            ret_type.into_iter().collect(),
            CallConv::SystemV,
        ))
    }
}

/// Get Type from C type specifiers
fn type_from_specifiers(specifiers: &[DeclarationSpecifier]) -> Result<Option<Type>> {
    for spec in specifiers {
        if let DeclarationSpecifier::TypeSpecifier(ts) = spec {
            match ts {
                TypeSpecifier::Void => return Ok(None),
                TypeSpecifier::Char => return Ok(Some(Type::I8)),
                TypeSpecifier::Short => return Ok(Some(Type::I16)),
                TypeSpecifier::Long => return Ok(Some(Type::I64)),
                TypeSpecifier::Float => return Ok(Some(Type::F32)),
                TypeSpecifier::Double => return Ok(Some(Type::F64)),
                _ => {}
            }
        }
    }
    Ok(Some(Type::I32))
}

impl Default for CodeGenContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Compile C source code to IR module
pub fn compile_to_ir(source: &str) -> Result<Module> {
    let tu = crate::parse(source)?;
    let mut ctx = CodeGenContext::new();
    ctx.generate(&tu)
}
