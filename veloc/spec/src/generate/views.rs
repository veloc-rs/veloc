//! Storage-independent borrowed view declarations.
//!
//! Layout adapters supply field types and optional opcode subsets. This layer
//! owns Rust declarations and lifetime propagation, not physical decoding.
use std::fmt::Write;

pub(crate) struct Field {
    pub name: String,
    pub ty: String,
}

pub(crate) struct Variant {
    pub name: String,
    pub fields: Vec<Field>,
    pub borrowed: bool,
    pub opcodes: Vec<String>,
}

pub(crate) enum Representation {
    /// Fields live directly in enum variants.
    Inline,
    /// Named records are also usable independently of the enclosing enum.
    Records,
}

pub(crate) struct View {
    pub name: String,
    pub variants: Vec<Variant>,
    pub representation: Representation,
}

impl View {
    pub fn lifetime(&self) -> &'static str {
        lifetime(self.variants.iter().any(|v| v.borrowed))
    }

    pub fn generate(&self) -> String {
        let mut out = String::new();
        writeln!(
            out,
            "#[derive(Debug, Clone, Copy)] pub enum {}{} {{",
            self.name,
            self.lifetime()
        )
        .unwrap();
        for variant in &self.variants {
            match self.representation {
                Representation::Inline => {
                    if variant.fields.is_empty() {
                        writeln!(out, "{},", variant.name).unwrap();
                    } else {
                        writeln!(out, "{} {{", variant.name).unwrap();
                        fields(&mut out, &variant.fields, "");
                        out.push_str("},\n");
                    }
                }
                Representation::Records => {
                    writeln!(
                        out,
                        "{}({}Inst{}),",
                        variant.name,
                        variant.name,
                        lifetime(variant.borrowed)
                    )
                    .unwrap();
                }
            }
        }
        out.push_str("}\n");
        for variant in &self.variants {
            if !variant.opcodes.is_empty() {
                writeln!(out, "#[allow(non_camel_case_types)] #[derive(Debug, Clone, Copy, PartialEq, Eq)] pub enum {}Opcode {{", variant.name).unwrap();
                for opcode in &variant.opcodes {
                    writeln!(out, "{opcode},").unwrap();
                }
                out.push_str("}\n");
            }
            if matches!(self.representation, Representation::Records) {
                writeln!(
                    out,
                    "#[derive(Debug, Clone, Copy)] pub struct {}Inst{} {{",
                    variant.name,
                    lifetime(variant.borrowed)
                )
                .unwrap();
                fields(&mut out, &variant.fields, "pub ");
                out.push_str("}\n");
            }
        }
        out
    }
}

fn lifetime(borrowed: bool) -> &'static str {
    if borrowed { "<'a>" } else { "" }
}

fn fields(out: &mut String, fields: &[Field], visibility: &str) {
    for field in fields {
        writeln!(out, "{visibility}{}: {},", field.name, field.ty).unwrap();
    }
}
