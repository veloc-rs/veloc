//! Checked payload constructors shared by generic and target code generation.
use super::operands::Shape;
use crate::{
    Error, model,
    syntax::{Kind, Node},
};
use std::collections::BTreeSet;

#[derive(Debug, Clone)]
pub(crate) struct Payloads {
    pub rust: String,
    pub host: String,
    layouts: Vec<Layout>,
}
#[derive(Debug, Clone)]
struct Layout {
    fields: Vec<(String, Shape)>,
    value: Node,
}

impl Payloads {
    pub fn compile(
        source: &str,
        node: Node,
        rust: String,
        host: String,
        variants: &BTreeSet<String>,
    ) -> Result<Self, Error> {
        let mut layouts = Vec::new();
        for node in model::list(source, node)? {
            let error = |message| Error::at(source, node.offset, message);
            let Kind::Record(mut fields) = node.kind else {
                return Err(error("expected a field layout record"));
            };
            let signature = fields
                .remove("fields")
                .ok_or_else(|| error("missing layout fields"))?;
            let value = fields
                .remove("value")
                .ok_or_else(|| error("missing layout value"))?;
            if !fields.is_empty() {
                return Err(error("unknown field layout property"));
            }
            let mut signature_fields = Vec::new();
            for field in model::list(source, signature)? {
                let (name, shape) = match field.kind {
                    Kind::Name(name) => (name, Shape::One),
                    Kind::Call(name, args) if name == "sequence" && args.len() == 1 => {
                        (model::name(source, args[0].clone())?, Shape::Sequence)
                    }
                    _ => return Err(error("expected an attribute variant or sequence(Variant)")),
                };
                if !variants.contains(&name) {
                    return Err(error("unknown attribute variant"));
                }
                signature_fields.push((name, shape));
            }
            if layouts
                .iter()
                .any(|layout: &Layout| layout.fields == signature_fields)
            {
                return Err(error("duplicate field layout"));
            }
            let arguments: Vec<_> = (0..signature_fields.len())
                .map(|i| format!("field{i}"))
                .collect();
            render(&value, &arguments, "writer", &rust)
                .map_err(|e| Error::at(source, value.offset, e))?;
            layouts.push(Layout {
                fields: signature_fields,
                value,
            });
        }
        Ok(Self {
            rust,
            host,
            layouts,
        })
    }

    pub fn construct(
        &self,
        fields: &[(&str, String, Shape)],
        writer: &str,
        rust: &str,
    ) -> Result<String, String> {
        let layout = self
            .layouts
            .iter()
            .find(|layout| {
                layout.fields.len() == fields.len()
                    && layout.fields.iter().zip(fields).all(
                        |((name, shape), (other, _, other_shape))| {
                            name == other && shape == other_shape
                        },
                    )
            })
            .ok_or_else(|| {
                format!(
                    "no payload layout for {:?}",
                    fields
                        .iter()
                        .map(|(name, _, shape)| (name, shape))
                        .collect::<Vec<_>>()
                )
            })?;
        render(
            &layout.value,
            &fields
                .iter()
                .map(|(_, value, _)| value.clone())
                .collect::<Vec<_>>(),
            writer,
            rust,
        )
    }
}

fn render(node: &Node, arguments: &[String], writer: &str, rust: &str) -> Result<String, String> {
    let path = |name: &str| -> Result<String, String> {
        if let Some(variant) = name.strip_prefix("payload::") {
            return Ok(format!("{rust}::{variant}"));
        }
        if let Some(method) = name.strip_prefix("host::") {
            return Ok(format!("{writer}.{method}"));
        }
        if matches!(name, "None" | "Some") {
            return Ok(name.to_owned());
        }
        Err(format!("invalid payload constructor path `{name}`"))
    };
    Ok(match &node.kind {
        Kind::Name(name) if name.starts_with("field") => {
            let index: usize = name[5..].parse().map_err(|_| "expected field index")?;
            arguments
                .get(index)
                .ok_or("payload field index out of bounds")?
                .clone()
        }
        Kind::Name(name) => path(name)?,
        Kind::Call(name, args) => format!(
            "{}({})",
            path(name)?,
            args.iter()
                .map(|arg| render(arg, arguments, writer, rust))
                .collect::<Result<Vec<_>, _>>()?
                .join(", ")
        ),
        Kind::Object(name, fields) => format!(
            "{} {{ {} }}",
            path(name)?,
            fields
                .iter()
                .map(|(name, value)| Ok(format!(
                    "{name}: {}",
                    render(value, arguments, writer, rust)?
                )))
                .collect::<Result<Vec<_>, String>>()?
                .join(", ")
        ),
        Kind::List(values) => format!(
            "[{}]",
            values
                .iter()
                .map(|value| render(value, arguments, writer, rust))
                .collect::<Result<Vec<_>, _>>()?
                .join(", ")
        ),
        _ => return Err("expected a payload constructor, field or list".into()),
    })
}
