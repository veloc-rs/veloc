//! Shared input parsers for the runtime command and differential runner.
use crate::{Val, engine::OptLevel};
use anyhow::{Context, Result, anyhow, bail};
use std::path::Path;

pub fn parse_opt_level(value: &str) -> Result<OptLevel, String> {
    match value {
        "0" => Ok(OptLevel::None),
        "1" => Ok(OptLevel::Default),
        _ => Err("expected optimization level 0 or 1".into()),
    }
}

pub fn read_wasm(path: &Path) -> Result<Vec<u8>> {
    if path.extension().and_then(|extension| extension.to_str()) != Some("wat") {
        return std::fs::read(path).with_context(|| format!("failed to read {}", path.display()));
    }
    #[cfg(feature = "wat")]
    {
        wat::parse_file(path).with_context(|| format!("failed to parse {}", path.display()))
    }
    #[cfg(not(feature = "wat"))]
    {
        bail!("WAT input requires rebuilding veloc-wasm with `--features wat`")
    }
}

pub fn parse_val(arg: &str) -> Result<Val> {
    let (ty, value) = arg
        .split_once(':')
        .ok_or_else(|| anyhow!("argument `{arg}` must have the form <type>:<value>"))?;
    match ty {
        "i32" => {
            Ok(Val::I32(value.parse().with_context(|| {
                format!("invalid i32 literal `{value}`")
            })?))
        }
        "i64" => {
            Ok(Val::I64(value.parse().with_context(|| {
                format!("invalid i64 literal `{value}`")
            })?))
        }
        "f32" => {
            Ok(Val::F32(value.parse().with_context(|| {
                format!("invalid f32 literal `{value}`")
            })?))
        }
        "f64" => {
            Ok(Val::F64(value.parse().with_context(|| {
                format!("invalid f64 literal `{value}`")
            })?))
        }
        _ => bail!("unsupported argument type `{ty}`; expected i32, i64, f32 or f64"),
    }
}
