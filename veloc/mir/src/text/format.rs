use crate::types::{ScalarType, Type};

pub fn parse_type(s: &str) -> Option<Type> {
    match s {
        "i8" => Some(Type::I8),
        "i16" => Some(Type::I16),
        "i32" => Some(Type::I32),
        "i64" => Some(Type::I64),
        "f32" => Some(Type::F32),
        "f64" => Some(Type::F64),
        "bool" => Some(Type::BOOL),
        "ptr" => Some(Type::PTR),
        _ if s.contains('<') => {
            let lt_pos = s.find('<')?;
            let base = &s[..lt_pos];
            let rest = &s[lt_pos + 1..s.len().checked_sub(1)?];
            let scalar = if base == "mask" {
                ScalarType::Bool
            } else {
                parse_scalar_type(base)?
            };
            let (is_scalable, lanes_str) = if let Some(lanes) = rest.strip_prefix("scalable ") {
                (true, lanes)
            } else {
                (false, rest)
            };
            Type::new_vector(scalar, lanes_str.parse().ok()?, is_scalable)
        }
        _ => None,
    }
}

pub fn parse_scalar_type(s: &str) -> Option<ScalarType> {
    match s {
        "i8" => Some(ScalarType::I8),
        "i16" => Some(ScalarType::I16),
        "i32" => Some(ScalarType::I32),
        "i64" => Some(ScalarType::I64),
        "f32" => Some(ScalarType::F32),
        "f64" => Some(ScalarType::F64),
        "bool" => Some(ScalarType::Bool),
        "ptr" => Some(ScalarType::Ptr),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::parse_type;
    use crate::{ScalarType, Type};
    use alloc::string::ToString;

    #[test]
    fn vector_types_round_trip() {
        for ty in [
            Type::new_vector(ScalarType::I32, 4, false).unwrap(),
            Type::new_vector(ScalarType::F32, 4, true).unwrap(),
            Type::new_mask(8, false).unwrap(),
        ] {
            let text = ty.to_string();
            assert_eq!(parse_type(&text), Some(ty));
        }
    }

    #[test]
    fn invalid_vector_shapes_are_rejected() {
        assert_eq!(parse_type("i32<3>"), None);
        assert_eq!(parse_type("ptr<4>"), None);
    }
}
