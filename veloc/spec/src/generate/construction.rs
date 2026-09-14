//! Shared call emission; storage adapters supply already resolved arguments.

pub(crate) struct Write {
    pub callee: String,
    pub args: Vec<String>,
}

impl Write {
    pub fn emit(&self) -> String {
        if self.args.is_empty() {
            return format!("{}()", self.callee);
        }
        let names = (0..self.args.len())
            .map(|i| format!("_arg{i}"))
            .collect::<Vec<_>>()
            .join(", ");
        // Prepare fields before consuming the writer: pool insertion can borrow
        // it. One tuple binding prevents locals from shadowing later arguments.
        format!(
            "{{ let ({names},) = ({},); {}({names}) }}",
            self.args.join(", "),
            self.callee
        )
    }
}

/// Preserve a borrowed tail when no prefix needs to be assembled.
pub(crate) fn slice(ty: &str, items: &[String], tail: Option<String>) -> String {
    let values = items.join(", ");
    match tail {
        None => format!("&[{values}]"),
        Some(tail) if items.is_empty() => tail,
        Some(tail) => format!(
            "&{{ let (_head, _tail) = ([{values}], {tail}); let mut _items = smallvec::SmallVec::<[{ty}; 4]>::from_slice(&_head); _items.extend_from_slice(_tail); _items }}"
        ),
    }
}
