//! Shared build-time constant pool handling for rule bytecode backends.
pub(crate) fn intern<T: PartialEq>(items: &mut Vec<T>, item: T) -> usize {
    if let Some(id) = items.iter().position(|old| *old == item) {
        id
    } else {
        let id = items.len();
        items.push(item);
        id
    }
}
