use veloc_interpreter::bytecode::format_module;
use veloc_mir::ModuleParser;

#[test]
fn direct_stack_bytecode_requires_an_in_bounds_nonvolatile_access() {
    let module = ModuleParser::new()
        .parse(
            "local function ranges(index: i32) -> void
             stack-slot p, size=8, align=4
             block0():
               store i32(1), p, offset=4
               last: i32 = load p, offset=4
               store i32(1), p, offset=6
               past: i32 = load p, offset=8
               before: ptr = ptr-offset p, -1
               low: i32 = load before, offset=0
               store i32(1), p, offset=4294967295
               store.volatile i32(1), p, offset=0
               element: ptr = ptr-index p, index, scale=4, offset=0
               dynamic: i32 = load element, offset=0
               return",
        )
        .unwrap();
    let text = format_module(&module).unwrap();
    let memory_ops: Vec<_> = text
        .lines()
        .filter_map(|line| line.split_whitespace().nth(1))
        .filter(|op| op.ends_with("load") || op.ends_with("store"))
        .collect();
    assert_eq!(
        memory_ops,
        [
            "stackstore",
            "stackload",
            "i32store",
            "i32load",
            "i32load",
            "i32store",
            "i32store",
            "i32load"
        ]
    );
}
