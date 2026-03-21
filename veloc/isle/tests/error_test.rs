use veloc_isle;

#[test]
fn test_error_reporting() {
    let input = "(select-rule (match (G_ADD $x $y @n)) (covers (@n)))";
    let res = veloc_isle::compile(input, "x86_64");
    if let Err(e) = res {
        println!("Expected error message:\n{}", e);
    } else {
        panic!("Should have failed to compile");
    }
}
