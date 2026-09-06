use std::{
    fs,
    path::Path,
    time::{SystemTime, UNIX_EPOCH},
};

#[test]
fn generated_files_are_formatted_together_and_invalid_syntax_is_reported() {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("veloc-rustfmt-{}-{unique}", std::process::id()));
    fs::create_dir(&dir).unwrap();
    let files = [dir.join("functions.rs"), dir.join("types.rs")];
    // Generated files can declare modules without owning their source files.
    fs::write(
        &files[0],
        "mod external;\nfn choose(x:bool)->u32{match x{true=>1,false=>2}}\n",
    )
    .unwrap();
    fs::write(&files[1], "struct Example{value:u32}\n").unwrap();
    let config = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../rustfmt.toml");
    veloc_opgen::format_rust(&files, &config).unwrap();
    let formatted = files
        .each_ref()
        .map(|path| fs::read_to_string(path).unwrap());
    assert!(formatted[0].contains("    match x {\n        true => 1,"));
    assert!(formatted[1].contains("struct Example {\n    value: u32,\n}"));
    veloc_opgen::format_rust(&files, &config).unwrap();
    assert_eq!(
        formatted,
        files
            .each_ref()
            .map(|path| fs::read_to_string(path).unwrap())
    );

    fs::write(&files[0], "fn broken( {").unwrap();
    let error = veloc_opgen::format_rust(&files, &config).unwrap_err();
    assert!(error.to_string().contains("rustfmt failed"));
    for path in files {
        fs::remove_file(path).unwrap();
    }
    fs::remove_dir(dir).unwrap();
}
