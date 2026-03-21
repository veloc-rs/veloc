use std::env;
use std::fs;
use std::path::Path;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: veloc-isle <input_file> <output_file> [arch]");
        std::process::exit(1);
    }

    let input_path = &args[1];
    let output_path = &args[2];
    let arch = if args.len() > 3 { &args[3] } else { "x86_64" };

    let input = fs::read_to_string(input_path).expect("Failed to read input file");
    match veloc_isle::compiler::compile(&input, arch) {
        Ok(output) => {
            fs::write(output_path, output).expect("Failed to write output file");
            println!("Successfully compiled {} to {}", input_path, output_path);
        }
        Err(e) => {
            eprintln!("Compilation error: {}", e);
            std::process::exit(1);
        }
    }
}
