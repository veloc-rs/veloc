//! Execute emitted ELF objects, including ABI calls and forced spills.
#![cfg(all(feature = "std", target_arch = "x86_64", target_os = "linux"))]
use std::{
    fs,
    path::PathBuf,
    process::Command,
    sync::atomic::{AtomicUsize, Ordering},
};
use veloc_codegen::{CodegenOptions, CodegenPipeline, TargetConfig, create_target_machine};
use veloc_mir::ModuleParser;

struct Workspace(PathBuf);

#[test]
fn branch_layout_preserves_boundaries_and_symbolic_fixups() {
    use veloc_codegen::FixupTarget as Target;
    use veloc_encoder::x86_64::*;
    use veloc_lir::SymbolId;
    use veloc_mir::Block;

    let descriptor = Branch {
        map: OpcodeMap::Primary,
        near: 0xe9,
        short: 0xeb,
    };
    let short = encode_branch(descriptor, true).unwrap();
    let long = encode_branch(descriptor, false).unwrap();
    let nop = encode(
        Legacy {
            prefix: Prefix::None,
            map: OpcodeMap::Primary,
            opcode: 0x90,
            wide: false,
        },
        Form::None,
        Immediate::None,
    )
    .unwrap();
    let call = encode(
        Legacy {
            prefix: Prefix::None,
            map: OpcodeMap::Primary,
            opcode: 0xe8,
            wide: false,
        },
        Form::None,
        Immediate::Relative(7),
    )
    .unwrap();
    let label = Block::from_u32(0);
    for padding in [0, 125, 126, 127, 128, 256] {
        for backwards in [false, true] {
            let mut emitter = veloc_codegen::Emitter::new();
            if backwards {
                emitter.mark_block(label);
            }
            if !backwards {
                emitter.branch(label, &short, &long).unwrap();
            }
            for _ in 0..padding {
                emitter.instruction(&nop, None).unwrap();
            }
            if backwards {
                emitter.branch(label, &short, &long).unwrap();
            }
            if !backwards {
                emitter.mark_block(label);
            }
            emitter
                .instruction(&call, Some(Target::Symbol(SymbolId::from_u32(0))))
                .unwrap();
            let code = emitter.finish().unwrap();
            let compact = if backwards {
                padding <= 126
            } else {
                padding <= 127
            };
            let branch_len = if compact { 2 } else { 5 };
            let start = if backwards { padding } else { 0 };
            assert_eq!(code.data[start], if compact { 0xeb } else { 0xe9 });
            let displacement = if compact {
                code.data[start + 1] as i8 as i64
            } else {
                i32::from_le_bytes(code.data[start + 1..start + 5].try_into().unwrap()) as i64
            };
            assert_eq!(
                displacement,
                if backwards {
                    -(padding as i64 + branch_len as i64)
                } else {
                    padding as i64
                }
            );
            assert_eq!(code.data.len(), padding + branch_len + 5);
            assert_eq!(code.relocations.len(), 1);
            assert_eq!(
                code.relocations[0].offset,
                (padding + branch_len + 1) as u64
            );
            assert_eq!(code.relocations[0].addend, 3);
        }
    }
    // Widening the inner jump must force a second layout iteration for the outer one.
    let mut emitter = veloc_codegen::Emitter::new();
    let far = Block::from_u32(1);
    emitter.branch(label, &short, &long).unwrap();
    emitter.branch(far, &short, &long).unwrap();
    for _ in 0..125 {
        emitter.instruction(&nop, None).unwrap();
    }
    emitter.mark_block(label);
    for _ in 0..128 {
        emitter.instruction(&nop, None).unwrap();
    }
    emitter.mark_block(far);
    let code = emitter.finish().unwrap();
    assert_eq!(code.data[0], 0xe9);
    assert_eq!(code.data[5], 0xe9);
    assert_eq!(i32::from_le_bytes(code.data[1..5].try_into().unwrap()), 130);
    assert_eq!(
        i32::from_le_bytes(code.data[6..10].try_into().unwrap()),
        253
    );
}

#[test]
fn full_unsigned_memory_offsets_do_not_sign_extend_disp32() {
    let mut source = String::new();
    let mut harness = String::from("#include <stdint.h>\n#include <assert.h>\n");
    for offset in [0x7fff_ffffu32, 0x8000_0000, 0xffff_ffff] {
        source += &format!(
            "
export function offset_{offset}(ptr, i64) -> i64
block0(v0: ptr, v1: i64):
  store.volatile v1, v0, offset={offset}
  v2: i64 = load.volatile v0, offset={offset}
  return v2
"
        );
        harness += &format!("extern uint64_t offset_{offset}(void *, uint64_t);\n");
    }
    harness += "int main(void) { uint64_t value=0;\n";
    for offset in [0x7fff_ffffu32, 0x8000_0000, 0xffff_ffff] {
        harness += &format!(
            "for(uint64_t n=0;n<100;n++) {{
            void *base=(void *)((uintptr_t)&value-UINT64_C({offset}));
            assert(offset_{offset}(base,n)==n);
            assert(value==n);
        }}\n"
        );
    }
    harness += "}";
    run(&source, &harness);
}

#[test]
fn narrow_memory_and_negative_pointer_offsets_execute() {
    let mut source = String::new();
    let mut harness = String::from("#include <stdint.h>\n#include <assert.h>\n");
    for width in [8, 16] {
        source += &format!(
            "
export function copy{width}(ptr, i{width}) -> i{width}
block0(v0: ptr, v1: i{width}):
  store.volatile v1, v0, offset=1
  v2: i{width} = load.volatile v0, offset=1
  return v2
"
        );
        harness += &format!("extern uint{width}_t copy{width}(void *, uint{width}_t);\n");
    }
    source += "
export function previous(ptr, i64) -> i64
block0(v0: ptr, v1: i64):
  v2: ptr = ptr-offset v0, -8
  store v1, v2, offset=0
  v3: i64 = load v2, offset=0
  return v3

export function pointer_slot(ptr, ptr) -> ptr
block0(v0: ptr, v1: ptr):
  store v1, v0, offset=0
  v2: ptr = load v0, offset=0
  return v2
";
    harness += "extern uint64_t previous(void *, uint64_t);
extern void *pointer_slot(void **, void *);
int main(void) {
  uint8_t bytes[4]={0xaa,0,0,0xbb};
  for(uint32_t n=0;n<65536;n++) {
    bytes[2]=0xcc;
    assert(copy8(bytes,(uint8_t)n)==(uint8_t)n);
    assert(bytes[0]==0xaa && bytes[1]==(uint8_t)n && bytes[2]==0xcc && bytes[3]==0xbb);
    assert(copy16(bytes,(uint16_t)n)==(uint16_t)n);
    assert(bytes[0]==0xaa && bytes[1]==(uint8_t)n && bytes[2]==(uint8_t)(n>>8) && bytes[3]==0xbb);
  }
  uint64_t words[2]={0,123};
  assert(previous(&words[1],42)==42);
  assert(words[0]==42 && words[1]==123);
  void *slot=0;
  assert(pointer_slot(&slot,words)==words && slot==words);
}";
    run(&source, &harness);
}

#[test]
fn extension_encodings_match_system_assembler_for_every_register_pair() {
    use veloc_codegen::target::x86_64::isle::*;
    use veloc_lir::{MachineFunction, MachineOpcode, Writable};
    let regs = [
        REG_RAX, REG_RCX, REG_RDX, REG_RBX, REG_RSP, REG_RBP, REG_RSI, REG_RDI, REG_R8, REG_R9,
        REG_R10, REG_R11, REG_R12, REG_R13, REG_R14, REG_R15,
    ];
    let bytes = [
        "al", "cl", "dl", "bl", "spl", "bpl", "sil", "dil", "r8b", "r9b", "r10b", "r11b", "r12b",
        "r13b", "r14b", "r15b",
    ];
    let words = [
        "ax", "cx", "dx", "bx", "sp", "bp", "si", "di", "r8w", "r9w", "r10w", "r11w", "r12w",
        "r13w", "r14w", "r15w",
    ];
    let dwords = [
        "eax", "ecx", "edx", "ebx", "esp", "ebp", "esi", "edi", "r8d", "r9d", "r10d", "r11d",
        "r12d", "r13d", "r14d", "r15d",
    ];
    let qwords = [
        "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12",
        "r13", "r14", "r15",
    ];
    let mut f = MachineFunction::new("encoding".into());
    let mut emitter = veloc_codegen::Emitter::new();
    let mut assembly = String::from(".text\n");
    let mut cases = Vec::new();
    for (opcode, mnemonic, sources, destinations) in [
        (TargetInst::X86Movzx8to32, "movzbl", &bytes, &dwords),
        (TargetInst::X86Movsx8to32, "movsbl", &bytes, &dwords),
        (TargetInst::X86Movzx16to32, "movzwl", &words, &dwords),
        (TargetInst::X86Movsx16to32, "movswl", &words, &dwords),
        (TargetInst::X86Movsx8to64, "movsbq", &bytes, &qwords),
        (TargetInst::X86Movsx16to64, "movswq", &words, &qwords),
    ] {
        for src in 0..16 {
            for dst in 0..16 {
                let line = format!("{mnemonic} %{}, %{}\n", sources[src], destinations[dst]);
                cases.push((emitter.position(), line.clone()));
                assembly.push_str(&line);
                let inst = f.writer().unary(
                    MachineOpcode::Target(opcode.as_u32()),
                    Writable(regs[dst]),
                    regs[src],
                );
                opcode.emit(&mut emitter, &f.inst(inst), &f).unwrap();
            }
        }
    }
    let dir = Workspace::new();
    fs::write(dir.0.join("reference.s"), assembly).unwrap();
    for (program, args) in [
        ("cc", vec!["-c", "reference.s", "-o", "reference.o"]),
        (
            "objcopy",
            vec![
                "-O",
                "binary",
                "--only-section=.text",
                "reference.o",
                "reference.bin",
            ],
        ),
    ] {
        let output = Command::new(program)
            .args(args)
            .current_dir(&dir.0)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let reference = fs::read(dir.0.join("reference.bin")).unwrap();
    let encoded = emitter.finish().unwrap().data;
    if encoded != reference {
        let offset = encoded
            .iter()
            .zip(&reference)
            .position(|(a, b)| a != b)
            .unwrap_or(encoded.len().min(reference.len()));
        let case = cases
            .iter()
            .rev()
            .find(|(start, _)| *start <= offset)
            .unwrap();
        panic!(
            "encoding differs at byte {offset}, instruction {}: got {:?}, expected {:?}",
            case.1.trim(),
            encoded.get(offset),
            reference.get(offset)
        );
    }
}

#[test]
#[ignore = "manual whole-backend benchmark; run with --release --ignored --nocapture"]
fn backend_benchmark() {
    let module = ModuleParser::new()
        .parse(include_str!("../examples/sum.mir"))
        .unwrap();
    module.validate().unwrap();
    let target = create_target_machine(TargetConfig::default()).unwrap();
    for optimize in [false, true] {
        let pipeline = CodegenPipeline::with_options(
            &*target,
            CodegenOptions {
                optimize,
                ..Default::default()
            },
        );
        let start = std::time::Instant::now();
        let mut bytes = 0;
        for _ in 0..200 {
            let code = pipeline.compile_functions(&module).unwrap();
            bytes = code.values().map(Vec::len).sum::<usize>();
            std::hint::black_box(code);
        }
        eprintln!(
            "sum: optimize={optimize}, 200 compilations={:?}, machine_code={bytes} bytes",
            start.elapsed()
        );
    }
}

#[test]
fn indirect_calls_and_escaping_stack_addresses() {
    run(
        r#"
export function indirect(ptr, i64) -> i64
block0(v0: ptr, v1: i64):
  v2: i64 = call-indirect v0(v1) : (i64) -> i64
  v3: i64 = iadd v2, v1
  return v3
import function fill(ptr, i64) -> void
export function address(i64) -> i64

block0(v0: i64):
  ss0: ptr = alloca size=16, align=8
  v1: ptr = ptr-offset ss0, 0
  call fill(v1, v0) : (ptr, i64) -> void
  v2: i64 = load ss0, offset=8
  return v2
"#,
        r#"
#include <stdint.h>
#include <assert.h>
extern uint64_t indirect(uint64_t (*)(uint64_t), uint64_t), address(uint64_t);
static uint64_t triple(uint64_t n) { return n*3; }
void fill(uint64_t *p, uint64_t n) { p[1]=n+42; }
int main(void) { for(uint64_t n=0;n<100;n++) { assert(indirect(triple,n)==n*4); assert(address(n)==n+42); } }
"#,
    );
}
impl Workspace {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "veloc-native-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Workspace {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn scalar_division_remainders_and_rotations_execute() {
    let mut source = String::new();
    let mut harness = String::from("#include <stdint.h>\n#include <assert.h>\n");
    let mut checks = String::new();
    for bits in [32, 64] {
        for (op, signed, expr) in [
            ("idiv-s", true, "a / b"),
            ("idiv-u", false, "a / b"),
            ("irem-s", true, "a % b"),
            ("irem-u", false, "a % b"),
            (
                "irotl",
                false,
                "(a << (b % WIDTH)) | (a >> ((WIDTH - b % WIDTH) % WIDTH))",
            ),
            (
                "irotr",
                false,
                "(a >> (b % WIDTH)) | (a << ((WIDTH - b % WIDTH) % WIDTH))",
            ),
        ] {
            let name = format!("{}_{bits}", op.replace('-', "_"));
            let ty = format!("{}int{bits}_t", if signed { "" } else { "u" });
            source += &format!(
                "\nexport function {name}(i{bits}, i{bits}) -> i{bits}\nblock0(v0: i{bits}, v1: i{bits}):\n  v2: i{bits} = {op} v0, v1\n  return v2\n"
            );
            harness += &format!("extern {ty} {name}({ty}, {ty});\n");
            checks += &format!(
                "{{ {ty} a=({ty})state, b=({ty})((state>>17)%123 + 1); assert({name}(a,b)==({})); }}\n",
                expr.replace("WIDTH", &bits.to_string())
            );
        }
    }
    harness += &format!(
        "int main(void) {{ uint64_t state=123; for(int i=0;i<1000;i++) {{ state=state*6364136223846793005ULL+1; {checks} }} }}"
    );
    run(&source, &harness);
}

#[test]
fn scalar_float_conversions_and_sign_bits_execute() {
    let mut source = String::new();
    let mut harness = String::from(
        "#include <stdint.h>\n#include <assert.h>\n#include <string.h>\n#include <stdio.h>\n",
    );
    let mut checks = String::new();
    for bits in [32, 64] {
        let cfloat = if bits == 32 { "float" } else { "double" };
        for integer in [32, 64] {
            for (sign, prefix) in [("s", ""), ("u", "u")] {
                let ctype = format!("{prefix}int{integer}_t");
                let name = format!("convert_{sign}_{integer}_{bits}");
                source += &format!(
                    "\nexport function {name}(i{integer}) -> f{bits}\nblock0(v0: i{integer}):\n  v1: f{bits} = int-to-float-{sign} v0\n  return v1\n"
                );
                harness += &format!("extern {cfloat} {name}({ctype});\n");
                checks += &format!("assert({name}(({ctype})state)==({cfloat})({ctype})state);\n");
                let name = format!("back_{sign}_{integer}_{bits}");
                source += &format!(
                    "\nexport function {name}(f{bits}) -> i{integer}\nblock0(v0: f{bits}):\n  v1: i{integer} = float-to-int-{sign} v0\n  return v1\n"
                );
                harness += &format!("extern {ctype} {name}({cfloat});\n");
                // Keep rounding away from overflow while covering both halves
                // of the unsigned range and positive/negative signed values.
                let discarded = if integer == 64 {
                    "0xffffffffffULL"
                } else {
                    "0xffff"
                };
                checks += &format!(
                    "{{ {ctype} n=({ctype})state; n &= ~(({ctype}){discarded}); {cfloat} x=({cfloat})n; {ctype} got={name}(x); if(got!=({ctype})x) {{ fprintf(stderr, \"{name}: x=%a got=%llx expected=%llx\\n\", (double)x, (unsigned long long)got, (unsigned long long)({ctype})x); return 1; }} }}\n"
                );
            }
        }
        for (op, mask) in [
            (
                "fneg",
                if bits == 32 {
                    "0x80000000ULL"
                } else {
                    "0x8000000000000000ULL"
                },
            ),
            (
                "fabs",
                if bits == 32 {
                    "0x7fffffffULL"
                } else {
                    "0x7fffffffffffffffULL"
                },
            ),
        ] {
            let name = format!("bits_{op}_{bits}");
            source += &format!(
                "\nexport function {name}(i{bits}) -> i{bits}\nblock0(v0: i{bits}):\n  v1: f{bits} = reinterpret v0\n  v2: f{bits} = {op} v1\n  v3: i{bits} = reinterpret v2\n  return v3\n"
            );
            harness += &format!("extern uint{bits}_t {name}(uint{bits}_t);\n");
            checks += &format!(
                "assert({name}((uint{bits}_t)state)==(((uint{bits}_t)state) {} {mask}));\n",
                if op == "fabs" { "&" } else { "^" }
            );
        }
    }
    harness += &format!(
        "int main(void) {{ uint64_t state=0; for(int i=0;i<10000;i++) {{ {checks} state=state*6364136223846793005ULL+1; }} }}"
    );
    run(&source, &harness);
}

#[test]
fn floating_comparisons_and_shared_select_inputs_execute() {
    let mut source = String::new();
    let mut harness = String::from("#include <stdint.h>\n#include <assert.h>\n#include <math.h>\n");
    let mut checks = String::new();
    for bits in [32, 64] {
        let ctype = if bits == 32 { "float" } else { "double" };
        for (cc, op) in [
            ("eq", "=="),
            ("ne", "!="),
            ("lt", "<"),
            ("le", "<="),
            ("gt", ">"),
            ("ge", ">="),
        ] {
            let name = format!("cmp_{cc}_{bits}");
            source += &format!(
                "\nexport function {name}(f{bits}, f{bits}) -> i32\nblock0(v0: f{bits}, v1: f{bits}):\n  v2: bool = fcmp {cc} v0, v1\n  v3: i32 = extendu v2\n  return v3\n"
            );
            harness += &format!("extern int {name}({ctype}, {ctype});\n");
            checks += &format!("assert({name}(a,b) == (a {op} b));\n");
        }
        source += &format!(
            "\nexport function select_{bits}(i{bits}, i{bits}, i{bits}) -> i{bits}\nblock0(v0: i{bits}, v1: i{bits}, v2: i{bits}):\n  v3: bool = icmp gtu v0, v1\n  v4: i{bits} = select v3, v1, v2\n  v5: i{bits} = iadd v4, v1\n  v6: i{bits} = iadd v5, v2\n  v7: i{bits} = extendu v3\n  v8: i{bits} = iadd v6, v7\n  return v8\n"
        );
        harness += &format!(
            "extern uint{bits}_t select_{bits}(uint{bits}_t,uint{bits}_t,uint{bits}_t);\n"
        );
    }
    harness += &format!(
        "int main(void) {{ double values[]={{-INFINITY,-3.5,-0.0,0.0,1.25,INFINITY,NAN}}; for(int i=0;i<7;i++) for(int j=0;j<7;j++) {{ double a=values[i], b=values[j]; {checks} }} for(uint64_t i=0;i<100;i++) {{ assert(select_32(i,42,9)==(i>42?42:9)+42+9+(i>42)); assert(select_64(i,42,9)==(i>42?42:9)+42+9+(i>42)); }} }}"
    );
    run(&source, &harness);
}

#[test]
fn loop_parameters_preserve_parallel_assignment() {
    run(
        r#"
export function rotate(i64, i64, i64, i64) -> i64
block0(v0: i64, v1: i64, v2: i64, v3: i64):
  jump block1(v0, v1, v2, v3)
block1(v4: i64, v5: i64, v6: i64, v7: i64):
  v8: i64 = iconst 0
  v9: bool = icmp eq v7, v8
  br v9, block2(), block3()
block3():
  v10: i64 = iconst 1
  v11: i64 = isub v7, v10
  jump block1(v5, v6, v4, v11)
block2():
  v12: i64 = iconst 100
  v13: i64 = imul v4, v12
  v14: i64 = iconst 10
  v15: i64 = imul v5, v14
  v16: i64 = iadd v13, v15
  v17: i64 = iadd v16, v6
  return v17

export function chain(i64, i64, i64, i64) -> i64
block0(v0: i64, v1: i64, v2: i64, v3: i64):
  jump block1(v0, v1, v2, v3)
block1(v4: i64, v5: i64, v6: i64, v7: i64):
  v8: i64 = iconst 0
  v9: bool = icmp eq v7, v8
  br v9, block2(), block3()
block3():
  v10: i64 = iconst 1
  v11: i64 = isub v7, v10
  v12: i64 = iadd v6, v10
  jump block1(v5, v6, v12, v11)
block2():
  v13: i64 = iconst 100
  v14: i64 = imul v4, v13
  v15: i64 = iconst 10
  v16: i64 = imul v5, v15
  v17: i64 = iadd v14, v16
  v18: i64 = iadd v17, v6
  return v18
"#,
        r#"
#include <stdint.h>
#include <assert.h>
extern uint64_t rotate(uint64_t,uint64_t,uint64_t,uint64_t);
extern uint64_t chain(uint64_t,uint64_t,uint64_t,uint64_t);
int main(void) {
  const uint64_t rotations[]={123,231,312};
  for(uint64_t n=0;n<100;n++) {
    assert(rotate(1,2,3,n)==rotations[n%3]);
    assert(chain(1,2,3,n)==(n+1)*100+(n+2)*10+n+3);
  }
}
"#,
    );
}

#[test]
fn ssa_edges_execute_spilled_cycles_and_duplicate_targets() {
    const N: usize = 40;
    let mut source = String::from("export function rotate_spilled(i64) -> i64\nblock0(v0: i64):\n");
    for i in 1..=N {
        source += &format!("  v{i}: i64 = iconst {i}\n");
    }
    let initial = (1..=N)
        .map(|i| format!("v{i}"))
        .collect::<Vec<_>>()
        .join(", ");
    let params = (N + 1..=2 * N + 1)
        .map(|i| format!("v{i}: i64"))
        .collect::<Vec<_>>()
        .join(", ");
    source += &format!(
        "  jump block1({initial}, v0)\nblock1({params}):\n  v82: i64 = iconst 0\n  v83: bool = icmp eq v81, v82\n  br v83, block2(), block3()\nblock3():\n  v84: i64 = iconst 1\n  v85: i64 = isub v81, v84\n"
    );
    let rotation = (N + 2..=2 * N)
        .map(|i| format!("v{i}"))
        .chain([format!("v{}", N + 1)])
        .collect::<Vec<_>>()
        .join(", ");
    source += &format!("  jump block1({rotation}, v85)\nblock2():\n  v86: i64 = iconst 0\n");
    let mut sum = 86;
    for i in 0..N {
        let weight = 87 + i * 3;
        let product = weight + 1;
        let next = weight + 2;
        source += &format!(
            "  v{weight}: i64 = iconst {}\n  v{product}: i64 = imul v{}, v{weight}\n  v{next}: i64 = iadd v{sum}, v{product}\n",
            i + 1,
            N + 1 + i
        );
        sum = next;
    }
    source += &format!("  return v{sum}\n");
    source += r#"
export function entry_loop(i64, i64) -> i64
block0(v0: i64, v1: i64):
  v2: i64 = iconst 0
  v3: bool = icmp eq v0, v2
  br v3, block1(), block2()
block1():
  return v1
block2():
  v4: i64 = iconst 1
  v5: i64 = isub v0, v4
  v6: i64 = iadd v1, v4
  jump block0(v5, v6)

export function same_target(i64, i64, i64) -> i64
block0(v0: i64, v1: i64, v2: i64):
  v3: i64 = iconst 0
  v4: bool = icmp eq v0, v3
  br v4, block1(v1, v2), block1(v2, v1)
block1(v5: i64, v6: i64):
  v7: i64 = isub v5, v6
  return v7
"#;
    run(
        &source,
        r#"
#include <stdint.h>
#include <assert.h>
extern int64_t rotate_spilled(int64_t);
extern int64_t same_target(int64_t,int64_t,int64_t);
extern int64_t entry_loop(int64_t,int64_t);
int main(void) {
  for(int n=0;n<83;n++) {
    int64_t expected=0;
    for(int i=0;i<40;i++) expected+=(i+1)*((i+n)%40+1);
    assert(rotate_spilled(n)==expected);
    assert(same_target(n,42,9)==(n==0?33:-33));
    assert(entry_loop(n,17)==n+17);
  }
}
"#,
    );
}

fn run(source: &str, harness: &str) {
    let module = ModuleParser::new().parse(source).unwrap();
    module.validate().unwrap();
    let target = create_target_machine(TargetConfig::default()).unwrap();
    for optimize in [false, true] {
        let pipeline = CodegenPipeline::with_options(
            &*target,
            CodegenOptions {
                optimize,
                ..Default::default()
            },
        );
        let object = pipeline.compile_object(&module).unwrap();
        let dir = Workspace::new();
        fs::write(dir.0.join("code.o"), object).unwrap();
        fs::write(dir.0.join("main.c"), harness).unwrap();
        let link = Command::new("cc")
            .current_dir(&dir.0)
            .args(["-O2", "-no-pie", "main.c", "code.o", "-o", "run"])
            .output()
            .expect("native tests require a C compiler");
        assert!(
            link.status.success(),
            "{}",
            String::from_utf8_lossy(&link.stderr)
        );
        let mut child = Command::new(dir.0.join("run"))
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while child.try_wait().unwrap().is_none() {
            if std::time::Instant::now() > deadline {
                child.kill().unwrap();
                child.wait().unwrap();
                panic!("native execution timed out (optimize={optimize})");
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let result = child.wait_with_output().unwrap();
        assert!(
            result.status.success(),
            "native execution (optimize={optimize}): {:?}\n{}",
            result.status,
            String::from_utf8_lossy(&result.stderr)
        );
    }
}

#[test]
fn incoming_and_outgoing_stack_arguments() {
    run(
        r#"
import function weighted(i64, i64, i64, i64, i64, i64, i64, i64) -> i64
export function forward(i64, i64, i64, i64, i64, i64, i64, i64) -> i64
block0(v0: i64, v1: i64, v2: i64, v3: i64, v4: i64, v5: i64, v6: i64, v7: i64):
  v8: i64 = call weighted(v7, v6, v5, v4, v3, v2, v1, v0) : (i64, i64, i64, i64, i64, i64, i64, i64) -> i64
  v9: i64 = iadd v8, v0
  return v9
"#,
        r#"
#include <stdint.h>
#include <assert.h>
extern uint64_t forward(uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t);
uint64_t weighted(uint64_t a,uint64_t b,uint64_t c,uint64_t d,uint64_t e,uint64_t f,uint64_t g,uint64_t h) {
 return a+2*b+3*c+4*d+5*e+6*f+7*g+8*h;
}
int main(void) { for(uint64_t n=0;n<100;n++) assert(forward(n,2,3,4,5,6,7,8)==weighted(8,7,6,5,4,3,2,n)+n); }
"#,
    );
}

#[test]
fn loops_branches_and_stack_memory() {
    run(
        r#"
export function sum(i64) -> i64
block0(v0: i64):
  v1: i64 = iconst 0
  v2: i64 = iconst 1
  jump block1(v0, v1)
block1(v3: i64, v4: i64):
  v5: bool = icmp eq v3, v1
  br v5, block3(v4), block2()
block2():
  v6: i64 = iadd v3, v4
  v7: i64 = isub v3, v2
  jump block1(v7, v6)
block3(v8: i64):
  return v8

export function memory(i64) -> i64

block0(v0: i64):
  ss0: ptr = alloca size=16, align=8
  store v0, ss0, offset=8
  v1: i64 = load ss0, offset=8
  return v1
"#,
        r#"
#include <stdint.h>
#include <assert.h>
extern uint64_t sum(uint64_t), memory(uint64_t);
int main(void) { for(uint64_t n=0;n<200;n++) { assert(sum(n)==n*(n+1)/2); assert(memory(n)==n); } }
"#,
    );
}

#[test]
fn conditional_edges_do_not_depend_on_block_layout() {
    run(
        r#"
export function choose(i64) -> i64
block0(v0: i64):
  v1: i64 = iconst 0
  v2: i64 = iconst 11
  v3: i64 = iconst 22
  v4: bool = icmp eq v0, v1
  br v4, block2(v2), block3(v3)
block1():
  unreachable
block2(v5: i64):
  return v5
block3(v6: i64):
  return v6
"#,
        r#"
#include <stdint.h>
#include <assert.h>
extern uint64_t choose(uint64_t);
int main(void) { for(uint64_t n=0;n<200;n++) assert(choose(n)==(n==0 ? 11 : 22)); }
"#,
    );
}

#[test]
fn volatile_memory_accesses_preserve_width_offsets_and_order() {
    run(
        r#"
export function memory_order(ptr, i32) -> i32
block0(v0: ptr, v1: i32):
  store.volatile.align4 v1, v0, offset=4
  v2: i32 = load.volatile.align4 v0, offset=4
  v3: i32 = iconst 1
  v4: i32 = iadd v2, v3
  store.volatile.align4 v4, v0, offset=4
  return v2
"#,
        r#"
#include <stdint.h>
#include <assert.h>
extern uint32_t memory_order(uint32_t *, uint32_t);
int main(void) {
  uint32_t words[3] = {0x12345678, 0, 0x87654321};
  for(uint32_t n=0;n<1000;n++) {
    assert(memory_order(words,n)==n);
    assert(words[1]==n+1);
    assert(words[0]==0x12345678 && words[2]==0x87654321);
  }
}
"#,
    );
}

#[test]
fn calls_and_high_register_pressure() {
    let mut source = String::from(
        "import function smash(i64) -> i64\nexport function pressure(i64) -> i64\nblock0(v0: i64):\n",
    );
    for i in 0..24 {
        source += &format!(
            "  v{}: i64 = iconst {}\n  v{}: i64 = imul v0, v{}\n",
            2 * i + 1,
            i + 3,
            2 * i + 2,
            2 * i + 1
        );
    }
    source += "  v49: i64 = call smash(v0) : (i64) -> i64\n";
    for i in 0..24 {
        source += &format!("  v{}: i64 = iadd v{}, v{}\n", 50 + i, 49 + i, 2 * i + 2);
    }
    source += "  return v73\n";
    run(
        &source,
        r#"
#include <stdint.h>
#include <assert.h>
extern uint64_t pressure(uint64_t);
__attribute__((noinline)) uint64_t smash(uint64_t x) {
  uint64_t y=x+17;
  __asm__ volatile("xor %%rax,%%rax; xor %%rcx,%%rcx; xor %%rdx,%%rdx; xor %%rsi,%%rsi; xor %%rdi,%%rdi; xor %%r8,%%r8; xor %%r9,%%r9; xor %%r10,%%r10; xor %%r11,%%r11"
    : : : "rax","rcx","rdx","rsi","rdi","r8","r9","r10","r11","cc");
  return y;
}
int main(void) { for(uint64_t n=0;n<200;n++) assert(pressure(n)==n*348+n+17); }
"#,
    );
}

#[test]
fn floating_values_survive_calls() {
    run(
        r#"
import function twice(f64) -> f64
export function floats(f64) -> f64
block0(v0: f64):
  v1: f64 = call twice(v0) : (f64) -> f64
  v2: f64 = fadd v0, v1
  return v2
"#,
        r#"
#include <assert.h>
extern double floats(double);
__attribute__((noinline)) double twice(double x) { return x*2; }
int main(void) { for(int n=-100;n<100;n++) assert(floats(n*0.25)==n*0.75); }
"#,
    );
}
