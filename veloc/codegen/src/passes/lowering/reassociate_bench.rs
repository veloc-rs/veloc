use super::reassociate as run;
use crate::pipeline::FunctionAnalysisCtx;
use alloc::vec::Vec;
use veloc_lir::stages::RawLir;
use veloc_lir::{GenericOpcode, MachineBlock, MachineFunction, MachineOpcode, Writable};
use veloc_mir::{Block, Type};

#[test]
#[ignore = "manual lowering microbenchmark; run with --release --ignored --nocapture"]
fn reassociate_benchmark() {
    let mut source = MachineFunction::<RawLir>::new("assoc_bench".into());
    source.blocks.push(MachineBlock::new(Block(0)));
    let leaves: Vec<_> = (0..12).map(|_| source.alloc_vreg(Type::I64)).collect();
    let mut acc = leaves[11];
    for &leaf in leaves[..11].iter().rev() {
        let result = source.alloc_vreg(Type::I64);
        {
            let id = source.writer().binary(
                MachineOpcode::Generic(GenericOpcode::G_ADD),
                Writable(result),
                acc,
                leaf,
            );
            source.append_inst_id_to_block(0, id);
            id
        };
        acc = result;
    }
    let start = std::time::Instant::now();
    let mut count = 0;
    for _ in 0..20 {
        let mut function = source.clone();
        count += run(&mut function, &mut FunctionAnalysisCtx::default());
        std::hint::black_box(function);
    }
    std::eprintln!(
        "20 x 12-leaf trees: {:?}; rewrites={count}",
        start.elapsed()
    );
}
