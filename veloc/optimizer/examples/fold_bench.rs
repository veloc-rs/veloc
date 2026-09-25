mod offline {
    use veloc_mir::{IntCC, Opcode};
    include!(concat!(env!("OUT_DIR"), "/semantics.rs"));
}
// Compare generated evaluation with the former per-fold graph construction path.
// cargo run --release -p veloc-optimizer --example fold_bench
use std::{hint::black_box, time::Instant};

use smallvec::SmallVec;
use veloc_mir::constant::ScalarConst;
use veloc_mir::{Opcode, Type};
use veloc_semantics::{Outcome, Sort, Value};

fn graph(op: Opcode, args: &[ScalarConst], results: &[Type]) -> Option<SmallVec<[ScalarConst; 2]>> {
    let types = args.iter().map(|c| c.ty()).collect::<SmallVec<[Type; 4]>>();
    op.validate_types(&types, results).ok()?;
    let sort = |ty: Type| {
        if ty == Type::BOOL {
            Sort::Bool
        } else {
            Sort::bv(ty.min_bit_width().unwrap() as u16).unwrap()
        }
    };
    let inputs = types
        .iter()
        .copied()
        .map(sort)
        .collect::<SmallVec<[Sort; 4]>>();
    let outputs = results
        .iter()
        .copied()
        .map(sort)
        .collect::<SmallVec<[Sort; 2]>>();
    let function = offline::SPECS
        .iter()
        .find(|s| s.opcode == op)?
        .program
        .instantiate(&inputs, &outputs, &[])
        .ok()?;
    let values = args
        .iter()
        .map(|c| match c.as_bool() {
            Some(b) => Value::Bool(b),
            None => Value::Bv(c.to_bits() as u128),
        })
        .collect::<SmallVec<[Value; 4]>>();
    let Outcome::Values(values) = function.execute(&values).ok()? else {
        return None;
    };
    Some(
        values
            .into_iter()
            .zip(results)
            .map(|(value, &ty)| match value {
                Value::Bool(b) => ScalarConst::from(b),
                Value::Bv(v) => match ty {
                    Type::I64 => ScalarConst::from(v as i64),
                    _ => unreachable!("benchmark uses i64 results"),
                },
            })
            .collect(),
    )
}

fn main() {
    const N: usize = 100_000;
    for op in [
        Opcode::IAdd,
        Opcode::IRotl,
        Opcode::IDivS,
        Opcode::IAddWithOverflow,
    ] {
        let results = if op == Opcode::IAddWithOverflow {
            &[Type::I64, Type::BOOL][..]
        } else {
            &[Type::I64][..]
        };
        // Construct MIR once: the timed path only queries facts and folds it.
        let mut body = veloc_mir::FuncBody::new(&[Type::I64, Type::I64]);
        let block = body.entry_block();
        let inputs = body.dfg().block_params(block).to_vec();
        let inst = body
            .edit()
            .append_inst(block, |w| w.from_values(op, &inputs).unwrap(), results);
        let evaluate = |fast, i| {
            let args = [
                ScalarConst::from(i as i64 ^ i64::MIN),
                ScalarConst::from(13i64),
            ];
            let (op, args, results) = black_box((op, args, results));
            if fast {
                veloc_optimizer::evaluate::fold(body.dfg(), inst, |value| {
                    inputs
                        .iter()
                        .position(|&input| input == value)
                        .map(|index| args[index])
                })
            } else {
                graph(op, &args, results)
            }
        };
        for i in 0..1000 {
            assert_eq!(evaluate(true, i), evaluate(false, i));
        }
        let mut times = [Vec::new(), Vec::new()];
        for round in 0..5 {
            // Alternate order to reduce systematic warmup/order effects.
            for index in [round % 2, 1 - round % 2] {
                let start = Instant::now();
                for i in 0..N {
                    black_box(evaluate(index == 0, i));
                }
                times[index].push(start.elapsed().as_nanos() as f64 / N as f64);
            }
        }
        for values in &mut times {
            values.sort_by(f64::total_cmp);
        }
        println!(
            "{op:?}: generated {:.1} ns/fold, graph {:.1} ns/fold ({:.1}x)",
            times[0][2],
            times[1][2],
            times[1][2] / times[0][2]
        );
    }
}
