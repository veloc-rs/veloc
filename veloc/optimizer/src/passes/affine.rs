//! Prove scalar functions affine over GF(2), then synthesize byte lookup tables.
//!
//! This is an optional native-object optimization: it creates immutable data.
//! Recognition symbolically executes MIR, including bounded loops and direct
//! calls. It does not depend on function names, CRC polynomials or source syntax.
//! Unsupported operations, nonlinear results, or exhausted work limits leave the
//! function unchanged. Polynomial arithmetic gives an exact proof, not sampling.
use crate::{ModulePass, OptConfig, PassOutcome, Profile};
use smallvec::{SmallVec, smallvec};
use std::{collections::BTreeMap, rc::Rc};
use veloc_analyzer::graph::PostDominatorTree;
use veloc_mir::{
    Block, FuncBody, FuncId, GlobalData, Inst, InstView, Int, IntCC, Linkage, MemFlags, Module,
    Opcode, ScalarConst, Type, TypeInfo, Value,
};

pub struct AffinePass;

// A monomial is a set of input bits; repeated factors satisfy x*x = x.
// The empty monomial (0) denotes one, and an empty polynomial denotes zero.
// Each polynomial stores sorted, unique monomials with coefficient one.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Bit(SmallVec<[u64; 2]>);
// Symbolic words are immutable. Branches and SSA reads share their storage;
// splitting a path copies only its value map, not every polynomial in scope.
type Word = Rc<[Bit]>;
type State = Vec<Option<Word>>;

impl Bit {
    fn one() -> Self {
        Self(smallvec![0])
    }
    fn xor(&self, other: &Self) -> Self {
        // Canonical sorted monomials; matching terms cancel over GF(2).
        let mut terms = SmallVec::new();
        let (mut a, mut b) = (self.0.iter().peekable(), other.0.iter().peekable());
        while let (Some(&x), Some(&y)) = (a.peek(), b.peek()) {
            match x.cmp(y) {
                core::cmp::Ordering::Less => {
                    terms.push(*x);
                    a.next();
                }
                core::cmp::Ordering::Greater => {
                    terms.push(*y);
                    b.next();
                }
                core::cmp::Ordering::Equal => {
                    a.next();
                    b.next();
                }
            }
        }
        terms.extend(a.chain(b).copied());
        Self(terms)
    }
    fn and(&self, other: &Self) -> Option<Self> {
        if self.0.len() * other.0.len() > 4096 {
            return None;
        }
        if let Some(bit) = self.constant() {
            return Some(if bit { other.clone() } else { Self::default() });
        }
        if let Some(bit) = other.constant() {
            return Some(if bit { self.clone() } else { Self::default() });
        }
        let mut products = Vec::with_capacity(self.0.len() * other.0.len());
        for a in &self.0 {
            for b in &other.0 {
                products.push(a | b);
            }
        }
        products.sort_unstable();
        let mut out = SmallVec::new();
        for term in products {
            if out.last() == Some(&term) {
                out.pop();
            } else {
                out.push(term);
            }
        }
        (out.len() <= 128).then_some(Self(out))
    }
    fn or(&self, other: &Self) -> Option<Self> {
        Some(self.xor(other).xor(&self.and(other)?))
    }
    fn constant(&self) -> Option<bool> {
        if self.0.is_empty() {
            Some(false)
        } else if self.0.len() == 1 && self.0.contains(&0) {
            Some(true)
        } else {
            None
        }
    }
}
fn constant(value: u64, bits: usize) -> Word {
    (0..bits)
        .map(|bit| {
            if value >> bit & 1 == 1 {
                Bit::one()
            } else {
                Bit::default()
            }
        })
        .collect()
}
fn number(word: &Word) -> Option<u64> {
    word.iter()
        .enumerate()
        .try_fold(0, |n, (i, b)| Some(n | ((b.constant()? as u64) << i)))
}
fn width(ty: Type) -> Option<usize> {
    (ty.is_integer() || ty == Type::BOOL)
        .then(|| ty.element_bits())
        .flatten()
        .filter(|&bits| bits <= 64)
        .map(|bits| bits as usize)
}
fn select(cond: &Bit, yes: &Word, no: &Word) -> Option<Word> {
    if yes.len() != no.len() {
        return None;
    }
    yes.iter()
        .zip(no.iter())
        .map(|(a, b)| Some(b.xor(&cond.and(&a.xor(b))?)))
        .collect()
}

struct Symbolic<'a> {
    module: &'a Module,
    remaining: usize,
    stack: Vec<FuncId>,
}
enum Exit {
    Join(State),
    Return(Word),
}

impl Symbolic<'_> {
    fn call(&mut self, id: FuncId, args: Vec<Word>) -> Option<Word> {
        if self.stack.len() == 8 || self.stack.contains(&id) {
            return None;
        }
        let body = self.module.function(id).body?;
        let mut state = vec![None; body.dfg().values().len()];
        if args.len() != body.params().len() {
            return None;
        }
        for (&param, arg) in body.params().iter().zip(args) {
            state[param.0 as usize] = Some(arg);
        }
        let post = PostDominatorTree::compute(body.cfg());
        self.stack.push(id);
        let result = self.run(body, &post, body.entry_block(), state, None, 0);
        self.stack.pop();
        match result? {
            Exit::Return(value) => Some(value),
            _ => None,
        }
    }
    fn value(body: &FuncBody, state: &State, value: Value) -> Option<Word> {
        if let Some(c) = body.dfg().as_scalar_const(value) {
            Some(constant(c.to_bits(), width(c.ty())?))
        } else {
            state.get(value.0 as usize)?.clone()
        }
    }
    fn edge(body: &FuncBody, state: &State, edge: veloc_mir::Successor<'_>) -> Option<State> {
        let args = edge
            .args
            .iter()
            .map(|&v| Self::value(body, state, v))
            .collect::<Option<Vec<_>>>()?;
        let mut next = state.clone();
        for (&param, arg) in body.dfg().block_params(edge.block).iter().zip(args) {
            next[param.0 as usize] = Some(arg);
        }
        Some(next)
    }
    fn run(
        &mut self,
        body: &FuncBody,
        post: &PostDominatorTree<Block>,
        mut block: Block,
        mut state: State,
        stop: Option<Block>,
        depth: usize,
    ) -> Option<Exit> {
        if depth > 32 {
            return None;
        }
        loop {
            if Some(block) == stop {
                return Some(Exit::Join(state));
            }
            for inst in body.layout().block_insts(block) {
                self.remaining = self.remaining.checked_sub(1)?;
                let view = body.dfg().inst(inst);
                match view {
                    InstView::Jump { dest } => {
                        state = Self::edge(body, &state, dest)?;
                        block = dest.block;
                        break;
                    }
                    InstView::Br {
                        condition,
                        then_dest,
                        else_dest,
                    } => {
                        let cond = Self::value(body, &state, condition)?.first()?.clone();
                        if let Some(taken) = cond.constant() {
                            let edge = if taken { then_dest } else { else_dest };
                            state = Self::edge(body, &state, edge)?;
                            block = edge.block;
                            break;
                        }
                        // The nearest strict post-dominator is the rendezvous.
                        // Both paths must actually reach it within the work limit.
                        let candidates: Vec<_> = body
                            .layout()
                            .block_order()
                            .filter(|&b| b != block && post.post_dominates(b, block))
                            .collect();
                        let join = candidates
                            .iter()
                            .copied()
                            .find(|&b| candidates.iter().all(|&p| post.post_dominates(p, b)));
                        let yes = self.run(
                            body,
                            post,
                            then_dest.block,
                            Self::edge(body, &state, then_dest)?,
                            join,
                            depth + 1,
                        )?;
                        let no = self.run(
                            body,
                            post,
                            else_dest.block,
                            Self::edge(body, &state, else_dest)?,
                            join,
                            depth + 1,
                        )?;
                        match (yes, no) {
                            (Exit::Return(a), Exit::Return(b)) => {
                                return Some(Exit::Return(select(&cond, &a, &b)?));
                            }
                            (Exit::Join(a), Exit::Join(b)) => {
                                state = a
                                    .into_iter()
                                    .zip(b)
                                    .map(|(a, b)| match (a, b) {
                                        (Some(a), Some(b)) => Some(Some(select(&cond, &a, &b)?)),
                                        _ => Some(None),
                                    })
                                    .collect::<Option<_>>()?;
                                block = join?;
                                break;
                            }
                            _ => return None,
                        }
                    }
                    InstView::Return { values: [value] } => {
                        return Some(Exit::Return(Self::value(body, &state, *value)?));
                    }
                    _ => {
                        let value = self.operation(body, &state, inst)?;
                        let [result] = body.dfg().inst_results(inst) else {
                            return None;
                        };
                        state[result.0 as usize] = Some(value);
                    }
                }
            }
        }
    }
    fn operation(&mut self, body: &FuncBody, state: &State, inst: Inst) -> Option<Word> {
        let view = body.dfg().inst(inst);
        if let InstView::Call { func_id, args } = view {
            return self.call(
                func_id,
                args.iter()
                    .map(|&v| Self::value(body, state, v))
                    .collect::<Option<_>>()?,
            );
        }
        let [result] = body.dfg().inst_results(inst) else {
            return None;
        };
        let bits = width(body.dfg().value_type(*result))?;
        let args: Vec<_> = body
            .dfg()
            .operands(inst)
            .iter()
            .map(|&v| Self::value(body, state, v))
            .collect::<Option<_>>()?;
        if crate::evaluate::can_reduce(body.dfg(), inst) {
            if let Some(values) = crate::evaluate::fold(body.dfg(), inst, |v| {
                ScalarConst::from_bits(
                    body.dfg().value_type(v),
                    number(&Self::value(body, state, v)?)?,
                )
            }) {
                if let [value] = &values[..] {
                    return Some(constant(value.to_bits(), bits));
                }
            }
        }
        let a = args.first()?;
        match view.opcode() {
            Opcode::IAnd | Opcode::IOr | Opcode::IXor => a
                .iter()
                .zip(args.get(1)?.iter())
                .map(|(a, b)| match view.opcode() {
                    Opcode::IAnd => a.and(b),
                    Opcode::IOr => a.or(b),
                    _ => Some(a.xor(b)),
                })
                .collect(),
            Opcode::ExtendU | Opcode::ExtendS | Opcode::Wrap => {
                let extension = if view.opcode() == Opcode::ExtendS {
                    a.last()?.clone()
                } else {
                    Bit::default()
                };
                let mut result = a.to_vec();
                result.resize(bits, extension);
                Some(result.into())
            }
            Opcode::IShl | Opcode::IShrU | Opcode::IShrS => {
                let shift = number(args.get(1)?)? as usize % a.len();
                let extension = if view.opcode() == Opcode::IShrS {
                    a.last()?.clone()
                } else {
                    Bit::default()
                };
                Some(
                    (0..bits)
                        .map(|i| {
                            if view.opcode() == Opcode::IShl {
                                i.checked_sub(shift)
                                    .and_then(|i| a.get(i))
                                    .cloned()
                                    .unwrap_or_default()
                            } else {
                                a.get(i + shift)
                                    .cloned()
                                    .unwrap_or_else(|| extension.clone())
                            }
                        })
                        .collect(),
                )
            }
            Opcode::Icmp => {
                let InstView::IntCompare { kind, .. } = view else {
                    return None;
                };
                if !matches!(kind, IntCC::Eq | IntCC::Ne) {
                    return None;
                }
                let mut unequal = Bit::default();
                for (a, b) in a.iter().zip(args.get(1)?.iter()) {
                    unequal = unequal.or(&a.xor(b))?;
                }
                Some(Rc::from([if kind == IntCC::Eq {
                    unequal.xor(&Bit::one())
                } else {
                    unequal
                }]))
            }
            Opcode::Select => select(a.first()?, args.get(1)?, args.get(2)?),
            _ => None,
        }
    }
}

struct Transform {
    id: FuncId,
    inputs: Vec<(usize, usize)>,
    columns: Vec<u64>,
    bias: u64,
    ty: Type,
}

impl ModulePass for AffinePass {
    fn name(&self) -> &'static str {
        "AffinePass"
    }
    fn run(&self, module: &mut Module, config: &OptConfig, metrics: &Profile) -> PassOutcome {
        let Some(layout) = config.data_layout else {
            return PassOutcome::Unchanged;
        };
        let mut transforms = Vec::new();
        for (id, func) in module.functions() {
            let Some(body) = func.body else {
                continue;
            };
            let sig = &module.signatures()[func.decl.signature];
            if sig.variadic {
                continue;
            }
            let [ty] = sig.returns() else {
                continue;
            };
            if !matches!(*ty, Type::I8 | Type::I16 | Type::I32 | Type::I64) {
                continue;
            }
            let Some(widths) = sig
                .params()
                .iter()
                .map(|&ty| width(ty))
                .collect::<Option<Vec<_>>>()
            else {
                continue;
            };
            let n: usize = widths.iter().sum();
            if n == 0 || n > 64 || body.dfg().inst_count() > 300 {
                continue;
            }
            let inputs: Vec<_> = widths
                .iter()
                .enumerate()
                .flat_map(|(i, &w)| (0..w).map(move |b| (i, b)))
                .collect();
            let mut next = 0;
            let args = widths
                .iter()
                .map(|&w| {
                    (0..w)
                        .map(|_| {
                            let bit = Bit(smallvec![1 << next]);
                            next += 1;
                            bit
                        })
                        .collect()
                })
                .collect();
            let mut interpreter = Symbolic {
                module,
                remaining: 8192,
                stack: vec![],
            };
            let Some(result) = interpreter.call(id, args) else {
                continue;
            };
            if interpreter.remaining > 8192 - 80 {
                continue;
            }
            let mut columns = vec![0; n];
            let mut bias = 0;
            let mut affine = true;
            for (bit, p) in result.iter().enumerate() {
                for &term in &p.0 {
                    if term == 0 {
                        bias |= 1 << bit;
                    } else if term.is_power_of_two() {
                        columns[term.trailing_zeros() as usize] |= 1 << bit;
                    } else {
                        affine = false;
                    }
                }
            }
            if affine {
                transforms.push(Transform {
                    id,
                    inputs,
                    columns,
                    bias,
                    ty: *ty,
                });
            }
        }
        let changed = transforms.len();
        for transform in transforms {
            synthesize(module, transform, layout.little_endian);
        }
        metrics.count("affine.functions", changed as u64);
        if changed == 0 {
            PassOutcome::Unchanged
        } else {
            PassOutcome::Changed
        }
    }
}

// Pack selected input bits. Group equal shifts so a byte slice becomes one
// shift/mask, rather than eight independent bit extractions.
fn pack(
    cursor: &mut veloc_mir::function::InstCursor<'_, '_>,
    params: &[Value],
    fields: &[(usize, usize, usize)],
) -> Value {
    let mut groups = BTreeMap::<(usize, isize), u64>::new();
    for &(param, source, dest) in fields {
        *groups
            .entry((param, dest as isize - source as isize))
            .or_default() |= 1 << source;
    }
    let mut value = cursor.iconst(Int::from_bits(Type::I64, 0).unwrap());
    for ((param, shift), mask) in groups {
        let input = params[param];
        let input = if cursor.value_type(input) == Type::I64 {
            input
        } else {
            cursor.extendu(input, Type::I64)
        };
        let mask = cursor.iconst(Int::from_bits(Type::I64, mask).unwrap());
        let mut part = cursor.iand(input, mask);
        if shift != 0 {
            let count =
                cursor.iconst(Int::from_bits(Type::I64, shift.unsigned_abs() as u64).unwrap());
            part = if shift > 0 {
                cursor.ishl(part, count)
            } else {
                cursor.ishr_u(part, count)
            };
        }
        value = cursor.ixor(value, part);
    }
    value
}
fn synthesize(module: &mut Module, t: Transform, little_endian: bool) {
    let mut direct = Vec::new();
    let mut grouped = BTreeMap::<u64, Vec<(usize, usize)>>::new();
    for (&column, &(param, bit)) in t.columns.iter().zip(&t.inputs) {
        if column.is_power_of_two() {
            direct.push((param, bit, column.trailing_zeros() as usize));
        } else if column != 0 {
            grouped.entry(column).or_default().push((param, bit));
        }
    }
    let mut groups: Vec<_> = grouped.into_iter().collect();
    groups.sort_by_key(|(_, sources)| sources[0]);
    let mut tables = Vec::new();
    for (index, chunk) in groups.chunks(8).enumerate() {
        let mut bytes = Vec::new();
        // The ABI result may be wider than every table entry. Store only the
        // bits occupied by this chunk's columns; XOR cannot introduce others.
        let mask = chunk.iter().fold(0, |bits, (column, _)| bits | column);
        let ty = match mask {
            0..=0xff => Type::I8,
            0..=0xffff => Type::I16,
            0..=0xffff_ffff => Type::I32,
            _ => Type::I64,
        };
        let size = width(ty).unwrap() / 8;
        for key in 0..1usize << chunk.len() {
            let value =
                chunk.iter().enumerate().fold(
                    0,
                    |v, (bit, (column, _))| if key >> bit & 1 == 1 { v ^ column } else { v },
                );
            if little_endian {
                bytes.extend_from_slice(&value.to_le_bytes()[..size]);
            } else {
                bytes.extend_from_slice(&value.to_be_bytes()[8 - size..]);
            }
        }
        let name = format!(".L.affine.{}.{}", t.id.0, index);
        let global = module.add_global(name, Type::PTR, Linkage::Local);
        module.define_global(
            global,
            GlobalData {
                bytes,
                align: size as u64,
                writable: false,
                relocations: vec![],
            },
        );
        let fields = chunk
            .iter()
            .enumerate()
            .flat_map(|(dest, (_, sources))| {
                sources.iter().map(move |&(param, bit)| (param, bit, dest))
            })
            .collect::<Vec<_>>();
        tables.push((global, fields, ty));
    }
    let sig = &module.signatures()[module.function(t.id).decl.signature];
    let mut body = FuncBody::new(sig.params());
    let params = body.params().to_vec();
    let entry = body.entry_block();
    let mut cursor = body
        .edit()
        .at_end(entry, module.decls(), module.signatures());
    let mut result = pack(&mut cursor, &params, &direct);
    let bias = cursor.iconst(Int::from_bits(Type::I64, t.bias).unwrap());
    result = cursor.ixor(result, bias);
    for (global, fields, ty) in tables {
        let index = pack(&mut cursor, &params, &fields);
        let base = cursor.global_addr(global);
        let address = cursor.ptr_index(
            base,
            index,
            veloc_mir::inst::PtrIndexImm {
                offset: 0,
                scale: (width(ty).unwrap() / 8) as u32,
            },
        );
        let part = cursor.load(address, 0, MemFlags::new().with_notrap(true), ty);
        let part = if ty == Type::I64 {
            part
        } else {
            cursor.extendu(part, Type::I64)
        };
        result = cursor.ixor(result, part);
    }
    let result = if t.ty == Type::I64 {
        result
    } else {
        cursor.wrap(result, t.ty)
    };
    cursor.ret(&[result]);
    *module.body_mut(t.id).unwrap() = body;
}
