//! Canonicalize pure, single-use integer trees before instruction selection.
//!
//! Associativity/commutativity alone have a direct normal form: flatten, sort
//! leaves by register identity, and rebuild. Searching an e-graph for that form
//! adds no target choices. Costed target patterns belong in instruction selection.

use crate::analysis::{ChangeSet, FunctionAnalysisCtx};

use alloc::vec;
use alloc::vec::Vec;
use hashbrown::HashMap;
use veloc_lir::{GenericOpcode, InstId, MachineFunction, MachineOpcode, Reg};
use veloc_mir::TypeInfo;

#[derive(Clone, Copy)]
struct Node {
    id: InstId,
    dst: Reg,
    lhs: Reg,
    rhs: Reg,
}

struct Tree {
    opcode: GenericOpcode,
    nodes: Vec<Node>,
    leaves: Vec<Reg>,
}

fn binary(f: &MachineFunction, id: InstId, opcode: GenericOpcode) -> Option<Node> {
    let inst = &f.inst(id);
    if inst.generic_opcode() != Some(opcode) || f.inst_extra(id).is_some() {
        return None;
    }
    let [lhs, rhs] = inst.inputs() else {
        return None;
    };
    let &[dst] = inst.results() else {
        return None;
    };
    if ![dst, *lhs, *rhs].iter().all(|r| r.is_vreg()) {
        return None;
    }
    let ty = f.vreg_data(dst).ty;
    if !ty.is_scalar()
        || !ty.is_integer()
        || f.vreg_data(*lhs).ty != ty
        || f.vreg_data(*rhs).ty != ty
    {
        return None;
    }
    Some(Node {
        id,
        dst,
        lhs: *lhs,
        rhs: *rhs,
    })
}

fn tree(f: &MachineFunction, positions: &HashMap<InstId, usize>, root: InstId) -> Option<Tree> {
    let opcode = f.inst(root).generic_opcode()?;
    if !matches!(
        opcode,
        GenericOpcode::Add
            | GenericOpcode::Mul
            | GenericOpcode::And
            | GenericOpcode::Or
            | GenericOpcode::Xor
    ) {
        return None;
    }
    let first = binary(f, root, opcode)?;
    if f.defs(first.dst).single().map(|site| site.inst()) != Some(root) {
        return None;
    }
    let ty = f.vreg_data(first.dst).ty;
    let mut nodes = vec![first];
    let mut leaves = Vec::new();
    // An explicit stack handles long expressions without recursive stack growth.
    let mut pending = vec![(first.rhs, root), (first.lhs, root)];
    while let Some((reg, parent)) = pending.pop() {
        if !reg.is_vreg() || f.vreg_data(reg).ty != ty || f.defs(reg).nth(1).is_some() {
            // After phi destruction a register can have multiple definitions.
            // Reassociation is not allowed to change which definition it reads.
            return None;
        }
        if let Some(def) = f.defs(reg).single().map(|site| site.inst())
            && let Some(&pos) = positions.get(&def)
        {
            if pos >= positions[&parent] {
                return None;
            }
            if f.uses(reg)
                .single()
                .is_some_and(|site| site.inst() == parent)
                && let Some(node) = binary(f, def, opcode)
            {
                nodes.push(node);
                pending.push((node.rhs, def));
                pending.push((node.lhs, def));
                continue;
            }
        }
        // Shared expressions and other opcodes are leaves, never duplicated.
        leaves.push(reg);
    }
    nodes.sort_unstable_by_key(|n| positions[&n.id]);
    leaves.sort_unstable_by_key(|r| r.0);
    Some(Tree {
        opcode,
        nodes,
        leaves,
    })
}

impl Tree {
    fn changed(&self) -> bool {
        let mut acc = self.leaves[0];
        for (node, &rhs) in self.nodes.iter().zip(&self.leaves[1..]) {
            if node.lhs != acc || node.rhs != rhs {
                return true;
            }
            acc = node.dst;
        }
        false
    }

    fn emit(&self, f: &mut MachineFunction, output: &mut Vec<InstId>) {
        let mut acc = self.leaves[0];
        for (node, &rhs) in self.nodes.iter().zip(&self.leaves[1..]) {
            f.editor().rewriter(node.id).write(
                MachineOpcode::Generic(self.opcode),
                &[node.dst],
                &[acc, rhs],
                &[],
            );
            output.push(node.id);
            acc = node.dst;
        }
    }
}

/// Store-maintained references, one maximal-tree traversal and one layout commit.
/// Existing instruction IDs and virtual registers are reused.
pub(crate) fn reassociate(f: &mut MachineFunction, analyses: &mut FunctionAnalysisCtx) -> usize {
    let mut changes = 0;
    for block in f.blocks().collect::<Vec<_>>() {
        let ids = f.block_insts(block).collect::<Vec<_>>();
        let positions: HashMap<_, _> = ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
        let mut visited = vec![false; ids.len()];
        let mut removed = vec![false; ids.len()];
        let mut plans: HashMap<InstId, Tree> = HashMap::new();
        for &root in ids.iter().rev() {
            if visited[positions[&root]] {
                continue;
            }
            let Some(tree) = tree(f, &positions, root) else {
                continue;
            };
            // Claim even already-canonical trees: don't rescan every subtree.
            for node in &tree.nodes {
                visited[positions[&node.id]] = true;
            }
            if tree.changed() {
                for node in &tree.nodes {
                    removed[positions[&node.id]] = true;
                }
                plans.insert(root, tree);
            }
        }
        if plans.is_empty() {
            continue;
        }
        changes += plans.len();
        let mut output = Vec::with_capacity(ids.len());
        for (i, id) in ids.into_iter().enumerate() {
            if let Some(plan) = plans.get(&id) {
                plan.emit(f, &mut output);
            } else if !removed[i] {
                output.push(id);
            }
        }
        f.editor().reorder_block(block, &output);
    }
    if changes != 0 {
        analyses
            .apply(ChangeSet::INST_SEMANTICS | ChangeSet::INST_OPERANDS | ChangeSet::BLOCK_LAYOUT);
    }
    changes
}
