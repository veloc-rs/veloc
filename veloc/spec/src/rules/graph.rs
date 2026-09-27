//! Ordered decision IR shared by legalization and instruction selection.
//! Tests/actions are interned by each frontend; this layer preserves priority
//! and shares contiguous pure prefixes without exponential Boolean expansion.
use std::collections::{BTreeMap, HashMap};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Node {
    Reject,
    Accept(usize),
    Check { test: usize, yes: usize, no: usize },
}

pub(crate) type Candidates = Vec<(usize, Vec<usize>)>;

#[derive(Default)]
pub(crate) struct Graph {
    pub nodes: Vec<Node>,
    interned: HashMap<Node, usize>,
    states: BTreeMap<(Candidates, usize), usize>,
}

impl Graph {
    fn node(&mut self, node: Node) -> usize {
        if let Some(&id) = self.interned.get(&node) {
            return id;
        }
        let id = self.nodes.len();
        self.nodes.push(node);
        self.interned.insert(node, id);
        id
    }

    pub fn compile(&mut self, candidates: Candidates) -> usize {
        let reject = self.node(Node::Reject);
        self.sequence(candidates, reject)
    }

    fn sequence(&mut self, candidates: Candidates, fallback: usize) -> usize {
        let key = (candidates.clone(), fallback);
        if let Some(&id) = self.states.get(&key) {
            return id;
        }
        let id = match candidates.first() {
            None => fallback,
            Some((rule, tests)) if tests.is_empty() => self.node(Node::Accept(*rule)),
            Some((_, tests)) => {
                // Share contiguous prefixes without expanding a Boolean
                // decision diagram (which can grow exponentially). The suffix
                // remains the fallback for every failure within this group.
                let test = tests[0];
                let count = candidates
                    .iter()
                    .take_while(|(_, ts)| ts.first() == Some(&test))
                    .count();
                let no = self.sequence(candidates[count..].to_vec(), fallback);
                let yes = candidates[..count]
                    .iter()
                    .map(|(r, ts)| (*r, ts[1..].to_vec()))
                    .collect();
                let yes = self.sequence(yes, no);
                self.node(Node::Check { test, yes, no })
            }
        };
        self.states.insert(key, id);
        id
    }
}
