//! Shared solver for logical IR value types. Matched variables are rigid;
//! construction variables are inferred, never allowed to narrow source domains.
use crate::{
    Error,
    schema::{Signature, Term, TypeSet},
};
use std::collections::BTreeMap;

struct Ty<A> {
    parent: usize,
    domain: TypeSet,
    anchor: Option<A>,
}
pub(crate) struct Infer<A> {
    types: Vec<Ty<A>>,
}
impl<A> Default for Infer<A> {
    fn default() -> Self {
        Self { types: Vec::new() }
    }
}
impl<A: Copy> Infer<A> {
    pub(crate) fn fresh(&mut self, domain: &TypeSet, anchor: Option<A>) -> usize {
        let id = self.types.len();
        self.types.push(Ty {
            parent: id,
            domain: domain.clone(),
            anchor,
        });
        id
    }

    fn root(&self, mut ty: usize) -> usize {
        while self.types[ty].parent != ty {
            ty = self.types[ty].parent;
        }
        ty
    }

    pub(crate) fn unify(
        &mut self,
        a: usize,
        b: usize,
        offset: usize,
        source: &str,
    ) -> Result<(), Error> {
        let a = self.root(a);
        let b = self.root(b);
        if a == b {
            return Ok(());
        }
        let left = &self.types[a];
        let right = &self.types[b];
        match (left.anchor, right.anchor) {
            (Some(_), Some(_)) => {
                if left.domain != right.domain || !left.domain.is_singleton() {
                    return Err(Error::at(
                        source,
                        offset,
                        "rule requires independent source types to be equal",
                    ));
                }
                self.types[b].parent = a;
            }
            (Some(_), None) => {
                if !left.domain.subset_of(&right.domain) {
                    return Err(Error::at(
                        source,
                        offset,
                        "target type domain does not cover every source type",
                    ));
                }
                self.types[b].parent = a;
            }
            (None, Some(_)) => return self.unify(b, a, offset, source),
            (None, None) => {
                let mut domain = left.domain.clone();
                domain.intersect(&right.domain);
                if domain.is_empty() {
                    return Err(Error::at(source, offset, "incompatible type domains"));
                }
                self.types[a].domain = domain;
                self.types[b].parent = a;
            }
        }
        Ok(())
    }

    pub(crate) fn signature(
        &mut self,
        signature: &Signature,
        anchor: impl Fn(bool, usize) -> Option<A>,
    ) -> (Vec<usize>, Vec<usize>) {
        let mut variables = BTreeMap::new();
        let mut term = |this: &mut Self, term: &Term, anchor| {
            if let Some(id) = term.variable {
                if let Some(&ty) = variables.get(&id) {
                    return ty;
                }
            }
            let ty = this.fresh(&term.domain, anchor);
            if let Some(id) = term.variable {
                variables.insert(id, ty);
            }
            ty
        };
        let inputs = signature
            .inputs
            .iter()
            .enumerate()
            .map(|(i, t)| term(self, t, anchor(false, i)))
            .collect();
        let results = signature
            .results
            .iter()
            .enumerate()
            .map(|(i, t)| term(self, t, anchor(true, i)))
            .collect();
        (inputs, results)
    }

    pub(crate) fn anchor(&self, ty: usize) -> Option<A> {
        self.types[self.root(ty)].anchor
    }
}
