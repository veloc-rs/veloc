//! Context-local, immutable signatures. The index owns IDs, never signature copies.
use crate::{Type, TypeInfo};
use alloc::{boxed::Box, vec::Vec};
use core::{
    fmt,
    hash::{BuildHasher, Hash, Hasher},
    ops::Index,
};
use hashbrown::{HashTable, hash_map::DefaultHashBuilder};

/// Direct index into the owning signature store. IDs from different stores
/// must be remapped or structurally compared, not compared as raw integers.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SigId(pub u32);

impl fmt::Display for SigId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "sig{}", self.0)
    }
}
impl fmt::Debug for SigId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CallConv {
    SystemV,
}
impl fmt::Display for CallConv {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SystemV => f.write_str("system_v"),
        }
    }
}

/// Parameters followed by results in one allocation. The split is part of
/// signature identity; (A) -> B differs from (A, B) -> ().
#[derive(Clone, PartialEq, Eq)]
pub struct Signature {
    types: Box<[Type]>,
    params: u32,
    pub call_conv: CallConv,
}
impl Signature {
    #[inline]
    pub fn params(&self) -> &[Type] {
        &self.types[..self.params as usize]
    }
    #[inline]
    pub fn returns(&self) -> &[Type] {
        &self.types[self.params as usize..]
    }
    #[inline]
    pub fn types(&self) -> &[Type] {
        &self.types
    }

    pub fn new(
        params: impl AsRef<[Type]>,
        returns: impl AsRef<[Type]>,
        call_conv: CallConv,
    ) -> Self {
        let (params, returns) = (params.as_ref(), returns.as_ref());
        let split = u32::try_from(params.len()).expect("too many signature parameters");
        let types = params.iter().chain(returns).copied().collect();
        Self {
            types,
            params: split,
            call_conv,
        }
    }
}
impl Hash for Signature {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (self.params(), self.returns(), self.call_conv).hash(state);
    }
}
impl fmt::Debug for Signature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Signature")
            .field("params", &self.params())
            .field("returns", &self.returns())
            .field("call_conv", &self.call_conv)
            .finish()
    }
}

/// Append-only within a compilation context. Reading or comparing known IDs
/// performs no hashing, allocation, synchronization or structural traversal.
#[derive(Debug, Clone)]
pub struct Signatures {
    entries: Vec<Signature>,
    index: HashTable<SigId>,
    hasher: DefaultHashBuilder,
}
impl Default for Signatures {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            index: HashTable::new(),
            hasher: DefaultHashBuilder::default(),
        }
    }
}
impl Signatures {
    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    #[inline]
    pub fn get(&self, id: SigId) -> Option<&Signature> {
        self.entries.get(id.0 as usize)
    }
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (SigId, &Signature)> {
        self.entries
            .iter()
            .enumerate()
            .map(|(i, sig)| (SigId(i as u32), sig))
    }

    fn find(
        &self,
        hash: u64,
        params: &[Type],
        returns: &[Type],
        call_conv: CallConv,
    ) -> Option<SigId> {
        self.index
            .find(hash, |id| {
                let sig = &self[*id];
                sig.call_conv == call_conv && sig.params() == params && sig.returns() == returns
            })
            .copied()
    }
    fn append(&mut self, hash: u64, signature: Signature) -> SigId {
        let id = SigId(u32::try_from(self.entries.len()).expect("too many signatures"));
        self.entries.push(signature);
        self.index.insert_unique(hash, id, |id| {
            self.hasher.hash_one(&self.entries[id.0 as usize])
        });
        id
    }
    /// Transfer an already constructed signature without cloning its payload.
    pub(crate) fn insert(&mut self, signature: Signature) -> SigId {
        let hash = self.hasher.hash_one(&signature);
        self.find(
            hash,
            signature.params(),
            signature.returns(),
            signature.call_conv,
        )
        .unwrap_or_else(|| self.append(hash, signature))
    }

    /// Probe borrowed slices first: a hit does not allocate a signature.
    pub(crate) fn intern(
        &mut self,
        params: &[Type],
        returns: &[Type],
        call_conv: CallConv,
    ) -> SigId {
        let hash = self.hasher.hash_one((params, returns, call_conv));
        self.find(hash, params, returns, call_conv)
            .unwrap_or_else(|| self.append(hash, Signature::new(params, returns, call_conv)))
    }

    /// Remap another context once. Nested signatures are interned before users,
    /// so subsequent comparisons in this context are comparisons of IDs.
    /// Validate the graph before appending anything to the destination.
    pub(crate) fn import(&mut self, source: &Self) -> Result<Vec<SigId>, SignatureError> {
        let order = source.dependency_order()?;
        let mut ids = alloc::vec![SigId(u32::MAX); source.len()];
        let mut types = Vec::new();
        for id in order {
            let sig = &source[id];
            let has_references = sig.types().iter().any(|ty| ty.is_callable());
            let imported = if has_references {
                types.clear();
                types.extend(sig.types().iter().map(|ty| match ty.as_callable() {
                    Some((id, kind)) => Type::callable(ids[id.0 as usize], kind),
                    None => *ty,
                }));
                let (params, returns) = types.split_at(sig.params().len());
                self.intern(params, returns, sig.call_conv)
            } else {
                self.intern(sig.params(), sig.returns(), sig.call_conv)
            };
            ids[id.0 as usize] = imported;
        }
        Ok(ids)
    }
}
impl Index<SigId> for Signatures {
    type Output = Signature;
    fn index(&self, id: SigId) -> &Self::Output {
        &self.entries[id.0 as usize]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignatureError {
    Unknown { source: SigId, target: SigId },
    Cycle { source: SigId, target: SigId },
}
impl fmt::Display for SignatureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unknown { source, target } => write!(
                f,
                "signature {source} references unknown signature {target}"
            ),
            Self::Cycle { source, target } => write!(
                f,
                "signature {source} references active signature {target}: recursive callable signature requires an explicit recursive type"
            ),
        }
    }
}
impl core::error::Error for SignatureError {}

impl Signatures {
    /// Postorder without Rust recursion, including forward references.
    pub fn dependency_order(&self) -> Result<Vec<SigId>, SignatureError> {
        let mut marks = alloc::vec![0u8; self.len()];
        let mut order = Vec::with_capacity(self.len());
        let mut stack = Vec::new();
        for (root, _) in self.iter() {
            if marks[root.0 as usize] != 0 {
                continue;
            }
            marks[root.0 as usize] = 1;
            stack.push((root, 0));
            while let Some((id, next)) = stack.last_mut() {
                let source = *id;
                let Some(ty) = self[source].types().get(*next) else {
                    marks[source.0 as usize] = 2;
                    order.push(source);
                    stack.pop();
                    continue;
                };
                *next += 1;
                let Some((target, _)) = ty.as_callable() else {
                    continue;
                };
                let Some(mark) = marks.get_mut(target.0 as usize) else {
                    return Err(SignatureError::Unknown { source, target });
                };
                match *mark {
                    0 => {
                        *mark = 1;
                        stack.push((target, 0));
                    }
                    1 => return Err(SignatureError::Cycle { source, target }),
                    _ => {}
                }
            }
        }
        Ok(order)
    }
}
