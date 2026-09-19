//! Recyclable, typed storage for privately owned payloads.
/// Type-indexed pool handles. IDs belong to their owning pool and must not be
/// used after removal: slots are recycled without generation counters.
pub struct PoolId<T>(u32, core::marker::PhantomData<fn() -> T>);
impl<T> Copy for PoolId<T> {}
impl<T> Clone for PoolId<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> core::fmt::Debug for PoolId<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Debug, Clone)]
pub struct Pool<T> {
    slots: alloc::vec::Vec<Slot<T>>,
    free: Option<u32>,
}
impl<T> Default for Pool<T> {
    fn default() -> Self {
        Self {
            slots: alloc::vec::Vec::new(),
            free: None,
        }
    }
}
#[derive(Debug, Clone)]
enum Slot<T> {
    Live(T),
    Free(Option<u32>),
}
impl<T> Pool<T> {
    pub fn push(&mut self, value: T) -> PoolId<T> {
        let index = if let Some(index) = self.free {
            let Slot::Free(next) = self.slots[index as usize] else {
                unreachable!("free pool slot")
            };
            self.free = next;
            self.slots[index as usize] = Slot::Live(value);
            index
        } else {
            let index = self.slots.len().try_into().expect("too many pooled fields");
            self.slots.push(Slot::Live(value));
            index
        };
        PoolId(index, core::marker::PhantomData)
    }
    pub fn get(&self, id: PoolId<T>) -> &T {
        let Slot::Live(value) = &self.slots[id.0 as usize] else {
            unreachable!("live pool slot")
        };
        value
    }
    pub fn get_mut(&mut self, id: PoolId<T>) -> &mut T {
        let Slot::Live(value) = &mut self.slots[id.0 as usize] else {
            unreachable!("live pool slot")
        };
        value
    }
    pub fn remove(&mut self, id: PoolId<T>) {
        assert!(matches!(self.slots[id.0 as usize], Slot::Live(_)));
        self.slots[id.0 as usize] = Slot::Free(self.free);
        self.free = Some(id.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cloning_and_recycling() {
        assert_eq!(core::mem::size_of::<PoolId<u64>>(), 4);
        let mut pool = Pool::<u64>::default();
        let id = pool.push(7);
        let cloned = pool.clone();
        pool.remove(id);
        let next = pool.push(9);
        assert_eq!(id.0, next.0);
        assert_eq!(*pool.get(next), 9);
        assert_eq!(*cloned.get(id), 7);
        *pool.get_mut(next) = 11;
        assert_eq!(*pool.get(next), 11);
        assert_eq!(pool.slots.len(), 1);
    }
}
