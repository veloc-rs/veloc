use veloc::backend::Backend;

/// 编译策略
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    Auto,
    Jit,
    Interpreter,
}

/// Engine 配置
#[derive(Debug, Clone)]
pub struct Config {
    pub strategy: Strategy,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            strategy: Strategy::Auto,
        }
    }
}

/// Engine 持有全局编译器配置
pub struct Engine {
    pub backend: Backend,
    pub config: Config,
}

impl Engine {
    pub fn new() -> Self {
        Self::with_config(Config::default())
    }

    pub fn with_config(config: Config) -> Self {
        Self {
            backend: Backend::new(),
            config,
        }
    }

    pub fn grow_memory(
        &self,
        vmctx_ptr: *mut crate::vm::VMContext,
        index: u32,
        num_memories: u32,
        delta: u32,
        offsets: &crate::vm::VMOffsets,
    ) -> Option<u32> {
        unsafe {
            let vm_memories = (*vmctx_ptr).memories_mut(offsets, num_memories as usize);
            if index >= num_memories {
                return None;
            }
            let def_ptr = vm_memories[index as usize];
            let def = &mut *def_ptr;
            let old_size = def.current_length;
            let old_pages = (old_size / 65536) as u32;
            let new_pages = old_pages.checked_add(delta)?;

            if new_pages > def.maximum_pages {
                return None;
            }

            let new_size = (new_pages as usize) * 65536;

            if delta > 0 {
                // 借助虚拟内存预留，我们只需要 mprotect 即可 commit 更多物理内存
                // 基地址 (def.base) 保持不变，不需要 memcpy。
                let grow_ptr = def.base.add(old_size);
                let grow_size = new_size - old_size;

                if libc::mprotect(
                    grow_ptr as *mut _,
                    grow_size,
                    libc::PROT_READ | libc::PROT_WRITE,
                ) != 0
                {
                    return None;
                }
            }

            def.current_length = new_size;
            Some(old_pages)
        }
    }
}
