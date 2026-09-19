//! A small synchronous WASI Preview 1 host.
//!
//! Veloc only needs a host adapter here, not Wasmtime's runtime, async bridge,
//! capability filesystem, or WITX code generator. Keeping the boundary in this
//! crate avoids pulling an entire second compiler/runtime dependency graph into
//! every `veloc-wasm` build. Unsupported calls are still registered with their
//! standard signatures and report `ENOSYS`, so modules fail at the call site
//! rather than during instantiation.

use std::io::{Read, Write};
use std::sync::OnceLock;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::Caller;
use crate::instance::VMInstance;
use crate::linker::Linker;
use crate::store::Store;

const WASI: &str = "wasi_snapshot_preview1";

const SUCCESS: i32 = 0;
const ERRNO_BADF: i32 = 8;
const ERRNO_FAULT: i32 = 21;
const ERRNO_INVAL: i32 = 28;
const ERRNO_IO: i32 = 29;
const ERRNO_NOSYS: i32 = 52;
const ERRNO_SPIPE: i32 = 70;

const FILETYPE_CHARACTER_DEVICE: u8 = 2;

/// Process state exposed to WASI guests.
///
/// Arguments and environment entries are stored without trailing NUL bytes;
/// the Preview 1 ABI adapter adds those bytes when copying into guest memory.
#[derive(Debug, Default)]
pub struct WasiCtx {
    args: Vec<Vec<u8>>,
    env: Vec<Vec<u8>>,
}

impl WasiCtx {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_args(mut self, args: impl IntoIterator<Item = impl Into<Vec<u8>>>) -> Self {
        self.args = args.into_iter().map(Into::into).collect();
        self
    }

    pub fn with_env(
        mut self,
        env: impl IntoIterator<Item = (impl AsRef<str>, impl AsRef<str>)>,
    ) -> Self {
        self.env = env
            .into_iter()
            .map(|(key, value)| format!("{}={}", key.as_ref(), value.as_ref()).into_bytes())
            .collect();
        self
    }
}

#[inline]
fn memory<'a>(caller: &'a Caller<'_>) -> Option<&'a mut [u8]> {
    // A host call has exclusive access to its instance. The returned slice is
    // used only for the duration of that call.
    unsafe { VMInstance::from_vmctx(caller.vmctx()).get_memory_mut(0) }
}

#[inline]
fn range(memory: &[u8], ptr: i32, len: usize) -> Option<std::ops::Range<usize>> {
    let start = usize::try_from(ptr).ok()?;
    let end = start.checked_add(len)?;
    (end <= memory.len()).then_some(start..end)
}

#[inline]
fn read_u32(memory: &[u8], ptr: i32) -> Option<u32> {
    let bytes: [u8; 4] = memory.get(range(memory, ptr, 4)?)?.try_into().ok()?;
    Some(u32::from_le_bytes(bytes))
}

#[inline]
fn write_bytes(memory: &mut [u8], ptr: i32, bytes: &[u8]) -> bool {
    let Some(target) = range(memory, ptr, bytes.len()) else {
        return false;
    };
    memory[target].copy_from_slice(bytes);
    true
}

#[inline]
fn write_u32(memory: &mut [u8], ptr: i32, value: u32) -> bool {
    write_bytes(memory, ptr, &value.to_le_bytes())
}

#[inline]
fn write_u64(memory: &mut [u8], ptr: i32, value: u64) -> bool {
    write_bytes(memory, ptr, &value.to_le_bytes())
}

fn write_string_table(memory: &mut [u8], table: i32, buffer: i32, values: &[Vec<u8>]) -> i32 {
    let mut cursor = match u32::try_from(buffer) {
        Ok(value) => value,
        Err(_) => return ERRNO_FAULT,
    };
    for (index, value) in values.iter().enumerate() {
        let slot = match i32::try_from(index)
            .ok()
            .and_then(|index| index.checked_mul(4))
            .and_then(|offset| table.checked_add(offset))
        {
            Some(slot) => slot,
            None => return ERRNO_FAULT,
        };
        if !write_u32(memory, slot, cursor) || !write_bytes(memory, cursor as i32, value) {
            return ERRNO_FAULT;
        }
        cursor = match cursor.checked_add(value.len() as u32) {
            Some(value) => value,
            None => return ERRNO_FAULT,
        };
        if !write_bytes(memory, cursor as i32, &[0]) {
            return ERRNO_FAULT;
        }
        cursor += 1;
    }
    SUCCESS
}

fn table_size(memory: &mut [u8], count_ptr: i32, bytes_ptr: i32, values: &[Vec<u8>]) -> i32 {
    let Some(bytes) = values.iter().try_fold(0u32, |total, value| {
        total.checked_add(u32::try_from(value.len()).ok()?.checked_add(1)?)
    }) else {
        return ERRNO_INVAL;
    };
    if write_u32(memory, count_ptr, values.len() as u32) && write_u32(memory, bytes_ptr, bytes) {
        SUCCESS
    } else {
        ERRNO_FAULT
    }
}

fn clock_nanos(id: i32) -> Option<u64> {
    match id {
        0 => SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .ok()
            .and_then(|duration| u64::try_from(duration.as_nanos()).ok()),
        1..=3 => {
            static START: OnceLock<Instant> = OnceLock::new();
            u64::try_from(START.get_or_init(Instant::now).elapsed().as_nanos()).ok()
        }
        _ => None,
    }
}

fn fd_write(caller: Caller<'_>, fd: i32, iovs: i32, iovs_len: i32, written: i32) -> i32 {
    let Some(memory) = memory(&caller) else {
        return ERRNO_FAULT;
    };
    let Ok(count) = usize::try_from(iovs_len) else {
        return ERRNO_INVAL;
    };
    let mut total = 0u32;
    let result = match fd {
        1 => {
            let mut output = std::io::stdout().lock();
            write_iovecs(memory, iovs, count, &mut output, &mut total)
        }
        2 => {
            let mut output = std::io::stderr().lock();
            write_iovecs(memory, iovs, count, &mut output, &mut total)
        }
        _ => return ERRNO_BADF,
    };
    if result.is_err() {
        return ERRNO_IO;
    }
    if !write_u32(memory, written, total) {
        return ERRNO_FAULT;
    }
    SUCCESS
}

fn write_iovecs(
    memory: &[u8],
    iovs: i32,
    count: usize,
    output: &mut impl Write,
    total: &mut u32,
) -> std::io::Result<()> {
    for index in 0..count {
        let Some(slot) = i32::try_from(index)
            .ok()
            .and_then(|index| index.checked_mul(8))
            .and_then(|offset| iovs.checked_add(offset))
        else {
            return Err(std::io::ErrorKind::InvalidInput.into());
        };
        let (Some(ptr), Some(len)) = (read_u32(memory, slot), read_u32(memory, slot + 4)) else {
            return Err(std::io::ErrorKind::InvalidInput.into());
        };
        let Some(bytes) =
            range(memory, ptr as i32, len as usize).and_then(|range| memory.get(range))
        else {
            return Err(std::io::ErrorKind::InvalidInput.into());
        };
        output.write_all(bytes)?;
        *total = total
            .checked_add(len)
            .ok_or(std::io::ErrorKind::InvalidInput)?;
    }
    output.flush()
}

fn fd_read(caller: Caller<'_>, fd: i32, iovs: i32, iovs_len: i32, read: i32) -> i32 {
    if fd != 0 {
        return ERRNO_BADF;
    }
    let Some(memory) = memory(&caller) else {
        return ERRNO_FAULT;
    };
    let Ok(count) = usize::try_from(iovs_len) else {
        return ERRNO_INVAL;
    };
    let mut input = std::io::stdin().lock();
    let mut total = 0u32;
    for index in 0..count {
        let Some(slot) = i32::try_from(index)
            .ok()
            .and_then(|index| index.checked_mul(8))
            .and_then(|offset| iovs.checked_add(offset))
        else {
            return ERRNO_FAULT;
        };
        let (Some(ptr), Some(len)) = (read_u32(memory, slot), read_u32(memory, slot + 4)) else {
            return ERRNO_FAULT;
        };
        let Some(buffer) = range(memory, ptr as i32, len as usize) else {
            return ERRNO_FAULT;
        };
        let count = match input.read(&mut memory[buffer]) {
            Ok(count) => count,
            Err(_) => return ERRNO_IO,
        };
        total += count as u32;
        if count != len as usize {
            break;
        }
    }
    if write_u32(memory, read, total) {
        SUCCESS
    } else {
        ERRNO_FAULT
    }
}

macro_rules! stub_funcs {
    ($linker:ident, $store:ident; $(fn $name:ident($($arg:ident: $ty:ty),*);)*) => {$({
        #[allow(unused_variables)]
        $linker.func_wrap(
            $store,
            WASI,
            stringify!($name),
            move |_caller: Caller<'_>, $($arg: $ty),*| -> i32 { ERRNO_NOSYS },
        );
    })*};
}

pub fn add_to_linker(linker: &mut Linker, store: &mut Store) -> crate::error::Result<()> {
    let ctx = store
        .wasi_ctx
        .clone()
        .ok_or_else(|| crate::error::Error::Message("WASI context not set in store".into()))?;

    let args = ctx.clone();
    linker.func_wrap(
        store,
        WASI,
        "args_get",
        move |caller: Caller<'_>, argv: i32, buffer: i32| {
            let Some(memory) = memory(&caller) else {
                return ERRNO_FAULT;
            };
            write_string_table(memory, argv, buffer, &args.args)
        },
    );
    let args = ctx.clone();
    linker.func_wrap(
        store,
        WASI,
        "args_sizes_get",
        move |caller: Caller<'_>, count: i32, bytes: i32| {
            let Some(memory) = memory(&caller) else {
                return ERRNO_FAULT;
            };
            table_size(memory, count, bytes, &args.args)
        },
    );
    let env = ctx.clone();
    linker.func_wrap(
        store,
        WASI,
        "environ_get",
        move |caller: Caller<'_>, table: i32, buffer: i32| {
            let Some(memory) = memory(&caller) else {
                return ERRNO_FAULT;
            };
            write_string_table(memory, table, buffer, &env.env)
        },
    );
    let env = ctx;
    linker.func_wrap(
        store,
        WASI,
        "environ_sizes_get",
        move |caller: Caller<'_>, count: i32, bytes: i32| {
            let Some(memory) = memory(&caller) else {
                return ERRNO_FAULT;
            };
            table_size(memory, count, bytes, &env.env)
        },
    );

    linker.func_wrap(
        store,
        WASI,
        "clock_res_get",
        move |caller: Caller<'_>, id: i32, out: i32| {
            if clock_nanos(id).is_none() {
                return ERRNO_INVAL;
            }
            let Some(memory) = memory(&caller) else {
                return ERRNO_FAULT;
            };
            if write_u64(memory, out, 1) {
                SUCCESS
            } else {
                ERRNO_FAULT
            }
        },
    );
    linker.func_wrap(
        store,
        WASI,
        "clock_time_get",
        move |caller: Caller<'_>, id: i32, _precision: i64, out: i32| {
            let Some(now) = clock_nanos(id) else {
                return ERRNO_INVAL;
            };
            let Some(memory) = memory(&caller) else {
                return ERRNO_FAULT;
            };
            if write_u64(memory, out, now) {
                SUCCESS
            } else {
                ERRNO_FAULT
            }
        },
    );
    linker.func_wrap(
        store,
        WASI,
        "fd_close",
        move |_caller: Caller<'_>, fd: i32| -> i32 {
            if (0..=2).contains(&fd) {
                SUCCESS
            } else {
                ERRNO_BADF
            }
        },
    );
    linker.func_wrap(
        store,
        WASI,
        "fd_fdstat_get",
        move |caller: Caller<'_>, fd: i32, out: i32| {
            if !(0..=2).contains(&fd) {
                return ERRNO_BADF;
            }
            let Some(memory) = memory(&caller) else {
                return ERRNO_FAULT;
            };
            let mut stat = [0u8; 24];
            stat[0] = FILETYPE_CHARACTER_DEVICE;
            stat[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
            stat[16..24].copy_from_slice(&u64::MAX.to_le_bytes());
            if write_bytes(memory, out, &stat) {
                SUCCESS
            } else {
                ERRNO_FAULT
            }
        },
    );
    linker.func_wrap(
        store,
        WASI,
        "fd_seek",
        move |_caller: Caller<'_>, fd: i32, _offset: i64, _whence: i32, _out: i32| -> i32 {
            if (0..=2).contains(&fd) {
                ERRNO_SPIPE
            } else {
                ERRNO_BADF
            }
        },
    );
    linker.func_wrap(
        store,
        WASI,
        "fd_tell",
        move |_caller: Caller<'_>, fd: i32, _out: i32| -> i32 {
            if (0..=2).contains(&fd) {
                ERRNO_SPIPE
            } else {
                ERRNO_BADF
            }
        },
    );
    linker.func_wrap(store, WASI, "fd_write", fd_write);
    linker.func_wrap(store, WASI, "fd_read", fd_read);
    linker.func_wrap(
        store,
        WASI,
        "fd_datasync",
        move |_caller: Caller<'_>, fd: i32| -> i32 {
            if (0..=2).contains(&fd) {
                SUCCESS
            } else {
                ERRNO_BADF
            }
        },
    );
    linker.func_wrap(
        store,
        WASI,
        "fd_sync",
        move |_caller: Caller<'_>, fd: i32| -> i32 {
            if (0..=2).contains(&fd) {
                SUCCESS
            } else {
                ERRNO_BADF
            }
        },
    );
    linker.func_wrap(
        store,
        WASI,
        "random_get",
        move |caller: Caller<'_>, ptr: i32, len: i32| {
            let Ok(len) = usize::try_from(len) else {
                return ERRNO_INVAL;
            };
            let Some(memory) = memory(&caller) else {
                return ERRNO_FAULT;
            };
            let Some(buffer) = range(memory, ptr, len) else {
                return ERRNO_FAULT;
            };
            match std::fs::File::open("/dev/urandom")
                .and_then(|mut file| file.read_exact(&mut memory[buffer]))
            {
                Ok(()) => SUCCESS,
                Err(_) => ERRNO_IO,
            }
        },
    );
    linker.func_wrap(store, WASI, "sched_yield", move || -> i32 {
        std::thread::yield_now();
        SUCCESS
    });
    linker.func_wrap(store, WASI, "proc_exit", move |code: i32| -> () {
        std::process::exit(code);
    });

    stub_funcs!(linker, store;
        fn fd_advise(fd: i32, offset: i64, len: i64, advice: i32);
        fn fd_allocate(fd: i32, offset: i64, len: i64);
        fn fd_fdstat_set_flags(fd: i32, flags: i32);
        fn fd_fdstat_set_rights(fd: i32, base: i64, inheriting: i64);
        fn fd_filestat_get(fd: i32, out: i32);
        fn fd_filestat_set_size(fd: i32, size: i64);
        fn fd_filestat_set_times(fd: i32, atim: i64, mtim: i64, flags: i32);
        fn fd_pread(fd: i32, iovs: i32, iovs_len: i32, offset: i64, read: i32);
        fn fd_prestat_get(fd: i32, out: i32);
        fn fd_prestat_dir_name(fd: i32, path: i32, path_len: i32);
        fn fd_pwrite(fd: i32, iovs: i32, iovs_len: i32, offset: i64, written: i32);
        fn fd_readdir(fd: i32, buffer: i32, buffer_len: i32, cookie: i64, used: i32);
        fn fd_renumber(fd: i32, to: i32);
        fn path_create_directory(fd: i32, path: i32, path_len: i32);
        fn path_filestat_get(fd: i32, flags: i32, path: i32, path_len: i32, out: i32);
        fn path_filestat_set_times(fd: i32, flags: i32, path: i32, path_len: i32, atim: i64, mtim: i64, fst_flags: i32);
        fn path_link(old_fd: i32, old_flags: i32, old_path: i32, old_path_len: i32, new_fd: i32, new_path: i32, new_path_len: i32);
        fn path_open(fd: i32, dirflags: i32, path: i32, path_len: i32, oflags: i32, rights_base: i64, rights_inheriting: i64, fdflags: i32, opened_fd: i32);
        fn path_readlink(fd: i32, path: i32, path_len: i32, buffer: i32, buffer_len: i32, used: i32);
        fn path_remove_directory(fd: i32, path: i32, path_len: i32);
        fn path_rename(fd: i32, old_path: i32, old_path_len: i32, new_fd: i32, new_path: i32, new_path_len: i32);
        fn path_symlink(old_path: i32, old_path_len: i32, fd: i32, new_path: i32, new_path_len: i32);
        fn path_unlink_file(fd: i32, path: i32, path_len: i32);
        fn poll_oneoff(input: i32, output: i32, subscriptions: i32, events: i32);
        fn proc_raise(signal: i32);
        fn sock_accept(fd: i32, flags: i32, out: i32);
        fn sock_recv(fd: i32, iovs: i32, iovs_len: i32, flags: i32, read: i32, out_flags: i32);
        fn sock_send(fd: i32, iovs: i32, iovs_len: i32, flags: i32, written: i32);
        fn sock_shutdown(fd: i32, how: i32);
    );

    Ok(())
}

pub fn default_wasi_ctx() -> WasiCtx {
    WasiCtx::new()
}
