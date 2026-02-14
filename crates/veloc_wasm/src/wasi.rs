use crate::Caller;
use crate::instance::VMInstance;
use crate::linker::Linker;
use crate::store::Store;
use futures::executor::block_on;
use std::cell::UnsafeCell;
pub use wasi_common::WasiCtx;
use wiggle::{BorrowHandle, GuestError, GuestMemory, Region};

struct WasiMemory<'a>(&'a mut [u8]);

unsafe impl<'a> GuestMemory for WasiMemory<'a> {
    fn base(&self) -> &[UnsafeCell<u8>] {
        unsafe {
            std::slice::from_raw_parts(self.0.as_ptr() as *const UnsafeCell<u8>, self.0.len())
        }
    }
    fn has_outstanding_borrows(&self) -> bool {
        false
    }
    fn is_mut_borrowed(&self, _: Region) -> bool {
        false
    }
    fn is_shared_borrowed(&self, _: Region) -> bool {
        false
    }
    fn mut_borrow(&self, _: Region) -> Result<BorrowHandle, GuestError> {
        Ok(BorrowHandle(0))
    }
    fn shared_borrow(&self, _: Region) -> Result<BorrowHandle, GuestError> {
        Ok(BorrowHandle(0))
    }
    fn mut_unborrow(&self, _: BorrowHandle) {}
    fn shared_unborrow(&self, _: BorrowHandle) {}
}

macro_rules! call_wasi_func {
    (proc_exit, $result:expr, $ret:ty) => {
        match block_on($result) {
            Ok(_) => (),
            Err(e) => {
                if let Some(exit) = e.downcast_ref::<wasi_common::I32Exit>() {
                    std::process::exit(exit.0);
                }
            }
        }
    };
    ($fname:ident, $result:expr, $ret:ty) => {
        match block_on($result) {
            Ok(_) => 0,
            Err(_) => 1,
        }
    };
}

macro_rules! add_funcs_to_linker {
    (
        $(
            $( #[$docs:meta] )*
            fn $fname:ident ($( $arg:ident : $typ:ty ),* $(,)?) -> $ret:tt
        );+ $(;)?
    ) => {
        pub fn add_to_linker(linker: &mut Linker, store: &mut Store) -> crate::error::Result<()> {
            const WASI: &'static str = "wasi_snapshot_preview1";
            let wasi_ctx = store.wasi_ctx.clone().expect("WASI context not set in store");
            $(
                let ctx = wasi_ctx.clone();
                linker.func_wrap(
                    store,
                    WASI, stringify!($fname),
                    move |caller: Caller, $($arg : $typ,)*| -> $ret {
                        let mut wasi_ctx = ctx.lock().unwrap();
                        let instance = unsafe { VMInstance::from_vmctx(caller.vmctx()) };
                        let memory_slice = instance.get_memory_mut(0).expect("Failed to get memory 0");
                        let mut memory = WasiMemory(memory_slice);

                        let result = async {
                            wasi_common::snapshots::preview_1::wasi_snapshot_preview1::$fname(&mut *wasi_ctx, &mut memory, $($arg,)*).await
                        };
                        call_wasi_func!($fname, result, $ret)
                    }
                );
            )*
            Ok(())
        }
    }
}

add_funcs_to_linker!(
    fn args_get(argv: i32, argv_buf: i32) -> i32;
    fn args_sizes_get(offset0: i32, offset1: i32) -> i32;
    fn environ_get(environ: i32, environ_buf: i32) -> i32;
    fn environ_sizes_get(offset0: i32, offset1: i32) -> i32;
    fn clock_res_get(id: i32, offset0: i32) -> i32;
    fn clock_time_get(id: i32, precision: i64, offset0: i32) -> i32;
    fn fd_advise(fd: i32, offset: i64, len: i64, advice: i32) -> i32;
    fn fd_allocate(fd: i32, offset: i64, len: i64) -> i32;
    fn fd_close(fd: i32) -> i32;
    fn fd_datasync(fd: i32) -> i32;
    fn fd_fdstat_get(fd: i32, offset0: i32) -> i32;
    fn fd_fdstat_set_flags(fd: i32, flags: i32) -> i32;
    fn fd_fdstat_set_rights(fd: i32, fs_rights_base: i64, fs_rights_inheriting: i64) -> i32;
    fn fd_filestat_get(fd: i32, offset0: i32) -> i32;
    fn fd_filestat_set_size(fd: i32, size: i64) -> i32;
    fn fd_filestat_set_times(fd: i32, atim: i64, mtim: i64, fst_flags: i32) -> i32;
    fn fd_pread(fd: i32, iov_buf: i32, iov_buf_len: i32, offset: i64, offset0: i32) -> i32;
    fn fd_prestat_get(fd: i32, offset0: i32) -> i32;
    fn fd_prestat_dir_name(fd: i32, path: i32, path_len: i32) -> i32;
    fn fd_pwrite(fd: i32, ciov_buf: i32, ciov_buf_len: i32, offset: i64, offset0: i32) -> i32;
    fn fd_read(fd: i32, iov_buf: i32, iov_buf_len: i32, offset1: i32) -> i32;
    fn fd_readdir(fd: i32, buf: i32, buf_len: i32, cookie: i64, offset0: i32) -> i32;
    fn fd_renumber(fd: i32, to: i32) -> i32;
    fn fd_seek(fd: i32, offset: i64, whence: i32, offset0: i32) -> i32;
    fn fd_sync(fd: i32) -> i32;
    fn fd_tell(fd: i32, offset0: i32) -> i32;
    fn fd_write(fd: i32, ciov_buf: i32, ciov_buf_len: i32, offset0: i32) -> i32;
    fn path_create_directory(fd: i32, offset: i32, length: i32) -> i32;
    fn path_filestat_get(fd: i32, flags: i32, offset: i32, length: i32, offset0: i32) -> i32;
    fn path_filestat_set_times(
        fd: i32,
        flags: i32,
        offset: i32,
        length: i32,
        atim: i64,
        mtim: i64,
        fst_flags: i32,
    ) -> i32;
    fn path_link(
        old_fd: i32,
        old_flags: i32,
        old_offset: i32,
        old_length: i32,
        new_fd: i32,
        new_offset: i32,
        new_length: i32,
    ) -> i32;
    fn path_open(
        fd: i32,
        dirflags: i32,
        offset: i32,
        length: i32,
        oflags: i32,
        fs_rights_base: i64,
        fdflags: i64,
        fs_rights_inheriting: i32,
        offset0: i32,
    ) -> i32;
    fn path_readlink(
        fd: i32,
        offset: i32,
        length: i32,
        buf: i32,
        buf_len: i32,
        offset0: i32,
    ) -> i32;
    fn path_remove_directory(fd: i32, offset: i32, length: i32) -> i32;
    fn path_rename(
        fd: i32,
        old_offset: i32,
        old_length: i32,
        new_fd: i32,
        new_offset: i32,
        new_length: i32,
    ) -> i32;
    fn path_symlink(
        old_offset: i32,
        old_length: i32,
        fd: i32,
        new_offset: i32,
        new_length: i32,
    ) -> i32;
    fn path_unlink_file(fd: i32, offset: i32, length: i32) -> i32;
    fn poll_oneoff(in_: i32, out: i32, nsubscriptions: i32, offset0: i32) -> i32;
    fn proc_exit(rval: i32) -> ();
    fn proc_raise(sig: i32) -> i32;
    fn sched_yield() -> i32;
    fn random_get(buf: i32, buf_len: i32) -> i32;
    fn sock_accept(fd: i32, flags: i32, offset0: i32) -> i32;
    fn sock_recv(
        fd: i32,
        iov_buf: i32,
        iov_buf_len: i32,
        ri_flags: i32,
        offset0: i32,
        offset1: i32,
    ) -> i32;
    fn sock_send(fd: i32, ciov_buf: i32, ciov_buf_len: i32, si_flags: i32, offset0: i32) -> i32;
    fn sock_shutdown(fd: i32, how: i32) -> i32;
);

pub fn default_wasi_ctx() -> WasiCtx {
    wasi_cap_std_sync::WasiCtxBuilder::new()
        .inherit_stdout()
        .inherit_stderr()
        .inherit_stdin()
        .build()
}
