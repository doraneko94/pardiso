//! Wrappers for [PARDISO].
//! 
//! [PARDISO]: https://www.pardiso-project.org/

extern crate pardiso_sys as ffi;
extern crate libc;
extern crate num_complex as num;
use libc::{c_void};

#[allow(non_camel_case_types)]
pub type c64 = num::Complex<f64>;

pub trait Bit64{}

impl Bit64 for f64 {}
impl Bit64 for c64 {}

#[inline]
pub unsafe fn pardiso<T: Bit64>(
    pt: &mut [isize], 
    maxfct: i32,
    mnum: i32,
    mtype: i32,
    phase: i32,
    n: i32,
    a: &[T],
    ia: &[i32],
    ja: &[i32],
    perm: &mut [i32],
    nrhs: i32,
    iparm: &mut [i32],
    msglvl: i32,
    b: &mut [T],
    x: &mut [T],
    error: &mut i32,
) {
    ffi::pardiso(
        pt.as_mut_ptr() as *mut _ as *mut c_void,
        &maxfct,
        &mnum,
        &mtype,
        &phase,
        &n,
        a.as_ptr() as *const _ as *const c_void,
        ia.as_ptr(),
        ja.as_ptr(),
        perm.as_mut_ptr(),
        &nrhs,
        iparm.as_mut_ptr(),
        &msglvl,
        b.as_mut_ptr() as *mut _ as *mut c_void,
        x.as_mut_ptr() as *mut _ as *mut c_void,
        error,
    )
}

#[inline]
pub unsafe fn pardisoinit(
    pt: &mut [isize],
    mtype: i32,
    iparm: &mut [i32],
) {
    ffi::pardisoinit(
        pt.as_mut_ptr() as *mut _ as *mut c_void,
        &mtype,
        iparm.as_mut_ptr(),
    )
}

#[inline]
pub unsafe fn pardiso_64<T: Bit64>(
    pt: &mut [isize], 
    maxfct: i64,
    mnum: i64,
    mtype: i64,
    phase: i64,
    n: i64,
    a: &[T],
    ia: &[i64],
    ja: &[i64],
    perm: &mut [i64],
    nrhs: i64,
    iparm: &mut [i64],
    msglvl: i64,
    b: &mut [T],
    x: &mut [T],
    error: &mut i64,
) {
    ffi::pardiso_64(
        pt.as_mut_ptr() as *mut _ as *mut c_void,
        &maxfct,
        &mnum,
        &mtype,
        &phase,
        &n,
        a.as_ptr() as *const _ as *const c_void,
        ia.as_ptr(),
        ja.as_ptr(),
        perm.as_mut_ptr(),
        &nrhs,
        iparm.as_mut_ptr(),
        &msglvl,
        b.as_mut_ptr() as *mut _ as *mut c_void,
        x.as_mut_ptr() as *mut _ as *mut c_void,
        error,
    )
}

#[inline]
pub unsafe fn pardiso_handle_store_64(
    pt: &mut [isize],
    dirname: &str,
    error: &mut i64,
) {
    let cstr = std::ffi::CString::new(dirname).unwrap();
    ffi::pardiso_handle_store_64(
        pt.as_mut_ptr() as *mut _ as *mut c_void,
        cstr.as_ptr() as *const i8,
        error,
    )
}

#[inline]
pub unsafe fn pardiso_handle_restore_64(
    pt: &mut [isize],
    dirname: &str,
    error: &mut i64,
) {
    let cstr = std::ffi::CString::new(dirname).unwrap();
    ffi::pardiso_handle_restore_64(
        pt.as_mut_ptr() as *mut _ as *mut c_void,
        cstr.as_ptr() as *const i8,
        error,
    )
}

#[inline]
pub unsafe fn pardiso_handle_delete_64(
    dirname: &str,
    error: &mut i64,
) {
    let cstr = std::ffi::CString::new(dirname).unwrap();
    ffi::pardiso_handle_delete_64(
        cstr.as_ptr() as *const i8,
        error,
    )
}

#[inline]
pub unsafe fn pardiso_handle_store(
    pt: &mut [isize],
    dirname: &str,
    error: &mut i32,
) {
    let cstr = std::ffi::CString::new(dirname).unwrap();
    ffi::pardiso_handle_store(
        pt.as_mut_ptr() as *mut _ as *mut c_void,
        cstr.as_ptr() as *const i8,
        error,
    )
}

#[inline]
pub unsafe fn pardiso_handle_restore(
    pt: &mut [isize],
    dirname: &str,
    error: &mut i32,
) {
    let cstr = std::ffi::CString::new(dirname).unwrap();
    ffi::pardiso_handle_restore(
        pt.as_mut_ptr() as *mut _ as *mut c_void,
        cstr.as_ptr() as *const i8,
        error,
    )
}

#[inline]
pub unsafe fn pardiso_handle_delete(
    dirname: &str,
    error: &mut i32,
) {
    let cstr = std::ffi::CString::new(dirname).unwrap();
    ffi::pardiso_handle_delete(
        cstr.as_ptr() as *const i8,
        error,
    )
}

#[inline]
pub unsafe fn mkl_pardiso_pivot<T: Bit64>(
    aii: T,
    bii: &mut T,
    eps: T,
) -> i32
{
    ffi::mkl_pardiso_pivot(
        &aii as *const _ as *const c_void,
        bii as *mut _ as *mut c_void,
        &eps as *const _ as *const c_void,
    )
}

#[inline]
pub unsafe fn pardiso_getdiag<T: Bit64>(
    pt: &mut [isize], 
    df: &mut [T],
    da: &mut [T],
    mnum: i32,
    error: &mut i32,
) {
    ffi::pardiso_getdiag(
        pt.as_mut_ptr() as *mut _ as *mut c_void,
        df.as_mut_ptr() as *mut _ as *mut c_void,
        da.as_mut_ptr() as *mut _ as *mut c_void,
        &mnum,
        error,
    )
}

/// haven't tested yet.
#[inline]
pub unsafe fn pardiso_export<T: Bit64>(
    pt: &mut [isize], 
    values: &mut [T],
    rows: &mut [i32],
    columns: &mut [i32],
    step: i32,
    iparm: &[i32],
    error: &mut i32,
) {
    ffi::pardiso_export(
        pt.as_mut_ptr() as *mut _ as *mut c_void,
        values.as_mut_ptr() as *mut _ as *mut c_void,
        rows.as_mut_ptr(),
        columns.as_mut_ptr(),
        &step,
        iparm.as_ptr(),
        error,
    )
}