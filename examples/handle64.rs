// ref. pardiso_handle_store_restore_c.c
// in Intel(R) MKL PARDISO C example
// (in this example, we will use a real matrix.
//  and `*_64` functions)
use pardiso::*;
use ndarray::*;
extern crate pardiso_src;

fn main() {
    /* Matrix data. */
    let n: i64 = 5;
    let ia: Array1<i64> = Array::from(vec![1, 4, 6, 9, 12, 14]);
    let ja: Array1<i64> = Array::from(vec![
        1, 2,    4,
        1, 2,
              3, 4, 5,
        1,    3, 4,
           2,       5,
    ]);
    let a: Array1<f64> = Array::from(vec![
         1.0,-1.0,     -3.0,
        -2.0, 5.0,
                   4.0, 6.0, 4.0,
        -4.0,      2.0, 7.0,
              8.0,          -5.0,
    ]);
    let mtype: i64 = 11;    // Real unsymmetric matrix
    let nrhs: i64 = 1;  // Number of right hand sides.

    /* Internal solver memory pointer pt (C lang), */
    /* 32-bit: int pt[64]; 64-bit: long int pt[64] */
    /* or void *pt[64] should be OK on both architectures */
    let mut pt: Array1<isize> = Array::zeros(64);
    /* Pardiso control parameters. */
    let mut iparm: Array1<i64> = Array1::zeros(64);
    let maxfct: i64 = 1;
    let mnum: i64 = 1;
    let msglvl: i64 = 0;
    let mut error: i64 = 0;

    let ddum0: Array1<f64> = Array::zeros(n as usize);      // f64 dummy 0
    let mut ddum1: Array1<f64> = Array::zeros(n as usize);  // f64 dummy 1
    let mut ddum2: Array1<f64> = Array::zeros(n as usize);  // f64 dummy 2
    let mut idum: Array1<i64> = Array::zeros(n as usize);   // i64 dummy.
/* -------------------------------------------------------------------- */
/* .. Setup Pardiso control parameters. */
/* -------------------------------------------------------------------- */    
    iparm[0] = 1;   // No solver default
    iparm[1] = 2;   // Fill-in reordering from METIS
    iparm[7] = 2;   // Max numbers of iterative refinement steps
    iparm[9] = 13;  // Perturb the pivot elements with 1E-13
    iparm[10] = 1;  // Use nonsymmetric permutation and scaling MPS
    iparm[12] = 1;  // Maximum weighted matching algorithm is switched-on (default for non-symmetric)
    iparm[17] = -1; // Output: Number of nonzeros in the factor LU
    iparm[18] = -1; // Output: Mflops for LU factorization
/* -------------------------------------------------------------------- */
/* ..  Reordering and Symbolic Factorization.  This step also allocates */
/*     all memory that is necessary for the factorization.              */
/* -------------------------------------------------------------------- */
    let phase = 11;
    unsafe {
        pardiso_64(
            pt.as_slice_memory_order_mut().unwrap(),
            maxfct,
            mnum,
            mtype,
            phase,
            n,
            a.as_slice_memory_order().unwrap(),
            ia.as_slice_memory_order().unwrap(),
            ja.as_slice_memory_order().unwrap(),
            idum.as_slice_memory_order_mut().unwrap(),
            nrhs,
            iparm.as_slice_memory_order_mut().unwrap(),
            msglvl,
            ddum1.as_slice_memory_order_mut().unwrap(),
            ddum2.as_slice_memory_order_mut().unwrap(),
            &mut error
        )
    }
    if error != 0 {
        panic!("ERROR during symbolic factorization: {}", error);
    }
    println!("Reordering completed ... ");
    println!("Number of nonzeros in factors = {}", iparm[17]);
    println!("Number of factorization MFLOPS = {}", iparm[18]);

/* -------------------------------------------------------------------- */
/* ..  storing current handle in work folder, deallocate it and         */
/*     restoring new handle                                             */
/* -------------------------------------------------------------------- */
    unsafe { pardiso_handle_store_64(pt.as_slice_memory_order_mut().unwrap(), "", &mut error); }
    if error != 0 {
        panic!("ERROR during storing PARDISO structures to a file: {}", error);
    }
    println!("\nStoring completed ... ");

    let phase = -1; // Release internal memory
    unsafe {
        pardiso_64(
            pt.as_slice_memory_order_mut().unwrap(),
            maxfct,
            mnum,
            mtype,
            phase,
            n,
            ddum0.as_slice_memory_order().unwrap(),
            ia.as_slice_memory_order().unwrap(),
            ja.as_slice_memory_order().unwrap(),
            idum.as_slice_memory_order_mut().unwrap(),
            nrhs,
            iparm.as_slice_memory_order_mut().unwrap(),
            msglvl,
            ddum1.as_slice_memory_order_mut().unwrap(),
            ddum2.as_slice_memory_order_mut().unwrap(),
            &mut error
        )
    }

    let mut pt_new: Array1<isize> = Array::zeros(64);

    unsafe { pardiso_handle_restore_64(pt_new.as_slice_memory_order_mut().unwrap(), "", &mut error); }
    if error != 0 {
        panic!("ERROR during restoring PARDISO structures from a file: {}", error);
    }
    println!("Restoring completed ... ");

    unsafe { pardiso_handle_delete_64("", &mut error); }
    if error != 0 {
        panic!("ERROR during deleting a file of PARDISO structures: {}", error);
    }
    println!("Deleting completed ... ");

/* -------------------------------------------------------------------- */
/* ..  Numerical factorization.                                         */
/* -------------------------------------------------------------------- */
    let phase = 22;
    unsafe {
        pardiso_64(
            pt_new.as_slice_memory_order_mut().unwrap(),
            maxfct,
            mnum,
            mtype,
            phase,
            n,
            a.as_slice_memory_order().unwrap(),
            ia.as_slice_memory_order().unwrap(),
            ja.as_slice_memory_order().unwrap(),
            idum.as_slice_memory_order_mut().unwrap(),
            nrhs,
            iparm.as_slice_memory_order_mut().unwrap(),
            msglvl,
            ddum1.as_slice_memory_order_mut().unwrap(),
            ddum2.as_slice_memory_order_mut().unwrap(),
            &mut error
        )
    }
    if error != 0 {
        panic!("ERROR during numerical factorization: {}", error);
    }
    println!("\nFactorization completed ... ");
}