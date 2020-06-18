// ref. pardiso_unsym_diag_pivot_c.c
// in Intel(R) MKL PARDISO C example
use pardiso::{pardiso, pardiso_getdiag};
use ndarray::*;
extern crate pardiso_src;

fn main() {
    /* Matrix data. */
    let n: i32 = 5;
    let ia: Array1<i32> = Array::from(vec![1, 4, 6, 9, 12, 14]);
    let ja: Array1<i32> = Array::from(vec![
        1, 2,    4,
        1, 2,
              3, 4, 5,
        1,    3, 4,
           2,       5,
    ]);
    let a1: Array1<f64> = Array::from(vec![
         21.0,  -1.0,         -3.0,
         -2.0,  35.0,
                       44.0,   6.0,   4.0,
         -4.0,          2.0,  57.0,
                 8.0,               -65.0,
    ]);
    let a2: Array1<f64> = Array::from(vec![
       -21.0,  -1.0,         -3.0,
        -2.0, -35.0,
                     -44.0,   6.0,   4.0,
        -4.0,          2.0, -57.0,
                8.0,                65.0,
    ]);

    let mtype = 11;  // Real symmetric matrix
    /* RHS and solution vectors. */
    let nrhs = 1;    // Number of right hand side.
    let mut df: Array1<f64> = Array::zeros(n as usize);
    let mut da: Array1<f64> = Array::zeros(n as usize);
    let mut er_d = 0;
    /* Internal solver memory pointer pt (C lang), */
    /* 32-bit: int pt[64]; 64-bit: long int pt[64] */
    /* or void *pt[64] should be OK on both architectures */
    let mut pt: Array1<isize> = Array::zeros(64);
    /* Pardiso control parameters. */
    let mut iparm: Array1<i32> = Array::zeros(64);
    let maxfct = 2;     // Maximum number of numerical factorizations.
    let mnum = 1;       // Which factorization to use.
    let msglvl = 0;     // Print stastical information in file
    let mut error = 0;  // Initialize error flag
    /* Auxiliary variables. */
    let mut ddum1: Array1<f64> = Array::zeros(n as usize);  // f64 dummy 1
    let mut ddum2: Array1<f64> = Array::zeros(n as usize);  // f64 dummy 2
    let mut idum: Array1<i32> = Array::zeros(n as usize);   // i32 dummy.
/* -------------------------------------------------------------------- */
/* .. Setup Pardiso control parameters. */
/* -------------------------------------------------------------------- */
    iparm[0] = 1;         // No solver default
    iparm[1] = 2;         // Fill-in reordering from METIS
    iparm[3] = 0;         // No iterative-direct algorithm
    iparm[4] = 0;         // No user fill-in reducing permutation
    iparm[5] = 0;         // Write solution into x
    iparm[6] = 0;         // Not in use
    iparm[7] = 2;         // Max numbers of iterative refinement steps
    iparm[8] = 0;         // Not in use
    iparm[9] = 13;        // Perturb the pivot elements with 1E-13
    iparm[10] = 1;        // Use nonsymmetric permutation and scaling MPS
    iparm[11] = 0;        // Not in use
    iparm[12] = 0;        // Maximum weighted matching algorithm is switched-off (default for symmetric). Try iparm[12] = 1 in case of inappropriate accuracy
    iparm[13] = 0;        // Output: Number of perturbed pivots
    iparm[14] = 0;        // Not in use
    iparm[15] = 0;        // Not in use
    iparm[16] = 0;        // Not in use
    iparm[17] = -1;       // Output: Number of nonzeros in the factor LU
    iparm[18] = -1;       // Output: Mflops for LU factorization
    iparm[19] = 0;        // Output: Numbers of CG Iterations
    iparm[55] = 1;        // Pivoting control -> You must set to 1 when you use pardiso_getdiag
/* -------------------------------------------------------------------- */
/* .. Reordering and Symbolic Factorization. This step also allocates */
/* all memory that is necessary for the factorization. */
/* -------------------------------------------------------------------- */
    let phase = 11;
    unsafe {
        pardiso(
            pt.as_slice_memory_order_mut().unwrap(),
            maxfct,
            mnum,
            mtype,
            phase,
            n,
            a1.as_slice_memory_order().unwrap(),
            ia.as_slice_memory_order().unwrap(),
            ja.as_slice_memory_order().unwrap(),
            idum.as_slice_memory_order_mut().unwrap(),
            nrhs,
            iparm.as_slice_memory_order_mut().unwrap(),
            msglvl,
            ddum1.as_slice_memory_order_mut().unwrap(),
            ddum2.as_slice_memory_order_mut().unwrap(),
            &mut error
        );
    }
    if error != 0 {
        panic!("ERROR during symbolic factorization: {}", error);
    }
    println!("Reordering completed ... ");
    println!("Number of nonzeros in factors = {}", iparm[17]);
    println!("Number of factorization MFLOPS = {}", iparm[18]);
/* -------------------------------------------------------------------- */
/* .. Numerical factorization. */
/* -------------------------------------------------------------------- */
    let phase = 22;
    unsafe {
        pardiso(
            pt.as_slice_memory_order_mut().unwrap(),
            maxfct,
            mnum,
            mtype,
            phase,
            n,
            a1.as_slice_memory_order().unwrap(),
            ia.as_slice_memory_order().unwrap(),
            ja.as_slice_memory_order().unwrap(),
            idum.as_slice_memory_order_mut().unwrap(),
            nrhs,
            iparm.as_slice_memory_order_mut().unwrap(),
            msglvl,
            ddum1.as_slice_memory_order_mut().unwrap(),
            ddum2.as_slice_memory_order_mut().unwrap(),
            &mut error
        );
    }
    if error != 0 {
        panic!("ERROR during numerical factorization: {}", error);
    }
    println!("\nFirst Factorization completed ... ");

    let phase = 22;
    let mnum = 2;
    unsafe {
        pardiso(
            pt.as_slice_memory_order_mut().unwrap(),
            maxfct,
            mnum,
            mtype,
            phase,
            n,
            a2.as_slice_memory_order().unwrap(),
            ia.as_slice_memory_order().unwrap(),
            ja.as_slice_memory_order().unwrap(),
            idum.as_slice_memory_order_mut().unwrap(),
            nrhs,
            iparm.as_slice_memory_order_mut().unwrap(),
            msglvl,
            ddum1.as_slice_memory_order_mut().unwrap(),
            ddum2.as_slice_memory_order_mut().unwrap(),
            &mut error
        );
    }
    if error != 0 {
        panic!("ERROR during numerical factorization: {}", error);
    }
    println!("\nSecond Factorization completed ... ");

/* -------------------------------------------------------------------- */
/* .. Back substitution and iterative refinement. */
/* -------------------------------------------------------------------- */
    let mnum = 1;

    unsafe {
        pardiso_getdiag(
            pt.as_slice_memory_order_mut().unwrap(),
            df.as_slice_memory_order_mut().unwrap(),
            da.as_slice_memory_order_mut().unwrap(),
            mnum,
            &mut er_d
        );
    }
    println!("\nFirst");
    for i in 0..(n as usize) {
        println!("d_fact[{}]={:8.3}, d_a={:8.3}", i, df[i], da[i]);
    }

    let mnum = 2;

    unsafe {
        pardiso_getdiag(
            pt.as_slice_memory_order_mut().unwrap(),
            df.as_slice_memory_order_mut().unwrap(),
            da.as_slice_memory_order_mut().unwrap(),
            mnum,
            &mut er_d
        );
    }
    println!("\nSecond");
    for i in 0..(n as usize) {
        println!("d_fact[{}]={:8.3}, d_a={:8.3}", i, df[i], da[i]);
    }
}