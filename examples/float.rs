// ref. pardiso_unsym_c.c
// in Intel(R) MKL PARDISO C example
use pardiso::pardiso;
use ndarray::*;
use ndarray_linalg::*;
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
    let a: Array1<f64> = Array::from(vec![
         1.0,-1.0,     -3.0,
        -2.0, 5.0,
                   4.0, 6.0, 4.0,
        -4.0,      2.0, 7.0,
              8.0,          -5.0,
    ]);
    let mtype: i32 = 11;    // Real unsymmetric matrix
    /* RHS and solution vectors */
    let mut b: Array1<f64> = Array::ones(n as usize);
    let mut x: Array1<f64> = Array::zeros(n as usize);
    let nrhs: i32 = 1;  // Number of right hand sides.

    /* Internal solver memory pointer pt (C lang), */
    /* 32-bit: int pt[64]; 64-bit: long int pt[64] */
    /* or void *pt[64] should be OK on both architectures */
    let mut pt: Array1<isize> = Array1::zeros(64);
    /* Pardiso control parameters. */
    let mut iparm: Array1<i32> = Array1::zeros(64);
    let maxfct: i32 = 1;
    let mnum: i32 = 1;
    let msglvl: i32 = 0;
    let mut error = 0;

    let mut ddum1: Array1<f64> = Array::zeros(n as usize);  // f64 dummy 1
    let mut ddum2: Array1<f64> = Array::zeros(n as usize);  // f64 dummy 2
    let mut idum: Array1<i32> = Array::zeros(n as usize);   // i32 dummy.
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
/* -------------------------------------------------------------------- */
/* .. Back substitution and iterative refinement. */
/* -------------------------------------------------------------------- */
    let phase = 33;

    let a2: Array2<f64> = arr2(&[
        [ 1.0,-1.0, 0.0,-3.0, 0.0],
        [-2.0, 5.0, 0.0, 0.0, 0.0],
        [ 0.0, 0.0, 4.0, 6.0, 4.0],
        [-4.0, 0.0, 2.0, 7.0, 0.0],
        [ 0.0, 8.0, 0.0, 0.0,-5.0],
     ]);
     let b2: Array1<f64> = Array1::ones(5);

    for i in 0..3 {
        iparm[11] = i;
        println!("\nSolving system with iparm[11] = {} ... ", iparm[11]);
        unsafe {
            pardiso(
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
                b.as_slice_memory_order_mut().unwrap(),
                x.as_slice_memory_order_mut().unwrap(),
                &mut error
            )
        }
        if error != 0 {
            panic!("ERROR during solution: {}", error);
        }

        let x2 = match i {
            0 => {
                println!("*** A x = b ***");
                a2.solve(&b2).unwrap()
            },
            1 => {
                println!("*** A^H x = b ***");
                a2.solve_h(&b2).unwrap()
            },
            _ => {
                println!("*** A^T x = b ***");
                a2.solve_t(&b2).unwrap()
            },
        };

        println!("The solution of the system is: ");
        println!("{:?}", x);

        println!("\nThe solution by LAPACKE is: ");
        println!("{:?}", x2);
    }
}