// ref. pardiso_unsym_complex_c.c
// in Intel(R) MKL PARDISO C example
use pardiso::{pardiso, pardisoinit};
use ndarray::*;
use ndarray_linalg::*;
extern crate pardiso_src;

fn main() {
    /* Matrix data. */
    let n: i32 = 8;
    let ia: Array1<i32> = Array::from(vec![1, 5, 8, 10, 12, 13, 16, 18, 21]);
    let ja: Array1<i32> = Array::from(vec![
        1,    3,       6, 7,
           2, 3,    5,
              3,             8,
                 4,       7,
           2,
              3,       6,    8,
           2,             7,
              3,          7, 8,
    ]);
    let a: Array1<c64> = Array::from(vec![
        c64::new( 7.0,  1.0),   //  0
        c64::new( 1.0,  1.0),   //  1
        c64::new( 2.0,  1.0),   //  2
        c64::new( 7.0,  1.0),   //  3
        c64::new(-4.0,  0.0),   //  4
        c64::new( 8.0,  1.0),   //  5
        c64::new( 2.0,  1.0),   //  6
        c64::new( 1.0,  1.0),   //  7
        c64::new( 5.0,  1.0),   //  8
        c64::new( 7.0,  0.0),   //  9
        c64::new( 9.0,  1.0),   // 10
        c64::new(-4.0,  1.0),   // 11
        c64::new( 7.0,  1.0),   // 12
        c64::new( 3.0,  1.0),   // 13
        c64::new( 8.0,  0.0),   // 14
        c64::new( 1.0,  1.0),   // 15
        c64::new(11.0,  1.0),   // 16
        c64::new(-3.0,  1.0),   // 17
        c64::new( 2.0,  1.0),   // 18
        c64::new( 5.0,  0.0),   // 19
    ]);
    let mut a2: Array2<c64> = Array::zeros((n as usize, n as usize));
    a2[[0, 0]] = c64::new( 7.0,  1.0);  //  0
    a2[[0, 2]] = c64::new( 1.0,  1.0);  //  1
    a2[[0, 5]] = c64::new( 2.0,  1.0);  //  2
    a2[[0, 6]] = c64::new( 7.0,  1.0);  //  3
    a2[[1, 1]] = c64::new(-4.0,  0.0);  //  4
    a2[[1, 2]] = c64::new( 8.0,  1.0);  //  5
    a2[[1, 4]] = c64::new( 2.0,  1.0);  //  6
    a2[[2, 2]] = c64::new( 1.0,  1.0);  //  7
    a2[[2, 7]] = c64::new( 5.0,  1.0);  //  8
    a2[[3, 3]] = c64::new( 7.0,  0.0);  //  9
    a2[[3, 6]] = c64::new( 9.0,  1.0);  // 10
    a2[[4, 1]] = c64::new(-4.0,  1.0);  // 11
    a2[[5, 2]] = c64::new( 7.0,  1.0);  // 12
    a2[[5, 5]] = c64::new( 3.0,  1.0);  // 13
    a2[[5, 7]] = c64::new( 8.0,  0.0);  // 14
    a2[[6, 1]] = c64::new( 1.0,  1.0);  // 15
    a2[[6, 6]] = c64::new(11.0,  1.0);  // 16
    a2[[7, 2]] = c64::new(-3.0,  1.0);  // 17
    a2[[7, 6]] = c64::new( 2.0,  1.0);  // 18
    a2[[7, 7]] = c64::new( 5.0,  0.0);  // 19

    let mtype: i32 = 13;    // Real complex unsymmetric matrix
    /* RHS and solution vectors */
    let mut b: Array1<c64> = Array::from(vec![c64::new(1.0, 1.0); n as usize]);
    let mut x: Array1<c64> = Array::zeros(n as usize);
    let nrhs: i32 = 1;  // Number of right hand sides.

    /* Internal solver memory pointer pt (C lang), */
    /* 32-bit: int pt[64]; 64-bit: long int pt[64] */
    /* or void *pt[64] should be OK on both architectures */
    let mut pt: Array1<isize> = Array::zeros(64);
    /* Pardiso control parameters. */
    let mut iparm: Array1<i32> = Array1::zeros(64);
    let maxfct: i32 = 1;
    let mnum: i32 = 1;
    let msglvl: i32 = 0;
    let mut error = 0;

    let mut idum: Array1<i32> = Array::zeros(n as usize);   // i32 dummy.
/* -------------------------------------------------------------------- */
/* .. Setup Pardiso control parameters. */
/* .. Initialize the internal solver memory pointer. This is only */
/* necessary for the FIRST call of the PARDISO solver. */
/* ..  Reordering and Symbolic Factorization.  This step also allocates */
/*     all memory that is necessary for the factorization.              */
/* ..  Numerical factorization.                                         */
/* ..  Back substitution and iterative refinement.                      */
/* -------------------------------------------------------------------- */
    let phase = 13;

    for i in 0..3 {
        unsafe { 
            pardisoinit(
                pt.as_slice_memory_order_mut().unwrap(),
                mtype,
                iparm.as_slice_memory_order_mut().unwrap()
            );
            iparm[11] = i;
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
            panic!("ERROR during analysis, factorization, and solve: {}", error);
        }
        let x2 = match i {
            0 => {
                println!("*** A x = b ***");
                a2.solve(&b).unwrap()
            },
            1 => {
                println!("*** A^H x = b ***");
                a2.solve_h(&b).unwrap()
            },
            _ => {
                println!("*** A^T x = b ***");
                a2.solve_t(&b).unwrap()
            },
        };
        println!("\nThe solution of the system is: ");
        for c in x.iter() {
            println!("{:5.2} + {:5.2}i", c.re, c.im);
        }
        
        println!("\nThe solution by LAPACKE is: ");
        for c in x2.iter() {
            println!("{:5.2} + {:5.2}i", c.re, c.im);
        }
        println!("");
    }
}