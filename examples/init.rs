use pardiso::*;
use ndarray::*;
extern crate pardiso_src;

fn main() {
    let mut pt: Array1<isize> = Array::zeros(64);
    let mut iparm: Array1<i32> = Array1::zeros(64);

    let mtype = 1;
    unsafe { pardisoinit(pt.as_slice_memory_order_mut().unwrap(), mtype, iparm.as_slice_memory_order_mut().unwrap()) }
    println!("\n=== iparm (Real and structurally symmetric) ===");
    println!("{:?}", iparm);

    let mtype = 2;
    unsafe { pardisoinit(pt.as_slice_memory_order_mut().unwrap(), mtype, iparm.as_slice_memory_order_mut().unwrap()) }
    println!("\n=== iparm (Real and symmetric positive definite) ===");
    println!("{:?}", iparm);

    let mtype = -2;
    unsafe { pardisoinit(pt.as_slice_memory_order_mut().unwrap(), mtype, iparm.as_slice_memory_order_mut().unwrap()) }
    println!("\n=== iparm (Real and symmetric indefinite) ===");
    println!("{:?}", iparm);

    let mtype = 3;
    unsafe { pardisoinit(pt.as_slice_memory_order_mut().unwrap(), mtype, iparm.as_slice_memory_order_mut().unwrap()) }
    println!("\n=== iparm (Complex and structurally symmetric) ===");
    println!("{:?}", iparm);

    let mtype = 4;
    unsafe { pardisoinit(pt.as_slice_memory_order_mut().unwrap(), mtype, iparm.as_slice_memory_order_mut().unwrap()) }
    println!("\n=== iparm (Complex and Hermitian positive definite) ===");
    println!("{:?}", iparm);

    let mtype = -4;
    unsafe { pardisoinit(pt.as_slice_memory_order_mut().unwrap(), mtype, iparm.as_slice_memory_order_mut().unwrap()) }
    println!("\n=== iparm (Complex and Hermitian indefinite) ===");
    println!("{:?}", iparm);

    let mtype = 6;
    unsafe { pardisoinit(pt.as_slice_memory_order_mut().unwrap(), mtype, iparm.as_slice_memory_order_mut().unwrap()) }
    println!("\n=== iparm (Complex and symmetric matrix) ===");
    println!("{:?}", iparm);

    let mtype = 11;
    unsafe { pardisoinit(pt.as_slice_memory_order_mut().unwrap(), mtype, iparm.as_slice_memory_order_mut().unwrap()) }
    println!("\n=== iparm (Real and nonsymmetric matrix) ===");
    println!("{:?}", iparm);

    let mtype = 13;
    unsafe { pardisoinit(pt.as_slice_memory_order_mut().unwrap(), mtype, iparm.as_slice_memory_order_mut().unwrap()) }
    println!("\n=== iparm (Complex and nonsymmetric matrix) ===");
    println!("{:?}", iparm);
}