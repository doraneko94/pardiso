use pardiso::*;
extern crate pardiso_src;

fn main() {
    let a = 1.00;
    let mut b = 0.01;
    let eps = 0.10;
    let ret = unsafe { mkl_pardiso_pivot(a, &mut b, eps) };
    println!("0.01 = b < eps = 0.10: b -> {:.2}", b);
    println!("returns: {}\n", ret);

    b = 0.2;
    let ret = unsafe { mkl_pardiso_pivot(a, &mut b, eps) };
    println!("0.20 = b > eps = 0.10: b -> {:.2}", b);
    println!("returns: {}", ret);
}