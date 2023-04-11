#![allow(dead_code)]

use core::ops::{Add,Mul};
use std::vec::Vec;
use std::fmt::Display;
use serde::{Serialize, Deserialize};            // optional

// trait Add <Output = Self> + Mul<Output = Self> + Display + Copy + Sized{}
#[derive(Serialize, Deserialize, Clone)]
pub struct Matrix<T: Add <Output = T> + Mul<Output = T> + Display + Copy + Sized> {
    h : usize,
    w : usize,
    pub mat : Vec<T>,
}

// General Definition for Mat
impl<T: Add <Output = T> + Mul<Output = T> + Display + Copy + Sized > Matrix<T>{
    // Creates an Vector
    pub fn new_v( m: Vec<T>) -> Self{
        return Self {h: m.len(), w: 1, mat: m };
    }

    // Can Create Vector or a Matrix
    pub fn new_m(y: usize, x: usize, m: Vec<T>) -> Self{
        if m.len() != y*x{
            panic!("The given Array isnt the same size as the target");
        }
        return Self { h: y, w: x, mat: m };
    }

    // prints out the matrix
    pub fn pout(&self){
        for y in 0..self.h{
            print!("[");
            for x in 0..self.w{
                print!(" {} ",self.mat[self.w*y+x]);
            }
            println!("]");
        }
    }

    pub fn transpose(&self) -> Matrix<T>{
        let oldw = self.w;
        let oldh = self.h;

        if self.w == 1 || self.h == 1 {
            return Self::new_m(self.w, self.h, self.mat.clone());
        }

        let mut mat: Vec<T> = Vec::with_capacity(oldw*oldh);

        for j in 0..oldw{
            for k in 0..oldh{
                mat.push(self.mat[k*oldw+j]);
            }
        }

        return Self::new_m(self.w, self.h, mat);
    }

    pub fn mulnum(mut self, a: T) -> Matrix<T>{
        for x in 0..self.h*self.w{
            self.mat[x] = self.mat[x] * a;
        }
        return self;
    }

    pub fn geth(&self) -> usize{
        return self.h;
    }
    
    pub fn getw(&self) -> usize{
        return self.w;
    }

    pub fn map(mut self, func: fn(T) -> T) -> Self{
        for l in 0..self.h*self.w{
            self.mat[l] = func(self.mat[l]);
        }
        return self;
    }

    pub fn hadamard(&self, o: &Self) -> Self{
        if self.h != o.h || self.w != o.w{
            panic!("Wrong Dimensions!");
        }

        let mut mat: Vec<T> = Vec::with_capacity(o.w*o.h);

        for j in 0..o.h*o.w{
            mat.push(self.mat[j] * o.mat[j]);
        }

        return Self::new_m(o.h,o.w,mat);
    }
}

// Simple Matrixaddition
// The pointer version of addition doesnt own a and b -> "c = &a + &b"
impl<T: Add <Output = T> + Mul<Output = T> + Display + Copy + Sized > Add for &Matrix<T>{
    type Output = Matrix<T>;

    fn add(self, o: Self) -> Matrix<T> {
        if self.h != o.h || self.w != o.w{
            panic!("Can't add matrices of different size!!! ({}:{}) | ({}:{})",self.w,self.h,o.w,o.h)

        }

        let mut tmp: Vec<T> = Vec::with_capacity(o.h*o.w);

        for y in 0..self.h{
            for x in 0..self.w{
                tmp.push(self.mat[o.w*y+x] + o.mat[o.w*y+x]);
            }
        }
        return Matrix {h: self.h, w: self.w, mat: tmp};
    }
}

// Simple Matrixaddition
// The non pointer Version doesnt owns Objects a and b -> "c = &a + &b"
impl<T: Add <Output = T> + Mul<Output = T> + Display + Copy + Sized > Add for Matrix<T>{
    type Output = Self;

    fn add(self, o: Self) -> Self {
        if self.h != o.h || self.w != o.w{
            panic!("Can't add matrices of different size!!! ({}:{}) | ({}:{})",self.w,self.h,o.w,o.h)
        }

        let mut tmp: Vec<T> = Vec::with_capacity(o.h*o.w);

        for y in 0..self.h{
            for x in 0..self.w{
                tmp.push(self.mat[o.w*y+x] + o.mat[o.w*y+x]);
            }
        }
        return Matrix {h: self.h, w: self.w, mat: tmp};
    }
}

// Simple Matrix/ Matrix-Vector Mutliplication
// This should be used for small matrices and matrics-vector Multiplication, for large nxn matrices use Fast-Matrix_Multiplication
// The non pointer Version owns and destroys both Objects a and b -> "c = &a + &b"
impl<T: Add <Output = T> + Mul<Output = T> + Display + Copy + Sized > Mul for &Matrix<T>{
    type Output = Matrix<T>;

    fn mul(self, o: Self) -> Matrix<T> {
        if self.w != o.h{
            panic!("Wrong Matrixdimensions for Multiplication!!! ({}:{})",self.w,o.h);
        }

        let h = self.h;
        let w = o.w;

        let mut tmp: Vec<T> = Vec::with_capacity(h*w);

        for hself in 0..self.h{
            for wo in 0..o.w{
                let mut t: T = self.mat[hself*self.w] * o.mat[wo];
                for c in 1..o.h{
                    t = t + self.mat[hself*self.w+c] * o.mat[c*o.w+wo];
                }
                tmp.push(t);
            }
        }
        return Matrix { h: h, w: w, mat: tmp};
    }

}

// Simple Matrix/ Matrix-Vector Mutliplication
// This should be used for small matrices and matrics-vector Multiplication, for large nxn matrices use Fast-Matrix_Multiplication
// The pointer Version owns and destroys both Objects a and b -> "c = &a + &b"
impl<T: Add <Output = T> + Mul<Output = T> + Display + Copy + Sized > Mul for Matrix<T>{
    type Output = Self;

    fn mul(self, o: Self) -> Self {
        if self.w != o.h{
            panic!("Wrong Matrixdimensions for Multiplication!!!");
        }

        let h = self.h;
        let w = o.w;

        let mut tmp: Vec<T> = Vec::with_capacity(h*w);

        for hself in 0..self.h{
            for wo in 0..o.w{
                let mut t: T = self.mat[hself*self.w] * o.mat[wo];
                for c in 1..o.h{
                    t = t + self.mat[hself*self.w+c] * o.mat[c*o.w+wo];
                }
                tmp.push(t);
            }
        }
        return Self { h: h, w: w, mat: tmp};
    }

}