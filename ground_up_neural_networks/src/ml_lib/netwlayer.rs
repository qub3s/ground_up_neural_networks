#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_comparisons)]

#[path = "matrix.rs"] mod matrix;

use matrix::Matrix;
use rand::Rng;
use std::fs;
use serde::{Serialize, Deserialize};

type Mval = f64;                    // f32 or f64

#[derive(Serialize, Deserialize, Clone)]
pub struct NetwLayer{
    input: usize,
    nodes: usize,
    pub b: Matrix<Mval>,             // biases
    pub w: Matrix<Mval>              // weights
}

#[derive(Serialize, Deserialize, Clone)]
pub struct NeuralNetwork{
    nodesperlayer: Vec<usize>,              // firstlayer is the amount of inputs
    pub layer: Vec<NetwLayer>
    // sigmoid vs relu
    // normalization
}

#[derive(Clone)]
pub struct Deltawb{
    pub w: Vec<Matrix<Mval>>,
    pub b: Vec<Matrix<Mval>>
}

impl NeuralNetwork{
    // thlayer -> the amount of hiddenlayers (number of layer -1);
    // nodesperlayer -> len = thlayer +2 (first element ist the amount of inputs) 
    pub fn new( tnodesperlayer: Vec<usize> ) -> Self{
        let thlayer = tnodesperlayer.len() -2;

        let mut rng = rand::thread_rng();                                                               // rng instance

        let mut tlayer: Vec<NetwLayer> = Vec::with_capacity(tnodesperlayer.len()-1);                                 // creates 

        for i in 1..tnodesperlayer.len(){
            let mut tbias: Vec<Mval> = Vec::with_capacity(tnodesperlayer[i]);                            // creates temporary bias
            let mut tmat: Vec<Mval> = Vec::with_capacity(tnodesperlayer[i]*tnodesperlayer[i-1]);         // creates temporary matrice

            // fills bias and matrix with random numbers
            for _ in 0..tnodesperlayer[i]{
                tbias.push(rng.gen_range(0.0..0.2)); //
            }

            for _ in 0.. tnodesperlayer[i]*tnodesperlayer[i-1]{
                tmat.push(rng.gen_range(0.0..(2.0/tnodesperlayer[i-1] as f64).sqrt()))
            }
            
            // creates struct
            tlayer.push(NetwLayer{input: tnodesperlayer[i-1], nodes: tnodesperlayer[i], b: Matrix::new_v(tbias), w: Matrix::new_m(tnodesperlayer[i], tnodesperlayer[i-1], tmat)});
        }

        return Self {nodesperlayer: tnodesperlayer, layer: tlayer};
    }

    pub fn get_input_size(&self) -> usize{
        return self.nodesperlayer[0];
    }

    pub fn get_num_layer(&self) -> usize{
        return self.layer.len();
    }

    // returns only the layers not the input size
    pub fn get_layer_size(&self) -> Vec<usize>{
        let mut tmp: Vec<usize> = vec![0; self.nodesperlayer.len()-1];
        tmp.clone_from_slice(&self.nodesperlayer[1..]);
        return tmp;
    }

    // evaluates input
    pub fn eval(&self, input: Vec<Mval>) -> Matrix<Mval>{
        let mut i = Matrix::new_v(input);

        for l in &self.layer{
            let z = &(&l.w * &i) + &l.b;

            i = z.map(Self::activfunc);
        }

        return Self::softmax(i);
    }

    pub fn backprop(&self, input: Vec<Mval>, mut target: Vec<Mval>, rate: Mval) -> Deltawb{

        let mut a = Matrix::new_v(input);
        let mut zv: Vec<Matrix<Mval>> = Vec::with_capacity(0);                      // calculate capacity
        let mut av: Vec<Matrix<Mval>> = Vec::with_capacity(0);                      // calculate capacity
        let mut biases: Vec<Matrix<Mval>> = Vec::with_capacity(0);                  // calculate capacity
        let mut weights: Vec<Matrix<Mval>> = Vec::with_capacity(0);                 // calculate capacity
        
        av.push(a.clone());                                                         // first element
        for l in &self.layer{
            let z = &(&l.w * &a) + &l.b;
            zv.push(z.clone());                                                     // clone entfernen
            a = z.map(Self::activfunc);
            av.push(a.clone());                                                     // clone entfernen
        }
        let x = av.len()-1;
        av[x] = Self::softmax(av[x].clone());                                       // clone entfernen
             
    
        let mut delta = &av[x] + &Matrix::new_v(target).mulnum(-1.0);               // delta berechnen /vielleicht extra funktion

        let layer = av.len()-1;

        for l in 0..self.layer.len(){
            let y = self.layer.len()-1 -1*l;
            
            let talm1 = &av[av.len()-2-l].clone().transpose();                                                       // transposed Layeractivationvalues transpose  
            zv[l] = zv[l].clone().map(Self::derivactivfunc);                                                            // use relu derivation
            
            if l == 0{
                delta = delta.hadamard(&zv[y]).mulnum(-rate);                                                           // first round  (doppel - entfernen)
            }
            else{
                delta = (&(&self.layer[y+1].w.transpose() * &delta)).hadamard(&zv[y]);                                  // propagate the error backwards
            }

            biases.push(delta.clone());
            weights.push((&delta*&talm1).clone())                                                                // change biases
        }

        return Deltawb {w: weights, b: biases};
    }

    pub fn applydeltawb(mut self, delta: Deltawb) -> Self{
        for l in 0..delta.b.len(){
            self.layer[l].b = &self.layer[l].b + &delta.b[delta.b.len()-1-l];                                                  // change biases
            self.layer[l].w = &self.layer[l].w + &delta.w[delta.b.len()-1-l];    
        }
        return self;
    }

    pub fn minibatch(mut self, inputs: Vec<Vec<Mval>>, targets: Vec<Vec<Mval>>, rate: Mval) -> Self{
        if inputs.len() != targets.len(){
            panic!("Wrong dimensions!!!");
        }

        let mut deltas = self.backprop(inputs[0].clone(),targets[0].clone(),rate);
        for l in 1..inputs.len(){
            let tmp = self.backprop(inputs[l].clone(),targets[l].clone(),rate);
            
            for m in 0..deltas.w.len(){
                deltas.w[m] = &deltas.w[m] + &tmp.w[m];
                deltas.b[m] = &deltas.b[m] + &tmp.b[m];                
            }
        }
        /*
        for m in 0..deltas.w.len(){
            deltas.w[m] = deltas.w[m].clone().mulnum(1.0/inputs.len() as f64);
            deltas.b[m] = deltas.b[m].clone().mulnum(1.0/inputs.len() as f64);                
        }
        */
        return self.applydeltawb(deltas);

    }

    // gets percentages in the last layer
    pub fn softmax( mut vec: Matrix<Mval>) -> Matrix<Mval>{
        let mut sum = 0.0;
        let mut max = vec.mat[0];
        
        for i in 1..vec.mat.len(){
            if vec.mat[i] > max{
                max = vec.mat[i];
            }
        }

        for i in 0..vec.mat.len(){
            sum += (vec.mat[i]-max).exp();
        }

        for i in 0..vec.mat.len(){
            let mut tmp = (vec.mat[i]-max).exp();
            if tmp < 0.00000000001{
                vec.mat[i] = 0.0;
            }
            else{
                vec.mat[i] = tmp/sum;
            }
        }

        return vec;
    }

    // activationfunction for values
    pub fn activfunc( mut v: Mval) -> Mval{
        if v < 0.0{
            v = 0.0;
        }
        return v;
    }

    // derivative of the activationfunction
    pub fn derivactivfunc( v: Mval) -> Mval{
        if v < 0.0{
            return 0.0;
        }
        return 1.0;
    } 

    /*
    pub fn activfunc( mut v: Mval) -> Mval{
        println!("{}",1.0/(1.0+(-v).exp()));
        println!("{}",v);
        return 1.0/(1.0+(-v).exp());
    }

    pub fn derivactivfunc( v: Mval) -> Mval{
        return ( 1.0/(1.0+(-v).exp()) ) * ( 1.0 - (1.0/(1.0+(-v).exp())) );
    }
    */
    

    // write an store function
    pub fn store(self){
        fs::write("NN",serde_yaml::to_string(&self).expect("Cant Serialize!")).expect("Can't write File!");
    }

    pub fn load() -> Self{
        let yaml = fs::read("NN").expect("Can't read File!");
        let yamlstr = std::str::from_utf8(&yaml).unwrap();
        let nn: NeuralNetwork = serde_yaml::from_str(yamlstr).unwrap();

        return nn;
    }

}