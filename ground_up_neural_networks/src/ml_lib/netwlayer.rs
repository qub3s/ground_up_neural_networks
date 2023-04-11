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
pub struct Result{
    pub nodevalues: Vec<Matrix<Mval>>,
    pub res: Vec<Mval>
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
                tbias.push(rng.gen_range(0.0..1.0));
            }

            for _ in 0.. tnodesperlayer[i]*tnodesperlayer[i-1]{
                tmat.push(rng.gen_range(0.0..(tnodesperlayer[i-1] as f64).sqrt()))
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
    pub fn eval(&self, input: Vec<Mval>) -> Result{
        let mut i = Matrix::new_v(input);
        let mut res: Vec<Mval> = Vec::with_capacity(0);                                      // calc capa
        let mut nodevalues: Vec<Matrix<Mval>> = Vec::with_capacity(0);                      // calculate capacity

        nodevalues.push(i.clone());
        for l in &self.layer{
            let z = &(&l.w * &i) + &l.b;
            
            /*
            println!("Input: ");
            i.pout();
            
            println!("Weights: ");
            l.w.pout();
            
            println!("Biases: ");
            l.b.pout();

            println!("Z: ");
            z.pout();
            */

            i = z.map(Self::activfunc);

            //println!("Activation: ");
            //i.pout();

            nodevalues.push(i.clone());
            //println!();
        }
         
        for l in 0..(&i).geth(){
            res.push((&i).mat[l]);
        }

        return Result { res: Self::softmax(res), nodevalues: nodevalues };
    }

    pub fn learn(mut self, res: Result, mut target: Vec<Mval>, rate: Mval) -> NeuralNetwork{

        // calculate error
        for l in 0..target.len(){
            target[l] = target[l] - res.res[l];
        }

        // create Matrix from Error
        let mut delta = Matrix::new_v(target);

        let layer = self.layer.len();
        
        let mut z : Vec<Matrix<Mval> > = Vec::with_capacity(layer);
        // Calculate all the values before activation function
        for x in 0..layer{
            let y = layer-1 -1*x;

            let lw = &self.layer[y].w;                                                                  
            let lb = &self.layer[y].b;
            let lam1 = &res.nodevalues[&res.nodevalues.len()-2-1*x].clone();

            z.push(&(lw * &lam1) + lb);
        }

        for l in 0..layer{

            let y = self.layer.len()-1 -1*l;
            let talm1 = &res.nodevalues[&res.nodevalues.len()-2-1*l].clone().transpose();                   // transposed Layeractivationvalues transpose  
            z[l] = z[l].clone().map(Self::derivactivfunc);                                                  // use relu derivation
            
            if l == 0{
                delta = delta.hadamard(&z[l]);                                                              // first round 
            }
            else{
                delta = (&self.layer[y+1].w.transpose() * &delta).hadamard(&z[l]);                          // propagate the error backwards
            }

            delta = delta.mulnum(rate);                                                                     // multiply the learning rate
            self.layer[y].b = &self.layer[y].b + &delta;                                                    // change biases
            self.layer[y].w = &self.layer[y].w + &(&delta*&talm1);                                          // change weights
        }
        
        return self;
    }

    // gets percentages in the last layer
    pub fn softmax( mut vec: Vec<Mval>) -> Vec<Mval>{
        let mut sum = 0.0;
        let mut max = vec[0];
        
        for i in 1..vec.len(){
            if vec[i] > max{
                max = vec[i];
            }
        }

        for i in 0..vec.len(){
            sum += (vec[i]-max).exp();
        }

        

        for i in 0..vec.len(){
            let mut tmp = (vec[i]-max).exp();
            if tmp < 0.00000000001{
                vec[i] = 0.0;
            }
            else{
                vec[i] = tmp/sum;
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