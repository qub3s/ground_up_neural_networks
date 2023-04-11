#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]
#![allow(unused_assignments)]
#![allow(unused_mut)]

#[path = "ml_lib/matrix.rs"] mod matrix;
#[path = "ml_lib/netwlayer.rs"] mod netw_layer;

use std::time::Instant;
use netw_layer::NeuralNetwork;
use rand::Rng;
use std::env;
use std::fs;
use json;
use std::io::{stdout, Write};
//use matrix::Matrix;

/*
Neural Network that learns to read numbers;

Neural Network that learns to play Blackjack;
- 16 Inputs (count cards, 1 player hand value)
- dealer up card

- Split
- Double 
- surrender
*/

// read on bias neurons
fn print_type_of<T>(_: &T) {
    println!("{}", std::any::type_name::<T>())
}

fn readnumberdataset() -> Vec<Vec<Vec<f64>>>{
    let mut jsvec: Vec<Vec<Vec<f64>>> = Vec::new();

    for x in 0..3{
        let ch = x.to_string();
        let fc = fs::read("json_dataset/".to_owned() + &ch + ".json").expect("Can't read File!");
        let fc = std::str::from_utf8(&fc).unwrap();
        let json = json::parse(fc).expect("error");
        let mut tmp1 = Vec::with_capacity(100);
        for j in 0..100{
            let mut tmp2 = Vec::with_capacity(1024);
            for l in 0..1024{
                tmp2.push((json[format!("{}",j as i32 )][l].as_f64().unwrap()-127.0)/255.0);
            }
            tmp1.push(tmp2);
        }
        jsvec.push(tmp1);
    }
    return jsvec;
}

fn scheresteinpapier(){
    let mut nn = NeuralNetwork::new(vec![2,10,10,3]);
    let mut rng = rand::thread_rng();

    let mut cost = 0.0;

    let reps = 1000000;
    let rate = 0.001;

    
    for x in 1..reps+1{
        let a = rng.gen_range(0..5);
        let b = rng.gen_range(0..5);
        
        let mut c = nn.eval(vec![a as f64,b as f64]);
        println!();
        println!("Result: {} : {} : {}",c.res[0],c.res[1],c.res[2]);

        if a == b{
            nn = nn.learn(c.clone(),vec![0.0,1.0,0.0],rate);
            println!("0 : 1 : 0");
        }
        else if a > b{
            nn = nn.learn(c.clone(),vec![1.0,0.0,0.0],rate);        // a wins
            println!("1 : 0 : 0");
        }
        else{
            nn = nn.learn(c.clone(),vec![0.0,0.0,1.0],rate);        // a wins
            println!("0 : 0 : 1");
        }

        let mut c = nn.eval(vec![a as f64,b as f64]);

        println!("Result: {} : {} : {}",c.res[0],c.res[1],c.res[2]);
        println!();
    }

    nn.store();
 
}

fn recognizenumbers(){
    let jsvec = readnumberdataset();
    let rate = 0.0000000001;
    let mut rng = rand::thread_rng();
    let outsize = 3;
    let mut nn = NeuralNetwork::new(vec![1024,100,10,outsize]);
    //nn = NeuralNetwork::load();

    for x in 0..100000{
        let number = rng.gen_range(0..outsize);
        println!();
        println!("N: {}",number);
        let example = 0; //rng.gen_range(0..100);

        let mut target = Vec::with_capacity(10);
        for l in 0..outsize{
            if l == number{
                target.push(1.0);
            }
            else{
                target.push(0.0);
            }
        }

        // jsvec[number][example].clone()
        let c = nn.eval(jsvec[number][example].clone());
        println!("Res:");
        for x in 0..outsize{
            println!("{}:{}",x,c.res[x]);
        }
        

        nn = nn.learn(c.clone(),target,rate);
        
        let c = nn.eval(jsvec[number][example].clone());

        println!("After learn:");
        for x in 0..outsize{
            println!("{}:{}",x,c.res[x]);
        }

    }

    /*
    for y in 0..3{
        let c = nn.eval(jsvec[y][0].clone());

        for x in 0..outsize{
            println!("{}:{}",x,c.res[x]);
        }
        println!();
    }
    */
    
    nn.store();
}

fn overunder100(){
    let mut nn = NeuralNetwork::new(vec![1,100,2]);
    let mut rng = rand::thread_rng();
    let reps = 1000000;
    let rate = 1.0/reps as f64;
    let out = 10;

    let mut sum = 0.0;
    let mut cost = 0.0;

    for x in 1..reps+1{
        let c = nn.eval(vec![sum as f64]);
        let prevsum = sum;
        sum += rng.gen_range(0.0..100.0);

        
        if sum > 100.0{
            nn = nn.learn(c.clone(),vec![0.0,1.0],rate);
            cost = cost + c.res[0];
            /*
            println!("under : over");
            println!("Result:   0  : 1 ({}:{})",prevsum,sum);
            println!("MyResult: {} : {}",c.res[0],c.res[1]);
            let c = nn.eval(vec![sum as f64]);
            println!("After learn : {}:{}",c.res[0],c.res[1]);
            println!();
            */
        }
        else{
            nn = nn.learn(c.clone(),vec![1.0,0.0],rate);        // a wins
            cost = cost + c.res[1];
            
            /*
            println!("under : over");
            println!("Result: 1 : 0 ({}:{})",prevsum,sum);
            println!("TargMyResultet: {} : {}",c.res[0],c.res[1]);
            
            
            let c = nn.eval(vec![sum as f64]);
            println!("After learn : {}:{}",c.res[0],c.res[1]);
            println!();
            */
        }

        sum = rng.gen_range(0.0..100.0);

        if x % (reps/out) as i32 == 0{
            println!();
            println!("{}% : {}",((x as f64/reps as f64)*100.0) as i32,cost/(reps/out) as f64);
            cost = 0.0;
            let v = [10.0,20.0,30.0,40.0,50.0,60.0,70.0,80.0,90.0];

            for x in v{
                let c = nn.eval(vec![x]);
                //println!("{} : {} : {}",x,c.res[0],c.res[1]);
            }
        }
    }

    let v = [10.0,20.0,30.0,40.0,50.0,60.0,70.0,80.0,90.0];

    println!();
    for x in v{
        let c = nn.eval(vec![x]);
        println!("{} : {} : {}",x,c.res[0],c.res[1]);
    }

    nn.store();
}

fn main() {
    env::set_var("RUST_BACKTRACE", "1");

    //recognizenumbers();
    //scheresteinpapier();
    overunder100();
}