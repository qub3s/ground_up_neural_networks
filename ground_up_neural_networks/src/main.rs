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
use plotters::prelude::*;
use chrono::{Utc, TimeZone};


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

    for x in 0..10{
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

fn plot(datay : Vec<f64>, steps : i32){ 
    let len = steps * datay.len() as i32;
    
    
    let mut datax : Vec<i32> = Vec::new();

    for x in 0..len{
        datax.push(x*steps);
    }

    let values: Vec<(i32, f64)>= datax.iter().cloned().zip(datay.iter().cloned()).collect();

    let root_area = BitMapBackend::new("plots/abcd.png", (1200, 600)).into_drawing_area();
    root_area.fill(&WHITE).unwrap();

    let mut ctx = ChartBuilder::on(&root_area)
        .set_label_area_size(LabelAreaPosition::Left, 30.0)
        .set_label_area_size(LabelAreaPosition::Bottom, 30.0)
        .set_label_area_size(LabelAreaPosition::Right, 30.0)
        .set_label_area_size(LabelAreaPosition::Top, 30.0)
        .caption("Gradient Descent", ("sans-serif", 30.0))
        .build_cartesian_2d(0..len-steps, 0.0..2.0 )
        .unwrap();

    ctx.configure_mesh().draw().unwrap();

    // Draw Scatter Plot
    ctx.draw_series(
        LineSeries::new(values.iter().map(|(x, y)| (*x,*y)), &BLUE)
    ).unwrap();

    root_area.present().expect("Unable to write result to file, please make sure 'plots' dir exists under current dir");
}

fn recognizenumbers(){
    let jsvec = readnumberdataset();
    let mut rng = rand::thread_rng();
    let outsize = 10;
    let mut nn = NeuralNetwork::new(vec![1024,10,10,outsize]);

    let reps = 700000;
    let rate = 0.00001;
    let out = 1000;
    let mut costs = Vec::with_capacity(out);    
    let mut cost = 0.0;

    let mut minibatchtargets = Vec::new();
    let mut minibatchinputs = Vec::new();
    let minibatchsize = 5;
    
    for x in 1..reps+1{
        let number = rng.gen_range(0..outsize);
        let example = rng.gen_range(0..100);

        let mut target = Vec::with_capacity(outsize);
        for l in 0..outsize{
            if l == number{
                target.push(1.0);
            }
            else{
                target.push(0.0);
            }
        }

        let c = nn.eval(jsvec[number][example].clone());       

        let mut tmp = 0.0;
        for x in 0..target.len(){
            tmp += ( target[0] - c.mat[0] ).abs();
        }
        cost += tmp as f64;

        minibatchinputs.push(jsvec[number][example].clone());
        minibatchtargets.push(target);

        if x % minibatchsize == 0{
            nn = nn.minibatch(minibatchinputs.clone(),minibatchtargets,rate).clone();
            minibatchtargets = Vec::new();
            minibatchinputs = Vec::new(); 
        }
        
        let c = nn.eval(jsvec[number][example].clone());
        

        if x % ( reps as f64 / out as f64 ) as i32 == 0{
            costs.push(cost / ( reps as f64 /out as f64 ));
            cost = 0.0;
        }
    }

    plot(costs,( reps as f64 /out as f64 ) as i32 );


    nn.store();
    
}


fn main() {
    env::set_var("RUST_BACKTRACE", "1");

    recognizenumbers();
    //scheresteinpapier();
    //overunder100();
}
