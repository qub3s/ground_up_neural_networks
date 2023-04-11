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

fn plot(datay : Vec<f64>, steps : i32){
    let len = steps * datay.len() as i32;
    
    
    let mut datax : Vec<i32> = Vec::new();

    for x in 0..len{
        datax.push(x*steps);
    }

    let values: Vec<(i32, f64)>= datax.iter().cloned().zip(datay.iter().cloned()).collect();

    let root_area = BitMapBackend::new("plots/abc.png", (1200, 600)).into_drawing_area();
    root_area.fill(&WHITE).unwrap();

    let mut ctx = ChartBuilder::on(&root_area)
        .set_label_area_size(LabelAreaPosition::Left, 30.0)
        .set_label_area_size(LabelAreaPosition::Bottom, 30.0)
        .set_label_area_size(LabelAreaPosition::Right, 30.0)
        .set_label_area_size(LabelAreaPosition::Top, 30.0)
        .caption("Gradient Descent", ("sans-serif", 30.0))
        .build_cartesian_2d(0..len-steps, 0.0..1.0 )
        .unwrap();

    ctx.configure_mesh().draw().unwrap();

    // Draw Scatter Plot
    ctx.draw_series(
        LineSeries::new(values.iter().map(|(x, y)| (*x,*y)), &BLUE)
    ).unwrap();

    root_area.present().expect("Unable to write result to file, please make sure 'plots' dir exists under current dir");
}

fn scheresteinpapier(){
    let mut nn = NeuralNetwork::new(vec![2,10,10,3]);
    let mut rng = rand::thread_rng();

    let mut cost = 0.0;

    let reps = 1000000;
    let rate = 0.0001;
    let out = 100;
    let mut costs = Vec::with_capacity(out);

    
    for x in 1..reps+1{
        let a = rng.gen_range(0..5);
        let b = rng.gen_range(0..5);
        
        let mut c = nn.eval(vec![a as f64,b as f64]);

        if a == b{
            nn = nn.learn(c.clone(),vec![0.0,1.0,0.0],rate);
            cost += c.res[0] + (c.res[1]-1.0).abs() + c.res[2];
        }
        else if a > b{
            nn = nn.learn(c.clone(),vec![1.0,0.0,0.0],rate);        // a wins
            cost += c.res[1] + (c.res[0]-1.0).abs() + c.res[2];
        }
        else{
            nn = nn.learn(c.clone(),vec![0.0,0.0,1.0],rate);        // a wins
            cost += c.res[1] + (c.res[2]-1.0).abs() + c.res[0];
        }

        let mut c = nn.eval(vec![a as f64,b as f64]);

        if x % (reps as f64 /out as f64) as i32 == 0{
            costs.push(cost/(reps as f64 /out as f64)as f64);
            cost = 0.0;
        }
    }

    plot(costs,( reps as f64 /out as f64 ) as i32 );
    nn.store();
 
}

fn overunder100(){
    let mut nn = NeuralNetwork::new(vec![1,10,2]);
    let mut rng = rand::thread_rng();
    let reps = 300000;
    let rate = 0.00006;
    let out = 100;
    let mut costs = Vec::with_capacity(out);

    let mut sum = 0.0;
    let mut cost = 0.0;
    

    for x in 1..reps+1{
        sum = rng.gen_range(0.0..100.0);
        let c = nn.eval(vec![sum as f64]);
        let x = x as i32;
        sum += rng.gen_range(0.0..100.0);

        
        if sum > 100.0{
            nn = nn.learn(c.clone(),vec![0.0,1.0],rate);
            cost = cost + (c.res[0]*c.res[0]);
        }
        else{
            nn = nn.learn(c.clone(),vec![1.0,0.0],rate);        // a wins
            cost = cost + (c.res[1]*c.res[1]);
        }

        if x % (reps/out) as i32 == 0{
            costs.push(cost/(reps/out)as f64);
            //println!("Costs: {}",rng.gen_range(0.0..100.0));
            cost = 0.0;
        }

        /*
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
        */
    }

    plot(costs,( reps/out ) as i32 );
    /*
    let v = [10.0,20.0,30.0,40.0,50.0,60.0,70.0,80.0,90.0];

    println!();
    for x in v{
        let c = nn.eval(vec![x]);
        println!("{} : {} : {}",x,c.res[0],c.res[1]);
    }
    */
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


fn main() {
    env::set_var("RUST_BACKTRACE", "1");

    //recognizenumbers();
    scheresteinpapier();
    //overunder100();
}
