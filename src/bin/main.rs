


use std::sync::Mutex;

use egui_wgpu::wgpu::naga::Type;
use simple_neural_network_project::{back, backprop, neural_network::{self, NeuralNetwork, TypeNeuronValue}, traning_handeler::TraningHandeler};
use actix_web::{get, post, web, App, HttpResponse, HttpServer, Responder};
use actix_files::{Files, NamedFile};
use serde::Serialize;
use csv::*;


// ideas for vizualisation
//https://www.chartjs.org/docs/latest/samples/other-charts/bubble.html
// chart.js
//

#[actix_web::main]
async fn main() -> std::io::Result<()> {

    HttpServer::new(|| {
        App::new()
        
            .app_data(web::Data::new(
                Mutex::new(
                    NeuralNetwork::builder()
                        .with_input_layer(2)
                        .with_hidden_layer(2)
                        .with_output_layer(2)
                        .build_network().unwrap()
                )
            ))
            
            .route("/network_generated_values.csv", web::get().to(generate_network_values))
            .route("/dataset.csv", web::get().to(dataset))
            .service(
                // prefixes all resources and routes attached to it...
                web::scope("/app")
                    // ...so this handles requests for `GET /app/index.html`
                    .route("/visualize", web::get().to(d3))
            )
            .service(Files::new("/", "./static"))

    })
    .bind(("127.0.0.1", 8080))?
    .run()
    .await
}



#[derive(serde::Serialize)]
struct Row<'a> {
    pos_x: &'a str,
    pos_y: &'a str,
    color: &'a str,
}


async fn generate_network_values(neural_network_mutex: web::Data<Mutex<NeuralNetwork>>) -> impl Responder {

    
    if let Some(neural_network) = neural_network_mutex.lock().as_mut().ok(){


        println!("Inside");

        let correct_output = vec![vec![1.0, 0.0], vec![1.0, 0.0], vec![1.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0], vec![0.0, 1.0], vec![0.0, 1.0], vec![0.0, 1.0], vec![0.0, 1.0], vec![0.0, 1.0]];
        let input = vec![vec![0.84,0.54344], vec![0.7,0.5], vec![0.75,0.4], vec![0.7,0.5], vec![0.88,0.88], vec![0.22,0.33], vec![0.30,0.14], vec![0.27,0.28], 
            vec![0.78,0.2], vec![0.9,0.1]]; // 


        let mut traning_handeler = TraningHandeler::new(&neural_network);
        traning_handeler.insert_traning_data(input, correct_output);
        traning_handeler.train_neural_network(neural_network, 300);



        let mut wtr = Writer::from_path("C:/Users/Henrik/Repos/Simple-Neural-Network-Project/src/bin/web/network_generated_values.csv").unwrap();
        for x in 0..100 {
            for y in 0..100 {
                let x_pos = x as f64 / 100.0; // between 0 and 1
                let y_pos = y as f64 / 100.0; // between 0 and 1

                let output_neuron_1 = neural_network.output_layer.as_ref().unwrap().neurons.get(0).unwrap();
                let output_neuron_2 = neural_network.output_layer.as_ref().unwrap().neurons.get(1).unwrap();


                let feedforward_values_map = neural_network.feedforward_to_map(&vec![x_pos as f32, y_pos as f32]).unwrap();

                let probability_classified_class_1 = feedforward_values_map.get(&(output_neuron_1, TypeNeuronValue::A)).unwrap();
                let probability_classified_class_2 = feedforward_values_map.get(&(output_neuron_2, TypeNeuronValue::A)).unwrap();

                //println!("class 1: {}, class 2: {}", probability_classified_class_1, probability_classified_class_2);
                
                if probability_classified_class_1 > probability_classified_class_2{
                    wtr.serialize(Row {
                        pos_x: f64::to_string(&x_pos).as_str(),
                        pos_y: f64::to_string(&y_pos).as_str(),
                        color: "#FF6666",
                    }).unwrap();
                } else {
                    wtr.serialize(Row {
                        pos_x: f64::to_string(&x_pos).as_str(),
                        pos_y: f64::to_string(&y_pos).as_str(),
                        color: "#AAA666",
                    }).unwrap();
                }
                
                
            }
        }

        let _ = wtr.flush();
    }

    /* 
    let mut wtr = Writer::from_path("C:/Users/Henrik/Repos/Simple-Neural-Network-Project/src/bin/web/network_generated_values.csv").unwrap();
        for x in 0..100 {
            for y in 0..100 {
                let x_pos = x as f64 / 100.0; // between 0 and 1
                let y_pos = y as f64 / 100.0; // between 0 and 1

                    wtr.serialize(Row {
                        pos_x: f64::to_string(&x_pos).as_str(),
                        pos_y: f64::to_string(&y_pos).as_str(),
                        color: "#FF6666",
                    }).unwrap();
                
                
                
            }
        }

    let _ = wtr.flush();
    */
    
    NamedFile::open_async("C:/Users/Henrik/Repos/Simple-Neural-Network-Project/src/bin/web/network_generated_values.csv").await.unwrap()
}


async fn d3() -> impl Responder {
    NamedFile::open_async("C:/Users/Henrik/Repos/Simple-Neural-Network-Project/src/bin/web/visualize.html").await.unwrap()
    //NamedFile::open_async("C:/Users/Henri/Repos/webWithRust/src/text.html").await.unwrap()
}

async fn dataset() -> impl Responder {
    NamedFile::open_async("C:/Users/Henrik/Repos/Simple-Neural-Network-Project/src/bin/web/dataset.csv").await.unwrap()
    //NamedFile::open_async("C:/Users/Henri/Repos/webWithRust/src/text.html").await.unwrap()
}


/* 
fn main() {


    
    println!("Hello, world!");
    
    let mut nw = NeuralNetwork::builder()
        .with_input_layer(2)
        .with_hidden_layer(5)
        .with_output_layer(3)
        .build_network().unwrap();

    let correct_output = vec![vec![0.0, 1.0, 0.0], vec![1.0, 0.0, 0.0]];
    let input = vec![vec![1.0, 1.0], vec![0.0, 0.5]]; // 

    let mut traning_handeler = TraningHandeler::new(&nw);
    traning_handeler.insert_traning_data(input, correct_output);
    traning_handeler.train_neural_network(&mut nw, 1000);

    let correct_output = vec![0.0, 1.0, 0.0];
    let input = vec![1.0, 1.0]; // 

    println!("cost: {}", back::calculate_cost(&correct_output, &input, &nw));

    let correct_output = vec![1.0, 0.0, 0.0];
    let input = vec![0.0, 0.5]; // 

    println!("cost: {}", back::calculate_cost(&correct_output, &input, &nw));

    /* 
    let mut gradient = backprop::backpropagate(&correct_output, &input, &nw);

    println!("Initial cost: {}", back::calculate_cost(&correct_output, &input, &nw));
    
    println!("Weight updated; {:?}", nw.output_layer.as_ref().unwrap().neurons.get(0).unwrap().weights.as_ref().unwrap().get(0).unwrap());

    gradient = backprop::backpropagate(&correct_output, &input, &nw);
    backprop::update_neural_network(&mut nw, &gradient);
    println!("{:?}", nw.calculate_values_of_all_neurons(&input));
    println!("cost: {}", back::calculate_cost(&correct_output, &input, &nw));
    


    for i in 0..10000 {
        gradient = backprop::backpropagate(&correct_output, &input, &nw);
        backprop::update_neural_network(&mut nw, &gradient);

        if i % 1000 == 1{
            println!("cost: {}", back::calculate_cost(&correct_output, &input, &nw));
        }
        //println!("{:?}", nw.calculate_values_of_all_neurons(&input));
        //println!("cost: {}", back::calculate_cost(&correct_output, &input, &nw));
    }

    gradient = backprop::backpropagate(&correct_output, &input, &nw);
    backprop::update_neural_network(&mut nw, &gradient);
    println!("{:?}", nw.calculate_values_of_all_neurons(&input));
    println!("cost: {}", back::calculate_cost(&correct_output, &input, &nw));


    println!("{:?}", nw.calculate_values_of_all_neurons(&input).unwrap());
    */
    /* 
    //let mut traning_handeler = TraningHandeler::new(SquishFunction::sigmoid, &mut nw);
    

    //let traning_data_input = vec![vec![1.0], vec![0.0], vec![0.5]];
    //let traning_data_correct_output = vec![vec![1.0], vec![0.0], vec![0.5]];

    //traning_handeler.insert_traning_data(traning_data_input, traning_data_correct_output);

    //traning_handeler.backpropagate_network(&mut nw);

    let correct_output = vec![0.0, 1.0];
    let input = vec![0.5, 0.5]; // 

    println!("initial cost: {}", back::calculate_cost(&correct_output, &input, &nw));
    println!("{:?}", nw.calculate_values_of_all_neurons(&input));
    let mut gradient = back::backpropagate_weights_bias(&correct_output, &input, &nw);
    back::update_neural_network(&mut nw, &gradient);
    println!("cost: {}", back::calculate_cost(&correct_output, &input, &nw));

    /*

    println!("Weight updated; {:?}", nw.output_layer.as_ref().unwrap().neurons.get(0).unwrap().weights.as_ref().unwrap().get(0).unwrap());

    gradient = back::backpropagate_weights_bias(&correct_output, &input, &nw);
    back::update_neural_network(&mut nw, &gradient);
    println!("{:?}", nw.calculate_values_of_all_neurons(&input));
    println!("cost: {}", back::calculate_cost(&correct_output, &input, &nw));
    */


    for i in 0..100000 {
        gradient = back::backpropagate_weights_bias(&correct_output, &input, &nw);
        back::update_neural_network(&mut nw, &gradient);

        if i % 10000 == 1{
            println!("cost: {}", back::calculate_cost(&correct_output, &input, &nw));
        }
        //println!("{:?}", nw.calculate_values_of_all_neurons(&input));
        //println!("cost: {}", back::calculate_cost(&correct_output, &input, &nw));
    }

    gradient = back::backpropagate_weights_bias(&correct_output, &input, &nw);
    back::update_neural_network(&mut nw, &gradient);
    println!("{:?}", nw.calculate_values_of_all_neurons(&input));
    println!("cost: {}", back::calculate_cost(&correct_output, &input, &nw));
    */
    
}
    */

    


