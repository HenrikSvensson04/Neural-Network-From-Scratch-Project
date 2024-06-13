



mod neural_network;
mod neuron;
mod layer;
mod Layer;

use crate::neural_network::NeuralNetwork;


fn main() {
    println!("Hello, world!");

    let nw = NeuralNetwork::builder()
        .with_input_layer(1)
        .with_hidden_layer(1)
        .with_output_layer(1)
        .build_network().unwrap();
}
