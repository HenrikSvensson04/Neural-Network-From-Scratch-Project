



mod neural_network;
mod neuron;
mod layer;
mod backpropagation;
//mod Layer;

use backpropagation::{SquishFunction, TraningHandeler};

use crate::neural_network::NeuralNetwork;


fn main() {
    println!("Hello, world!");
    
    let mut nw = NeuralNetwork::builder()
        .with_input_layer(1)
        .with_hidden_layer(1)
        .with_hidden_layer(2)
        .with_output_layer(1)
        .build_network().unwrap();

    let mut traning_handeler = TraningHandeler::new(SquishFunction::sigmoid, &mut nw);
    

    let traning_data_input = vec![vec![1.0], vec![0.0], vec![0.5]];
    let traning_data_correct_output = vec![vec![1.0], vec![0.0], vec![0.5]];

    traning_handeler.insert_traning_data(traning_data_input, traning_data_correct_output);

    traning_handeler.backpropagate_network(&mut nw);
    
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_number_of_neurons_and_connections_for_layers() {
        let mut nw = NeuralNetwork::builder()
            .with_input_layer(1)
            .with_hidden_layer(5)
            .with_hidden_layer(2)
            .with_output_layer(1)
            .build_network().unwrap();

            //println!("{:?}", nw.hidden_layers.get_mut(0).unwrap().neurons.len());
            // check that hidden layers has correct number neurons
            assert_eq!(nw.hidden_layers.get_mut(0).unwrap().neurons.len(), 5);
            assert_eq!(nw.hidden_layers.get_mut(1).unwrap().neurons.len(), 2);

            let number_of_connections_per_neuron_first_hidden_layer = nw.hidden_layers.get_mut(0).unwrap().neurons.get_mut(0).unwrap().weights.clone().unwrap().len();
            let number_of_connections_per_neuron_second_hidden_layer = nw.hidden_layers.get_mut(1).unwrap().neurons.get_mut(0).unwrap().weights.clone().unwrap().len();
            let number_of_connections_per_neuron_output_layer = nw.output_layer.unwrap().neurons.get_mut(0).unwrap().weights.clone().unwrap().len();
            // check that neurons have correct number of connections
            assert_eq!(number_of_connections_per_neuron_first_hidden_layer, 1);
            assert_eq!(number_of_connections_per_neuron_second_hidden_layer, 5);
            assert_eq!(number_of_connections_per_neuron_output_layer, 2);
    }
}





