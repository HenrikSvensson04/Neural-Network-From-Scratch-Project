



mod neural_network;
mod neuron;
mod layer;
mod backpropagation;
mod back;
mod Layer;
mod Neuron;
//mod Layer;

use backpropagation::{SquishFunction, TraningHandeler};

use crate::neural_network::NeuralNetwork;


fn main() {
    println!("Hello, world!");
    
    let mut nw = NeuralNetwork::builder()
        .with_input_layer(1)
        .with_hidden_layer(1)
        //.with_hidden_layer(2)
        .with_output_layer(1)
        .build_network().unwrap();

    //let mut traning_handeler = TraningHandeler::new(SquishFunction::sigmoid, &mut nw);
    

    //let traning_data_input = vec![vec![1.0], vec![0.0], vec![0.5]];
    //let traning_data_correct_output = vec![vec![1.0], vec![0.0], vec![0.5]];

    //traning_handeler.insert_traning_data(traning_data_input, traning_data_correct_output);

    //traning_handeler.backpropagate_network(&mut nw);

    back::backpropagate_weights_bias(&vec![1.0], &vec![0.5], &nw);
    
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_number_of_neurons_and_connections_for_layers() {
        let mut nw = NeuralNetwork::builder()
            .with_input_layer(1)
            .with_hidden_layer(5)
            .with_hidden_layer(3)
            .with_output_layer(1)
            .build_network().unwrap();

            //println!("{:?}", nw.hidden_layers.get_mut(0).unwrap().neurons.len());
            // check that hidden layers has correct number neurons
            assert_eq!(nw.hidden_layers.get_mut(0).unwrap().neurons.len(), 5);
            assert_eq!(nw.hidden_layers.get_mut(1).unwrap().neurons.len(), 3);

            let number_of_connections_per_neuron_first_hidden_layer = nw.hidden_layers.get_mut(0).unwrap().neurons.get_mut(0).unwrap().weights.clone().unwrap().len();
            let number_of_connections_per_neuron_second_hidden_layer = nw.hidden_layers.get_mut(1).unwrap().neurons.get_mut(0).unwrap().weights.clone().unwrap().len();
            let number_of_connections_per_neuron_output_layer = nw.output_layer.unwrap().neurons.get_mut(0).unwrap().weights.clone().unwrap().len();
            // check that neurons have correct number of connections
            assert_eq!(number_of_connections_per_neuron_first_hidden_layer, 1);
            assert_eq!(number_of_connections_per_neuron_second_hidden_layer, 5);
            assert_eq!(number_of_connections_per_neuron_output_layer, 3);
    }


    #[test]
    fn test_correct_values_of_neurons() {

        let number_of_neurons_input_layer = 2;
        let number_of_neurons_hidden_layer_1 = 3;
        let mut nw = NeuralNetwork::builder()
            .with_input_layer(number_of_neurons_input_layer)
            .with_hidden_layer(number_of_neurons_hidden_layer_1)
            .with_output_layer(1)
            .build_network().unwrap();

            
            // change weights to 1.0 and bias to 5.0
            {
                // hidden
                nw.hidden_layers.get_mut(0).unwrap().neurons.iter_mut().for_each(|neuron|{
                    neuron.weights = Some(vec![1.0; number_of_neurons_input_layer as usize]);
                    neuron.bias = Some(5.0);
                });
                
                //output
                nw.output_layer.as_mut().unwrap().neurons.iter_mut().for_each(|neuron|{
                    neuron.weights = Some(vec![1.0; number_of_neurons_hidden_layer_1 as usize]);
                    neuron.bias = Some(5.0);
                });
            }
            // set input neuron values
            let input = vec![2.0; number_of_neurons_input_layer as usize];

            // calculate values of all neurons in network
            let neuron_values = nw.calculate_values_of_all_neurons(&input).unwrap();

            let mut traning_handeler = TraningHandeler::new(SquishFunction::sigmoid, &mut nw);

            // I1 -> n11 
            // I2 -> n12 -> n21
            //    -> n13
            // let's calculate dn11/dw1
            //let dn11dw = traning_handeler.calculate_partial_derivative(10.0, (1, 1, 1, 0), &nw, &neuron_values);
            // should be = 2 * (A(L) - y)
            
            



    }

    
    #[test]
    fn test_correct_partial_derivatives() {

        let number_of_neurons_input_layer = 2;
        let number_of_neurons_hidden_layer_1 = 3;
        let mut nw = NeuralNetwork::builder()
            .with_input_layer(number_of_neurons_input_layer)
            .with_hidden_layer(number_of_neurons_hidden_layer_1)
            .with_output_layer(1)
            .build_network().unwrap();

            
            // change weights to 1.0 and bias to 5.0
            {
                // hidden
                nw.hidden_layers.get_mut(0).unwrap().neurons.iter_mut().for_each(|neuron|{
                    neuron.weights = Some(vec![1.0; number_of_neurons_input_layer as usize]);
                    neuron.bias = Some(5.0);
                });
                
                //output
                nw.output_layer.as_mut().unwrap().neurons.iter_mut().for_each(|neuron|{
                    neuron.weights = Some(vec![1.0; number_of_neurons_hidden_layer_1 as usize]);
                    neuron.bias = Some(5.0);
                });
            }
            // set input neuron values
            let input = vec![2.0; number_of_neurons_input_layer as usize];

            //------------------------------------------------------------------
            // TEST VECTOR VERSION: calculate values of all neurons in network
            let neuron_values = nw.calculate_values_of_all_neurons(&input).unwrap();
            // Thus we should have
            // I1 -> n11 
            // I2 -> n12 -> n21
            //    -> n13
            //
            // where: I1 = I2 = 2.0, 
            // n11 = n12 = n13 = I1 * 1.0 + I2 * 1.0 + 5 = 9;
            // n21 = n12 * 1.0 + .. + n13 * 1.0 + 5 = 9 + 9 + 9 + 5 = 32
            println!("{:?}", neuron_values);
            assert_eq!(neuron_values.get(0).unwrap().get(0).unwrap().clone(), 2.0 as f32); // Thus input I1 = I2 = 2.0
            assert_eq!(neuron_values.get(1).unwrap().get(0).unwrap().clone(), 9.0 as f32); // Thus n11 or n12 or n13 = 9
            assert_eq!(neuron_values.get(2).unwrap().get(0).unwrap().clone(), 32.0 as f32); // Thus n21 = 32

            //-----------------------------------------------------------------
            // TEST HASHMAP VERSION: calculate values of all neurons in network 
            let neuron_values = nw.calculate_values_of_all_neurons_map(&input).unwrap();
            // Thus we should have
            // I1 -> n11 
            // I2 -> n12 -> n21
            //    -> n13
            //
            // where: I1 = I2 = 2.0, 
            // n11 = n12 = n13 = I1 * 1.0 + I2 * 1.0 + 5 = 9;
            // n21 = n12 * 1.0 + .. + n13 * 1.0 + 5 = 9 + 9 + 9 + 5 = 32

            let I1 = nw.input_layer.as_ref().unwrap().neurons.get(0).unwrap();
            assert_eq!(neuron_values.get(I1).unwrap().clone(), 2.0 as f32); // Thus input I1 = I2 = 2.0

            let n11 = nw.hidden_layers.get(0).unwrap().neurons.get(0).unwrap();
            assert_eq!(neuron_values.get(n11).unwrap().clone(), 9.0 as f32); // Thus n11 or n12 or n13 = 9

            let n21 = nw.output_layer.as_ref().unwrap().neurons.get(0).unwrap();
            assert_eq!(neuron_values.get(n21).unwrap().clone(), 32.0 as f32); // Thus n21 = 32

    }
}





