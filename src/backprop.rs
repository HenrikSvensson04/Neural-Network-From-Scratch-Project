use std::{collections::HashMap, iter::zip};

use crate::{layer::Layer, neural_network::NeuralNetwork, neuron::Neuron, util};



use nalgebra::{DMatrix, DVector};



type Gradient = (Vec<f32>, f32);



pub fn backpropagate(correct_output_values : &Vec<f32>, input_to_neural_network : &Vec<f32>, neural_network : &NeuralNetwork) -> HashMap<Neuron, Gradient>{

    let mut gradient : HashMap<Neuron, Gradient> = HashMap::new(); //Where: type Gradient = (Vec<f32>, f32); Thus: (weights, bias)
    
    let neurons_values_map = neural_network.calculate_values_of_all_neurons_map(input_to_neural_network).unwrap(); 
    let neurons_z_values_map = neural_network.calculate_values_of_all_neurons_z_value_map(input_to_neural_network).unwrap();

    // calculate partial derivatives between output layer and cost function
    let partials_output_layer : Vec<f32> = zip(neural_network.output_layer.as_ref().unwrap().neurons.iter(), correct_output_values.iter()).map(|(neuron, correct_output)|{

        let dCdA = 2.0 * (neurons_values_map.get(neuron).unwrap() - correct_output);

        let z = neurons_z_values_map.get(neuron).unwrap();

        let dCdZ = dCdA * util::derivative_of_sigmoid(z.clone());
        // calculate dw and db from partial dC/dz(L) and add to gradient
        calculate_final_partials_to_gradient(dCdZ, &neuron, neural_network.output_layer.as_ref().unwrap(), &neurons_values_map, &neural_network, &mut gradient);
        dCdZ
    }).collect();

    // transform into nalgebra type partials_vector
    let mut partials_vector = DVector::from_column_slice(&partials_output_layer); // this is partials_vector of previous layer's partials dC/dZ(L+1)
    //println!("{:?}", partials_vector);

    let mut previous_layer = neural_network.output_layer.as_ref().unwrap();

    neural_network.hidden_layers.iter().rev().for_each(|hidden_layer|{
        
        let mut matrix = DMatrix::zeros(hidden_layer.neurons.len(), partials_vector.nrows());

        let mut row = 0; // we also use this for direct indexing to locate weights in neuron.rs
        let mut col = 0;

        // this iterates neurons in A(L-1)
        hidden_layer.neurons.iter().for_each(|neuron|{
            // this iterates neurons upper layer, thus for A(L)
            col = 0;
            previous_layer.neurons.iter().for_each(|neuron_in_previous_layer|{

                // Calculate partial: where z(L)(n) / z(L-1)(k) = weight (from layer L to L-1)
                // Where: z(L)(n) / z(L-1)(k) = (z(L)(n) / a(L-1)(k)) * (a(L-1)(k) / z(L-1)(k))
                let z_lower_value = neurons_z_values_map.get(neuron).unwrap().clone(); // Thus value of: z(L-1)(k)

                //println!("{:?} and: row: {}", neuron_in_previous_layer.weights, row);
                //println!("");
                //println!("row: {}, column: {}", row, col);
                //println!("Rows in matrix: {}, Columns in matrix: {}", matrix.nrows(), matrix.ncols());
                //println!("{:?}", matrix);
                let dz_upper_dz_lower = neuron_in_previous_layer.weights.as_ref().unwrap().get(row).unwrap().clone() * util::derivative_of_sigmoid(z_lower_value);
                matrix[(row, col)] = dz_upper_dz_lower;
                col += 1;
            });

            row += 1;
        });

        let partials = &matrix * &partials_vector;

        // add to gradient
        zip(partials.iter(), hidden_layer.neurons.iter()).for_each(|(partial, neuron)|{

            // calculate dw and db from partial dC/dz(L) and add to gradient
            calculate_final_partials_to_gradient(partial.clone(), &neuron, &hidden_layer, &neurons_values_map, &neural_network, &mut gradient)
        });

        // update partials_vector for next iteration with the calulated partials dC/dA(L), from this layer L
        partials_vector = DVector::from_column_slice(partials.as_slice());

        // switch previous layer to the current layer
        previous_layer = hidden_layer;

    });

    gradient

}


/// calculates dw and db from partial dC/dz(L)
/// inserts into gradient
pub fn calculate_final_partials_to_gradient(partial : f32, current_neuron : &Neuron, current_layer : &Layer, neurons_values_map : &HashMap<&Neuron, f32>, neural_network : &NeuralNetwork, gradient : &mut HashMap<Neuron, Gradient>){
    // relative to bias 
    let dCdB = partial * 1.0;

    // relative to weights
    let preceding_layer = neural_network.get_preceding_layer(current_layer).unwrap(); // the layer below
    let weights_derivatives_vec : Vec<f32> = preceding_layer.neurons.iter().map(|neuron_in_preceding_layer|{
        let dCdw = partial * neurons_values_map.get(neuron_in_preceding_layer).unwrap(); // Thus dZ(L-1) / dW(k) = a(L-2)(k)
        dCdw
    }).collect();
    //println!("{:?}", weights_derivatives_vec);
    // add to hashmap gradient
    gradient.insert(current_neuron.clone(), (weights_derivatives_vec, dCdB));
}


    



