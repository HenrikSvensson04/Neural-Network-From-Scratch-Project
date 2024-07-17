use std::{collections::HashMap, iter::zip, ops::Add};

use crate::{layer::Layer, neural_network::NeuralNetwork, neural_network::TypeNeuronValue, neuron::Neuron, util};
use nalgebra::{DMatrix, DVector};


type PartialDerivatives = (Vec<f32>, f32);
pub struct Gradient{
    pub inside_map : HashMap<Neuron, PartialDerivatives>
}

impl Gradient{
    pub fn new() -> Gradient{
        Gradient{
            inside_map : HashMap::new()
        }
    }
}

impl Add for Gradient {
    type Output = Gradient;
    fn add(self, mut other: Gradient) -> Gradient {
        let mut new_gradient : HashMap<Neuron, PartialDerivatives> = HashMap::new();

        self.inside_map.into_iter().for_each(|(neuorn, partial_derivative_self)|{

            if let Some(partial_derivative_other) = other.inside_map.remove(&neuorn){
                let new_weights_derivatives : Vec<f32> = zip(partial_derivative_other.0, partial_derivative_self.0).map(|(a, b)|{
                    a + b
                }).collect();

                let new_bias_derivative = partial_derivative_other.1 + partial_derivative_self.1;
                new_gradient.insert(neuorn, (new_weights_derivatives, new_bias_derivative));
            }
        });

        Gradient{
            inside_map : new_gradient
        }
    }
}

/// Calculate the cost of a neural network given a test case
/// This calculates the: Mean Absolute Error
pub fn calculate_cost(correct_output_values : &Vec<f32>, input_to_neural_network : &Vec<f32>, neural_network : &NeuralNetwork) -> f32{

    // values of neurons in neural network given the input
    let neurons_values_map = neural_network.feedforward_to_map(input_to_neural_network).unwrap(); // layer -> neuron

    let cost = zip(correct_output_values, neural_network.output_layer.as_ref().unwrap().neurons.iter()).map(|(correct_output, neuron)|{
        let inner =  neurons_values_map.get(&(neuron, TypeNeuronValue::A)).unwrap() - correct_output;
        return inner * inner;
    }).sum();

    return cost;
}


/// Update a neural network with gradient
pub fn update_neural_network(neural_network : &mut NeuralNetwork, gradient : &Gradient, learning_rate : &f32){

    // Updates the neural network with the derivatives
    // Note: The cost function reduces if we move in the direction of the negative gradient
    // Moreover, the gradient tells us how much the cost function change given a change to input variables.

    let iter_neurons = neural_network.hidden_layers.iter_mut().chain(neural_network.output_layer.iter_mut());
    iter_neurons.for_each(|layer|{
        layer.neurons.iter_mut().for_each(|neuron|{
            if let Some(gradient_for_neuron) = gradient.inside_map.get(neuron){
                zip(neuron.weights.as_mut().unwrap(), &gradient_for_neuron.0).into_iter().for_each(|(weight, weight_derivative)|{
                    *weight = (*weight) -  (1.0) * weight_derivative * learning_rate;
                }); 

                if let Some(bias) = &mut neuron.bias{
                    *bias = (*bias) -  (1.0) * gradient_for_neuron.1 * learning_rate;
                }
            }
        });
    });
}

/// Backpropagate a neural network
/// Returns a gradient 
pub fn backpropagate(correct_output_values : &Vec<f32>, input_to_neural_network : &Vec<f32>, neural_network : &NeuralNetwork) -> Gradient{

    let mut gradient : Gradient = Gradient::new(); 
    let feedforward_values = neural_network.feedforward_to_map(input_to_neural_network).unwrap();

    // calculate partial derivatives between output layer and cost function
    let partials_output_layer : Vec<f32> = zip(neural_network.output_layer.as_ref().unwrap().neurons.iter(), correct_output_values.iter()).map(|(neuron, correct_output)|{

        let dCdA = 2.0 * (feedforward_values.get(&(neuron, TypeNeuronValue::A)).unwrap() - correct_output);
        let z = feedforward_values.get(&(neuron, TypeNeuronValue::Z)).unwrap();
        let dCdZ = dCdA * util::derivative_of_sigmoid(z.clone());

        // calculate dw and db from partial dC/dz(L) and add to gradient
        calculate_final_partials_to_gradient(dCdZ, &neuron, neural_network.output_layer.as_ref().unwrap(), &feedforward_values, &neural_network, &mut gradient);
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
                let z_lower_value = feedforward_values.get(&(neuron, TypeNeuronValue::Z)).unwrap().clone(); // Thus value of: z(L-1)(k)
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
            calculate_final_partials_to_gradient(partial.clone(), &neuron, &hidden_layer, &feedforward_values, &neural_network, &mut gradient)
        });

        // update partials_vector for next iteration with the calulated partials dC/dA(L), from this layer L
        partials_vector = DVector::from_column_slice(partials.as_slice());

        // switch previous layer to the current layer
        previous_layer = hidden_layer;
    });

    gradient

}


/// Calculates a neurons derivatives dW and dB from existing partial dC/dZ(L)
/// Then inserts values into gradient
fn calculate_final_partials_to_gradient(partial : f32, current_neuron : &Neuron, current_layer : &Layer, neurons_values_map : &HashMap<(&Neuron, TypeNeuronValue), f32>, neural_network : &NeuralNetwork, gradient : &mut Gradient){
    
    // relative to bias 
    let dCdB = partial * 1.0;

    // relative to weights
    let preceding_layer = neural_network.get_preceding_layer(current_layer).unwrap(); // the layer below
    let weights_derivatives_vec : Vec<f32> = preceding_layer.neurons.iter().map(|neuron_in_preceding_layer|{
        let dCdw = partial * neurons_values_map.get(&(neuron_in_preceding_layer, TypeNeuronValue::A)).unwrap(); // Thus dZ(L-1) / dW(k) = a(L-2)(k)
        dCdw
    }).collect();

    // add to hashmap gradient
    gradient.inside_map.insert(current_neuron.clone(), (weights_derivatives_vec, dCdB));
}