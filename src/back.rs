use std::{collections::HashMap, iter::{zip, Map}};

use crate::{layer::{self, Layer}, neural_network::{self, NeuralNetwork}, neuron::Neuron};

type Gradient = (Vec<f32>, f32);


enum TypeVariable{
    Weight, 
    Bias
}

struct VariableTreePosition<'a>{
    layer : Option<&'a Layer>,
    neuron : Option<&'a Neuron>, // if None, then this is the top of the variable tree: Thus at C
    variable : Option<TypeVariable>, 
    index : Option<i32> // if variable is weight, this then specify which weight
}

impl<'a> VariableTreePosition<'a>{
    pub fn new(layer : Option<&'a Layer>, neuron : Option<&'a Neuron>, variable : Option<TypeVariable>, index : Option<i32>) -> VariableTreePosition<'a>{
        VariableTreePosition{
            layer,
            neuron,
            variable,
            index
        }
    }
}




// return the gradient of neural network with a test case
pub fn backpropagate_weights_bias(correct_output_values : &Vec<f32>, input_to_neural_network : &Vec<f32>, neural_network : &NeuralNetwork) -> HashMap<Neuron, (Vec<f32>, f32)> /* key: Neuron, value: gradient(weights, bias)*/{

    // gradient
    let gradient : HashMap<Neuron, Gradient> = HashMap::new();

    // both hidden layers and output layer
    let mut layers : Vec<&Layer> = Vec::new();
    layers.extend(&neural_network.hidden_layers);
    layers.push(&neural_network.output_layer.as_ref().unwrap());

    // includes the first input layer
    let neurons_values_map = neural_network.calculate_values_of_all_neurons_map(input_to_neural_network).unwrap(); // layer -> neuron


    // calculate gradient in respect to weights
    layers.iter().for_each(|layer|{

        layer.neurons.iter().for_each(|neuron|{

            // gradient relative to weight
            for i in 0..neuron.weights.as_ref().unwrap().len() {
                let variable_tree_start = VariableTreePosition::new(None, None, None, None);
                let variable_tree_destination = 
                    VariableTreePosition::new(
                        Some(&layer),
                        Some(&neuron), 
                        Some(TypeVariable::Weight), 
                        Some(i as i32)
                );

                let derivative = calculate_partial_derivative(&variable_tree_start, &variable_tree_destination, neural_network, &neurons_values_map, &correct_output_values);
                println!("Some derivative weight: {}", derivative);
            }

            // gradient relative to bias
            let variable_tree_start = VariableTreePosition::new(None, None, None, None);
            let variable_tree_destination = 
                VariableTreePosition::new(
                    Some(&layer),
                    Some(&neuron), 
                    Some(TypeVariable::Bias), 
                    None
            );
            let derivative = calculate_partial_derivative(&variable_tree_start, &variable_tree_destination, neural_network, &neurons_values_map, &correct_output_values);
            println!("Some derivative bias: {}", derivative);

        })
    });
    return gradient;
}

pub fn derivative_of_sigmoid(input_value : f32) -> f32{
    let e_power_negative_x = f32::powf(std::f32::consts::E, input_value);
    return e_power_negative_x / f32::powf(1.0 + e_power_negative_x, 2.0);
}


fn calculate_partial_derivative(variable_tree_start : &VariableTreePosition, variable_tree_destination : &VariableTreePosition, neural_network : &NeuralNetwork, neurons_values : &HashMap<&Neuron, f32>, correct_output_values : &Vec<f32>) -> f32{
        // 
        //         Variable Tree
        //             C 
        //             |
        //             .
        //             .
        //             .
        //          -- |
        //         y   a(L)
        //             |
        //             z(L)
        //       ----------------
        //    w1..wn   |       b
        //             a(L-1)
        //             .
        //       ...   .   ...
        // 

    if variable_tree_start.neuron.is_some(){
        // Now we do not start at the top!

        // check if we should continue deeper in variable tree
        if variable_tree_start.layer.is_some() && variable_tree_start.layer == variable_tree_destination.layer{
            // Now we have reached our final neuron, becuase our destination variable level == current variable level

            // set our layer
            let current_layer = variable_tree_start.layer.unwrap();
            // set out current neuron
            let current_neuron = variable_tree_destination.neuron.unwrap();

            // check which variable type our partial derivative require
            match variable_tree_destination.variable.as_ref().unwrap(){
                TypeVariable::Weight => {

                    let weight_index = variable_tree_destination.index.unwrap();
                    let next_neuron = neural_network // This is the neuron a(L-1)(n)
                        .get_preceding_layer(current_layer)
                        .unwrap()
                        .neurons.get(weight_index as usize)
                        .unwrap();

                    // dZ(L)/dWn = a(L-1)(n), Thus return value of a(L-1)(n):
                    return neurons_values.get(next_neuron).unwrap().clone();
                }, 
                TypeVariable::Bias => {
                    return 1.0; // Because dz(L)/db = 1
                }
            }
        } else {
            // we should now continue deeper in variable tree, becuase our searched variable is at a preceding layer

            // check if we are at input layer
            if variable_tree_start.layer.is_none(){
                0.0 // We are at input layer with no weights or bias
            } else {
                // we should now continue deeper in variable tree, becuase our searched variable is at a preceding layer
                let current_neuron = variable_tree_start.neuron.unwrap();

                // this is dZ(L-1)(n) / da(L-1)(k)
                // Value of this is weight from neuron a(L)(n) to a(L-1)(k) = w(nk)
                let dZda = current_neuron.weights.as_ref().unwrap().iter().map(|weight|{

                    let preceding_layer = neural_network.get_preceding_layer(variable_tree_start.layer.unwrap()).unwrap();
                    
                    let new_variable_tree_start = VariableTreePosition::new(Some(&preceding_layer), variable_tree_destination.neuron, None, None);
                    return weight * calculate_partial_derivative(&new_variable_tree_start, variable_tree_destination, neural_network, neurons_values, correct_output_values);
                }).sum(); // according to chain rule
                return dZda;
            }
        }
    } else {

        // we start at the top, at the variable C
        //         Variable Tree
        //             C 
        //             |
        //             .
        //             .
        //             .
        //          -- |
        //         y   a(L)
        //             |
        //             z(L)
        //       ----------------
        //    w1..wn   |       b
        //             a(L-1)
        //             .
        //       ...   .   ...
        // 

        let output_layer = neural_network.output_layer.as_ref().unwrap();

        // calculate dC/da(L)(1), ... , dC/da(L)(n)
        let derivative : f32 = zip(output_layer.neurons.iter(), correct_output_values).map(|(neuron, correct_value)|{

            let dCdA = 2.0 * (neurons_values.get(neuron).unwrap() - correct_value); 
            let dAdZ = derivative_of_sigmoid(neurons_values.get(neuron).unwrap().clone());

            // create new position for next iteration in the variable tree, Thus walk down one step
            let new_variable_tree_start = 
                // for the next iteration we set out output layer as our start layer in our variable tree
                // Thus we start at z(L) in next iteration
                VariableTreePosition::new(Some(&output_layer), variable_tree_destination.neuron, None, None);

            // recursively calculate next partials
            return dCdA * dAdZ * calculate_partial_derivative(&new_variable_tree_start, variable_tree_destination, neural_network, neurons_values, correct_output_values);

        }).sum(); // according to chain rule

        return derivative;
    }
}
