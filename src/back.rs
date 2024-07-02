use std::{collections::HashMap, iter::{zip, Map}};

use crate::{layer::{self, Layer}, neural_network::{self, NeuralNetwork}, neuron::{self, Neuron}, util};

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

/// Calculate the cost a neural network given a test case
pub fn calculate_cost(correct_output_values : &Vec<f32>, input_to_neural_network : &Vec<f32>, neural_network : &NeuralNetwork) -> f32{

    // values of neurons in neural network given the input
    let neurons_values_map = neural_network.calculate_values_of_all_neurons_map(input_to_neural_network).unwrap(); // layer -> neuron

    let cost = zip(correct_output_values, neural_network.output_layer.as_ref().unwrap().neurons.iter()).map(|(correct_output, neuron)|{
        

        let inner =  neurons_values_map.get(neuron).unwrap() - correct_output;
        return inner * inner;
    }).sum();

    return cost;
}

/// calculate the gradient of neural network given a test case - and optimize the neural network with the gradient
pub fn backpropagate_weights_bias<'a>(correct_output_values : &'a Vec<f32>, input_to_neural_network : &'a Vec<f32>, neural_network : &'a NeuralNetwork) -> HashMap<Neuron, Gradient>/* key: Neuron, value: gradient(weights, bias)*/{

    // gradient
    let mut gradient : HashMap<Neuron, Gradient> = HashMap::new(); //Where: type Gradient = (Vec<f32>, f32); Thus: (weights, bias)

    // includes the first input layer
    let neurons_values_map = neural_network.calculate_values_of_all_neurons_map(input_to_neural_network).unwrap(); // layer -> neuron
    let neurons_z_values_map = neural_network.calculate_values_of_all_neurons_z_value_map(input_to_neural_network).unwrap();

    {
        // both hidden layers and output layer
        let mut layers : Vec<&Layer> = Vec::new();
        layers.extend(&neural_network.hidden_layers);
        layers.push(&neural_network.output_layer.as_ref().unwrap());

        //println!("Lewngth: {:?}", layers.len());


        // calculate gradient in respect to weights
        layers.iter().for_each(|layer|{

            layer.neurons.iter().for_each(|neuron|{

                let mut local_gradient = (Vec::new(), 0.0);
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

                    let derivative = calculate_partial_derivative(&variable_tree_start, &variable_tree_destination, neural_network, &neurons_values_map, &neurons_z_values_map, &correct_output_values);
                    //println!("Some derivative weight: {}", derivative);

                    local_gradient.0.push(derivative);
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
                let derivative = calculate_partial_derivative(&variable_tree_start, &variable_tree_destination, neural_network, &neurons_values_map, &neurons_z_values_map, &correct_output_values);
                //println!("Some derivative bias: {}", derivative);

                local_gradient.1 = derivative;

                // bad solution? Should not really be a clone, it should reference to neural network, however references here does not permit changing the network later on.
                gradient.insert(neuron.clone(), local_gradient);

            })
        });
    }
    return gradient;
}


pub fn update_neural_network(neural_network : &mut NeuralNetwork, gradient : &HashMap<Neuron, Gradient>){

    // Update the neural network with the derivatives
    // Note: The cost function reduces if we move in the direction of the negative gradient
    // Moreover, the gradient tells us how much the cost function change given a change to input variables.

    let iter_neurons = neural_network.hidden_layers.iter_mut().chain(neural_network.output_layer.iter_mut());
    iter_neurons.for_each(|layer|{
        layer.neurons.iter_mut().for_each(|neuron|{
            if let Some(gradient_for_neuron) = gradient.get(neuron){
                zip(neuron.weights.as_mut().unwrap(), &gradient_for_neuron.0).into_iter().for_each(|(weight, weight_derivative)|{
                    *weight = (*weight) -  (1.0) * weight_derivative;
                    //println!("Derivative weight: {}", weight_derivative);
                }); 

                if let Some(bias) = &mut neuron.bias{
                    *bias = (*bias) -  (1.0) * gradient_for_neuron.1;
                }
            }
        });
    });
}


fn calculate_partial_derivative(variable_tree_start : &VariableTreePosition, variable_tree_destination : &VariableTreePosition, neural_network : &NeuralNetwork, neurons_values : &HashMap<&Neuron, f32>, neurons_z_values : &HashMap<&Neuron, f32>, correct_output_values : &Vec<f32>) -> f32{
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

    // check if we should calculate derivative relative to C or a neuron
    if variable_tree_start.neuron.is_some(){
        // Now we do not start at the top! Thus not at C variable

        // check if we should continue deeper in variable tree
        if variable_tree_start.layer.is_some() && variable_tree_start.layer == variable_tree_destination.layer{
            // Now we have reached our final neuron, becuase our destination variable level == current variable level

            //println!("Finished:");
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
                    println!("A(L-1)(n): {}", neurons_values.get(next_neuron).unwrap().clone());
                    return neurons_values.get(next_neuron).unwrap().clone();
                }, 
                TypeVariable::Bias => {
                    return 1.0; // Because dz(L)/db = 1
                }
            }
        } else {

            println!("Go deeper:");
            // we should now continue deeper in variable tree, becuase our searched variable is at a preceding layer

            // check if we are at input layer
            if variable_tree_start.layer.is_none(){
                0.0 // We are at input layer with no weights or bias
            } else {
                // we should now continue deeper in variable tree, becuase our searched variable is at a preceding layer
                let current_neuron = variable_tree_start.neuron.unwrap();
                let current_layer = variable_tree_start.layer.unwrap();
                let preceding_layer = neural_network.get_preceding_layer(current_layer).unwrap();

                // this is dZ(L-1)(n) / da(L-1)(k)
                // Value of this is weight from neuron a(L)(n) to a(L-1)(k) = w(nk)

                let dZda = zip(current_neuron.weights.as_ref().unwrap().iter(), preceding_layer.neurons.iter()).map(|(weight, neuron)|{
                    let new_variable_tree_start = VariableTreePosition::new(Some(&preceding_layer), Some(neuron), None, None);
                    println!("dz(L)/da(L-1) {:?}", weight);

                    // because we want what is inside the sigmoid : Thus z(L), use the inverse sigmoid
                    let z = neurons_z_values.get(neuron).unwrap().clone();
                    //let z = util::inverse_sigmoid(neurons_values.get(neuron).unwrap().clone()); // TODO: replace this, as it seems it gives value: inf, infinity sometimes
                    let dAdZ = util::derivative_of_sigmoid(z);
                    println!("Z(L): {}", z);
                    println!("dAdZ: {}", dAdZ);

                    let daLdz = weight;

                    return daLdz * dAdZ * calculate_partial_derivative(&new_variable_tree_start, variable_tree_destination, neural_network, neurons_values, neurons_z_values, correct_output_values);
                }).sum(); // according to chain rule
                
                /* 
                let dZda = current_neuron.weights.as_ref().unwrap().iter().map(|weight|{
                    let preceding_layer = neural_network.get_preceding_layer(variable_tree_start.layer.unwrap()).unwrap();
                    
                    let new_variable_tree_start = VariableTreePosition::new(Some(&preceding_layer), variable_tree_destination.neuron, None, None);
                    println!("dz(L)/da(L-1) {:?}", weight);
                    return weight * calculate_partial_derivative(&new_variable_tree_start, variable_tree_destination, neural_network, neurons_values, correct_output_values);
                }).sum(); // according to chain rule
                */
                return dZda;
            }
        }
    } else {

        //println!("Top:");

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
            println!("dCdA: {}", dCdA);
            // because we want what is inside the sigmoid : Thus z(L), use the inverse sigmoid
            //let z = util::inverse_sigmoid(neurons_values.get(neuron).unwrap().clone()); // TODO: replace this, as it seems it gives value: inf, infinity sometimes
            let z = neurons_z_values.get(neuron).unwrap().clone();
            let dAdZ = util::derivative_of_sigmoid(z);
            println!("Z(L): {}", z);
            println!("dAdZ: {}", dAdZ);

            // create new position for next iteration in the variable tree, Thus walk down one step
            let new_variable_tree_start = 
                // for the next iteration we set out output layer as our start layer in our variable tree
                // Thus we start at z(L) in next iteration
                VariableTreePosition::new(Some(&output_layer), Some(neuron), None, None);

            // recursively calculate next partials
            return dCdA * dAdZ * calculate_partial_derivative(&new_variable_tree_start, variable_tree_destination, neural_network, neurons_values, neurons_z_values, correct_output_values);

        }).sum(); // according to chain rule

        return derivative;
    }
}
