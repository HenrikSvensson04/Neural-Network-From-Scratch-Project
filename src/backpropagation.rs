use std::{iter::zip};

use crate::{backpropagation, layer::Layer, neural_network::{self, NeuralNetwork}};


type local_neuron_tuple = (Vec<f32>, Vec<f32>);

pub enum SquishFunction{
    sigmoid
}

pub struct TraningHandeler{
    squish_function : SquishFunction,
    error : Option<f32>,
    // todo: insert traning data
    traning_data_input : Vec<Vec<f32>>,
    traning_data_correct_output : Vec<Vec<f32>>,
    traning_data_num_neurons_input : u32,
    traning_data_num_neurons_output : u32
}

impl TraningHandeler{
    
    /// select the neural network that you wish to associate with this traningHandeler
    pub fn new(squish_function : SquishFunction, neural_network : &mut NeuralNetwork) -> TraningHandeler{
        TraningHandeler{
            squish_function : squish_function, 
            error : None,
            traning_data_input : Vec::new(),
            traning_data_correct_output : Vec::new(),
            traning_data_num_neurons_input : neural_network.input_layer.as_mut().unwrap().get_number_of_neurons() as u32,
            traning_data_num_neurons_output : neural_network.output_layer.as_mut().unwrap().get_number_of_neurons() as u32
        }
    }

    /// insert traning data -- verifies that data is compatible with selected neural network
    pub fn insert_traning_data(&mut self, traning_input_data : Vec<Vec<f32>>, traning_correct_output : Vec<Vec<f32>>) -> Option<()>{

        let mut mismatch_data_format = false;

        zip(traning_input_data.iter(), traning_input_data.iter()).for_each(|(input_data, correct_ouput)|{
            if input_data.len() as u32 != self.traning_data_num_neurons_input && correct_ouput.len() as u32 != self.traning_data_num_neurons_output{
                mismatch_data_format = true;
            }
        });

        if mismatch_data_format{ return None};

        self.traning_data_input = traning_input_data;
        self.traning_data_correct_output = traning_correct_output;
        return Some(());
    }


    /// backpropagate network.....
    pub fn backpropagate_network(&mut self, neural_network : &mut NeuralNetwork) -> Option<()>{

        if neural_network.input_layer.is_some() && neural_network.output_layer.is_some(){


            // Todo: scroll over each traning data
            //self.traning_data_input

            // generate gradient layout
            //type local_neuron_tuple = (Vec<f32>, Vec<f32>);           // layer -> neuron -> (weights, bias)per neuron
            let mut gradient : Vec<Vec<local_neuron_tuple>> = Vec::new(); // layer -> neuron -> (weights, bias), per neuron

            // for input layer - unneccesary???
            //gradient.push(self.initalize_gradient_for_layer(&neural_network.input_layer.as_ref().unwrap()));

            // for hidden layers
            neural_network.hidden_layers.iter().for_each(|layer|{
                let neurons : Vec<local_neuron_tuple> = self.initalize_gradient_for_layer(layer);
                gradient.push(neurons);
            });

            // for output layer
            gradient.push(self.initalize_gradient_for_layer(&neural_network.output_layer.as_ref().unwrap()));

            println!("{:?}", gradient);

            gradient = self.calculate_gradient(&gradient, &neural_network);

            // calculate gradient

            return Some(());
        } 
        
        
        return None;
    }

    pub fn initalize_gradient_for_layer(&self, layer : &Layer)-> Vec<local_neuron_tuple>{
        return layer.neurons.iter().map(|neuron|{
            /* 
            let tuple = zip(neuron.weights, neuron.bias).map(|(weights, bias)|{
                return 
            })
            */
            let mut tuple : local_neuron_tuple = (Vec::new(), Vec::new());
            for i in 0..neuron.weights.as_ref().unwrap().len(){
                tuple.0.push(-1.0);
                tuple.1.push(-1.0);
            }

            return tuple;
        }).collect();
    }

    /// calculate partial derivatives relative to all weights and bias of each neuron
    pub fn calculate_gradient(&self, gradient : &Vec<Vec<local_neuron_tuple>>, neural_network : &NeuralNetwork) -> Vec<Vec<local_neuron_tuple>>{

        // calculate derivatives to weights
        for i in 0..gradient.len(){
            let layer = gradient.get(i).unwrap();
            for j in 0..layer.len(){
                let neuron_weights_biases = layer.get(j).unwrap();
                for k in 0..neuron_weights_biases.0.len(){ // Note: becuse num weights = num biases
                    
                    let weight_pos = (i, j, k, 0 as usize); // layer, neuron, connection/edge, weight or bias
                    let bias_pos = (i, j, k, 1 as usize);

                    let weight_partial = self.calculate_partial_derivative(weight_pos, &neural_network);
    
                    
                }


            }
        }
        // calculate derivatives to weights

        Vec::new()
    }

    pub fn calculate_partial_derivative(&self, variable_pos : (usize, usize, usize, usize), neural_network : &NeuralNetwork) -> f32{


        10.0
    }


    pub fn derivative_of_sigmoid(&self, input_value : f32) -> f32{
        let e_power_negative_x = f32::powf(std::f32::consts::E, input_value);
        return e_power_negative_x / f32::powf(1.0 + e_power_negative_x, 2.0);
    }

    ///         Variable Tree
    ///           C or a(L+1)
    ///             |
    ///             .
    ///             .
    ///             .
    ///          -- |
    ///         y   a(L)
    ///             |
    ///             z(L)
    ///       ----------------
    ///    w1..wn   |      b1..bn
    ///             a(L-1)
    ///             .
    ///       ...   .   ...
    /// 
    /// 

    /// According to chain rule
    /// dC = dz = (dC/da(L)) * (da(L)/dz(L))
    pub fn derivative_of_z_from_top(&self, a : f32, y : f32, variable_pos : (usize, usize, usize, usize), neural_network : &NeuralNetwork) -> f32{

        // First this: dt/da(L)
        let dtda = 2.0 * (a-y);


        // where a = a(L) = sigmoid(z(L))
        // where z(L) = w1*a(L-1) + .... + wn*a(L-1) + b1*a(L-1) + .... + bn*a(L-1)
        // calculate z
        let neuron = neural_network.get_neuron_from_position(variable_pos).unwrap();
        let sum_w = {

            // calculate: w1*a(L-1) + .... + wn*a(L-1) + b1*a(L-1) + .... + bn*a(L-1)
            let mut sum = 0;
            for i in 0..neuron.weights.as_ref().unwrap().len() {
                //let corresponding_neuron_value = neural_network.get_neuron_from_position(variable_pos)
                //sum = 
            }
            neural_network.get_neuron_from_position(variable_pos).unwrap().weights.unwrap().into_iter()
        
        };

        let z = sum_w + neuron.bias;
        let dadz = self.derivative_of_sigmoid(z);

        return dtda * dadz;

    }


}