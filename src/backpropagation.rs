use std::{collections::HashMap, iter::zip};

use crate::{backpropagation, layer::Layer, neural_network::{self, NeuralNetwork}};


type local_neuron_tuple = (Vec<f32>, f32); // 

type PartialPosition = (i32, i32, i32, i32); 
// layer -> associated neuron -> type variable(C - cost function, weight, bias, neuron value A(L), inside sigmoid Z(L)) -> (if weight) specify weight

type PartialDerivative= (PartialPosition /* Position of variable a */, PartialPosition /* Position of variable b */); 
// Here we describe the partial deriativtive (da/sb) in relation to the graph tree of all variables
// Thus: Partial derivative from two nodes in the graph (da/db), determained by partial position

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
            //type local_neuron_tuple = (Vec<f32>, f32);
            let mut gradient : Vec<Vec<local_neuron_tuple>> = Vec::new(); // layer -> neuron -> (vec weights, bias), per neuron, where layer = 0 is first hidden layer

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
            let mut tuple : local_neuron_tuple = (Vec::new(), 1.0);
            for i in 0..neuron.weights.as_ref().unwrap().len(){
                tuple.0.push(1.0);
            }

            return tuple;
        }).collect();
    }


    pub fn calculate_gradient(&self, gradient : &Vec<Vec<local_neuron_tuple>>, neural_network : &NeuralNetwork) -> Vec<Vec<local_neuron_tuple>>{


        // calculate all values of neurons in network
        let neuron_values = neural_network.calculate_values_of_all_neurons(self.traning_data_input.get(0).unwrap()).unwrap();
        let correct_value_y = 10.0;
        /* 
        type PartialPosition = (i32, i32, i32, i32); 
        // layer -> associated neuron -> type variable(C - cost function, weight, bias, neuron value A(L), inside sigmoid Z(L)) -> (if weight) specify weight

        type PartialDerivative= (PartialPosition /* Position of variable a */, PartialPosition /* Position of variable b */); 
        // Here we describe the partial deriativtive (da/sb) in relation to the graph tree of all variables
        // Thus: Partial derivative from two nodes in the graph (da/db), determained by partial position
        */

        // lookup for all partial derivatives
        let mut lookup_partial_derivatives : HashMap<Option<f32>, PartialDerivative> = HashMap::new();














        // calculate partial derivatives to weights and bias
        for i in 0..gradient.len(){ // go through gradient
            let layer = gradient.get(i).unwrap();

            for j in 0..layer.len(){ // iterate through neurons
                let neuron_weights_bias = layer.get(j).unwrap();

                // Calculate partials relative to weights
                for k in 0..neuron_weights_bias.0.len(){ 

                    // Define start varible as cost function C
                    let partial_start_pos : PartialPosition = (gradient.len() as i32, -1, 0, 0);
                    let partial_end_pos : PartialPosition = (i as i32, j as i32, 1, k as i32); // at layer i, at neuron j, weight, weight k

                    let partial_derivative = calculate_partial_derivative(partial_start_pos, partial_end_pos);

                }


                // Calculate partials relative to bias


            }
        }
        // calculate derivatives to weights

        Vec::new()
    }

    pub fn calculate_partial_derivative(partial_start_pos : PartialPosition, partial_end_pos : PartialPosition, lookup_partial_derivatives : &mut HashMap<Option<f32>, PartialDerivative>) -> f32{

        match partial_start_pos.1{
            -1 => { // start at top C - cost function
                                                                                // select neurons 0..10
                let new_partial_pos = (partial_start_pos.0, 0..10, 3, 0);

            }, 
            0 => {

            },
            _=> {

            }
        }
        

        10
    }



    /*
    /// calculate partial derivatives relative to all weights and bias of each neuron
    pub fn calculate_gradient(&self, gradient : &Vec<Vec<local_neuron_tuple>>, neural_network : &NeuralNetwork) -> Vec<Vec<local_neuron_tuple>>{


        // calculate all values of neurons in network
        let neuron_values = neural_network.calculate_values_of_all_neurons(self.traning_data_input.get(0).unwrap()).unwrap();

        let correct_value_y = 10.0;

        // calculate derivatives to weights and bias
        for i in 0..gradient.len(){
            let layer = gradient.get(i).unwrap();
            for j in 0..layer.len(){
                let neuron_weights_biases = layer.get(j).unwrap();
                for k in 0..neuron_weights_biases.0.len(){ // Note: becuse num weights = num biases
                    
                    let weight_pos = (i, j, k, 0 as usize); // Where: layer, neuron, connection/edge, weight or bias(0 == weight, 1 == bias)
                    let bias_pos = (i, j, k, 1 as usize);
                    //let current_pos = (gradient.len(), )                    


                    let weight_partial = self.calculate_partial_derivative(correct_value_y, weight_pos, , &neural_network, &neuron_values);
    
                    
                }


            }
        }
        // calculate derivatives to weights

        Vec::new()
    }

    pub fn calculate_partial_derivative(&self, y : f32, variable_pos : (usize, usize, usize, usize), current_pos : (usize, usize, usize, usize), neural_network : &NeuralNetwork, neuron_values : &Vec<Vec<f32>>) -> f32{

        if variable_pos.3 == 0{ // calculate partial derivative relative to weight
            let dtopdz = self.derivative_of_z_from_top(y, variable_pos, neural_network, neuron_values);
            let dzdw = neuron_values.get(variable_pos.0).unwrap().get(variable_pos.1).unwrap();

            let dtopdw = dtopdz * dzdw;
            return dtopdw;
        } else { /*if variable_pos.3 == 1{ */ // calculate partial derivative relative to bias
            let dtopdz = self.derivative_of_z_from_top(y, variable_pos, neural_network, neuron_values);
            //let dzdb = /*****/
            return 10.0;
        }

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
    ///    w1..wn   |       b
    ///             a(L-1)
    ///             .
    ///       ...   .   ...
    /// 
    /// 

    /// According to chain rule
    /// dC = dz = (dC/da(L)) * (da(L)/dz(L))                
    pub fn derivative_of_z_from_top(&self, y : f32, variable_pos : (usize, usize, usize, usize), neural_network : &NeuralNetwork, neuron_values : &Vec<Vec<f32>>) -> f32{

        let a /*neuron a(L), the neuron we are at*/ : f32 = neuron_values.get(variable_pos.0+1).unwrap().get(variable_pos.1 +1).unwrap().clone();
        // First this: dt/da(L)
        let dtda = 2.0 * (a-y);

        // where a = a(L) = sigmoid(z(L))
        // where z(L) = w1*a(L-1) + .... + wn*a(L-1) + b1*a(L-1) + .... + bn*a(L-1)
        // calculate z
        let neuron = neural_network.get_neuron_from_position(variable_pos).unwrap();

        // calculate: w1*a(L-1) + .... + wn*a(L-1) + b
        let neuron_values_previous_layer_iterator = neuron_values.get(variable_pos.0).unwrap();
        let sum_weights : f32= zip(neuron_values_previous_layer_iterator, neuron.weights.as_ref().unwrap())
            .into_iter()
            .map(|(neuron_value, weight)|{
                neuron_value * weight
            }).sum(); // = w1*a(L-1) + .... + wn*a(L-1)
        
        let z = sum_weights + neuron.bias.unwrap(); // calculate: z(L) = w1*a(L-1) + .... + wn*a(L-1) + b
        let dadz = self.derivative_of_sigmoid(z);
        return dtda * dadz;
    }

    */
}