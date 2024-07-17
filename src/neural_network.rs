

use std::collections::HashMap;
use std::iter::{zip, Sum};
use std::path::Iter;

use crate::layer::Layer;
//use crate::backpropagation::{SquishFunction, TraningHandeler};
use crate::neuron::{self, Neuron};
use crate::util;
use serde::{Serialize, Deserialize};



#[derive(PartialEq, PartialOrd, Eq, Hash, Serialize, Deserialize)]
pub enum TypeNeuronValue{
    A, // A(L) with activation function 
    Z  // Z(L) without activation function
}

#[derive(Serialize, Deserialize)]
pub struct NeuralNetwork{
    pub hidden_layers : Vec<Layer>,
    pub input_layer : Option<Layer>,
    pub output_layer : Option<Layer>,
    id_neuron_count : u32
}

impl NeuralNetwork{
    pub fn builder() -> NeuralNetwork{
        NeuralNetwork { 
            hidden_layers: Vec::new(), 
            input_layer: None,
            output_layer: None,
            id_neuron_count : 0
        }
    }

    /// add hidden layer
    pub fn with_hidden_layer(mut self, number_of_neurons : u32) -> Self{
        self.hidden_layers.push(Layer::new(number_of_neurons, &mut self.id_neuron_count));
        self
    }

    /// add input layer
    pub fn with_input_layer(mut self, number_of_neurons : u32) -> Self{
        self.input_layer.insert(Layer::new(number_of_neurons, &mut self.id_neuron_count));
        self
    }

    /// add output layer
    pub fn with_output_layer(mut self, number_of_neurons : u32) -> Self{
        self.output_layer.insert(Layer::new(number_of_neurons, &mut self.id_neuron_count));
        self
    }

    /// Access neuron
    /// Where: variable_pos : (layer, neuron, connection/edge, weight or bias)
    pub fn get_neuron_from_position(&self, variable_pos : (usize, usize, usize, usize)) -> Option<&Neuron>{
        
        if self.input_layer.is_some() && self.output_layer.is_some(){
            if variable_pos.0 <= self.hidden_layers.len(){ // return neuron in hidden
                return self.hidden_layers.get(variable_pos.0).unwrap().neurons.get(variable_pos.1);
            } else if variable_pos.0 == self.hidden_layers.len()+1{ // return neuron in output
                return self.output_layer.as_ref().unwrap().neurons.get(variable_pos.1);
            }
        } 
        return None;
    }

    // To be removed!
    /// Calculate values of all neurons into a vec<vec<f32>>, with format: layer -> neuron -> value, this includes both hidden layers and output layer and input layer
    pub fn calculate_values_of_all_neurons(&self, input : &Vec<f32>) -> Option<Vec<Vec<f32>>>{

        if input.len() == self.input_layer.as_ref().unwrap().get_number_of_neurons(){
            let mut return_vec : Vec<Vec<f32>> = Vec::new();
            return_vec.push(input.clone()); // add input layer
            for i in 0..self.hidden_layers.len()+1 { // Note: all hidden layers + output layer

                let layer = {
                    if i < self.hidden_layers.len(){
                        self.hidden_layers.get(i).unwrap()
                    } else {
                        &self.output_layer.as_ref().unwrap()
                    }
                };

                let mut layer_values = Vec::new();
                
                for j in 0..layer.neurons.len() {
                    let neuron = layer.neurons.get(j).unwrap(); 

                    let input_layer_values = {
                        if i == 0{// use input layer
                            &input
                        } else { // use previous hidden layer as layer -- thus we use the value we calculated last iteration as input to this layer
                            return_vec.get(i).unwrap()
                        }
                    };
                    let mut neuron_value : f32 = zip(input_layer_values, neuron.weights.as_ref().unwrap()).into_iter().map(|(input_neuron_value, weight)|{
                            //println!("{} * {}, Number of neurons in previous layer: {}", input_neuron_value, weight, neuron.weights.as_ref().unwrap().len());
                            return input_neuron_value*weight;
                    }).sum();

                    neuron_value += neuron.bias.unwrap(); // add bias
                    neuron_value = util::sigmoid(neuron_value);

                    
                    layer_values.push(neuron_value);
                }
                return_vec.push(layer_values);
            }

            return Some(return_vec);
        } else {
            return None;
        }
    }


    
    /// Feedforward neural network
    /// 
    /// Calculates both neuron value with activation function, A(L)(N), and without activation function, Z(L)(N)
    /// 
    /// Returns a hashmap, where key is the corresponding neuron
    /// Calculates values of all neurons thus A(L)(N) 
    /// Includes neurons in both hidden, output and input layer
    pub fn feedforward_to_map(&self, input : &Vec<f32>) -> Option<HashMap<(&Neuron, TypeNeuronValue), f32>>{

        if input.len() == self.input_layer.as_ref().unwrap().get_number_of_neurons(){
            let mut return_map : HashMap<(&Neuron, TypeNeuronValue), f32> = HashMap::new();
            let mut return_vec : Vec<Vec<f32>> = Vec::new();
            return_vec.push(input.clone()); // add input layer

            // add input layer
            for i in 0..input.len(){
                // add A(L)
                return_map.insert((self.input_layer.as_ref().unwrap().neurons.get(i).unwrap(), TypeNeuronValue::Z), input.get(i).unwrap().clone());
                // add Z(L)
                return_map.insert((self.input_layer.as_ref().unwrap().neurons.get(i).unwrap(), TypeNeuronValue::A), util::sigmoid(input.get(i).unwrap().clone()));
            }

            for i in 0..self.hidden_layers.len()+1 { // Note: all hidden layers + output layer

                let layer = {
                    if i < self.hidden_layers.len(){
                        self.hidden_layers.get(i).unwrap()
                    } else {
                        &self.output_layer.as_ref().unwrap()
                    }
                };

                let mut layer_values = Vec::new();
                
                for j in 0..layer.neurons.len() {
                    let neuron = layer.neurons.get(j).unwrap(); 

                    let input_layer_values = {
                        if i == 0{// use input layer
                            &input
                        } else { // use previous hidden layer as layer -- thus we use the value we calculated last iteration as input to this layer
                            return_vec.get(i).unwrap()
                        }
                    };
                    let mut neuron_value : f32 = zip(input_layer_values, neuron.weights.as_ref().unwrap()).into_iter().map(|(input_neuron_value, weight)|{
                            //println!("{} * {}, Number of neurons in previous layer: {}", input_neuron_value, weight, neuron.weights.as_ref().unwrap().len());
                            return input_neuron_value*weight;
                    }).sum();

                    // add bias
                    neuron_value += neuron.bias.unwrap(); 

                    // add A(L)
                    return_map.insert((&neuron, TypeNeuronValue::Z), neuron_value);
                    // add Z(L)
                    return_map.insert((&neuron, TypeNeuronValue::A), util::sigmoid(neuron_value));
                    
                    layer_values.push(util::sigmoid(neuron_value));
                }

                //println!("\n");
                return_vec.push(layer_values);
            }

            return Some(return_map);
        } else {
            return None;
        }
    }

    /// Creates the network!
    /// Thus: creates all edges, verifies that there are input and output layers
    pub fn build_network(mut self) -> Option<Self>{

        if self.input_layer.is_some() && self.output_layer.is_some(){

            // create edges between neurons in the layers - set weights and bias random between -1 and 1
            let number_of_hidden_layers = self.hidden_layers.len().clone();
            for i in 0..self.hidden_layers.len()+1{
                let size = number_of_hidden_layers;
                    match i {
                        _size if _size == size => {  // create connections from output layer

                            let previous_layer_number_of_neurons = match size {
                                0 => { // this is the case if there are no hidden layers
                                    self.input_layer.as_ref().unwrap().get_number_of_neurons()
                                }, 
                                _=> {
                                    self.hidden_layers.get(i-1).unwrap().get_number_of_neurons()
                                }
                            };
                            let output_layer = self.output_layer.as_mut().unwrap();
                            output_layer.neurons.iter_mut().for_each(|neuron|{
                                neuron.set_random_weights_and_bias(previous_layer_number_of_neurons as u32);
                            });
                        }, 
                        0 => { // create connections to the input layer
                            let previous_layer_number_of_neurons = self.input_layer.as_mut().unwrap().get_number_of_neurons();
                            let current_layer = self.hidden_layers.get_mut(i).unwrap();
                            current_layer.neurons.iter_mut().for_each(|neuron|{
                                neuron.set_random_weights_and_bias(previous_layer_number_of_neurons as u32);
                            });   
                        }, 
                        
                        _=> { // create connections between hidden layers
                            let previous_layer_number_of_neurons = self.hidden_layers.get(i-1).unwrap().get_number_of_neurons();
                            let current_layer = self.hidden_layers.get_mut(i).unwrap();
                            current_layer.neurons.iter_mut().for_each(|neuron|{
                                neuron.set_random_weights_and_bias(previous_layer_number_of_neurons as u32);
                            });
                        }    
                    }
            }
            return Some(self);
        } 
        return None;
    }

    pub fn get_preceding_layer(&self, layer : &Layer) -> Option<&Layer>{ // return None if layer is the input layer

        if &*layer == self.output_layer.as_ref().unwrap(){
            match self.hidden_layers.len(){
                0 => {Some(self.input_layer.as_ref().unwrap())}, 
                _=> { Some(&self.hidden_layers.last().unwrap())}
            }
        } else if &*layer == self.input_layer.as_ref().unwrap(){
            None
        } else {
            let pos = self.hidden_layers.iter().position(|l| *l == *layer).unwrap();
            if pos == 0 {
                return Some(self.input_layer.as_ref().unwrap());
            } else {
                return Some(self.hidden_layers.get(pos-1).unwrap());
            }
        }
    }
}