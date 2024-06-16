

use std::iter::zip;

use crate::layer::Layer;
use crate::backpropagation::{SquishFunction, TraningHandeler};
use crate::neuron::Neuron;

pub struct NeuralNetwork{
    pub hidden_layers : Vec<Layer>,
    pub input_layer : Option<Layer>,
    pub output_layer : Option<Layer>,
}

impl NeuralNetwork{
    pub fn builder() -> NeuralNetwork{
        NeuralNetwork { 
            hidden_layers: Vec::new(), 
            input_layer: None,
            output_layer: None,
        }
    }

    /// add hidden layer
    pub fn with_hidden_layer(mut self, number_of_neurons : u32) -> Self{
        self.hidden_layers.push(Layer::new(number_of_neurons));
        self
    }

    /// add input layer
    pub fn with_input_layer(mut self, number_of_neurons : u32) -> Self{
        self.input_layer.insert(Layer::new(number_of_neurons));
        self
    }

    /// add output layer
    pub fn with_output_layer(mut self, number_of_neurons : u32) -> Self{
        self.output_layer.insert(Layer::new(number_of_neurons));
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

    pub fn calculate_values_of_all_neurons(&self) -> Vec<Vec<f32>>{


        for i in self.hidden_layers {
            
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
                        0 => { // create connections to the input layer
                            let previous_layer_number_of_neurons = self.input_layer.as_mut().unwrap().get_number_of_neurons();
                            let current_layer = self.hidden_layers.get_mut(i).unwrap();
                            current_layer.neurons.iter_mut().for_each(|neuron|{
                                neuron.set_random_weights_and_bias(previous_layer_number_of_neurons as u32);
                            });   
                        }, 
                        _size if _size == size => {  // create connections from output layer
                            let previous_layer_number_of_neurons = self.hidden_layers.get(i-1).unwrap().get_number_of_neurons();
                            let output_layer = self.output_layer.as_mut().unwrap();
                            output_layer.neurons.iter_mut().for_each(|neuron|{
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
}





