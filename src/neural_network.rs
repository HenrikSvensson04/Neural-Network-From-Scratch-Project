

use std::iter::zip;

use crate::layer::Layer;




pub struct NeuralNetwork{
    pub hidden_layers : Vec<Layer>,
    pub input_layer : Option<Layer>,
    pub output_layer : Option<Layer>
    
}

impl NeuralNetwork{

    pub fn builder() -> NeuralNetwork{
        NeuralNetwork { 
            hidden_layers: Vec::new(), 
            input_layer: None,
            output_layer: None
        }
    }

    /// add hidden layer
    pub fn with_hidden_layer(&mut self, number_of_neurons : u32) -> &mut NeuralNetwork{
        self.hidden_layers.push(Layer::new(number_of_neurons));
        return self;
    }

    /// add input layer
    pub fn with_input_layer(&mut self, number_of_neurons : u32) -> &mut NeuralNetwork{
        self.input_layer.insert(Layer::new(number_of_neurons));
        return self;
    }

    /// add output layer
    pub fn with_output_layer(&mut self, number_of_neurons : u32) -> &mut NeuralNetwork{
        self.output_layer.insert(Layer::new(number_of_neurons));
        return self;
    }

    /// Creates the network!
    /// Thus: creates all edges, verifies that there are input and output layers
    pub fn build_network(&mut self) -> Option<&mut NeuralNetwork>{

        if self.input_layer.is_some() && self.output_layer.is_some(){

            // create edges between neurons in the layers
            //self.hidden_layers.iter_mut().zip(other)  // Idea for this: https://stackoverflow.com/questions/66386013/how-to-iterate-over-two-elements-in-a-collection-stepping-by-one-using-iterator
            
            let number_of_hidden_layers = self.hidden_layers.len().clone();
            /* 
            zip(self.hidden_layers.iter_mut(), 0..number_of_hidden_layers)
                .for_each(|(hidden_layer, i)|{
                    let size = number_of_hidden_layers;
                    let current_layer = hidden_layer;
                    match i {
                        0 => { // create connections to the input layer

                            
                        }, 
                        size => {  // create connections to output layer

                        }, 
                        _=> { // create connections between hidden layers
                            let previous_layer_number_of_neurons = self.hidden_layers.get(i-1).unwrap().get_number_of_neurons();
                            current_layer.neurons.iter_mut().for_each(|neuron|{
                                neuron.set_random_weights_and_bias(previous_layer_number_of_neurons as u32);
                            });

                        }    
                    }
            });
            */
             
            for i in 0..self.hidden_layers.len(){

                let size = number_of_hidden_layers;

                    match i {
                        0 => { // create connections to the input layer

                            let previous_layer_number_of_neurons = self.input_layer.as_mut().unwrap().get_number_of_neurons();
                            let current_layer = self.hidden_layers.get_mut(i).unwrap();

                            current_layer.neurons.iter_mut().for_each(|neuron|{
                                neuron.set_random_weights_and_bias(previous_layer_number_of_neurons as u32);
                            });

                            
                        }, 
                        size => {  // create connections from output layer
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

                return Some(self);
            }
            

            // set all edges to random
            //self.set_network_random();
        } 
        return None;
    }

    fn set_network_random(&mut self){

    }
}





