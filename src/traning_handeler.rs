use std::iter::zip;

use crate::{backprop::{self, Gradient}, neural_network::NeuralNetwork};

pub struct TraningHandeler{
    // todo: insert traning data
    pub traning_data_input : Vec<Vec<f32>>,
    pub traning_data_correct_output : Vec<Vec<f32>>,
    traning_data_num_neurons_input : u32,
    traning_data_num_neurons_output : u32
}

impl TraningHandeler{
    
    /// Select the neural network that you wish to associate with this traningHandeler
    pub fn new(neural_network : &NeuralNetwork) -> TraningHandeler{
        TraningHandeler{
            traning_data_input : Vec::new(),
            traning_data_correct_output : Vec::new(),
            traning_data_num_neurons_input : neural_network.input_layer.as_ref().unwrap().get_number_of_neurons() as u32,
            traning_data_num_neurons_output : neural_network.output_layer.as_ref().unwrap().get_number_of_neurons() as u32
        }
    }

    /// Insert traning data 
    /// Verifies that data is compatible with the selected neural network
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

    // train neural network given the selected traning data
    pub fn train_neural_network(&self, neural_network : &mut NeuralNetwork, number_of_epochs : u32){
        for _i in 0..number_of_epochs {
            let mut gradient : Option<Gradient> = None;
            zip(&self.traning_data_input, &self.traning_data_correct_output).for_each(|(input, correct_output)|{

                match gradient.is_some(){
                    true => {
                        let mut redundant_swap : Option<Gradient> = None;
                        // swap references in memory - to make sure we don't have to make unnecceary clone()
                        std::mem::swap(&mut gradient, &mut redundant_swap); 
                        
                        // add gradients together
                        gradient = Some(redundant_swap.unwrap() + backprop::backpropagate(&correct_output, &input, &neural_network));
                    }, 
                    _=> {
                        gradient = Some(backprop::backpropagate(&correct_output, &input, &neural_network));
                    }
                }
            });

            if let Some(gradient_unwrapped) = gradient{
                // update neural network
                backprop::update_neural_network(neural_network, &gradient_unwrapped);
            }
        }
    }
}