


use std::iter::zip;

use wasm_bindgen::prelude::*;
use crate::{backprop, neural_network::NeuralNetwork, traning_handeler::TraningHandeler, neural_network::TypeNeuronValue};

//use Serde::{Serialize, DeSerialize};
use serde::{Serialize, Deserialize};
use serde_json;



// https://developer.mozilla.org/en-US/docs/WebAssembly/Rust_to_Wasm


// For visualiation and draw pixels and circle: html canvas

#[wasm_bindgen]
#[derive(Serialize, Deserialize)]
pub struct NeuralWrapper{
    neural_network : NeuralNetwork,
    traning_handeler : Option<TraningHandeler>
}

#[wasm_bindgen]
impl NeuralWrapper{

    #[wasm_bindgen(constructor)] 
    pub fn new(structure_network : String, type_of_creation : String, json : String) -> NeuralWrapper{

        // create either object from json sessionStorage or from scratch
        if type_of_creation == "json"{
            let neural_wrapper: NeuralWrapper = serde_json::from_str(&json).unwrap();
            neural_wrapper
        } else {
            let mut wrapper = NeuralWrapper{ 
                neural_network: {
                    let mut neural_network = NeuralNetwork::builder().with_input_layer(2);

                    // generate hidden layers from string input
                    let parts = structure_network.split("-");
                    let mut i = 0;
                    let length = parts.clone().count();
                    for part in parts{
                        if i != 0 && i != length-1{
                            if part.parse::<u32>().is_ok(){
                                let number_of_neurons = part.parse::<u32>().unwrap();
                                neural_network = neural_network.with_hidden_layer(number_of_neurons);
                            }
                        } 
                        i += 1;
                    }
                    // add output-layer
                    neural_network.with_output_layer(2).build_network().unwrap()
                }, 
                traning_handeler: None
            };
            wrapper.initalize_traning_handeler();
            wrapper
        }
    }

    #[wasm_bindgen]
    pub fn initalize_traning_handeler(&mut self){
        self.traning_handeler = Some(TraningHandeler::new(&self.neural_network, 1.0 /* Learning rate */))
    }

    #[wasm_bindgen]
    pub fn insert_traning_data(&mut self, input : Vec<f32>, correct_output : Vec<f32>){
        if let Some(traning_handeler) = &mut self.traning_handeler{
            traning_handeler.traning_data_input.push(input);
            traning_handeler.traning_data_correct_output.push(correct_output);
        }
    }

    #[wasm_bindgen]
    pub fn train(&mut self, epochs : u32){
        if let Some(traning_handeler) = &mut self.traning_handeler{
            traning_handeler.train_neural_network(&mut self.neural_network, epochs);
        }
    }

    /// TODO: Make this function more safe!
    #[wasm_bindgen]
    pub fn get_output(&mut self, input : Vec<f32>) -> Vec<f32>{

        // calculate values of neurons given the input
        let feedforward_values = self.neural_network.feedforward_to_map(&input);

        // extract the values of the output neurons
        let vec : Vec<f32> = if let Some(values) = feedforward_values{
            let vec : Vec<f32>= self.neural_network.output_layer.as_ref().unwrap().neurons.iter().map(|neuron|{
                values.get(&(neuron, TypeNeuronValue::A)).unwrap().clone()
            }).collect();
            return vec;
        } else {
            Vec::new()
        };
        return vec;
    }

    // Code just for proof of wasm's functionality. To be removed!
    #[wasm_bindgen]
    pub fn get(&self) -> f64{
        self.neural_network.output_layer.as_ref().unwrap().neurons.get(0).unwrap().bias.unwrap() as f64
    }

    #[wasm_bindgen]
    pub fn get_json_serialized(&self) -> String{
        serde_json::to_string(&self).expect("Works!")
    }

    #[wasm_bindgen]
    pub fn get_cost(&self) -> f32{
        let mut cost = -1.0;
        if let Some(traning_handeler) = &self.traning_handeler{
            cost = 0.0;
            let mut iterations = 0;
            zip(&traning_handeler.traning_data_input, &traning_handeler.traning_data_correct_output).for_each(|(inputs_vec, correct_outputs_vec)|{
                iterations += 1;
                cost += backprop::calculate_cost(&correct_outputs_vec, &inputs_vec, &self.neural_network);
            });
            cost = cost / iterations as f32;
        }
        return cost;
    }

    #[wasm_bindgen]
    pub fn set_learning_rate(&mut self, learning_rate : f32){
        if let Some(traning_handeler) = &mut self.traning_handeler{
            traning_handeler.learning_rate = learning_rate;
        }
    }
}