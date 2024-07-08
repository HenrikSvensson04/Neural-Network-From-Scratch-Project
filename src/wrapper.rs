


use wasm_bindgen::prelude::*;
use crate::{backprop, neural_network::NeuralNetwork, traning_handeler::TraningHandeler};



// https://developer.mozilla.org/en-US/docs/WebAssembly/Rust_to_Wasm


// For visualiation and draw pixels and circle: html canvas

#[wasm_bindgen]
pub struct NeuralWrapper{
    neural_network : NeuralNetwork,
    traning_handeler : Option<TraningHandeler>
}

#[wasm_bindgen]
impl NeuralWrapper{
    #[wasm_bindgen(constructor)]
    pub fn new(hidden_layers : u32) -> NeuralWrapper{
        let mut wrapper = NeuralWrapper{ 
            neural_network: {
                NeuralNetwork::builder()
                    .with_input_layer(2)
                    .with_hidden_layer(hidden_layers)
                    .with_output_layer(2)
                    .build_network()
                    .unwrap()
            }, 
            traning_handeler: None
        };
        wrapper.initalize_traning_handeler();
        wrapper
    }

    #[wasm_bindgen]
    pub fn initalize_traning_handeler(&mut self){
        self.traning_handeler = Some(TraningHandeler::new(&self.neural_network))
    }

    #[wasm_bindgen]
    pub fn insert_traning_data(&mut self, input : Vec<f32>, correct_output : Vec<f32>){
        if let Some(traning_handeler) = &mut self.traning_handeler{
            traning_handeler.traning_data_input.push(input);
            traning_handeler.traning_data_correct_output.push(correct_output);
        }
    }

    #[wasm_bindgen]
    pub fn get(&self) -> f64{
        self.neural_network.output_layer.as_ref().unwrap().neurons.get(0).unwrap().bias.unwrap() as f64
    }
}