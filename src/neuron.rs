

use rand::{thread_rng, Rng};
use std::hash::Hash;

use crate::layer::{Layer};

use serde::{Serialize, Deserialize};


#[derive(PartialOrd, Clone, Debug, Serialize, Deserialize)]
pub struct Neuron{
    pub weights : Option<Vec<f32>>, // using direct indexing to previous layer.
    pub bias : Option<f32>, // using direct indexing to previous layer., 
    id : u32

}

impl PartialEq for Neuron {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Ord for Neuron{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if self.id == other.id {
            std::cmp::Ordering::Equal
        } else if self.id > other.id{
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Less
        }
    }
}

impl Eq for Neuron {}

impl Hash for Neuron {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl Neuron{
    pub fn new(id : u32) -> Neuron{
        Neuron{
            weights : None, 
            bias : None, 
            id
        }
    }

    pub fn set_random_weights_and_bias(&mut self, number_of_neurons_previous_layer : u32){
        // create random weights
        self.weights = Some(
            (0..number_of_neurons_previous_layer as usize).map(|redundant_i|{
                let rand_weight = (thread_rng().gen::<f32>() - 0.5) * 2.0;  /*Now between -1 and 1 */ //* 100.0; // between -100 and 100
                return rand_weight as f32
            }).collect()
        );
        // create random bias
        self.bias = Some(((thread_rng().gen::<f32>() - 0.5) * 2.0) as f32);
    }
}