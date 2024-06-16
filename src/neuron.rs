

use rand::{thread_rng, Rng};



pub struct Neuron{
    pub weights : Option<Vec<f32>>, // using direct indexing to previous layer.
    pub bias : Option<Vec<f32>> // using direct indexing to previous layer.

}

impl Neuron{
    pub fn new() -> Neuron{
        Neuron{
            weights : None, 
            bias : None
        }
    }

    pub fn set_random_weights_and_bias(&mut self, number_of_neurons_previous_layer : u32){
        // create random weights
        self.weights = Some(
            (0..number_of_neurons_previous_layer as usize).map(|redundant_i|{
                let rand_weight = (thread_rng().gen::<f32>() * 0.5 - 1.0) * 2.0; // between -1 and 1
                return rand_weight;
            }).collect()
        );

        // create random bias
        self.bias = Some(
            (0..number_of_neurons_previous_layer as usize).map(|redundant_i|{
                let rand_bias = (thread_rng().gen::<f32>() * 0.5 - 1.0) * 2.0; // between -1 and 1
                return rand_bias;
            }).collect()
        );

    }
}