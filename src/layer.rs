

use crate::neuron::Neuron;


pub struct Layer{
    pub neurons : Vec<Neuron>
}

impl Layer{
    pub fn new(number_of_neurons : u32) -> Layer{
        return Layer{
            neurons : {
                // create the neurons vec
                let neurons_vec : Vec<Neuron> = (0..number_of_neurons)
                    .map(|neuron| {return Neuron::new();})
                    .collect();
                neurons_vec
            }
            
        };
    }

    pub fn get_number_of_neurons(&self) -> usize {
        return self.neurons.len();
    }
}