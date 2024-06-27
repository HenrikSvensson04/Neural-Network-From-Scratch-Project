

use crate::neuron::Neuron;


#[derive(PartialEq, Eq)]
pub struct Layer{
    pub neurons : Vec<Neuron> // direct indexing
}

impl Layer{
    pub fn new(number_of_neurons : u32, id_generator_count : &mut u32) -> Layer{
        return Layer{
            neurons : {
                // create the neurons vec
                let neurons_vec : Vec<Neuron> = (0..number_of_neurons)
                    .map(|neuron| {
                        *id_generator_count += 1; 
                        return Neuron::new(id_generator_count.clone());
                    })
                    .collect();
                neurons_vec
            }
        };
    }

    pub fn get_number_of_neurons(&self) -> usize {
        return self.neurons.len();
    }

    pub fn has_neuron(&self, neuron : &Neuron) -> bool{
        self.neurons.binary_search(neuron).is_ok()
    }
}
