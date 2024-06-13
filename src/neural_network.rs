

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
            self.hidden_layers.iter_mut().zip(other)  // Idea for this: https://stackoverflow.com/questions/66386013/how-to-iterate-over-two-elements-in-a-collection-stepping-by-one-using-iterator


            // set all edges to random
            self.set_network_random();
        } 
        return None;
    }

    fn set_network_random(&mut self){

    }
}





