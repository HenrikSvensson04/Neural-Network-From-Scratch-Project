


use simple_neural_network_project::{back, backprop, neural_network::NeuralNetwork, traning_handeler::TraningHandeler};


fn main() {
    println!("Hello, world!");
    
    let mut nw = NeuralNetwork::builder()
        .with_input_layer(2)
        .with_hidden_layer(5)
        .with_output_layer(3)
        .build_network().unwrap();

    let correct_output = vec![vec![0.0, 1.0, 0.0], vec![1.0, 0.0, 0.0]];
    let input = vec![vec![1.0, 1.0], vec![0.0, 0.5]]; // 


    let mut traning_handeler = TraningHandeler::new(&nw);
    traning_handeler.insert_traning_data(input, correct_output);
    traning_handeler.train_neural_network(&mut nw, 1000);


    let correct_output = vec![0.0, 1.0, 0.0];
    let input = vec![1.0, 1.0]; // 

    println!("cost: {}", back::calculate_cost(&correct_output, &input, &nw));

    let correct_output = vec![1.0, 0.0, 0.0];
    let input = vec![0.0, 0.5]; // 

    println!("cost: {}", back::calculate_cost(&correct_output, &input, &nw));
    
    


    /* 
    let mut gradient = backprop::backpropagate(&correct_output, &input, &nw);

    println!("Initial cost: {}", back::calculate_cost(&correct_output, &input, &nw));
    
    println!("Weight updated; {:?}", nw.output_layer.as_ref().unwrap().neurons.get(0).unwrap().weights.as_ref().unwrap().get(0).unwrap());

    gradient = backprop::backpropagate(&correct_output, &input, &nw);
    backprop::update_neural_network(&mut nw, &gradient);
    println!("{:?}", nw.calculate_values_of_all_neurons(&input));
    println!("cost: {}", back::calculate_cost(&correct_output, &input, &nw));
    


    for i in 0..10000 {
        gradient = backprop::backpropagate(&correct_output, &input, &nw);
        backprop::update_neural_network(&mut nw, &gradient);

        if i % 1000 == 1{
            println!("cost: {}", back::calculate_cost(&correct_output, &input, &nw));
        }
        //println!("{:?}", nw.calculate_values_of_all_neurons(&input));
        //println!("cost: {}", back::calculate_cost(&correct_output, &input, &nw));
    }

    gradient = backprop::backpropagate(&correct_output, &input, &nw);
    backprop::update_neural_network(&mut nw, &gradient);
    println!("{:?}", nw.calculate_values_of_all_neurons(&input));
    println!("cost: {}", back::calculate_cost(&correct_output, &input, &nw));


    println!("{:?}", nw.calculate_values_of_all_neurons(&input).unwrap());
    */
    /* 
    //let mut traning_handeler = TraningHandeler::new(SquishFunction::sigmoid, &mut nw);
    

    //let traning_data_input = vec![vec![1.0], vec![0.0], vec![0.5]];
    //let traning_data_correct_output = vec![vec![1.0], vec![0.0], vec![0.5]];

    //traning_handeler.insert_traning_data(traning_data_input, traning_data_correct_output);

    //traning_handeler.backpropagate_network(&mut nw);

    let correct_output = vec![0.0, 1.0];
    let input = vec![0.5, 0.5]; // 

    println!("initial cost: {}", back::calculate_cost(&correct_output, &input, &nw));
    println!("{:?}", nw.calculate_values_of_all_neurons(&input));
    let mut gradient = back::backpropagate_weights_bias(&correct_output, &input, &nw);
    back::update_neural_network(&mut nw, &gradient);
    println!("cost: {}", back::calculate_cost(&correct_output, &input, &nw));

    /*

    println!("Weight updated; {:?}", nw.output_layer.as_ref().unwrap().neurons.get(0).unwrap().weights.as_ref().unwrap().get(0).unwrap());

    gradient = back::backpropagate_weights_bias(&correct_output, &input, &nw);
    back::update_neural_network(&mut nw, &gradient);
    println!("{:?}", nw.calculate_values_of_all_neurons(&input));
    println!("cost: {}", back::calculate_cost(&correct_output, &input, &nw));
    */


    for i in 0..100000 {
        gradient = back::backpropagate_weights_bias(&correct_output, &input, &nw);
        back::update_neural_network(&mut nw, &gradient);

        if i % 10000 == 1{
            println!("cost: {}", back::calculate_cost(&correct_output, &input, &nw));
        }
        //println!("{:?}", nw.calculate_values_of_all_neurons(&input));
        //println!("cost: {}", back::calculate_cost(&correct_output, &input, &nw));
    }

    gradient = back::backpropagate_weights_bias(&correct_output, &input, &nw);
    back::update_neural_network(&mut nw, &gradient);
    println!("{:?}", nw.calculate_values_of_all_neurons(&input));
    println!("cost: {}", back::calculate_cost(&correct_output, &input, &nw));
    */
    
}


