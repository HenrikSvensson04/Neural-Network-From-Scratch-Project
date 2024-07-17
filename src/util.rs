
pub fn sigmoid(x : f32) -> f32{
        
    return 1.0 / (1.0 + f32::powf(std::f32::consts::E, -1.0 * x));
}

pub fn derivative_of_sigmoid(input_value : f32) -> f32{
    let mut e_power_negative_x = f32::powf(std::f32::consts::E, -1.0 * input_value);

    if f32::is_infinite(e_power_negative_x){
        e_power_negative_x = 1000000000.0;
    }
    
    return e_power_negative_x / (e_power_negative_x * (2.0 + e_power_negative_x) + 1.0);
}
