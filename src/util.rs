
pub fn sigmoid(x : f32) -> f32{
        
    return 1.0 / (1.0 + f32::powf(std::f32::consts::E, -1.0 * x));
}

pub fn derivative_of_sigmoid(input_value : f32) -> f32{
    let e_power_negative_x = f32::powf(std::f32::consts::E, -1.0 * input_value);
    return e_power_negative_x / (e_power_negative_x * (2.0 + e_power_negative_x) + 1.0);
}

pub fn inverse_sigmoid(x : f32) -> f32{
    f32::log(x / (1.0 - x), std::f32::consts::E)
}