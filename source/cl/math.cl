__kernel void array_sigmoid_f32(
    __global float *x, 
    __global float *result) {

    uintptr_t i = get_global_id(0);
    result[i] = 1.0 / (1.0 + exp(-x[i]));
}

__kernel void array_sigmoid_f64(
    __global double *x, 
    __global double *result) {

    uintptr_t i = get_global_id(0);
    result[i] = 1.0 / (1.0 + exp(-x[i]));
}