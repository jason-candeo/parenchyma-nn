extern crate parenchyma as pa;
extern crate parenchyma_nn as pann;

use pa::{Backend, Native, OpenCL, SharedTensor};

fn main() {
    let ref native: Backend = Backend::new::<Native>().unwrap();
    // Initialize an OpenCL or CUDA backend packaged with the NN extension.
    let ref backend = pann::Backend::new::<OpenCL>().unwrap();

    // Initialize two `SharedTensor`s.
    let shape = 5;
    let data = vec![3.5, 4., 5., 6., 7.];
    let ref x = SharedTensor::<f32>::with(backend, shape, data).unwrap();
    let ref mut result = SharedTensor::<f32>::new(shape);

    // Run the sigmoid operation, provided by the NN extension, on 
    // your OpenCL/CUDA enabled GPU (or CPU, which is possible through OpenCL)
    backend.sigmoid(x, result).unwrap();

    // Print the result: `[0.97068775] shape=[1], strides=[1]`
    println!("{:?}", result.read(native).unwrap().as_native().unwrap());
}