extern crate parenchyma as pa;
extern crate parenchyma_nn as pann;

use pa::{Backend, BackendConfig, Native, OpenCL, SharedTensor};
use pa::HardwareKind::GPU;
use pann::NNPackage;

fn main() {
    let ref native: Backend = Backend::new::<Native>().unwrap();

    // Initialize an OpenCL or CUDA backend packaged with the NN extension.
    let ref backend = {
        let framework = OpenCL::new().unwrap();
        let hardware = framework.available_hardware.clone();
        let configuration = BackendConfig::<OpenCL, NNPackage>::new(framework, hardware, GPU);

        Backend::try_from(configuration).unwrap()
    };

    let data: Vec<f64> = vec![3.5, 12.4, 0.5, 6.5];
    let length = data.len();

    // Initialize two `SharedTensor`s.
    let ref x = SharedTensor::with(backend, length, data).unwrap();
    let ref mut result = SharedTensor::new(length);

    // Run the sigmoid operation, provided by the NN extension, on 
    // your OpenCL/CUDA enabled GPU (or CPU, which is possible through OpenCL)
    backend.sigmoid(x, result).unwrap();

    // Print the result: `[0.97068775, 0.9999959, 0.62245935, 0.9984988] shape=[4], strides=[1]`
    println!("{:?}", result.read(native).unwrap().as_native().unwrap());
}