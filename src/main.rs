extern crate parenchyma as pa;
extern crate parenchyma_dnn as padnn;

use pa::{Backend, BackendConfig, HOST, OpenCL, SharedTensor};
use pa::HardwareKind::GPU;
use padnn::package::ParenchymaDeep;

fn main() {
    // Initialize an OpenCL or CUDA backend packaged with the NN extension.
    let ref backend = {
        let framework = OpenCL::new().unwrap();
        let hardware = framework.available_hardware.clone();
        
        let configuration = 
            BackendConfig::<OpenCL, ParenchymaDeep>::new(
                framework, 
                hardware, 
                GPU
            );

        Backend::try_from(configuration).unwrap()
    };

    let data: Vec<f32> = vec![3.5, 12.4, 0.5, 6.5];
    let length = data.len();

    // Initialize two `SharedTensor`s.
    let ref x = SharedTensor::with(backend, length, data).unwrap();
    let ref mut result = SharedTensor::new(length);

    // TODO reuse buf?

    // Run the sigmoid operation, provided by the NN extension, on 
    // your OpenCL/CUDA enabled GPU (or CPU, which is possible through OpenCL)
    backend.sigmoid(x, result).unwrap();

    // Print the result: `[0.97068775, 0.9999959, 0.62245935, 0.9984988] shape=[4], strides=[1]`
    println!("{:?}", result.read(HOST).unwrap().as_native().unwrap());
}