use {NN, NNPackage};
use parenchyma::{Build, Result, SharedTensor};
use parenchyma::opencl::OpenCLContext;
use parenchyma::opencl::high;
use parenchyma::utility::Uninitialized;

#[derive(Debug)]
pub struct OpenCLNNPackage {
    program: high::Program,
    sigmoid: high::Kernel,
}

impl NN for OpenCLContext<NNPackage> {

    /// Computes the [sigmoid function] over the input tensor `x`.
    ///
    /// Saves the `result`.
    ///
    /// [sigmoid function]: https://en.wikipedia.org/wiki/Sigmoid_function
    fn sigmoid(&self, x: &SharedTensor, result: &mut SharedTensor) -> Result {

        let tensor_x = x.read(self)?;
        let tensor_result = result.write(self)?;
        let sigmoid = unsafe { &self.package().cl.sigmoid };
        sigmoid.set_arg(0, tensor_x)?;
        sigmoid.set_arg(1, tensor_result)?;

        let global_work_size = &[1];
        let local_work_size = &[];

        // TODO
        let event_wait_list = &[];

        // TODO
        let event = self.device().queue()
            .enqueue_nd_range_kernel(sigmoid, global_work_size, local_work_size, event_wait_list)?;

        Ok(())
    }
}

impl Build<OpenCLContext<Uninitialized>> for NNPackage {

    fn build(cx: &mut OpenCLContext<Uninitialized>) -> Result<NNPackage> {
        let source = vec![include_str!("../../source/cl/math.cl")];
        let program = cx.create_program(&source)?;
        let sigmoid = program.create_kernel("array_sigmoid_f32")?;

        let cl_package = OpenCLNNPackage {
            program,
            sigmoid, 
        };

        Ok(NNPackage { cl: cl_package })
    }
}