use {NN, NNPackage};
use parenchyma::{Build, Result, SharedTensor};
use parenchyma::opencl::OpenCLContext;
use parenchyma::opencl::high;
use parenchyma::utility::Uninitialized;

#[derive(Debug)]
struct Precision {
    f32: high::Kernel,
    f64: high::Kernel,
}

#[derive(Debug)]
pub struct OpenCLNNPackage {
    program: high::Program,
    sigmoid: Precision,
}

macro_rules! impl_nn {
    ($($t:ident)|*) => {
        $(
            impl NN<$t> for OpenCLContext<NNPackage> {

                /// Computes the [sigmoid function] over the input tensor `x`.
                ///
                /// Saves the `result`.
                ///
                /// [sigmoid function]: https://en.wikipedia.org/wiki/Sigmoid_function
                fn sigmoid(&self, x: &SharedTensor<$t>, result: &mut SharedTensor<$t>) -> Result {

                    let kernel = unsafe { &self.package().cl.sigmoid.$t };

                    kernel.set_arg(0, x.read(self)?)?;
                    kernel.set_arg(1, result.write(self)?)?;

                    let global_work = x.shape().dimensions();
                    let local_work = &[];

                    // TODO event_wait_list
                    let events = &[];

                    // TODO
                    let event = self.device().queue()
                        .enqueue_nd_range_kernel(
                            kernel, global_work, local_work, events)?;

                    Ok(())
                }
            }
        )*
    }
}

impl_nn!(f32 | f64);

macro_rules! create_kernel {
    ($program:expr, $name:expr) => (Precision {
        f32: $program.create_kernel(&format!("{}_f32", $name))?,
        f64: $program.create_kernel(&format!("{}_f64", $name))?,
    })
}

impl Build<OpenCLContext<Uninitialized>> for NNPackage {

    fn build(cx: &mut OpenCLContext<Uninitialized>) -> Result<NNPackage> {
        let program = cx.create_program(&[include_str!("../../source/cl/math.cl")])?;

        let cl_package = OpenCLNNPackage {
            sigmoid: create_kernel!(program, "array_sigmoid"), 
            program,
        };

        Ok(NNPackage { cl: cl_package })
    }
}