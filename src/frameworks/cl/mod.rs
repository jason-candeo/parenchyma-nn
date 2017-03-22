use extension::{Activation, ActivationBackward, ActivationMode, Deep};
use package::ParenchymaDeep;

use parenchyma::{Build, Result, SharedTensor};
use parenchyma::opencl::OpenCLContext;
use parenchyma::opencl::high;
use parenchyma::utility::Uninitialized;

#[derive(Debug)]
pub struct Precision<T> {
    pub f32: T,
    pub f64: T,
}

#[derive(Debug)]
pub struct Package {
    program: high::Program,

    // === activation

    tanh: Precision<high::Kernel>,
    sigmoid: Precision<high::Kernel>,
    relu: Precision<high::Kernel>,
    elu: Precision<high::Kernel>,

    // === activation backward

    tanh_backward: Precision<high::Kernel>,
    sigmoid_backward: Precision<high::Kernel>,
    relu_backward: Precision<high::Kernel>,
    elu_backward: Precision<high::Kernel>,
}

impl Deep for OpenCLContext<ParenchymaDeep> { }

macro_rules! activation { ($($t:ident)|*) => ($(

impl Activation<$t> for OpenCLContext<ParenchymaDeep> {

    fn activation(
        &self, 
        mode: ActivationMode, 
        input: &SharedTensor<$t>, 
        output: &mut SharedTensor<$t>) -> Result {

        use extension::ActivationMode::*;

        let kernel = match mode {
            Tanh => unsafe { &self.package().cl.tanh.$t },
            Sigmoid => unsafe { &self.package().cl.sigmoid.$t },
            ReLu => unsafe { &self.package().cl.relu.$t },
            Elu => unsafe { &self.package().cl.elu.$t },
        };

        let length = input.shape.capacity();

        kernel.set_arg(0, input.read(self)?)?;
        kernel.set_arg(1, output.write(self)?)?;
        kernel.set_arg(2, &length)?;

        let global_work = &[length];
        let local_work = &[];

        // TODO event_wait_list
        let events = &[];

        // TODO
        let event = self.device().queue()
            .enqueue_nd_range_kernel(kernel, global_work, local_work, events)?;

        Ok(())
    }
}

impl ActivationBackward<$t> for OpenCLContext<ParenchymaDeep> {

    fn activation_backward(
        &self, 
        mode: ActivationMode, 
        input: &SharedTensor<$t>, 
        input_diff: &SharedTensor<$t>, 
        output_diff: &mut SharedTensor<$t>) -> Result {

        use extension::ActivationMode::*;

        let kernel = match mode {
            Tanh => unsafe { &self.package().cl.tanh_backward.$t },
            Sigmoid => unsafe { &self.package().cl.sigmoid_backward.$t },
            ReLu => unsafe { &self.package().cl.relu_backward.$t },
            Elu => unsafe { &self.package().cl.elu_backward.$t },
        };

        let length = input.shape.capacity();

        kernel.set_arg(0, input.read(self)?)?;
        kernel.set_arg(1, input_diff.read(self)?)?;
        kernel.set_arg(2, output_diff.write(self)?)?;
        kernel.set_arg(3, &length)?;

        let global_work = &[length];
        let local_work = &[];

        // TODO event_wait_list
        let events = &[];


        // TODO
        let event = self.device().queue()
            .enqueue_nd_range_kernel(kernel, global_work, local_work, events)?;

        Ok(())
    }
}

)*)}

activation!(f32 | f64);

macro_rules! create_kernel {
    ($program:expr, $name:expr) => (Precision {
        f32: $program.create_kernel(&format!("{}_float", $name))?,
        f64: $program.create_kernel(&format!("{}_double", $name))?,
    })
}

impl Build<OpenCLContext<Uninitialized>> for ParenchymaDeep {

    fn build(cx: &mut OpenCLContext<Uninitialized>) -> Result<ParenchymaDeep> {

        let program = cx.create_program(&[
            include_str!("source/activation.cl"),
            include_str!("source/activationBackward.cl")
        ])?;

        let cl_package = Package {
            tanh: create_kernel!(program, "tanh"),
            sigmoid: create_kernel!(program, "sigmoid"),
            relu: create_kernel!(program, "relu"),
            elu: create_kernel!(program, "elu"),

            tanh_backward: create_kernel!(program, "tanh_backward"),
            sigmoid_backward: create_kernel!(program, "sigmoid_backward"),
            relu_backward: create_kernel!(program, "relu_backward"),
            elu_backward: create_kernel!(program, "elu_backward"),

            program,
        };

        Ok(ParenchymaDeep { cl: cl_package })
    }
}