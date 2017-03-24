use parenchyma::{Result, SharedTensor};

/// Provides the functionality for a backend to support DNN related operations.
pub trait Deep: Forward + Backward { }

pub enum ActivationMode {
    Tanh,
    Sigmoid,
    ReLu,
    Elu,
}

pub trait Forward<F = f32> {

    fn activation(&self, ActivationMode, &SharedTensor<F>, output: &mut SharedTensor<F>) -> Result;

    /// Computes the [hyperbolic tangent] over the `input` tensor.
    ///
    /// Saves the `output`.
    ///
    /// [hyperbolic tangent]: https://en.wikipedia.org/wiki/Hyperbolic_function
    fn tanh(&self, input: &SharedTensor<F>, output: &mut SharedTensor<F>) -> Result {
        self.activation(ActivationMode::Tanh, input, output)
    }

    /// Computes the [sigmoid function] over the `input` tensor.
    ///
    /// Saves the `output`.
    ///
    /// [sigmoid function]: https://en.wikipedia.org/wiki/Sigmoid_function
    fn sigmoid(&self, input: &SharedTensor<F>, output: &mut SharedTensor<F>) -> Result {
        self.activation(ActivationMode::Sigmoid, input, output)
    }

    /// Computes the [rectified linear units] over the `input` tensor.
    ///
    /// Saves the `output`.
    ///
    /// [rectified linear units]: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
    fn relu(&self, input: &SharedTensor<F>, output: &mut SharedTensor<F>) -> Result {
        self.activation(ActivationMode::ReLu, input, output)
    }

    /// Computes the exponential linear unit [new] over the `input` tensor.
    ///
    /// Saves the `output`.
    fn elu(&self, input: &SharedTensor<F>, output: &mut SharedTensor<F>) -> Result {
        self.activation(ActivationMode::Elu, input, output)
    }

    /// Computes a [CNN convolution] over the input tensor `x`, and then saves the `result`.
    ///
    /// [CNN convolution]: https://en.wikipedia.org/wiki/Convolutional_neural_network
    fn convolution(
        &self, 
        filter: &SharedTensor, 
        x: &SharedTensor, 
        result: &mut SharedTensor) -> Result {

        unimplemented!()
    }

    /// Computes a logarithmic softmax over the input tensor `x`, and then saves the `result`.
    fn log_softmax(&self, x: &SharedTensor<F>, result: &mut SharedTensor<F>) -> Result {

        unimplemented!()
    }
}

pub trait Backward<F = f32> {

    fn activation_backward(
        &self, 
        mode: ActivationMode, 
        x: &SharedTensor<F>, 
        x_diff: &SharedTensor<F>, 
        result_diff: &mut SharedTensor<F>) -> Result;

    fn tanh_backward(
        &self, 
        x: &SharedTensor<F>, 
        x_diff: &SharedTensor<F>, 
        result_diff: &mut SharedTensor<F>) -> Result {

        self.activation_backward(ActivationMode::Tanh, x, x_diff, result_diff)
    }

    fn sigmoid_backward(
        &self, 
        x: &SharedTensor<F>, 
        x_diff: &SharedTensor<F>, 
        result_diff: &mut SharedTensor<F>) -> Result {

        self.activation_backward(ActivationMode::Sigmoid, x, x_diff, result_diff)
    }

    fn relu_backward(
        &self, 
        x: &SharedTensor<F>, 
        x_diff: &SharedTensor<F>, 
        result_diff: &mut SharedTensor<F>) -> Result {

        self.activation_backward(ActivationMode::ReLu, x, x_diff, result_diff)
    }

    fn elu_backward(
        &self, 
        x: &SharedTensor<F>, 
        x_diff: &SharedTensor<F>, 
        result_diff: &mut SharedTensor<F>) -> Result {

        self.activation_backward(ActivationMode::Elu, x, x_diff, result_diff)
    }

    // fn convolution_backward(&self, ..) -> Result {

    //     unimplemented!()
    // }

    fn log_softmax_backward(
        &self, 
        x: &SharedTensor<F>, 
        x_diff: &SharedTensor<F>, 
        result_diff: &mut SharedTensor<F>) -> Result {

        unimplemented!()
    }
}