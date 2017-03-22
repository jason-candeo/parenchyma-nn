use parenchyma::{Result, SharedTensor};

/// Provides the functionality for a backend to support DNN related operations.
pub trait Deep: Activation<f32>+Activation<f64>+ActivationBackward<f32>+ActivationBackward<f64> { }

pub enum ActivationMode {
    Tanh,
    Sigmoid,
    ReLu,
    Elu,
}

/// Provides activation functions.
pub trait Activation<F> {

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
}

pub trait ActivationBackward<F> {

    fn activation_backward(
        &self, 
        mode: ActivationMode, 
        input: &SharedTensor<F>, 
        input_diff: &SharedTensor<F>, 
        output_diff: &mut SharedTensor<F>) -> Result;

    fn tanh_backward(
        &self, 
        input: &SharedTensor<F>, 
        input_diff: &SharedTensor<F>, 
        output_diff: &mut SharedTensor<F>) -> Result {

        self.activation_backward(ActivationMode::Tanh, input, input_diff, output_diff)
    }

    fn sigmoid_backward(
        &self, 
        input: &SharedTensor<F>, 
        input_diff: &SharedTensor<F>, 
        output_diff: &mut SharedTensor<F>) -> Result {

        self.activation_backward(ActivationMode::Sigmoid, input, input_diff, output_diff)
    }

    fn relu_backward(
        &self, 
        input: &SharedTensor<F>, 
        input_diff: &SharedTensor<F>, 
        output_diff: &mut SharedTensor<F>) -> Result {

        self.activation_backward(ActivationMode::ReLu, input, input_diff, output_diff)
    }

    fn elu_backward(
        &self, 
        input: &SharedTensor<F>, 
        input_diff: &SharedTensor<F>, 
        output_diff: &mut SharedTensor<F>) -> Result {

        self.activation_backward(ActivationMode::Elu, input, input_diff, output_diff)
    }
}