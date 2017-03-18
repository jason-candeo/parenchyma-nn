use parenchyma::{Result, SharedTensor};

/// Provides the functionality for a backend to support Neural Network related operations.
pub trait NN<F> {
    /// Computes the [sigmoid function] over the input tensor `x`.
    ///
    /// Saves the `result`.
    ///
    /// [sigmoid function]: https://en.wikipedia.org/wiki/Sigmoid_function
    fn sigmoid(&self, x: &SharedTensor<F>, result: &mut SharedTensor<F>) -> Result;

    // /// Computes the gradient of a [sigmoid function] over the input tensor `x`.
    // ///
    // /// Saves the result to `result_diff`.
    // ///
    // /// [sigmoid function]: https://en.wikipedia.org/wiki/Sigmoid_function
    // fn sigmoid_grad(
    //     &self, 
    //     x: &SharedTensor<F>, 
    //     x_diff: &SharedTensor<F>,
    //     result: &SharedTensor<F>, 
    //     result_diff: &mut SharedTensor<F>) -> Result;
}

pub trait NNExtension: NN<f32> + NN<f64> { }

impl<I> NNExtension for I where I: NN<f32> + NN<f64> { }