use parenchyma::ExtensionPackage;
use std::fmt;
use super::NN;
use super::frameworks::opencl::OpenCLNNPackage;

pub union NNPackage {
    pub cl: OpenCLNNPackage,
}

impl ExtensionPackage for NNPackage {

    const PACKAGE_NAME: &'static str = "parenchyma/nn";

    type Extension = NN;
}

impl fmt::Debug for NNPackage {

    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "NNPackage")
    }
}