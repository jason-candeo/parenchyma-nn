use extension::Deep;
use frameworks::cl;
use parenchyma::ExtensionPackage;
use std::fmt;

pub union ParenchymaDeep {
    pub cl: cl::Package,
}

impl ExtensionPackage for ParenchymaDeep {

    const PACKAGE_NAME: &'static str = "parenchyma/dnn";

    type Extension = Deep;
}

impl fmt::Debug for ParenchymaDeep {

    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "ParenchymaDeep")
    }
}