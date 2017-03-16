#![allow(warnings)]
#![feature(associated_consts, untagged_unions)]

extern crate parenchyma;

pub use self::extension::NN;
pub use self::package::NNPackage;

mod extension;
mod frameworks;
mod package;

pub type Backend = parenchyma::Backend<NNPackage>;