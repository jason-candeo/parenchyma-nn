#![allow(warnings)]
#![feature(associated_consts, untagged_unions)]

extern crate parenchyma;

pub mod extension;
pub mod package;

mod frameworks;

pub type Backend = parenchyma::Backend<package::ParenchymaDeep>;