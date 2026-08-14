//! A Rust transliteration of the CuTe layout algebra.
//!
//! The source is PyCuTe (`pycute/`), and the modules here mirror it one
//! for one. Names, decomposition, and algorithms follow that source, so
//! each file is checkable against its Python counterpart. The test suite
//! is a port of the PyCuTe test suite.
//!
//! # What this crate does not do
//!
//! The algebra is colexicographic, as CuTe defines it. The crate adds no
//! row-major variant. A caller that wants row-major order converts at its
//! own boundary.
//!
//! The crate also carries no names for axes. A layout is positional.
//!
//! # Divergences from PyCuTe
//!
//! Python tells a tuple from a leaf at run time. Rust needs a type, so
//! [`HTuple`] makes the two cases variants of one enum.
//!
//! Python raises. This crate returns [`Result`](std::result::Result).
//!
//! [`ArithTuple`] stores its children as PyCuTe stores them, and its
//! equality extends trailing positions by zero. `Hash` therefore hashes a
//! trimmed form, so that equal values hash alike.

pub mod atuple;
pub mod error;
pub mod htuple;
pub mod stride;
pub mod typedefs;

pub use atuple::{ArithTuple, StrideScalar, basis_repr, e, is_basis, make_basis_like, proj, unit};
pub use error::{Error, Result};
pub use htuple::HTuple;
// `coalesce_z` stays behind its module, as PyCuTe keeps `_coalesce_z`
// private to `stride.py`; the exported `coalesce_z` is the `Layout` one.
pub use stride::{Coshape, coprofile, coshape, inner_product, prefix_product, stride};
pub use typedefs::{Int, IntTuple, Stride};
