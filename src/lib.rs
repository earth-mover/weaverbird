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
//!
//! PyCuTe's integers may be symbolic, and `typedefs.is_static` tells a
//! concrete one from a symbolic one. Here [`Int`] is the only integer, so
//! `is_static` would always be true. The crate omits it, and the ordering
//! helpers PyCuTe guards with it — the `_stride_key` sorts in `layout.py`
//! — collapse to a plain sort.

pub mod atuple;
pub mod error;
pub mod htuple;
pub mod shape;
pub mod stride;
pub mod typedefs;

pub use atuple::{
    ArithTuple, StrideScalar, basis_repr, e, is_basis, make_basis_like, proj, proj_tuple,
    proj_tuple_mut, unit,
};
pub use error::{Error, Result};
pub use htuple::HTuple;
pub use shape::{
    common_coarsening, common_refinement, compatible, coordinates, crd2idx, depth, idx2crd, rank,
    shape, size,
};
// `coalesce_z` stays behind its module, as PyCuTe keeps `_coalesce_z`
// private to `stride.py`; the exported `coalesce_z` is the `Layout` one.
pub use stride::{Coshape, coprofile, coshape, inner_product, prefix_product, stride};
pub use typedefs::{Int, IntTuple, Stride};
