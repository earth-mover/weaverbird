//! The type vocabulary. Mirrors `pycute/typedefs.py`.
//!
//! PyCuTe names its leaf types with abstract base classes, and tests
//! membership by registration. Rust has the types themselves, so the
//! predicates become pattern matches and the aliases become type aliases.

use crate::{atuple::StrideScalar, htuple::HTuple};

/// The integer leaf. One signed type serves shapes, coordinates, and
/// strides, as Python's `int` does.
pub type Int = i64;

/// A hierarchical tuple whose leaves are integers. Shapes and coordinates
/// share this carrier.
pub type IntTuple = HTuple<Int>;

/// The stride half of a layout. Congruent with the layout's shape.
pub type Stride = HTuple<StrideScalar>;
