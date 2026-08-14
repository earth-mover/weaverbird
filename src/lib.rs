//! The CuTe layout algebra in Rust.
//!
//! A [`Layout`] maps a coordinate domain to a codomain. It pairs a shape
//! — an [`IntTuple`] of extents — with a congruent stride. The algebra
//! composes, inverts, divides and multiplies those maps.
//!
//! ```
//! use weaverbird::{layout, Layout, Tiler};
//!
//! // A 4x8 tile, row-major, and the element it holds at (2, 3).
//! let a = layout!((4, 8):(8, 1));
//! assert_eq!(a.eval(&weaverbird::ht!((2, 3))).unwrap().as_int(), Some(19));
//!
//! // Split a 24-element domain into 4-element tiles of stride 2.
//! let split = layout!(24:1).logical_divide(layout!(4:2)).unwrap();
//! assert_eq!(split, layout!((4, (2, 3)):(2, (1, 8))));
//!
//! // A tiler dispatches by mode. `Tiler::Skip` leaves a mode alone.
//! let by_mode: Tiler = vec![Tiler::from(2), Tiler::Skip].into();
//! assert_eq!(a.logical_divide(by_mode).unwrap().shape, weaverbird::ht!(((2, 2), 8)));
//! ```
//!
//! # Conventions
//!
//! Fallible operations return [`Result`]. Nothing panics on bad input.
//!
//! # Origin
//!
//! The algorithms are a port of PyCuTe (`pycute/`), and the test suite is
//! a port of its test suite. The API is not: the dispatch facade, the
//! duck-typed accessors and the `None` arguments of the Python source are
//! methods, iterators and dedicated types here.

pub mod algebra;
pub mod atuple;
pub mod error;
pub mod htuple;
pub mod layout;
pub mod shape;
pub mod stride;
pub mod typedefs;

pub use algebra::{greatest_common_domain, layout_add};
pub use atuple::{ArithTuple, StrideScalar, e, scaled_basis};
pub use error::{Error, Result};
pub use htuple::{HTuple, Leaves};
pub use layout::{Layout, Profile, Scale, Tiler};
pub use shape::Coordinates;
pub use typedefs::{Int, IntTuple, Stride};
