//! What can go wrong, and the values that went wrong.
//!
//! Every variant carries the operands themselves, not a rendering of
//! them, so a caller can match on the failure and read the layouts back
//! out. The messages come from `Display`, through `thiserror`.

use thiserror::Error;

use crate::{
    atuple::StrideScalar,
    layout::{Layout, Profile, Tiler},
    typedefs::{Int, IntTuple, Stride},
};

/// The crate's result alias.
pub type Result<T> = std::result::Result<T, Error>;

/// Which divisibility condition a composition failed (Whitepaper,
/// Eqs. (20)-(21)).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Condition {
    Shape,
    Stride,
}

impl std::fmt::Display for Condition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Condition::Shape => write!(f, "shape"),
            Condition::Stride => write!(f, "stride"),
        }
    }
}

/// What went wrong.
///
/// The layouts sit behind a [`Box`], so the error stays small enough to
/// return by value.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum Error {
    /// A sum mixed an arithmetic tuple with a non-zero integer. The two
    /// live at different ranks, so the sum has no value.
    #[error("cannot add {lhs} and {rhs}: they sit at different ranks")]
    Incompatible { lhs: StrideScalar, rhs: StrideScalar },

    /// A product needed one integer operand and got two arithmetic
    /// tuples. `Z^S` is closed under scaling, not under multiplication.
    #[error("no product for {lhs} * {rhs}")]
    NoProduct { lhs: StrideScalar, rhs: StrideScalar },

    /// An operation needed a single scaled basis vector and got a sum of
    /// several.
    #[error("{value} is not a basis element")]
    NotBasis { value: StrideScalar },

    /// A path ran past a leaf, or off the end of a tuple.
    #[error("path {path:?} does not address {value}")]
    BadPath { path: Vec<usize>, value: String },

    /// A range ran backwards.
    #[error("range [{begin}, {end}) is reversed")]
    ReversedRange { begin: usize, end: usize },

    /// Two shapes have no common refinement: their ranks differ, their
    /// leaves disagree, or a leaf does not match the size beside it.
    #[error("no common refinement for {lhs} and {rhs}")]
    NoRefinement { lhs: IntTuple, rhs: IntTuple },

    /// Two shapes have no common coarsening, because their sizes differ.
    /// Equal sizes always meet: the integer size is the last resort.
    #[error("no common coarsening for {lhs} and {rhs}")]
    NoCoarsening { lhs: IntTuple, rhs: IntTuple },

    /// A coordinate does not index the shape beside it: the ranks
    /// differ, or a tuple coordinate met an integer shape.
    #[error("{crd} is not a coordinate of {shape}")]
    BadCoord { crd: IntTuple, shape: IntTuple },

    /// A shape and a stride do not share a profile.
    #[error("shape {shape} and stride {stride} are not congruent")]
    NotCongruent { shape: IntTuple, stride: Stride },

    /// A shape and a mode ordering do not share a profile.
    #[error("shape {shape} and order {order} are not congruent")]
    BadOrder { shape: IntTuple, order: IntTuple },

    /// A prefix product base does not weakly coarsen the shape.
    #[error("base {base} does not coarsen shape {shape}")]
    BadBase { shape: IntTuple, base: Stride },

    /// Two layouts that had to span the same domain do not.
    #[error("size mismatch between {lhs} and {rhs}")]
    SizeMismatch { lhs: Box<Layout>, rhs: Box<Layout> },

    /// A layout that had to be injective sends two coordinates to one
    /// codomain value.
    #[error("{op}({layout}) is non-injective")]
    NonInjective { op: &'static str, layout: Box<Layout> },

    /// A composition failed one of the divisibility conditions.
    #[error("{condition} condition: composition({a}, {b})")]
    Divisibility { condition: Condition, a: Box<Layout>, b: Box<Layout> },

    /// A left inverse met strides that do not form an ordered chain.
    #[error("left_inverse({layout}): the strides form no ordered chain")]
    UnorderedStrides { layout: Box<Layout> },

    /// A complement met a mode of extent zero, which it cannot divide by.
    #[error("complement({layout}): a mode of extent 0")]
    ZeroExtent { layout: Box<Layout> },

    /// A complement cannot grow to span the extension.
    #[error("complement({layout}) does not extend over {extend}")]
    ExtensionMismatch { layout: Box<Layout>, extend: IntTuple },

    /// A recast scale is not positive. The rescaling divides by it.
    #[error("scale {num}/{den} is not positive")]
    BadScale { num: Int, den: Int },

    /// A recast leaf divides unevenly: neither the stride nor the scale
    /// is a multiple of the other.
    #[error("recast {shape}:{stride} by {num}/{den} divides unevenly")]
    Recast { shape: Int, stride: StrideScalar, num: Int, den: Int },

    /// A by-mode tiler outranks the layout it dispatches over.
    #[error("rank mismatch: {op}({layout}, {tiler:?})")]
    TilerRank { op: &'static str, layout: Box<Layout>, tiler: Tiler },

    /// A coalesce profile outranks the layout it dispatches over.
    #[error("rank mismatch: {op}({layout}, {profile:?})")]
    ProfileRank { op: &'static str, layout: Box<Layout>, profile: Profile },
}
