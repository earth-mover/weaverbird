//! The crate's error type. PyCuTe raises `ValueError` and `TypeError`;
//! each raise becomes a variant here.

/// What went wrong.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    /// A sum mixed an [`ArithTuple`](crate::ArithTuple) with a non-zero
    /// integer. The two live at different ranks, so the sum has no value.
    #[error("arithmetic tuple incompatibility: {lhs} + {rhs}")]
    Incompatible { lhs: String, rhs: String },

    /// An operation needed a single scaled basis vector and got a sum of
    /// several.
    #[error("{value} is not a basis element")]
    NotBasis { value: String },

    /// A layout that had to be injective sends two coordinates to one
    /// codomain value. PyCuTe's "Non-injective layout" failures.
    #[error("non-injective layout: {detail}")]
    NonInjective { detail: String },

    /// A path ran past a leaf, or off the end of a tuple.
    #[error("path {path:?} does not address {value}")]
    BadPath { path: Vec<usize>, value: String },

    /// A range ran backwards.
    #[error("range [{begin}, {end}) is reversed")]
    ReversedRange { begin: usize, end: usize },

    /// Two shapes have no common refinement: their ranks differ, their
    /// leaves disagree, or a leaf does not match the size beside it.
    #[error("no common refinement for {lhs} and {rhs}")]
    NoRefinement { lhs: String, rhs: String },

    /// Two shapes have no common coarsening, because their sizes differ.
    /// Equal sizes always meet: the integer size is the last resort.
    #[error("no common coarsening for {lhs} and {rhs}")]
    NoCoarsening { lhs: String, rhs: String },

    /// A coordinate does not index the shape beside it: the ranks
    /// differ, or a tuple coordinate met an integer shape.
    #[error("idx2crd({idx}, {shape})")]
    BadCoord { idx: String, shape: String },

    /// Two layouts that had to span the same domain do not. The
    /// pre-condition of [`layout_add`](crate::layout_add).
    #[error("size mismatch between {lhs} and {rhs}")]
    SizeMismatch { lhs: String, rhs: String },

    /// Two hierarchical tuples that had to share a profile do not.
    #[error("{lhs} and {rhs} are not congruent")]
    NotCongruent { lhs: String, rhs: String },

    /// A product needed one integer operand and got two arithmetic
    /// tuples. `Z^S` is closed under scaling, not under multiplication.
    #[error("no product for {lhs} * {rhs}")]
    NoProduct { lhs: String, rhs: String },

    /// A division did not come out even. PyCuTe's "divisibility
    /// condition" failures.
    #[error("divisibility condition violated: {detail}")]
    Divisibility { detail: String },

    /// A by-mode operation got a profile of higher rank than the value
    /// it dispatches over. PyCuTe's "Rank mismatch" failures.
    #[error("rank mismatch: {op}({value}, {profile})")]
    RankMismatch {
        op: &'static str,
        value: String,
        profile: String,
    },
}

/// The crate's result alias.
pub type Result<T> = std::result::Result<T, Error>;
