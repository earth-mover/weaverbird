//! Arithmetic tuples. Mirrors `pycute/atuple.py`.
//!
//! An [`ArithTuple`] is an element of `Z^S`: a hierarchical tuple of
//! integers under elementwise addition and scalar multiplication, with
//! trailing positions implicitly zero.
//!
//! It is the second thing a layout may hold at a stride leaf. An integer
//! stride sends a coordinate to an offset. A basis stride sends the same
//! coordinate to a coordinate, because the sum lands in `Z^S` and not in
//! `Z`. That is what makes an identity tensor an identity.
//!
//! # Representation
//!
//! [`ArithTuple`] stores its children as PyCuTe stores them, so one
//! element admits several representations. `E(0)`, `ArithTuple(1, 0)`,
//! and `ArithTuple((1,))` all denote `1·e_0`, and all three hold
//! different children.
//!
//! Equality extends trailing positions by zero, so the three compare
//! equal. `Hash` therefore hashes [`StrideScalar::canonical`], which
//! trims those trailing zeros. Equal values hash alike.

use std::{
    cmp::Ordering,
    fmt::{self, Debug, Display},
    hash::{Hash, Hasher},
};

use crate::{
    error::{Error, Result},
    htuple::HTuple,
    typedefs::{Int, IntTuple},
};

// ---------------------------------------------------------------------------
// StrideScalar — the leaf a layout's stride may hold
// ---------------------------------------------------------------------------

/// An integer, or an arithmetic tuple.
///
/// The integer is the depth-0 case. `ScaledBasis(a, [])` is the integer
/// `a`, so an ordinary integer-strided layout is the empty-path case of
/// the same algebra. Nothing forks.
#[derive(Clone)]
pub enum StrideScalar {
    Int(Int),
    Arith(ArithTuple),
}

/// A hierarchical tuple of integers under elementwise addition and
/// scalar multiplication. Trailing positions are implicitly zero.
#[derive(Clone)]
pub struct ArithTuple {
    data: Vec<StrideScalar>,
}

impl ArithTuple {
    /// Stores `data` verbatim. PyCuTe's `_set`.
    pub fn from_data(data: Vec<StrideScalar>) -> Self {
        ArithTuple { data }
    }

    /// The children, as stored.
    pub fn data(&self) -> &[StrideScalar] {
        &self.data
    }

    /// The child at `i`, or the implicit zero past the end.
    fn child(&self, i: usize) -> StrideScalar {
        self.data.get(i).cloned().unwrap_or(StrideScalar::Int(0))
    }
}

impl From<Int> for StrideScalar {
    fn from(value: Int) -> Self {
        StrideScalar::Int(value)
    }
}

impl From<ArithTuple> for StrideScalar {
    fn from(value: ArithTuple) -> Self {
        StrideScalar::Arith(value)
    }
}

impl StrideScalar {
    /// True when every leaf is zero. The additive identity of `Z^S` at
    /// any rank.
    pub fn is_zero(&self) -> bool {
        match self {
            StrideScalar::Int(v) => *v == 0,
            StrideScalar::Arith(a) => a.data.iter().all(StrideScalar::is_zero),
        }
    }

    /// The unique representation of this element. Trims trailing zeros
    /// at every level, so an all-zero tuple becomes the integer `0`.
    ///
    /// Equality ignores those zeros, so `Hash` hashes this form.
    pub fn canonical(&self) -> StrideScalar {
        match self {
            StrideScalar::Int(v) => StrideScalar::Int(*v),
            StrideScalar::Arith(a) => {
                let mut trimmed: Vec<StrideScalar> =
                    a.data.iter().map(StrideScalar::canonical).collect();
                while trimmed.last().is_some_and(StrideScalar::is_zero) {
                    trimmed.pop();
                }
                match trimmed.is_empty() {
                    true => StrideScalar::Int(0),
                    false => StrideScalar::Arith(ArithTuple::from_data(trimmed)),
                }
            }
        }
    }

    /// This element as a plain hierarchical tuple of integers.
    ///
    /// A basis stride sends a coordinate to a coordinate, and this reads
    /// that coordinate back out.
    pub fn to_tuple(&self) -> IntTuple {
        match self {
            StrideScalar::Int(v) => HTuple::Leaf(*v),
            StrideScalar::Arith(a) => a.data().iter().map(StrideScalar::to_tuple).collect(),
        }
    }

    /// The integer value, for an element of rank zero.
    pub fn as_int(&self) -> Option<Int> {
        match self {
            StrideScalar::Int(v) => Some(*v),
            StrideScalar::Arith(_) => None,
        }
    }

    /// The elementwise sum.
    ///
    /// Adding a non-zero integer to an [`ArithTuple`] returns
    /// [`Error::Incompatible`]: the two sit at different ranks, so the
    /// sum has no value.
    pub fn add(&self, other: &StrideScalar) -> Result<StrideScalar> {
        match (self, other) {
            (StrideScalar::Int(x), StrideScalar::Int(y)) => Ok(StrideScalar::Int(x + y)),
            (StrideScalar::Arith(_), StrideScalar::Int(0)) => Ok(self.clone()),
            (StrideScalar::Int(0), StrideScalar::Arith(_)) => Ok(other.clone()),
            (StrideScalar::Arith(x), StrideScalar::Arith(y)) => {
                let width = x.data.len().max(y.data.len());
                (0..width)
                    .map(|i| x.child(i).add(&y.child(i)))
                    .collect::<Result<Vec<_>>>()
                    .map(|data| StrideScalar::Arith(ArithTuple::from_data(data)))
            }
            _ => Err(Error::Incompatible { lhs: self.clone(), rhs: other.clone() }),
        }
    }

    /// Every leaf multiplied by `factor`.
    pub fn scale(&self, factor: Int) -> StrideScalar {
        match self {
            StrideScalar::Int(v) => StrideScalar::Int(v * factor),
            StrideScalar::Arith(a) => StrideScalar::Arith(ArithTuple::from_data(
                a.data.iter().map(|c| c.scale(factor)).collect(),
            )),
        }
    }

    /// The sub-element at `path`. An empty path returns the whole
    /// element.
    pub fn get(&self, path: &[usize]) -> Result<&StrideScalar> {
        let bad_path = || Error::BadPath { path: path.to_vec(), value: format!("{self}") };
        match (path.split_first(), self) {
            (None, _) => Ok(self),
            (Some((&i, rest)), StrideScalar::Arith(a)) => {
                a.data.get(i).ok_or_else(bad_path)?.get(rest)
            }
            (Some(_), StrideScalar::Int(_)) => Err(bad_path()),
        }
    }
}

// ---------------------------------------------------------------------------
// Equality and order
// ---------------------------------------------------------------------------

/// Equality under implicit zero extension.
///
/// Two elements are equal when every explicit or implicit child agrees.
/// So `int 0` equals an all-zero tuple of any rank, while `int 5` equals
/// no tuple at all — a non-zero integer has no rank-1 form.
impl PartialEq for StrideScalar {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (StrideScalar::Int(x), StrideScalar::Int(y)) => x == y,
            (StrideScalar::Int(v), StrideScalar::Arith(_))
            | (StrideScalar::Arith(_), StrideScalar::Int(v)) => {
                *v == 0 && self.is_zero() && other.is_zero()
            }
            (StrideScalar::Arith(x), StrideScalar::Arith(y)) => {
                let width = x.data.len().max(y.data.len());
                (0..width).all(|i| x.child(i) == y.child(i))
            }
        }
    }
}

impl Eq for StrideScalar {}

impl PartialEq for ArithTuple {
    fn eq(&self, other: &Self) -> bool {
        StrideScalar::Arith(self.clone()) == StrideScalar::Arith(other.clone())
    }
}

impl Eq for ArithTuple {}

/// Hashes [`StrideScalar::canonical`], so that equal values hash alike.
impl Hash for StrideScalar {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self.canonical() {
            StrideScalar::Int(v) => {
                state.write_u8(0);
                v.hash(state);
            }
            StrideScalar::Arith(a) => {
                state.write_u8(1);
                a.data.hash(state);
            }
        }
    }
}

impl Hash for ArithTuple {
    fn hash<H: Hasher>(&self, state: &mut H) {
        StrideScalar::Arith(self.clone()).hash(state);
    }
}

impl StrideScalar {
    /// Colexicographic order: the highest position decides.
    ///
    /// The order is partial. A non-zero integer and a tuple sit at
    /// different ranks, so they do not compare and this returns
    /// [`Error::Incompatible`].
    pub fn try_cmp(&self, other: &Self) -> Result<Ordering> {
        match (colex_lt(self, other)?, colex_lt(other, self)?) {
            (true, _) => Ok(Ordering::Less),
            (_, true) => Ok(Ordering::Greater),
            (false, false) => Ok(Ordering::Equal),
        }
    }

    /// Colexicographic order for a sort, with incomparable elements
    /// reported equal.
    ///
    /// The crate sorts modes by stride, and a stable sort then holds an
    /// incomparable pair in its input order. Use [`Self::try_cmp`] where
    /// the answer must be an order.
    pub fn sort_cmp(&self, other: &Self) -> Ordering {
        self.try_cmp(other).unwrap_or(Ordering::Equal)
    }
}

// The type carries no `PartialOrd`. The order is partial, and `<` would
// hide the pairs that do not compare.

/// Strict colex order. Walks both elements from the highest position
/// down. Returns [`Error::Incompatible`] on a rank mismatch.
fn colex_lt(a: &StrideScalar, b: &StrideScalar) -> Result<bool> {
    let incompatible = || Error::Incompatible { lhs: a.clone(), rhs: b.clone() };
    match (a, b) {
        (StrideScalar::Int(x), StrideScalar::Int(y)) => Ok(x < y),
        (StrideScalar::Int(v), _) | (_, StrideScalar::Int(v)) if *v != 0 => Err(incompatible()),
        _ => {
            let empty: &[StrideScalar] = &[];
            let (x, y) = match (a, b) {
                (StrideScalar::Arith(x), StrideScalar::Arith(y)) => (x.data(), y.data()),
                (StrideScalar::Arith(x), _) => (x.data(), empty),
                (_, StrideScalar::Arith(y)) => (empty, y.data()),
                _ => (empty, empty),
            };
            let zero = StrideScalar::Int(0);
            for i in (0..x.len().max(y.len())).rev() {
                let (l, r) = (x.get(i).unwrap_or(&zero), y.get(i).unwrap_or(&zero));
                if colex_lt(l, r)? {
                    return Ok(true);
                }
                if colex_lt(r, l)? {
                    return Ok(false);
                }
            }
            Ok(false)
        }
    }
}

// ---------------------------------------------------------------------------
// Display
// ---------------------------------------------------------------------------

/// CuTe notation: a single scaled basis vector prints as
/// `value@p_n@…@p_0`, and anything else as a tuple.
impl Display for StrideScalar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (basis_repr(self).as_slice(), self) {
            ([(value, path)], _) => {
                write!(f, "{value}")?;
                path.iter().rev().try_for_each(|p| write!(f, "@{p}"))
            }
            (_, StrideScalar::Arith(a)) => {
                write!(f, "(")?;
                for (i, c) in a.data.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    Display::fmt(c, f)?;
                }
                write!(f, ")")
            }
            (_, StrideScalar::Int(v)) => write!(f, "{v}"),
        }
    }
}

/// The same notation as [`Display`], so a stride reads alike in a
/// message and in a dump.
impl Debug for StrideScalar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Display::fmt(self, f)
    }
}

impl Display for ArithTuple {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Display::fmt(&StrideScalar::Arith(self.clone()), f)
    }
}

impl Debug for ArithTuple {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Display::fmt(self, f)
    }
}

// ---------------------------------------------------------------------------
// Factories
// ---------------------------------------------------------------------------

/// A scaled basis vector at `path`.
///
/// ```text
/// scaled_basis(a, [])     == a
/// scaled_basis(a, [0])    == (a,0,0,…)
/// scaled_basis(a, [1])    == (0,a,0,…)
/// scaled_basis(a, [1,0])  == (0,(a,0,…),0,…)
/// ```
pub fn scaled_basis(value: Int, path: &[usize]) -> StrideScalar {
    path.iter().rev().fold(StrideScalar::Int(value), |acc, &i| {
        let mut data = vec![StrideScalar::Int(0); i];
        data.push(acc);
        StrideScalar::Arith(ArithTuple::from_data(data))
    })
}

/// The unit basis element at `path`. PyCuTe's `E`.
pub fn e(path: &[usize]) -> StrideScalar {
    scaled_basis(1, path)
}

/// Builds a unit basis element. `e!()` is the integer `1`; `e!(1, 0)` is
/// the unit at path `[1, 0]`.
#[macro_export]
macro_rules! e {
    ($($index:expr),* $(,)?) => { $crate::atuple::e(&[$($index),*]) };
}

// ---------------------------------------------------------------------------
// Basis accessors
// ---------------------------------------------------------------------------

/// Decomposes `x` into scaled basis vectors, so that
/// `x == sum(value · e(path))`.
///
/// Each entry is one non-zero leaf with its path. An all-zero element
/// decomposes to the single rank-zero term `(0, [])`.
pub(crate) fn basis_repr(x: &StrideScalar) -> Vec<(Int, Vec<usize>)> {
    fn walk(x: &StrideScalar, prefix: &[usize], out: &mut Vec<(Int, Vec<usize>)>) {
        match x {
            StrideScalar::Arith(a) => a.data().iter().enumerate().for_each(|(i, c)| {
                let mut path = prefix.to_vec();
                path.push(i);
                walk(c, &path, out);
            }),
            StrideScalar::Int(v) if *v != 0 => out.push((*v, prefix.to_vec())),
            StrideScalar::Int(_) => {}
        }
    }
    let mut out = Vec::new();
    walk(x, &[], &mut out);
    match out.is_empty() {
        true => vec![(0, vec![])],
        false => out,
    }
}

/// True when `x` is a single scaled basis vector. Every integer counts,
/// including zero.
#[cfg(test)]
pub(crate) fn is_basis(x: &StrideScalar) -> bool {
    basis_repr(x).len() == 1
}

/// The path of a single scaled basis vector.
///
/// Returns [`Error::NotBasis`] for a sum of several terms.
fn basis_path(profile: &StrideScalar) -> Result<Vec<usize>> {
    match basis_repr(profile).as_slice() {
        [(_, path)] => Ok(path.clone()),
        _ => Err(Error::NotBasis { value: profile.clone() }),
    }
}

/// The part of `x` at the position `profile` names.
///
/// `profile` must be a single scaled basis vector — typically a stride
/// leaf, which already is one. An integer profile names the empty path,
/// so it returns `x` unchanged.
pub(crate) fn proj<'a>(x: &'a StrideScalar, profile: &StrideScalar) -> Result<&'a StrideScalar> {
    x.get(&basis_path(profile)?)
}

/// The part of a hierarchical tuple at the position `profile` names.
///
/// The same projection as [`proj`], over an [`HTuple`] carrier instead
/// of a [`StrideScalar`] one. PyCuTe has one `proj`, because there `get`
/// walks any value.
///
/// Returns [`Error::BadPath`] when the path runs off `x`.
#[cfg(test)]
pub(crate) fn proj_tuple<'a, T>(x: &'a HTuple<T>, profile: &StrideScalar) -> Result<&'a HTuple<T>>
where
    T: Debug,
{
    let path = basis_path(profile)?;
    x.get(&path).ok_or_else(|| Error::BadPath { path, value: format!("{x:?}") })
}

/// The part of a hierarchical tuple at the position `profile` names,
/// for writing through. The mutable twin of [`proj_tuple`].
pub(crate) fn proj_tuple_mut<'a, T>(
    x: &'a mut HTuple<T>,
    profile: &StrideScalar,
) -> Result<&'a mut HTuple<T>>
where
    T: Debug,
{
    x.try_get_mut(&basis_path(profile)?)
}

/// The unit basis element at `profile`'s path. Ignores `profile`'s
/// coefficient.
pub(crate) fn unit(profile: &StrideScalar) -> Result<StrideScalar> {
    basis_path(profile).map(|path| e(&path))
}

/// A `profile`-shaped tuple of unit basis elements, one per leaf, each
/// at its own position. PyCuTe's `make_basis_like`.
///
/// This is the stride of an identity tensor.
pub(crate) fn make_basis_like<T>(profile: &HTuple<T>) -> HTuple<StrideScalar> {
    fn walk<T>(profile: &HTuple<T>, prefix: &[usize]) -> HTuple<StrideScalar> {
        match profile {
            HTuple::Leaf(_) => HTuple::Leaf(e(prefix)),
            HTuple::Tuple(modes) => HTuple::Tuple(
                modes
                    .iter()
                    .enumerate()
                    .map(|(i, m)| {
                        let mut path = prefix.to_vec();
                        path.push(i);
                        walk(m, &path)
                    })
                    .collect(),
            ),
        }
    }
    walk(profile, &[])
}

#[cfg(test)]
#[path = "tests/atuple.rs"]
mod tests;
