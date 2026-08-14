//! Functions for CuTe strides. Mirrors `pycute/stride.py`.
//!
//! A stride is congruent with a shape and holds a [`StrideScalar`] at
//! every leaf. The functions here read a stride, pair it with a shape,
//! and fold adjacent modes together.
//!
//! PyCuTe dispatches on attributes: `stride` looks for `.stride`, and
//! `coshape` for `._coshape`. Rust has traits, so the second becomes
//! [`Coshape`]. The first needs no trait yet — until `Layout` lands, the
//! only thing that has a stride is a stride.

use crate::{
    atuple::StrideScalar,
    error::{Error, Result},
    htuple::{HTuple, zip_transform_leaf},
    typedefs::{Int, IntTuple, Stride},
};

/// The sub-stride at `mode`. An empty mode returns the whole stride.
///
/// PyCuTe's `stride`, minus the `Layout` / `Tensor` dispatch, which
/// arrives with those types.
///
/// Returns [`Error::BadPath`] when `mode` does not address `obj`.
pub fn stride<'a>(obj: &'a Stride, mode: &[usize]) -> Result<&'a Stride> {
    obj.get(mode).ok_or_else(|| Error::BadPath {
        path: mode.to_vec(),
        value: format!("{obj:?}"),
    })
}

/// Sum of the leaf-wise products of two congruent HTuples: `sum(x*y)`.
///
/// Pre-conditions:
///   congruent(a, b)
///
/// Examples:
/// ```text
/// inner_product((1, 0, 1),    (1, 3, 6))       == 7
/// inner_product((2, 3),       (1, 4))          == 14
/// inner_product((1, (2, 3)),  (1, (10, 100)))  == 321
/// ```
///
/// Returns [`Error::BadPath`] when the two profiles differ, and
/// [`Error::Incompatible`] when the products do not sum.
pub fn inner_product(a: &IntTuple, b: &Stride) -> Result<StrideScalar> {
    zip_transform_leaf(&|x: &Int, y: &StrideScalar| y.scale(*x), a, b)?
        .leaves()
        .into_iter()
        .try_fold(StrideScalar::Int(0), |sum, term| sum.add(term))
}

/// Exclusive prefix product of the leaves of `a`, congruent with `a`.
///
/// `init` seeds the running product and may be:
///   -- a stride scalar (e.g. the default `1`), or
///   -- a tuple of stride scalars weakly congruent with `a`;
///      each mode is prefix-producted independently.
///
/// Pre-conditions:
///   weakly_congruent(init, a)
///
/// Examples:
/// ```text
/// prefix_product((3, 2, 4))           == (1, 3, 6)
/// prefix_product((3, (2, 4)))         == (1, (3, 6))
/// prefix_product((4, 8), 2)           == (2, 8)               # base 2
/// prefix_product(((2, 3), (4, 5)), (1, 100)) == ((1, 2), (100, 400))   # per-mode base
/// ```
///
/// Returns [`Error::Incompatible`] when `init` does not weakly coarsen
/// `a`.
pub fn prefix_product(a: &IntTuple, init: &Stride) -> Result<Stride> {
    let incompatible = || Error::Incompatible {
        lhs: format!("{a:?}"),
        rhs: format!("{init:?}"),
    };
    match (init, a) {
        (HTuple::Leaf(seed), _) => {
            // One running product per leaf, each emitted before it takes
            // that leaf in, so the fold stays exclusive.
            let mut running = a
                .leaves()
                .into_iter()
                .scan(seed.clone(), |product, v| {
                    let current = product.clone();
                    *product = product.scale(*v);
                    Some(current)
                })
                .collect::<Vec<_>>()
                .into_iter();
            // One value per leaf, so this cannot run dry.
            HTuple::unflatten(&mut running, a).ok_or_else(incompatible)
        }
        (HTuple::Tuple(seeds), HTuple::Tuple(modes)) if seeds.len() == modes.len() => modes
            .iter()
            .zip(seeds)
            .map(|(x, i)| prefix_product(x, i))
            .collect::<Result<Vec<_>>>()
            .map(HTuple::Tuple),
        _ => Err(incompatible()),
    }
}

/// A type that knows the shape of its own codomain.
///
/// PyCuTe asks `hasattr(obj, '_coshape')`; this is that question, asked
/// at compile time.
pub trait Coshape {
    /// Shape of the codomain. PyCuTe's `_coshape`.
    ///
    /// Fallible, because the coshape of a `Layout` is an inner product
    /// of its shape and its stride: an incongruent pair, or a stride
    /// that mixes an integer with a basis element, has none. PyCuTe
    /// raises in both cases.
    fn coshape(&self) -> Result<IntTuple>;
}

/// Shape of the codomain.
///
/// Returns [`Error::BadPath`] when `mode` does not address the coshape.
pub fn coshape<T: Coshape>(obj: &T, mode: &[usize]) -> Result<IntTuple> {
    let value = obj.coshape()?;
    value.get(mode).cloned().ok_or_else(|| Error::BadPath {
        path: mode.to_vec(),
        value: format!("{value:?}"),
    })
}

/// Profile of the codomain.
///
/// Returns [`Error::BadPath`] when `mode` does not address the coshape.
pub fn coprofile<T: Coshape>(obj: &T, mode: &[usize]) -> Result<IntTuple> {
    coshape(obj, mode)
}

/// Return a new shape and stride that are coalesced equivalents of the
/// input. This is the size-1-preserving ("_z") core fold.
///
/// Two adjacent modes may be merged only when the merge preserves the
/// layout's evaluation. The merge condition below verifies this with two
/// O(1) checks that are jointly necessary and sufficient:
///
///   1. `s_a*d_a == d_b`                       (linearity at `(0, 1)`)
///   2. `(s_a-1)*d_a + d_b == (2*s_a-1)*d_a`   (linearity at `(s_a-1, 1)`)
///
/// PyCuTe guards the merge with `is_static(s_a) == is_static(s_b)`, so
/// that a concrete shape never folds into a symbolic one. [`Int`] is
/// always concrete, so the guard is always satisfied here and is left
/// out.
///
/// Pre-conditions:
///   congruent(shape, stride)
///
/// PyCuTe's `_coalesce_z`, the stride-level half of `coalesce_z`.
pub fn coalesce_z(shape: &IntTuple, stride: &Stride) -> Result<(IntTuple, Stride)> {
    let (result_s, result_d) = shape.leaves().into_iter().zip(stride.leaves()).try_fold(
        (Vec::new(), Vec::new()),
        |(mut result_s, mut result_d): (Vec<Int>, Vec<StrideScalar>), (&s_b, d_b)| {
            // Drop trailing size-1 modes.
            while result_s.last() == Some(&1) {
                result_s.pop();
                result_d.pop();
            }
            let merged = match (result_s.last(), result_d.last()) {
                (Some(&s_a), Some(d_a)) => mergeable(s_a, d_a, d_b)?.then_some(s_a * s_b),
                _ => None,
            };
            match (merged, result_s.last_mut()) {
                // Merge mergeable modes.
                (Some(size), Some(last)) => *last = size,
                // Else, append.
                _ => {
                    result_s.push(s_b);
                    result_d.push(d_b.clone());
                }
            }
            Ok((result_s, result_d))
        },
    )?;
    Ok((
        HTuple::Tuple(result_s.into_iter().map(HTuple::Leaf).collect()),
        HTuple::Tuple(result_d.into_iter().map(HTuple::Leaf).collect()),
    ))
}

/// The two linearity checks of [`coalesce_z`], in order. The second is
/// reached only once the first holds, which is what keeps its sum
/// well-typed: `d_b` then agrees with a multiple of `d_a`, so the two
/// sit at the same rank.
fn mergeable(s_a: Int, d_a: &StrideScalar, d_b: &StrideScalar) -> Result<bool> {
    match d_a.scale(s_a) == *d_b {
        false => Ok(false),
        true => d_a
            .scale(s_a - 1)
            .add(d_b)
            .map(|sum| sum == d_a.scale(2 * s_a - 1)),
    }
}
