//! Strides: the map half of a layout.
//!
//! A stride is congruent with a shape and holds a [`StrideScalar`] at
//! every leaf. The methods here pair a stride with a shape, and the free
//! function folds adjacent modes together.

use crate::{
    atuple::StrideScalar,
    error::{Error, Result},
    htuple::HTuple,
    typedefs::{Int, IntTuple, Stride},
};

/// An integer tuple read as a stride, one [`StrideScalar::Int`] per leaf.
///
/// The depth-0 case of the stride algebra: every integer stride is an
/// arithmetic-tuple stride whose leaves happen to carry no basis.
///
/// ```
/// # use weaverbird::{ht, stride, Stride};
/// assert_eq!(Stride::from(ht!((8, 1))), stride!((8, 1)));
/// ```
impl From<IntTuple> for Stride {
    fn from(t: IntTuple) -> Self {
        t.transform_leaf(&|v: &Int| StrideScalar::Int(*v))
    }
}

impl IntTuple {
    /// Sum of the leaf-wise products of a shape and a stride: `sum(x*y)`.
    ///
    /// ```
    /// # use weaverbird::{ht, stride, HTuple, StrideScalar};
    /// let dot = ht!((1, 0, 1)).inner_product(&stride!((1, 3, 6))).unwrap();
    /// assert_eq!(dot, StrideScalar::Int(7));
    /// ```
    ///
    /// Returns [`Error::BadPath`] when the two profiles differ, and
    /// [`Error::Incompatible`] when the products do not sum.
    pub fn inner_product(&self, stride: &Stride) -> Result<StrideScalar> {
        self.zip_transform(stride, &|x: &Int, y: &StrideScalar| y.scale(*x))?
            .leaves()
            .try_fold(StrideScalar::Int(0), |sum, term| sum.add(term))
    }

    /// Exclusive prefix product of the extents, congruent with the shape.
    ///
    /// `init` seeds the running product. It is a stride scalar, or a
    /// tuple of them that weakly coarsens the shape; each mode then runs
    /// its own product.
    ///
    /// ```
    /// # use weaverbird::{ht, stride, HTuple, StrideScalar};
    /// assert_eq!(ht!((3, 2, 4)).prefix_product(&stride!(1)).unwrap(), stride!((1, 3, 6)));
    /// assert_eq!(ht!((4, 8)).prefix_product(&stride!(2)).unwrap(), stride!((2, 8)));
    /// ```
    ///
    /// Returns [`Error::Incompatible`] when `init` does not weakly
    /// coarsen the shape.
    pub fn prefix_product(&self, init: &Stride) -> Result<Stride> {
        let incompatible = || Error::BadBase { shape: self.clone(), base: init.clone() };
        match (init, self) {
            (HTuple::Leaf(seed), _) => {
                // One running product per leaf, each emitted before it
                // takes that leaf in, so the fold stays exclusive.
                let mut running = self
                    .leaves()
                    .scan(seed.clone(), |product, v| {
                        let current = product.clone();
                        *product = product.scale(*v);
                        Some(current)
                    })
                    .collect::<Vec<_>>()
                    .into_iter();
                // One value per leaf, so this cannot run dry.
                HTuple::unflatten(&mut running, self).ok_or_else(incompatible)
            }
            (HTuple::Tuple(seeds), HTuple::Tuple(modes)) if seeds.len() == modes.len() => {
                modes.iter().zip(seeds).map(|(mode, seed)| mode.prefix_product(seed)).collect()
            }
            _ => Err(incompatible()),
        }
    }

    /// The compact, column-major stride of this shape: the prefix product
    /// from a base of one.
    pub fn compact_stride(&self) -> Stride {
        self.compact_stride_from(&mut 1)
    }

    /// The compact stride, with `base` as the running product. `base`
    /// leaves the call holding the size of this shape.
    pub(crate) fn compact_stride_from(&self, base: &mut Int) -> Stride {
        match self {
            HTuple::Leaf(extent) => {
                let current = *base;
                *base *= extent;
                HTuple::Leaf(StrideScalar::Int(current))
            }
            HTuple::Tuple(modes) => {
                modes.iter().map(|mode| mode.compact_stride_from(base)).collect()
            }
        }
    }
}

/// The coalesced equivalent of `shape` and `stride`, keeping size-1
/// modes.
///
/// A merge of two adjacent modes must preserve the layout's evaluation.
/// Two O(1) checks decide it, and they are jointly necessary and
/// sufficient:
///
///   1. `s_a*d_a == d_b`                       (linearity at `(0, 1)`)
///   2. `(s_a-1)*d_a + d_b == (2*s_a-1)*d_a`   (linearity at `(s_a-1, 1)`)
///
/// Pre-conditions:
///   congruent(shape, stride)
pub(crate) fn coalesce_modes(shape: &IntTuple, stride: &Stride) -> Result<(IntTuple, Stride)> {
    let (result_s, result_d) = shape.leaves().zip(stride.leaves()).try_fold(
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
        result_s.into_iter().map(HTuple::Leaf).collect(),
        result_d.into_iter().map(HTuple::Leaf).collect(),
    ))
}

/// The two linearity checks of [`coalesce_modes`], in order. The second
/// runs only once the first holds, which is what keeps its sum
/// well-typed: `d_b` then agrees with a multiple of `d_a`, so the two sit
/// at the same rank.
fn mergeable(s_a: Int, d_a: &StrideScalar, d_b: &StrideScalar) -> Result<bool> {
    match d_a.scale(s_a) == *d_b {
        false => Ok(false),
        true => d_a.scale(s_a - 1).add(d_b).map(|sum| sum == d_a.scale(2 * s_a - 1)),
    }
}

#[cfg(test)]
#[path = "tests/stride.rs"]
mod tests;
