//! Shapes: the integer half of a layout.
//!
//! A shape is an [`IntTuple`] of extents. Every function here is a method
//! on that tuple, so a caller holding a [`Layout`](crate::Layout) reads
//! `layout.shape`, and a caller wanting one mode reads
//! `layout.shape.get(mode)`.

use crate::{
    atuple::StrideScalar,
    error::{Error, Result},
    htuple::HTuple,
    typedefs::{Int, IntTuple},
};

impl IntTuple {
    /// The size of the domain: the product of every extent.
    pub fn size(&self) -> Int {
        self.leaves().product()
    }

    /// The size of each top-level mode, keeping the rank.
    pub fn size_each(&self) -> IntTuple {
        match self {
            HTuple::Leaf(v) => HTuple::Leaf(*v),
            HTuple::Tuple(modes) => modes.iter().map(|m| HTuple::Leaf(m.size())).collect(),
        }
    }

    /// The longest path from the root to a leaf. A leaf has depth zero,
    /// and an empty tuple has depth one.
    pub fn depth(&self) -> usize {
        match self {
            HTuple::Leaf(_) => 0,
            HTuple::Tuple(modes) => 1 + modes.iter().map(IntTuple::depth).max().unwrap_or(0),
        }
    }

    /// A tuple that holds `value` at `path` and zero everywhere else.
    ///
    /// The inverse of [`HTuple::get`]: `lift(x, p).get(p) == x`.
    pub fn lift(value: Int, path: &[usize]) -> IntTuple {
        path.iter().rev().fold(HTuple::Leaf(value), |acc, &i| {
            let mut modes = vec![HTuple::Leaf(0); i];
            modes.push(acc);
            HTuple::Tuple(modes)
        })
    }

    /// True when `self` *coarsens* `other`.
    ///
    /// Compatibility, `a ≼ b`, is a partial order on shapes. It
    /// strengthens weak congruence by also requiring the sizes to agree.
    /// Every coordinate of `a` is then a coordinate of `b`.
    ///
    /// ```
    /// # use weaverbird::ht;
    /// assert!(ht!(30).compatible_with(&ht!((2, 15))));
    /// assert!(ht!((2, 15)).compatible_with(&ht!((2, (3, 5)))));
    /// assert!(!ht!(24).compatible_with(&ht!(32)));               // size mismatch
    /// assert!(!ht!((2, (3, 5))).compatible_with(&ht!(((3, 2), 5))));
    /// assert!(ht!(24).compatible_with(&ht!((24,))));             // int ≼ (int,)
    /// assert!(!ht!((24,)).compatible_with(&ht!(24)));            // not the reverse
    /// ```
    pub fn compatible_with(&self, other: &IntTuple) -> bool {
        match (self, other) {
            (HTuple::Tuple(x), HTuple::Tuple(y)) => {
                x.len() == y.len() && x.iter().zip(y).all(|(i, j)| i.compatible_with(j))
            }
            (HTuple::Leaf(v), _) => *v == other.size(),
            (HTuple::Tuple(_), HTuple::Leaf(_)) => false,
        }
    }

    /// The minimal shape that *refines* both shapes — their join under
    /// the compatibility order.
    ///
    /// ```
    /// # use weaverbird::ht;
    /// assert_eq!(ht!(30).common_refinement(&ht!((2, 15))).unwrap(), ht!((2, 15)));
    /// assert_eq!(ht!(10).common_refinement(&ht!((10,))).unwrap(), ht!((10,)));
    /// ```
    ///
    /// Returns [`Error::NoRefinement`] when no such shape exists.
    pub fn common_refinement(&self, other: &IntTuple) -> Result<IntTuple> {
        match (self, other) {
            (HTuple::Tuple(x), HTuple::Tuple(y)) if x.len() == y.len() => {
                x.iter().zip(y).map(|(a, b)| a.common_refinement(b)).collect()
            }
            (HTuple::Leaf(x), HTuple::Leaf(y)) if x == y => Ok(self.clone()),
            (HTuple::Leaf(x), HTuple::Tuple(_)) if *x == other.size() => Ok(other.clone()),
            (HTuple::Tuple(_), HTuple::Leaf(y)) if self.size() == *y => Ok(self.clone()),
            _ => Err(Error::NoRefinement { lhs: self.clone(), rhs: other.clone() }),
        }
    }

    /// The maximal shape that *coarsens* both shapes — their meet under
    /// the compatibility order.
    ///
    /// A meet exists exactly when the two sizes agree. In the worst case
    /// it is the integer size itself.
    ///
    /// ```
    /// # use weaverbird::ht;
    /// let a = ht!((4, (3, 5)));
    /// assert_eq!(a.common_coarsening(&ht!(((2, 2), 15))).unwrap(), ht!((4, 15)));
    /// assert_eq!(ht!((6, 5)).common_coarsening(&ht!((2, 15))).unwrap(), ht!(30));
    /// ```
    ///
    /// Returns [`Error::NoCoarsening`] when the sizes differ.
    pub fn common_coarsening(&self, other: &IntTuple) -> Result<IntTuple> {
        // The per-mode meet, when both sides are tuples of one rank and
        // every mode meets. Otherwise the integer size is the last
        // resort.
        let per_mode = match (self, other) {
            (HTuple::Tuple(x), HTuple::Tuple(y)) if x.len() == y.len() => x
                .iter()
                .zip(y)
                .map(|(a, b)| a.common_coarsening(b))
                .collect::<Result<IntTuple>>()
                .ok(),
            _ => None,
        };
        match (per_mode, self.size(), other.size()) {
            (Some(meet), _, _) => Ok(meet),
            (None, a, b) if a == b => Ok(HTuple::Leaf(a)),
            _ => Err(Error::NoCoarsening { lhs: self.clone(), rhs: other.clone() }),
        }
    }

    /// Maps any coordinate of this shape to a *natural* coordinate.
    ///
    /// The decomposition is colexicographic, so the leftmost mode varies
    /// fastest. The final mode skips its `mod` and keeps the whole
    /// quotient. An out-of-bounds index therefore does not wrap; the
    /// excess accumulates in the last leaf.
    ///
    /// ```
    /// # use weaverbird::ht;
    /// assert_eq!(ht!((3, 2, 4)).idx2crd(&ht!(7)).unwrap(), ht!((1, 0, 1)));
    /// assert_eq!(ht!((3, (2, 4))).idx2crd(&ht!(7)).unwrap(), ht!((1, (0, 1))));
    /// assert_eq!(ht!((3, 7, 2)).idx2crd(&ht!(42)).unwrap(), ht!((0, 0, 2)));
    /// ```
    ///
    /// Returns [`Error::BadCoord`] when `idx` does not index this shape.
    pub fn idx2crd(&self, idx: &IntTuple) -> Result<IntTuple> {
        let bad_coord = || Error::BadCoord { crd: idx.clone(), shape: self.clone() };
        match (idx, self) {
            (HTuple::Leaf(i), HTuple::Leaf(_)) => Ok(HTuple::Leaf(*i)),
            (HTuple::Tuple(is), HTuple::Tuple(ss)) if is.len() == ss.len() => {
                is.iter().zip(ss).map(|(i, s)| s.idx2crd(i)).collect()
            }
            (HTuple::Leaf(i), HTuple::Tuple(_)) => {
                let extents = self.leaves().copied().collect::<Vec<_>>();
                // Every extent but the last takes a `divmod`. The last
                // one keeps the quotient whole.
                let head = extents.split_last().map_or(&[][..], |(_, head)| head);
                let mut quotient = *i;
                let mut digits = head
                    .iter()
                    .map(|&s| {
                        let remainder = quotient.rem_euclid(s);
                        quotient = quotient.div_euclid(s);
                        remainder
                    })
                    .collect::<Vec<_>>();
                digits.push(quotient);
                HTuple::unflatten(&mut digits.into_iter(), self).ok_or_else(bad_coord)
            }
            _ => Err(bad_coord()),
        }
    }

    /// Maps any coordinate of this shape to an integral coordinate.
    ///
    /// The recomposition is colexicographic, so the leftmost mode varies
    /// fastest. It inverts [`Self::idx2crd`] on in-bounds input.
    ///
    /// `crd` need only weakly coarsen the shape: a leaf of `crd` may
    /// stand for a whole sub-tree, which then contributes its
    /// [`Self::size`]. That is what admits a flat coordinate against a
    /// nested shape.
    ///
    /// ```
    /// # use weaverbird::{ht, StrideScalar};
    /// let idx = ht!((3, (2, 4))).crd2idx(&ht!((1, (0, 1)))).unwrap();
    /// assert_eq!(idx, StrideScalar::Int(7));
    /// let flat = ht!((3, (2, 3))).crd2idx(&ht!((2, 5))).unwrap();
    /// assert_eq!(flat, StrideScalar::Int(17));
    /// ```
    ///
    /// Returns [`Error::BadCoord`] when `crd` does not coarsen the shape.
    pub fn crd2idx(&self, crd: &IntTuple) -> Result<StrideScalar> {
        // One extent per leaf of `crd`: the size of the sub-shape that
        // leaf stands for.
        let sizes = crd
            .zip_leaves(self)
            .map_err(|_| Error::BadCoord { crd: crd.clone(), shape: self.clone() })?
            .into_iter()
            .map(|(_, sub)| sub.size())
            .collect::<Vec<_>>();
        let extents = HTuple::unflatten(&mut sizes.into_iter(), crd)
            .ok_or_else(|| Error::BadCoord { crd: crd.clone(), shape: self.clone() })?;
        crd.inner_product(&extents.compact_stride())
    }

    /// Every natural coordinate of this shape, in colexicographic order.
    ///
    /// ```
    /// # use weaverbird::ht;
    /// let all: Vec<_> = ht!((3, 2)).coordinates().collect();
    /// assert_eq!(all.first(), Some(&ht!((0, 0))));
    /// assert_eq!(all.len(), 6);
    /// ```
    pub fn coordinates(&self) -> Coordinates<'_> {
        let extents = self.leaves().copied().collect::<Vec<_>>();
        Coordinates {
            digits: vec![0; extents.len()],
            done: extents.iter().any(|&s| s <= 0),
            extents,
            shape: self,
        }
    }
}

/// Every natural coordinate of a shape, in colexicographic order. Built
/// by [`IntTuple::coordinates`].
#[derive(Debug)]
pub struct Coordinates<'a> {
    shape: &'a IntTuple,
    extents: Vec<Int>,
    digits: Vec<Int>,
    done: bool,
}

impl Iterator for Coordinates<'_> {
    type Item = IntTuple;

    fn next(&mut self) -> Option<IntTuple> {
        if self.done {
            return None;
        }
        let crd = HTuple::unflatten(&mut self.digits.iter().copied(), self.shape)?;
        // The leftmost digit runs fastest, and a carry out of the last
        // one ends the walk.
        self.done = self.digits.iter_mut().zip(&self.extents).all(|(digit, &extent)| {
            *digit += 1;
            let carry = *digit == extent;
            if carry {
                *digit = 0;
            }
            carry
        });
        Some(crd)
    }
}

/// A stride from nested parentheses. Each leaf converts through
/// [`From`], so integers and basis elements both fit.
///
/// ```
/// use weaverbird::{e, stride, HTuple, StrideScalar};
/// let flat = stride!((1, 4));
/// let basis = stride!((1, e!(0), e!(1)));
/// let nested = stride!((1, (4, 12)));
/// # let _ = (flat, basis, nested);
/// ```
#[macro_export]
macro_rules! stride {
    (( $($body:tt)* )) => {
        $crate::htuple::HTuple::Tuple($crate::stride_modes!([] [] $($body)*))
    };
    ($leaf:expr) => {
        $crate::htuple::HTuple::Leaf($crate::atuple::StrideScalar::from($leaf))
    };
}

/// Splits the body of a [`stride!`] tuple on commas, so a leaf may span
/// several tokens.
#[macro_export]
#[doc(hidden)]
macro_rules! stride_modes {
    // The body ran out. An unflushed leaf closes the list.
    ([$($done:expr),*] []) => { vec![$($done),*] };
    ([$($done:expr),*] [$($leaf:tt)+]) => { vec![$($done,)* $crate::stride!($($leaf)+)] };
    // A comma flushes the leaf it closes.
    ([$($done:expr),*] [$($leaf:tt)+] , $($rest:tt)*) => {
        $crate::stride_modes!([$($done,)* $crate::stride!($($leaf)+)] [] $($rest)*)
    };
    // Anything else joins the leaf under construction.
    ([$($done:expr),*] [$($leaf:tt)*] $head:tt $($rest:tt)*) => {
        $crate::stride_modes!([$($done),*] [$($leaf)* $head] $($rest)*)
    };
}
