//! Hierarchical tuples. Mirrors `pycute/htuple.py`.
//!
//! A hierarchical tuple is a leaf, or a tuple of hierarchical tuples.
//! PyCuTe carries both as plain Python values and tells them apart with
//! `is_tuple`. Rust needs a type, so the two cases become variants of
//! [`HTuple`].

use std::fmt::{self, Debug};

use crate::{
    error::{Error, Result},
    typedefs::{Int, IntTuple},
};

/// A leaf, or a tuple of hierarchical tuples.
#[derive(Clone, PartialEq, Eq, Hash)]
pub enum HTuple<T> {
    Leaf(T),
    Tuple(Vec<HTuple<T>>),
}

/// Prints as PyCuTe prints: a leaf by its own `Debug`, a tuple as
/// `(a,b,c)`.
impl<T: Debug> Debug for HTuple<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HTuple::Leaf(v) => Debug::fmt(v, f),
            HTuple::Tuple(modes) => {
                write!(f, "(")?;
                for (i, m) in modes.iter().enumerate() {
                    if i > 0 {
                        write!(f, ",")?;
                    }
                    Debug::fmt(m, f)?;
                }
                write!(f, ")")
            }
        }
    }
}

impl<T> HTuple<T> {
    /// True for a tuple, false for a leaf. PyCuTe's `is_tuple`.
    pub fn is_tuple(&self) -> bool {
        matches!(self, HTuple::Tuple(_))
    }

    /// The top-level element count. A leaf counts as one.
    pub fn rank(&self) -> usize {
        match self {
            HTuple::Leaf(_) => 1,
            HTuple::Tuple(modes) => modes.len(),
        }
    }

    /// The sub-tuple at `path`. An empty path returns the whole tuple.
    /// Each index steps into one level.
    ///
    /// PyCuTe's `get`.
    pub fn get(&self, path: &[usize]) -> Option<&HTuple<T>> {
        match path.split_first() {
            None => Some(self),
            Some((&i, rest)) => match self {
                HTuple::Tuple(modes) => modes.get(i)?.get(rest),
                HTuple::Leaf(_) => None,
            },
        }
    }

    /// Every leaf, in pre-order.
    pub fn leaves(&self) -> Vec<&T> {
        match self {
            HTuple::Leaf(v) => vec![v],
            HTuple::Tuple(modes) => modes.iter().flat_map(HTuple::leaves).collect(),
        }
    }

    /// Every root-to-leaf path, in pre-order. The path inverse of
    /// [`Self::get`].
    pub fn leaf_paths(&self) -> Vec<Vec<usize>> {
        match self {
            HTuple::Leaf(_) => vec![vec![]],
            HTuple::Tuple(modes) => modes
                .iter()
                .enumerate()
                .flat_map(|(i, m)| {
                    m.leaf_paths().into_iter().map(move |mut p| {
                        p.insert(0, i);
                        p
                    })
                })
                .collect(),
        }
    }

    /// The first top-level element, or the leaf itself.
    pub fn front(&self) -> &HTuple<T> {
        match self {
            HTuple::Tuple(modes) if !modes.is_empty() => &modes[0],
            other => other,
        }
    }

    /// The last top-level element, or the leaf itself.
    pub fn back(&self) -> &HTuple<T> {
        match self {
            HTuple::Tuple(modes) if !modes.is_empty() => &modes[modes.len() - 1],
            other => other,
        }
    }

    /// The tuple with its first element replaced. A leaf becomes `value`.
    pub fn replace_front(self, value: HTuple<T>) -> HTuple<T> {
        match self {
            HTuple::Tuple(mut modes) if !modes.is_empty() => {
                modes[0] = value;
                HTuple::Tuple(modes)
            }
            _ => value,
        }
    }

    /// The tuple with its last element replaced. A leaf becomes `value`.
    pub fn replace_back(self, value: HTuple<T>) -> HTuple<T> {
        match self {
            HTuple::Tuple(mut modes) if !modes.is_empty() => {
                let last = modes.len() - 1;
                modes[last] = value;
                HTuple::Tuple(modes)
            }
            _ => value,
        }
    }

    /// The tuple form. A leaf becomes a one-element tuple.
    pub fn wrap(self) -> HTuple<T> {
        match self {
            HTuple::Leaf(_) => HTuple::Tuple(vec![self]),
            tuple => tuple,
        }
    }

    /// Strips one-element tuples, as deep as they go.
    pub fn unwrap(&self) -> &HTuple<T> {
        match self {
            HTuple::Tuple(modes) if modes.len() == 1 => modes[0].unwrap(),
            other => other,
        }
    }

    /// The tuple of every leaf, one level deep.
    pub fn flatten(&self) -> HTuple<T>
    where
        T: Clone,
    {
        HTuple::Tuple(
            self.leaves()
                .into_iter()
                .cloned()
                .map(HTuple::Leaf)
                .collect(),
        )
    }

    /// Applies `f` to every leaf and keeps the profile.
    pub fn transform_leaf<U>(&self, f: &impl Fn(&T) -> U) -> HTuple<U> {
        match self {
            HTuple::Leaf(v) => HTuple::Leaf(f(v)),
            HTuple::Tuple(modes) => {
                HTuple::Tuple(modes.iter().map(|m| m.transform_leaf(f)).collect())
            }
        }
    }

    /// Draws one value per leaf of `profile` and builds a tuple with
    /// `profile`'s shape. The inverse of [`Self::flatten`].
    ///
    /// Returns `None` when `values` runs dry.
    pub fn unflatten<P>(values: &mut impl Iterator<Item = T>, profile: &HTuple<P>) -> Option<Self> {
        match profile {
            HTuple::Leaf(_) => values.next().map(HTuple::Leaf),
            HTuple::Tuple(modes) => modes
                .iter()
                .map(|m| HTuple::unflatten(values, m))
                .collect::<Option<Vec<_>>>()
                .map(HTuple::Tuple),
        }
    }

    /// A tuple with `profile`'s shape and `value` at every leaf.
    pub fn repeat_like<P>(value: &T, profile: &HTuple<P>) -> Self
    where
        T: Clone,
    {
        profile.transform_leaf(&|_| value.clone())
    }

    /// The tuple of the sub-tuples at each of `modes`.
    ///
    /// Returns [`Error::BadPath`] when a mode does not address `self`.
    pub fn select(&self, modes: &[usize]) -> Result<HTuple<T>>
    where
        T: Clone + Debug,
    {
        modes
            .iter()
            .map(|&i| {
                self.get(&[i]).cloned().ok_or_else(|| Error::BadPath {
                    path: vec![i],
                    value: format!("{self:?}"),
                })
            })
            .collect::<Result<Vec<_>>>()
            .map(HTuple::Tuple)
    }

    /// The tuple of the sub-tuples in `begin..end`.
    ///
    /// Returns [`Error::ReversedRange`] when `begin` exceeds `end`.
    pub fn take(&self, begin: usize, end: usize) -> Result<HTuple<T>>
    where
        T: Clone + Debug,
    {
        if begin > end {
            return Err(Error::ReversedRange { begin, end });
        }
        self.select(&(begin..end).collect::<Vec<_>>())
    }
}

impl HTuple<Int> {
    /// The product of every leaf. A shape's domain size.
    pub fn product(&self) -> Int {
        self.leaves().into_iter().product()
    }

    /// The product of each top-level mode, keeping the rank.
    pub fn product_each(&self) -> HTuple<Int> {
        match self {
            HTuple::Leaf(v) => HTuple::Leaf(*v),
            HTuple::Tuple(modes) => {
                HTuple::Tuple(modes.iter().map(|m| HTuple::Leaf(m.product())).collect())
            }
        }
    }

    /// A tuple that holds `value` at `path` and zero everywhere else.
    ///
    /// The inverse of [`Self::get`]: `lift(x, p).get(p) == x`.
    pub fn lift(value: Int, path: &[usize]) -> IntTuple {
        path.iter().rev().fold(HTuple::Leaf(value), |acc, &i| {
            let mut modes = vec![HTuple::Leaf(0); i];
            modes.push(acc);
            HTuple::Tuple(modes)
        })
    }
}

/// True when `a` and `b` have the same profile. Leaf values do not
/// matter; only the shape of the tree does.
pub fn congruent<A, B>(a: &HTuple<A>, b: &HTuple<B>) -> bool {
    match (a, b) {
        (HTuple::Leaf(_), HTuple::Leaf(_)) => true,
        (HTuple::Tuple(x), HTuple::Tuple(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(i, j)| congruent(i, j))
        }
        _ => false,
    }
}

/// True when `a` coarsens `b`: every leaf of `a` may fold a sub-tree of
/// `b`. A leaf coarsens anything. A tuple never coarsens a leaf.
pub fn weakly_congruent<A, B>(a: &HTuple<A>, b: &HTuple<B>) -> bool {
    match (a, b) {
        (HTuple::Leaf(_), _) => true,
        (HTuple::Tuple(_), HTuple::Leaf(_)) => false,
        (HTuple::Tuple(x), HTuple::Tuple(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(i, j)| weakly_congruent(i, j))
        }
    }
}

/// Applies `f` to each pair of leaves and keeps the profile.
///
/// Returns [`Error::BadPath`] when the two profiles differ. PyCuTe pads
/// with `None` instead; no caller needs that, so this one is strict.
pub fn zip_transform_leaf<A, B, U>(
    f: &impl Fn(&A, &B) -> U,
    a: &HTuple<A>,
    b: &HTuple<B>,
) -> Result<HTuple<U>>
where
    A: Debug,
{
    match (a, b) {
        (HTuple::Leaf(x), HTuple::Leaf(y)) => Ok(HTuple::Leaf(f(x, y))),
        (HTuple::Tuple(x), HTuple::Tuple(y)) if x.len() == y.len() => x
            .iter()
            .zip(y)
            .map(|(i, j)| zip_transform_leaf(f, i, j))
            .collect::<Result<Vec<_>>>()
            .map(HTuple::Tuple),
        _ => Err(Error::BadPath {
            path: vec![],
            value: format!("{a:?}"),
        }),
    }
}

/// Builds an [`HTuple`] from nested parentheses.
///
/// ```
/// use pinstripe::{ht, HTuple};
/// let flat = ht!((2, 3));
/// let nested = ht!(((5, 4), 3));
/// let leaf = ht!(4);
/// # let _ = (flat, nested, leaf);
/// ```
#[macro_export]
macro_rules! ht {
    (( $($inner:tt),+ $(,)? )) => {
        $crate::htuple::HTuple::Tuple(vec![ $($crate::ht!($inner)),+ ])
    };
    ($value:expr) => {
        $crate::htuple::HTuple::Leaf($value)
    };
}
