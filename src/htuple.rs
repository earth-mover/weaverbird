//! Hierarchical tuples.
//!
//! A hierarchical tuple is a leaf, or a tuple of hierarchical tuples.
//! Shapes, strides, coordinates and profiles all use this one carrier.

use std::fmt::{self, Debug, Display};

use crate::error::{Error, Result};

/// A leaf, or a tuple of hierarchical tuples.
#[derive(Clone, PartialEq, Eq, Hash)]
pub enum HTuple<T> {
    Leaf(T),
    Tuple(Vec<HTuple<T>>),
}

/// CuTe notation: a leaf by its own `Display`, and a tuple as `(a,b,c)`.
impl<T: Display> Display for HTuple<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_modes(self, f, &Display::fmt)
    }
}

/// The same notation as [`Display`], for the leaf types that print one
/// way only.
impl<T: Debug> Debug for HTuple<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_modes(self, f, &Debug::fmt)
    }
}

/// Writes a leaf through `leaf`, and a tuple as `(a,b,c)`.
fn write_modes<T>(
    x: &HTuple<T>,
    f: &mut fmt::Formatter<'_>,
    leaf: &impl Fn(&T, &mut fmt::Formatter<'_>) -> fmt::Result,
) -> fmt::Result {
    match x {
        HTuple::Leaf(v) => leaf(v, f),
        HTuple::Tuple(modes) => {
            write!(f, "(")?;
            for (i, m) in modes.iter().enumerate() {
                if i > 0 {
                    write!(f, ",")?;
                }
                write_modes(m, f, leaf)?;
            }
            write!(f, ")")
        }
    }
}

/// Collects modes into a tuple.
impl<T> FromIterator<HTuple<T>> for HTuple<T> {
    fn from_iter<I: IntoIterator<Item = HTuple<T>>>(modes: I) -> Self {
        HTuple::Tuple(modes.into_iter().collect())
    }
}

impl<T> HTuple<T> {
    /// True for a tuple, false for a leaf.
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
    pub fn get(&self, path: &[usize]) -> Option<&HTuple<T>> {
        match path.split_first() {
            None => Some(self),
            Some((&i, rest)) => match self {
                HTuple::Tuple(modes) => modes.get(i)?.get(rest),
                HTuple::Leaf(_) => None,
            },
        }
    }

    /// The sub-tuple at `path`, for writing through.
    /// The mutable twin of [`Self::get`].
    pub fn get_mut(&mut self, path: &[usize]) -> Option<&mut HTuple<T>> {
        match path.split_first() {
            None => Some(self),
            Some((&i, rest)) => match self {
                HTuple::Tuple(modes) => modes.get_mut(i)?.get_mut(rest),
                HTuple::Leaf(_) => None,
            },
        }
    }

    /// The sub-tuple at `path`, for writing through, or the error that
    /// names it.
    ///
    /// [`Self::get_mut`] cannot report the error itself: the message
    /// renders `self`, which the mutable borrow already holds. The
    /// shared walk therefore decides first, so only the failing call
    /// pays for the formatting.
    pub fn try_get_mut(&mut self, path: &[usize]) -> Result<&mut HTuple<T>>
    where
        T: Debug,
    {
        match self.get(path) {
            None => Err(Error::BadPath { path: path.to_vec(), value: format!("{self:?}") }),
            #[expect(
                clippy::expect_used,
                reason = "the shared walk above reached the node, so the mutable walk does too"
            )]
            Some(_) => Ok(self.get_mut(path).expect("get reached the node, so get_mut reaches it")),
        }
    }

    /// Every leaf, in pre-order.
    pub fn leaves(&self) -> Leaves<'_, T> {
        Leaves { stack: vec![self] }
    }

    /// Every root-to-leaf path, in pre-order. The path inverse of
    /// [`Self::get`].
    pub fn leaf_paths(&self) -> Vec<Vec<usize>> {
        fn walk<T>(x: &HTuple<T>, prefix: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
            match x {
                HTuple::Leaf(_) => out.push(prefix.clone()),
                HTuple::Tuple(modes) => modes.iter().enumerate().for_each(|(i, m)| {
                    prefix.push(i);
                    walk(m, prefix, out);
                    prefix.pop();
                }),
            }
        }
        let mut out = Vec::new();
        walk(self, &mut Vec::new(), &mut out);
        out
    }

    /// The first top-level element, or the leaf itself.
    pub fn front(&self) -> &HTuple<T> {
        match self {
            HTuple::Tuple(modes) => modes.first().unwrap_or(self),
            leaf => leaf,
        }
    }

    /// The last top-level element, or the leaf itself.
    pub fn back(&self) -> &HTuple<T> {
        match self {
            HTuple::Tuple(modes) => modes.last().unwrap_or(self),
            leaf => leaf,
        }
    }

    /// The tuple with its first element replaced. A leaf becomes `value`.
    pub fn replace_front(self, value: HTuple<T>) -> HTuple<T> {
        self.replace_at(0, value)
    }

    /// The tuple with its last element replaced. A leaf becomes `value`.
    pub fn replace_back(self, value: HTuple<T>) -> HTuple<T> {
        let last = self.rank().saturating_sub(1);
        self.replace_at(last, value)
    }

    /// The tuple with mode `i` replaced. A leaf becomes `value`.
    fn replace_at(self, i: usize, value: HTuple<T>) -> HTuple<T> {
        match self {
            HTuple::Tuple(mut modes) if i < modes.len() => {
                modes[i] = value;
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
    pub fn strip_singletons(&self) -> &HTuple<T> {
        match self {
            HTuple::Tuple(modes) if modes.len() == 1 => modes[0].strip_singletons(),
            other => other,
        }
    }

    /// The tuple of every leaf, one level deep.
    pub fn flatten(&self) -> HTuple<T>
    where
        T: Clone,
    {
        self.leaves().cloned().map(HTuple::Leaf).collect()
    }

    /// Applies `f` to every leaf and keeps the profile.
    pub fn transform_leaf<U>(&self, f: &impl Fn(&T) -> U) -> HTuple<U> {
        match self {
            HTuple::Leaf(v) => HTuple::Leaf(f(v)),
            HTuple::Tuple(modes) => modes.iter().map(|m| m.transform_leaf(f)).collect(),
        }
    }

    /// Applies a fallible `f` to every leaf and keeps the profile.
    pub fn try_transform_leaf<U>(&self, f: &impl Fn(&T) -> Result<U>) -> Result<HTuple<U>> {
        match self {
            HTuple::Leaf(v) => f(v).map(HTuple::Leaf),
            HTuple::Tuple(modes) => modes.iter().map(|m| m.try_transform_leaf(f)).collect(),
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
                self.get(&[i])
                    .cloned()
                    .ok_or_else(|| Error::BadPath { path: vec![i], value: format!("{self:?}") })
            })
            .collect()
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

    /// True when `self` and `other` have the same profile. Leaf values do
    /// not matter; only the shape of the tree does.
    pub fn congruent<U>(&self, other: &HTuple<U>) -> bool {
        match (self, other) {
            (HTuple::Leaf(_), HTuple::Leaf(_)) => true,
            (HTuple::Tuple(x), HTuple::Tuple(y)) => {
                x.len() == y.len() && x.iter().zip(y).all(|(i, j)| i.congruent(j))
            }
            _ => false,
        }
    }

    /// True when `self` coarsens `other`: every leaf of `self` may fold a
    /// sub-tree of `other`. A leaf coarsens anything. A tuple never
    /// coarsens a leaf.
    pub fn weakly_congruent<U>(&self, other: &HTuple<U>) -> bool {
        match (self, other) {
            (HTuple::Leaf(_), _) => true,
            (HTuple::Tuple(_), HTuple::Leaf(_)) => false,
            (HTuple::Tuple(x), HTuple::Tuple(y)) => {
                x.len() == y.len() && x.iter().zip(y).all(|(i, j)| i.weakly_congruent(j))
            }
        }
    }

    /// Applies `f` to each pair of leaves and keeps the profile.
    ///
    /// Returns [`Error::BadPath`] when the two profiles differ.
    pub fn zip_transform<U, V>(
        &self,
        other: &HTuple<U>,
        f: &impl Fn(&T, &U) -> V,
    ) -> Result<HTuple<V>>
    where
        T: Debug,
    {
        match (self, other) {
            (HTuple::Leaf(x), HTuple::Leaf(y)) => Ok(HTuple::Leaf(f(x, y))),
            (HTuple::Tuple(x), HTuple::Tuple(y)) if x.len() == y.len() => {
                x.iter().zip(y).map(|(i, j)| i.zip_transform(j, f)).collect()
            }
            _ => Err(self.bad_path()),
        }
    }

    /// Pairs each leaf of `self` with whatever stands opposite it in
    /// `other`, in pre-order. A leaf of `self` may face a whole sub-tuple.
    ///
    /// Returns [`Error::BadPath`] when `self` does not coarsen `other`.
    pub fn zip_leaves<'a, U>(&'a self, other: &'a HTuple<U>) -> Result<Vec<(&'a T, &'a HTuple<U>)>>
    where
        T: Debug,
    {
        match (self, other) {
            (HTuple::Leaf(x), _) => Ok(vec![(x, other)]),
            (HTuple::Tuple(x), HTuple::Tuple(y)) if x.len() == y.len() => x
                .iter()
                .zip(y)
                .map(|(i, j)| i.zip_leaves(j))
                .collect::<Result<Vec<_>>>()
                .map(|nested| nested.concat()),
            _ => Err(self.bad_path()),
        }
    }

    /// Folds `f` over the pairs [`Self::zip_leaves`] yields, left to
    /// right.
    pub fn fold_leaves<U, V>(
        &self,
        other: &HTuple<U>,
        init: V,
        f: &impl Fn(V, &T, &HTuple<U>) -> V,
    ) -> Result<V>
    where
        T: Debug,
    {
        self.zip_leaves(other).map(|pairs| pairs.into_iter().fold(init, |acc, (x, y)| f(acc, x, y)))
    }

    /// The error a profile mismatch raises.
    fn bad_path(&self) -> Error
    where
        T: Debug,
    {
        Error::BadPath { path: vec![], value: format!("{self:?}") }
    }
}

impl<P: Debug> HTuple<Option<P>> {
    /// The parts of `other` that `self` leaves open, flattened one level.
    /// An absent leaf keeps its counterpart.
    ///
    /// A coordinate marks the modes a slice retains this way, so the
    /// profile is a tuple of optional values.
    pub fn slice<T: Clone>(&self, other: &HTuple<T>) -> Result<HTuple<T>> {
        self.filter_leaves(other, Option::is_none)
    }

    /// The parts of `other` that `self` pins down, flattened one level.
    /// The complement of [`Self::slice`].
    pub fn dice<T: Clone>(&self, other: &HTuple<T>) -> Result<HTuple<T>> {
        self.filter_leaves(other, Option::is_some)
    }

    /// The counterparts of the leaves that `keep` admits.
    fn filter_leaves<T: Clone>(
        &self,
        other: &HTuple<T>,
        keep: impl Fn(&Option<P>) -> bool,
    ) -> Result<HTuple<T>> {
        self.zip_leaves(other).map(|pairs| {
            pairs.into_iter().filter(|(mark, _)| keep(mark)).map(|(_, x)| x.clone()).collect()
        })
    }
}

/// Every leaf of an [`HTuple`], in pre-order. Built by
/// [`HTuple::leaves`].
#[derive(Debug)]
pub struct Leaves<'a, T> {
    stack: Vec<&'a HTuple<T>>,
}

impl<'a, T> Iterator for Leaves<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        loop {
            match self.stack.pop()? {
                HTuple::Leaf(v) => return Some(v),
                HTuple::Tuple(modes) => self.stack.extend(modes.iter().rev()),
            }
        }
    }
}

/// Builds an [`HTuple`] from nested parentheses. A leaf is any
/// expression.
///
/// ```
/// use weaverbird::{ht, HTuple};
/// let flat = ht!((2, 3));
/// let nested = ht!(((5, 4), 3));
/// let leaf = ht!(4);
/// let computed = ht!((2 * 3, 4));
/// # let _ = (flat, nested, leaf, computed);
/// ```
#[macro_export]
macro_rules! ht {
    (( $($body:tt)* )) => {
        $crate::htuple::HTuple::Tuple($crate::ht_modes!([] [] $($body)*))
    };
    ($leaf:expr) => {
        $crate::htuple::HTuple::Leaf($leaf)
    };
}

/// Splits the body of an [`ht!`] tuple on commas, so a leaf may span
/// several tokens.
#[macro_export]
#[doc(hidden)]
macro_rules! ht_modes {
    // The body ran out. An unflushed leaf closes the list.
    ([$($done:expr),*] []) => { vec![$($done),*] };
    ([$($done:expr),*] [$($leaf:tt)+]) => { vec![$($done,)* $crate::ht!($($leaf)+)] };
    // A comma flushes the leaf it closes.
    ([$($done:expr),*] [$($leaf:tt)+] , $($rest:tt)*) => {
        $crate::ht_modes!([$($done,)* $crate::ht!($($leaf)+)] [] $($rest)*)
    };
    // Anything else joins the leaf under construction.
    ([$($done:expr),*] [$($leaf:tt)*] $head:tt $($rest:tt)*) => {
        $crate::ht_modes!([$($done),*] [$($leaf)* $head] $($rest)*)
    };
}

#[cfg(test)]
#[path = "tests/htuple.rs"]
mod tests;
