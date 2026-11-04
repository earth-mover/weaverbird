//! `PyCuTe`'s accessor contract, and the tensor it builds.
//!
//! An [`Accessor`] maps an offset to a value; that is, it models random
//! access to a block of memory.
//!
//! - A [`Layout`] maps a coordinate to an offset;
//! - An [`Accessor`] maps that offset to a value.
//!
//! So the two stay independent: a layout can address plain memory,
//! a lazily-computed function, or a transformed view through an accessor.
//! A [`Tensor`] is the composition, coordinate to value.
//!
//! The transliteration stops at the contract. `PyCuTe` ships an
//! `ImplicitAccessor` and a `TransformAccessor`; a caller writes the
//! accessor its own storage needs.

use std::fmt::Debug;

use crate::atuple::{StrideScalar, e};
use crate::error::Result;
use crate::htuple::HTuple;
use crate::layout::{Layout, Tiler};
use crate::typedefs::{Int, IntTuple};

/// `PyCuTe`'s `Accessor` ABC: a read-only random-access handle.
///
/// Divergence from `PyCuTe`: `offset` and `get` return [`Result`], because
/// [`StrideScalar`] addition is partial. A non-zero integer and an arithmetic
/// tuple sit at different ranks, so their sum has no value; `PyCuTe`'s untyped
/// `+` would instead raise at that point.
pub trait Accessor: Clone + Debug + Send + Sync + 'static {
    /// What a read returns.
    type Value;

    /// `PyCuTe`'s `Accessor.__add__`: the same handle, advanced by `by`.
    /// It may refuse `by`; a [`Tensor`] that keeps its origin clones instead.
    fn offset(&self, by: &StrideScalar) -> Result<Self>;

    /// `PyCuTe`'s `Accessor.__getitem__`.
    fn get(&self, at: &StrideScalar) -> Result<Self::Value>;
}

/// `PyCuTe`'s `Tensor`: an [`Accessor`] composed with a [`Layout`].
/// The layout maps a coordinate to an offset; the accessor maps that
/// offset to a value.
#[derive(Debug)]
pub struct Tensor<A> {
    accessor: A,
    layout: Layout,
}

impl<A: Accessor> Tensor<A> {
    /// The `layout` over `accessor`, as the two are given.
    pub fn new(accessor: A, layout: Layout) -> Self {
        Tensor { accessor, layout }
    }

    /// `PyCuTe`'s `identity_tensor`, over any accessor: `(n0, n1, ..) : (E0, E1, ..)`.
    /// A size-1 axis keeps its basis stride, so a reader finds each base axis.
    pub fn identity(accessor: A, shape: &[usize]) -> Self {
        let modes = shape
            .iter()
            .enumerate()
            .map(|(k, &n)| Layout::from_parts(HTuple::Leaf(n as Int), HTuple::Leaf(e(&[k]))))
            .collect();
        Tensor { accessor, layout: Layout::from_modes(modes) }
    }

    /// `PyCuTe`'s `Tensor.accessor`, read through a method since the field is private here.
    pub fn accessor(&self) -> &A {
        &self.accessor
    }

    /// `PyCuTe`'s `Tensor.layout`, read through a method since the field is private here.
    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    /// `PyCuTe`'s `Tensor.shape`: one extent per top-level mode.
    /// A mode left nested by `compose` collapses to its size, so the result
    /// always has one entry per axis.
    pub fn shape(&self) -> Vec<usize> {
        self.layout.shape.size_each().leaves().map(|&n| n as usize).collect()
    }

    /// `PyCuTe`'s `Tensor.__getitem__`, rank-0 arm.
    pub fn get(&self, v: &[usize]) -> Result<A::Value> {
        let crd = IntTuple::Tuple(v.iter().map(|&i| HTuple::Leaf(i as Int)).collect());
        self.accessor.get(&self.layout.eval(&crd)?)
    }

    /// [`Layout::mode`], for `permute`/`squeeze`/`broadcast` to build on.
    /// Divergence from `PyCuTe`: `Tensor.get(mode=...)` returns a sub-tensor; this
    /// returns the raw sub-layout, which is all those three callers need.
    pub fn mode(&self, i: usize) -> Result<Layout> {
        self.layout.mode(i)
    }

    /// `PyCuTe`'s `accessor + offset`, lifted to the tensor.
    pub fn offset(&self, by: &StrideScalar) -> Result<Self> {
        Ok(Tensor { accessor: self.accessor.offset(by)?, layout: self.layout.clone() })
    }

    /// `PyCuTe`'s `Tensor._composition`.
    pub fn compose(&self, tiler: Tiler) -> Result<Self> {
        Ok(Tensor { accessor: self.accessor.clone(), layout: self.layout.compose(tiler)? })
    }

    /// No `PyCuTe` counterpart: axis reordering is the caller's vocabulary,
    /// built here from [`Layout::mode`] and [`Layout::from_modes`].
    /// `axes[i]` names the mode that becomes axis `i`.
    pub fn permute(&self, axes: &[usize]) -> Result<Self> {
        let modes = axes.iter().map(|&i| self.mode(i)).collect::<Result<Vec<_>>>()?;
        Ok(Tensor { accessor: self.accessor.clone(), layout: Layout::from_modes(modes) })
    }

    /// No `PyCuTe` counterpart: dropping an axis is the caller's vocabulary.
    /// Drops the named modes. Does not check their extent; a caller that
    /// must keep only size-1 axes checks that itself.
    pub fn squeeze(&self, axes: &[usize]) -> Result<Self> {
        let modes = (0..self.layout.shape.rank())
            .filter(|i| !axes.contains(i))
            .map(|i| self.mode(i))
            .collect::<Result<Vec<_>>>()?;
        Ok(Tensor { accessor: self.accessor.clone(), layout: Layout::from_modes(modes) })
    }

    /// No `PyCuTe` counterpart: broadcasting a new axis is the caller's vocabulary.
    /// Inserts a stride-0 axis of `extent` at `at`, clamped to the rank, so an
    /// index past the end appends.
    pub fn broadcast(&self, at: usize, extent: usize) -> Result<Self> {
        let mut modes =
            (0..self.layout.shape.rank()).map(|i| self.mode(i)).collect::<Result<Vec<_>>>()?;
        modes.insert(
            at.min(modes.len()),
            Layout::from_parts(HTuple::Leaf(extent as Int), HTuple::Leaf(StrideScalar::Int(0))),
        );
        Ok(Tensor { accessor: self.accessor.clone(), layout: Layout::from_modes(modes) })
    }
}

impl<A: Accessor> Clone for Tensor<A> {
    /// Hand-written: a derive would bound `A: Clone` again, which [`Accessor`] already gives.
    fn clone(&self) -> Self {
        Tensor { accessor: self.accessor.clone(), layout: self.layout.clone() }
    }
}

#[cfg(test)]
#[path = "tests/tensor.rs"]
mod tests;
