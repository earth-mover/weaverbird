//! Functions for CuTe shapes. Mirrors `pycute/shape.py`.
//!
//! A shape is an [`IntTuple`] of extents. PyCuTe reads a shape off any
//! object that carries one, and falls back to the object itself; Rust
//! has no such duck typing, so these functions take the [`IntTuple`].
//! The `.shape` lookup arrives with `Layout`.
//!
//! PyCuTe's `@ModeOpDecorator` gives each accessor an optional mode
//! path, so that `size[1](x)` reads mode 1. Here that is an explicit
//! `mode: &[usize]` argument, spelled the way [`HTuple::get`] spells it.
//!
use crate::{
    atuple::StrideScalar,
    error::{Error, Result},
    htuple::{HTuple, transform_apply_leaf},
    stride::{inner_product, prefix_product},
    typedefs::{Int, IntTuple, Stride},
};

/// An object's shape. PyCuTe's `shape`.
///
/// An [`IntTuple`] is its own shape, so this is the mode projection
/// alone.
///
/// Returns [`Error::BadPath`] when `mode` does not address `obj`.
pub fn shape<'a>(obj: &'a IntTuple, mode: &[usize]) -> Result<&'a IntTuple> {
    obj.get(mode).ok_or_else(|| Error::BadPath {
        path: mode.to_vec(),
        value: format!("{obj:?}"),
    })
}

/// An object's size: the product of its extents. PyCuTe's `size`.
pub fn size(obj: &IntTuple, mode: &[usize]) -> Result<Int> {
    shape(obj, mode).map(HTuple::product)
}

/// An object's rank: its top-level element count. A leaf has rank one.
/// PyCuTe's `rank`.
pub fn rank(obj: &IntTuple, mode: &[usize]) -> Result<usize> {
    shape(obj, mode).map(HTuple::rank)
}

/// An object's depth: the longest path from the root to a leaf. A leaf
/// has depth zero. PyCuTe's `depth`.
///
/// PyCuTe raises on an empty tuple, whose `reduce(max, ())` has no
/// seed. Here an empty tuple has depth one.
pub fn depth(obj: &IntTuple, mode: &[usize]) -> Result<usize> {
    fn walk(s: &IntTuple) -> usize {
        match s {
            HTuple::Leaf(_) => 0,
            HTuple::Tuple(modes) => 1 + modes.iter().map(walk).max().unwrap_or(0),
        }
    }
    shape(obj, mode).map(walk)
}

/// True when `a` *coarsens* `b`. PyCuTe's `compatible`.
///
/// Compatibility, `a ≼ b`, is a partial order on shapes. It strengthens
/// weak congruence by also requiring the sizes to agree. Equivalently,
/// every coordinate of `a` is a coordinate of `b`.
///
/// ```text
/// compatible(30, (2, 15))              == true
/// compatible((2, 15), (2, (3, 5)))     == true
/// compatible(24, 32)                   == false   // size mismatch
/// compatible((2, (3, 5)), ((3, 2), 5)) == false   // same size, still incompatible
/// compatible(24, (24,))                == true    // int ≼ (int,)
/// compatible((24,), 24)                == false   // but not the reverse
/// ```
pub fn compatible(a: &IntTuple, b: &IntTuple) -> bool {
    match (a, b) {
        (HTuple::Tuple(x), HTuple::Tuple(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(i, j)| compatible(i, j))
        }
        (HTuple::Leaf(v), _) => *v == b.product(),
        (HTuple::Tuple(_), HTuple::Leaf(_)) => false,
    }
}

/// The minimal shape that *refines* both `a` and `b` — their join under
/// the compatibility order. PyCuTe's `common_refinement`.
///
/// ```text
/// common_refinement(30, (2, 15))               == (2, 15)
/// common_refinement(10, (10,))                 == (10,)
/// common_refinement(((2, 3), 20), (6, (4, 5))) == ((2, 3), (4, 5))
/// ```
///
/// Returns [`Error::NoRefinement`] when no such shape exists.
pub fn common_refinement(a: &IntTuple, b: &IntTuple) -> Result<IntTuple> {
    let no_refinement = || Error::NoRefinement {
        lhs: format!("{a:?}"),
        rhs: format!("{b:?}"),
    };
    match (a, b) {
        (HTuple::Tuple(x), HTuple::Tuple(y)) if x.len() == y.len() => x
            .iter()
            .zip(y)
            .map(|(ai, bi)| common_refinement(ai, bi))
            .collect::<Result<Vec<_>>>()
            .map(HTuple::Tuple),
        (HTuple::Leaf(x), HTuple::Leaf(y)) if x == y => Ok(a.clone()),
        (HTuple::Leaf(x), HTuple::Tuple(_)) if *x == b.product() => Ok(b.clone()),
        (HTuple::Tuple(_), HTuple::Leaf(y)) if a.product() == *y => Ok(a.clone()),
        _ => Err(no_refinement()),
    }
}

/// The maximal shape that *coarsens* both `a` and `b` — their meet
/// under the compatibility order. PyCuTe's `common_coarsening`.
///
/// A meet exists exactly when the two sizes agree; in the worst case it
/// is the integer size itself.
///
/// ```text
/// common_coarsening((2, 15), (2, (3, 5)))      == (2, 15)
/// common_coarsening((4, (3, 5)), ((2, 2), 15)) == (4, 15)
/// common_coarsening((6, 5), (2, 15))           == 30   // mode 0 mismatch -> int
/// common_coarsening((2, 3), (2, 3, 1))         == 6    // rank mismatch -> int
/// ```
///
/// Returns [`Error::NoCoarsening`] when the sizes differ.
pub fn common_coarsening(a: &IntTuple, b: &IntTuple) -> Result<IntTuple> {
    // A per-mode meet, when both sides are tuples of the same rank and
    // every mode meets. PyCuTe swallows the `ValueError` and falls
    // through to the integer meet below.
    let per_mode = match (a, b) {
        (HTuple::Tuple(x), HTuple::Tuple(y)) if x.len() == y.len() => x
            .iter()
            .zip(y)
            .map(|(ai, bi)| common_coarsening(ai, bi))
            .collect::<Result<Vec<_>>>()
            .map(HTuple::Tuple)
            .ok(),
        _ => None,
    };
    match (per_mode, a.product(), b.product()) {
        (Some(c), _, _) => Ok(c),
        (None, sa, sb) if sa == sb => Ok(HTuple::Leaf(sa)),
        _ => Err(Error::NoCoarsening {
            lhs: format!("{a:?}"),
            rhs: format!("{b:?}"),
        }),
    }
}

/// Maps any coordinate to a *natural* coordinate of `shape`. PyCuTe's
/// `idx2crd`.
///
/// The index is decomposed in colexicographical order, so the leftmost
/// mode varies fastest. The final mode keeps the whole quotient — its
/// `mod` is skipped — so an out-of-bounds index does not wrap; the
/// excess accumulates in the last leaf.
///
/// ```text
/// idx2crd(7,  14)          == 7
/// idx2crd(7,  (3, 2, 4))   == (1, 0, 1)
/// idx2crd(7,  (3, (2, 4))) == (1, (0, 1))
/// idx2crd(42, (3, 7, 2))   == (0, 0, 2)   // the last leaf absorbs the excess
/// ```
///
/// PyCuTe also accepts `None`, and answers zeros shaped like `shape`.
/// An [`IntTuple`] carries no `None`, so that case is absent; a caller
/// that wants it writes `HTuple::repeat_like(&0, shape)`.
///
/// Returns [`Error::BadCoord`] when `idx` does not index `shape`.
pub fn idx2crd(idx: &IntTuple, shape: &IntTuple) -> Result<IntTuple> {
    let bad_coord = || Error::BadCoord {
        idx: format!("{idx:?}"),
        shape: format!("{shape:?}"),
    };
    match (idx, shape) {
        (HTuple::Leaf(i), HTuple::Leaf(_)) => Ok(HTuple::Leaf(*i)),
        (HTuple::Tuple(is), HTuple::Tuple(ss)) if is.len() == ss.len() => is
            .iter()
            .zip(ss)
            .map(|(i, s)| idx2crd(i, s))
            .collect::<Result<Vec<_>>>()
            .map(HTuple::Tuple),
        (HTuple::Leaf(i), HTuple::Tuple(_)) => {
            let extents = shape.leaves();
            // Every extent but the last takes a `divmod`; the last one
            // keeps the quotient whole.
            let head = extents.split_last().map_or(&[][..], |(_, head)| head);
            let mut quotient = *i;
            let mut digits = head
                .iter()
                .map(|&&s| {
                    let remainder = quotient.rem_euclid(s);
                    quotient = quotient.div_euclid(s);
                    remainder
                })
                .collect::<Vec<_>>();
            digits.push(quotient);
            HTuple::unflatten(&mut digits.into_iter(), shape).ok_or_else(bad_coord)
        }
        _ => Err(bad_coord()),
    }
}

/// Maps any coordinate of `shape` to an integral coordinate. PyCuTe's
/// `crd2idx`.
///
/// The recomposition is colexicographic, so the leftmost mode varies
/// fastest. It is the inverse of [`idx2crd`] on in-bounds input.
///
/// ```text
/// crd2idx((1, 0, 1),   (3, 2, 4))   == 7
/// crd2idx((1, (0, 1)), (3, (2, 4))) == 7
/// crd2idx(7,           (3, (2, 4))) == 7   // integral, passes through
/// crd2idx((2, 5),      (3, (2, 3))) == 17  // flat coord, nested shape
/// ```
///
/// `crd` need only weakly coarsen `shape`: a leaf of `crd` may stand for
/// a whole sub-tree of `shape`, and that sub-tree contributes its
/// [`size`]. That is what admits the flat coordinate above.
///
/// PyCuTe types the result `Integer`. Here it is a [`StrideScalar`],
/// which is what [`inner_product`] returns; the strides are the plain
/// integers of a [`prefix_product`], so the value is always the
/// [`StrideScalar::Int`] case.
///
/// Returns [`Error::BadCoord`] when `crd` does not coarsen `shape`.
pub fn crd2idx(crd: &IntTuple, shape: &IntTuple) -> Result<StrideScalar> {
    let bad_coord = || Error::BadCoord {
        idx: format!("{crd:?}"),
        shape: format!("{shape:?}"),
    };
    // One extent per leaf of `crd`: the size of the sub-shape that leaf
    // stands for. PyCuTe writes it `transform_leaf(lambda c,s: size(s))`.
    let extents = transform_apply_leaf(
        &HTuple::Tuple,
        &|_: Option<&IntTuple>, sub: Option<&IntTuple>| {
            sub.ok_or_else(bad_coord)
                .and_then(|s| size(s, &[]))
                .map(HTuple::Leaf)
        },
        Some(crd),
        Some(shape),
    )?;
    inner_product(crd, &prefix_product(&extents, &Stride::Leaf(1.into()))?)
}

/// Every natural coordinate of `shape`, in colexicographical order.
/// PyCuTe's `coordinates`.
///
/// ```text
/// coordinates(6)          == [0, 1, 2, 3, 4, 5]
/// coordinates((3, 2))     == [(0,0), (1,0), (2,0), (0,1), (1,1), (2,1)]
/// ```
///
/// PyCuTe's generator raises a `TypeError` once it runs dry — the
/// `raise` at the end of the function body sits after the `yield`s.
/// This one just ends.
pub fn coordinates(shape: &IntTuple) -> Vec<IntTuple> {
    match shape {
        HTuple::Leaf(n) => (0..*n).map(HTuple::Leaf).collect(),
        HTuple::Tuple(modes) => match modes.split_first() {
            None => vec![HTuple::Tuple(vec![])],
            Some((first, rest)) => {
                let heads = coordinates(first);
                coordinates(&HTuple::Tuple(rest.to_vec()))
                    .into_iter()
                    .flat_map(|tail| {
                        let tail_modes = match tail {
                            HTuple::Tuple(m) => m,
                            leaf => vec![leaf],
                        };
                        heads.iter().map(move |head| {
                            HTuple::Tuple(
                                std::iter::once(head.clone())
                                    .chain(tail_modes.iter().cloned())
                                    .collect(),
                            )
                        })
                    })
                    .collect()
            }
        },
    }
}
