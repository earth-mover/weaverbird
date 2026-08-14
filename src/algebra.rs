//! Generic algebraic operations over layouts. Mirrors
//! `pycute/algebra.py`.
//!
//! PyCuTe's `algebra.py` is a dispatch facade: each function looks for a
//! private method on its argument and promotes an int or a tuple through
//! `tiler_to_layout`. Only two of its functions carry an algorithm of
//! their own, and those two are here — [`layout_add`] and
//! [`greatest_common_domain`]. The rest of the facade lands with the
//! methods it dispatches to.

use crate::{
    atuple::StrideScalar,
    error::{Error, Result},
    htuple::HTuple,
    layout::{Layout, Profile},
    shape::size,
    typedefs::{Int, IntTuple, Stride},
};

/// The profile PyCuTe spells as the default `profile=1`: coalesce every
/// mode.
fn every_mode() -> Profile {
    HTuple::Leaf(Some(1))
}

/// Adds two layouts coordinate-wise. PyCuTe's `layout_add`.
///
/// Given layouts `A` and `B` with `size(A) == size(B)`, returns a layout
/// `R` with
///
/// ```text
/// size(R) == size(A) == size(B)
/// R(i)    == A(i) + B(i)        for i in 0..size(R)
/// ```
///
/// `A` and `B` need not be compatible, and `R` is not required to be
/// compatible with either.
///
/// PyCuTe takes `None` for either operand and returns the other, because
/// `_composition` seeds its accumulator with `None` and folds
/// [`layout_add`] over the basis terms of its right-hand side. Only the
/// accumulator is ever absent, so only the left operand is an
/// [`Option`] here, and the return is a plain layout — the fold reads
/// `acc = layout_add(acc.as_ref(), &term)?`.
///
/// Pre-conditions:
///   `size(a) == size(b)`.
///
/// Post-conditions:
///   `size(result) == size(a)`,
///   `result(i) == a(i) + b(i)` for every `i`,
///   symmetric: `layout_add(a, b) == layout_add(b, a)`.
///
/// ```text
/// layout_add(Layout(12, 1),          Layout(12, 1))          == Layout(12, 2)
/// layout_add(Layout(5, 0),           Layout(5, 1))           == Layout(5, 1)
/// layout_add(Layout((4, 3), (1, 4)), Layout((4, 3), (3, 1))) == Layout((4, 3), (4, 5))
/// ```
///
/// Returns [`Error::SizeMismatch`] when the two sizes disagree, and
/// [`Error::NoRefinement`] when no common refinement spans the whole
/// domain.
pub fn layout_add(a: Option<&Layout>, b: &Layout) -> Result<Layout> {
    let Some(a) = a else {
        return Ok(b.clone());
    };
    let size_a = size(&a.shape, &[])?;
    let size_b = size(&b.shape, &[])?;
    if size_a != size_b {
        return Err(Error::SizeMismatch {
            lhs: format!("{a}"),
            rhs: format!("{b}"),
        });
    }

    // Reduce A and B to canonical form so that greatest_common_domain
    // (which walks the *shapes*) sees the maximally-merged
    // factorizations actually exposed by each layout's strides.
    let a_co = a.coalesce(&every_mode())?;
    let b_co = b.coalesce(&every_mode())?;
    let g = greatest_common_domain(&a_co.shape, &b_co.shape);

    if size(&g.shape, &[])? != size_a {
        return Err(Error::NoRefinement {
            lhs: format!("{a_co}"),
            rhs: format!("{b_co}"),
        });
    }

    // G has shape S = (s_0, s_1, ...) and stride D = (d_0, d_1, ...).
    // Every i in [0, size(A)) decomposes uniquely as
    //     i = sum_k c_k * d_k    with  c_k in [0, s_k)
    // via G. Because size(G) == size(A) and G's leaves align with the
    // coalesced shapes of both A and B, A and B are linear on this
    // lattice:
    //     A(i) = sum_k c_k * A(d_k)        B(i) = sum_k c_k * B(d_k)
    // so the result has shape S and stride leaf-wise (A(d_k) + B(d_k)).
    let result_stride = transform_stride_leaf(&g.stride, &|d| {
        let d = as_coord(d, &g)?;
        a_co.call(&d)?.add(&b_co.call(&d)?)
    })?;

    Layout::set(g.shape, result_stride).coalesce(&every_mode())
}

/// A stride leaf of a [`greatest_common_domain`] result, read as the
/// coordinate PyCuTe passes straight to `A_co(d)`.
///
/// Those strides come out of integer arithmetic alone, so every leaf is
/// an integer and the other case does not arise. It is still spelled
/// out, because [`Stride`] admits an arithmetic tuple that no layout can
/// take as a coordinate.
fn as_coord(d: &StrideScalar, g: &Layout) -> Result<IntTuple> {
    match d {
        StrideScalar::Int(v) => Ok(HTuple::Leaf(*v)),
        other => Err(Error::BadCoord {
            idx: format!("{other:?}"),
            shape: format!("{:?}", g.shape),
        }),
    }
}

/// PyCuTe's `transform_leaf` over a stride, with a fallible leaf
/// function. [`HTuple::transform_leaf`] takes an infallible one.
fn transform_stride_leaf(
    stride: &Stride,
    f: &impl Fn(&StrideScalar) -> Result<StrideScalar>,
) -> Result<Stride> {
    let leaves = stride
        .leaves()
        .into_iter()
        .map(f)
        .collect::<Result<Vec<_>>>()?;
    // One value per leaf, so the rebuild cannot run dry.
    HTuple::unflatten(&mut leaves.into_iter(), stride).ok_or_else(|| Error::Incompatible {
        lhs: format!("{stride:?}"),
        rhs: String::from("stride"),
    })
}

/// A layout that selects the *greatest common domain* of two shapes.
/// PyCuTe's `greatest_common_domain`.
///
/// The result is a layout whose
///
/// -- shape is an ordered factorization of the common divisor that
///    `a` and `b` share *in order*, and
/// -- stride records the offset at which each common factor appears.
///
/// It depends on the two shapes alone. PyCuTe reads a shape off an int,
/// a tuple, a `Layout` or a `Tensor`; here the caller passes the shape,
/// so a layout operand arrives as `&layout.shape`.
///
/// When the leaves of `a` and `b` are pairwise coprime in their walk
/// order (e.g. `(5, 3)` against `(3, 5)`), no aligned common factor
/// exists and the result is the trivial singleton `Layout((1,), (0,))`.
///
/// Post-conditions:
///   symmetric in `a` and `b`,
///   `depth(result) == 1`,
///   `size(result)` divides `gcd(size(a), size(b))`.
///
/// ```text
/// greatest_common_domain((10,), (10,))        == Layout((10,), (1,))
/// greatest_common_domain((16, 3), (16, 3))    == Layout((16, 3), (1, 16))
/// greatest_common_domain((5, 3, 4), (10, 6))  == Layout((5, 2), (1, 30))
/// greatest_common_domain((5, 3), (3, 5))      == Layout((1,), (0,))
/// ```
pub fn greatest_common_domain(a: &IntTuple, b: &IntTuple) -> Layout {
    // A mutating walk over two flattened shapes, as PyCuTe writes it.
    // Each step divides a shared factor out of one leaf of each side, so
    // the two lists are consumed in place rather than folded over.
    let mut a: Vec<Int> = a.leaves().into_iter().copied().collect();
    let mut b: Vec<Int> = b.leaves().into_iter().copied().collect();

    let mut result_s: Vec<Int> = Vec::new();
    let mut result_d: Vec<Int> = Vec::new();

    // Running prefix products.
    let (mut p_a, mut p_b) = (1, 1);
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        // Skip size-1 leaves.
        if a[i] == 1 {
            i += 1;
            continue;
        }
        if b[j] == 1 {
            j += 1;
            continue;
        }

        let gcd_ab = gcd(a[i], b[j]);
        // Coprime leaves: nothing to do.
        if gcd_ab == 1 {
            p_a *= a[i];
            i += 1;
            p_b *= b[j];
            j += 1;
            continue;
        }

        // Largest factor of (p_a * a[i]) and (p_b * b[j]) that contains
        // the shared `gcd_ab`. We can only commit this factor when it is
        // consistent with both running prefix products.
        let gcd_pab = gcd(p_a * a[i], p_b * b[j]);
        if gcd_pab % p_a == 0 && gcd_pab % p_b == 0 {
            result_s.push(gcd_ab);
            result_d.push(gcd_pab / gcd_ab);
        }

        p_a *= gcd_ab;
        p_b *= gcd_ab;
        a[i] /= gcd_ab;
        b[j] /= gcd_ab;
    }

    match result_s.is_empty() {
        true => Layout::set(ht_of(&[1]), stride_of(&[0])),
        false => Layout::set(ht_of(&result_s), stride_of(&result_d)),
    }
}

/// A flat [`IntTuple`] of the given leaves.
fn ht_of(leaves: &[Int]) -> IntTuple {
    HTuple::Tuple(leaves.iter().copied().map(HTuple::Leaf).collect())
}

/// A flat [`Stride`] of the given integer leaves.
fn stride_of(leaves: &[Int]) -> Stride {
    HTuple::Tuple(
        leaves
            .iter()
            .map(|&d| HTuple::Leaf(StrideScalar::Int(d)))
            .collect(),
    )
}

/// The greatest common divisor of two non-negative integers. PyCuTe
/// calls `math.gcd`; `std` has no equivalent, and a whole dependency for
/// one Euclid loop is not worth it.
fn gcd(a: Int, b: Int) -> Int {
    match b {
        0 => a.abs(),
        _ => gcd(b, a % b),
    }
}
