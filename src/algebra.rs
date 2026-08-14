//! The two algebraic operations that belong to no single layout.
//!
//! Everything else in the algebra is a method on [`Layout`].

use crate::{
    atuple::StrideScalar,
    error::{Error, Result},
    htuple::HTuple,
    layout::Layout,
    typedefs::{Int, IntTuple},
};

/// Adds two layouts coordinate-wise.
///
/// Given layouts `A` and `B` with `size(A) == size(B)`, returns a layout
/// `R` with
///
/// ```text
/// size(R) == size(A) == size(B)
/// R(i)    == A(i) + B(i)        for i in 0..size(R)
/// ```
///
/// `A` and `B` need not be compatible. `R` need not be compatible with
/// either.
///
/// Pre-conditions:
///   `size(a) == size(b)`.
///
/// Post-conditions:
///   `size(result) == size(a)`,
///   `result(i) == a(i) + b(i)` for every `i`,
///   symmetric: `layout_add(a, b) == layout_add(b, a)`.
///
/// ```
/// # use weaverbird::{layout, layout_add};
/// assert_eq!(layout_add(&layout!(12:1), &layout!(12:1)).unwrap(), layout!(12:2));
/// assert_eq!(layout_add(&layout!(5:0), &layout!(5:1)).unwrap(), layout!(5:1));
/// let sum = layout_add(&layout!((4, 3):(1, 4)), &layout!((4, 3):(3, 1))).unwrap();
/// assert_eq!(sum, layout!((4, 3):(4, 5)));
/// ```
///
/// Returns [`Error::SizeMismatch`] when the two sizes disagree, and
/// [`Error::NoRefinement`] when no common refinement spans the whole
/// domain.
pub fn layout_add(a: &Layout, b: &Layout) -> Result<Layout> {
    let size_a = a.shape.size();
    if size_a != b.shape.size() {
        return Err(Error::SizeMismatch { lhs: Box::new(a.clone()), rhs: Box::new(b.clone()) });
    }

    // Reduce A and B to canonical form. greatest_common_domain walks the
    // shapes, so it must see the merged factorizations that the strides
    // expose.
    let a_co = a.coalesced()?;
    let b_co = b.coalesced()?;
    let g = greatest_common_domain(&a_co.shape, &b_co.shape);

    if g.shape.size() != size_a {
        return Err(Error::NoRefinement { lhs: a_co.shape.clone(), rhs: b_co.shape.clone() });
    }

    // G has shape S = (s_0, s_1, ...) and stride D = (d_0, d_1, ...).
    // Every i in [0, size(A)) decomposes uniquely as
    //     i = sum_k c_k * d_k    with  c_k in [0, s_k)
    // via G. size(G) == size(A), and G's leaves align with the coalesced
    // shapes of both A and B, so A and B are linear on this lattice:
    //     A(i) = sum_k c_k * A(d_k)        B(i) = sum_k c_k * B(d_k)
    // The result therefore has shape S, and stride (A(d_k) + B(d_k)) at
    // each leaf.
    let stride = g.stride.try_transform_leaf(&|d| {
        let d = as_coord(d)?;
        a_co.eval(&d)?.add(&b_co.eval(&d)?)
    })?;

    Layout::from_parts(g.shape, stride).coalesced()
}

/// A stride leaf of a [`greatest_common_domain`] result, read as a
/// coordinate.
///
/// Those strides come out of integer arithmetic alone, so every leaf is
/// an integer. The other arm stays for the type: a [`StrideScalar`]
/// admits an arithmetic tuple, which no layout takes as a coordinate.
fn as_coord(d: &StrideScalar) -> Result<IntTuple> {
    match d {
        StrideScalar::Int(v) => Ok(HTuple::Leaf(*v)),
        other => Err(Error::NotBasis { value: other.clone() }),
    }
}

/// A layout that selects the *greatest common domain* of two shapes.
///
/// The result is a layout whose
///
/// -- shape is an ordered factorization of the common divisor that
///    `a` and `b` share *in order*, and
/// -- stride records the offset at which each common factor appears.
///
/// It depends on the two shapes alone, so a layout operand arrives as
/// `&layout.shape`.
///
/// The leaves of `a` and `b` can be pairwise coprime in their walk order,
/// as `(5, 3)` is against `(3, 5)`. No aligned common factor exists then,
/// and the result is the trivial singleton `Layout((1,), (0,))`.
///
/// Post-conditions:
///   symmetric in `a` and `b`,
///   `depth(result) == 1`,
///   `size(result)` divides `gcd(size(a), size(b))`.
///
/// ```
/// # use weaverbird::{greatest_common_domain, ht, layout};
/// let g = greatest_common_domain(&ht!((16, 3)), &ht!((16, 3)));
/// assert_eq!(g, layout!((16, 3):(1, 16)));
/// assert_eq!(greatest_common_domain(&ht!((5, 3)), &ht!((3, 5))), layout!((1,):(0,)));
/// ```
pub fn greatest_common_domain(a: &IntTuple, b: &IntTuple) -> Layout {
    // Each step divides a shared factor out of one leaf of each side, so
    // the two lists are consumed in place rather than folded over.
    let mut a = a.leaves().copied().collect::<Vec<Int>>();
    let mut b = b.leaves().copied().collect::<Vec<Int>>();

    let mut extents: Vec<Int> = Vec::new();
    let mut offsets: Vec<Int> = Vec::new();

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

        // The largest factor of (p_a * a[i]) and (p_b * b[j]) that holds
        // the shared `gcd_ab`. It commits only when it agrees with both
        // running prefix products.
        let gcd_pab = gcd(p_a * a[i], p_b * b[j]);
        if gcd_pab % p_a == 0 && gcd_pab % p_b == 0 {
            extents.push(gcd_ab);
            offsets.push(gcd_pab / gcd_ab);
        }

        p_a *= gcd_ab;
        p_b *= gcd_ab;
        a[i] /= gcd_ab;
        b[j] /= gcd_ab;
    }

    match extents.is_empty() {
        true => Layout::from_parts(ht_of(&[1]), stride_of(&[0])),
        false => Layout::from_parts(ht_of(&extents), stride_of(&offsets)),
    }
}

/// A flat [`IntTuple`] of the given leaves.
fn ht_of(leaves: &[Int]) -> IntTuple {
    leaves.iter().copied().map(HTuple::Leaf).collect()
}

/// A flat stride of the given integer leaves.
fn stride_of(leaves: &[Int]) -> HTuple<StrideScalar> {
    leaves.iter().map(|&d| HTuple::Leaf(StrideScalar::Int(d))).collect()
}

/// The greatest common divisor of two integers. `std` has none, and one
/// Euclid loop does not justify a dependency.
fn gcd(a: Int, b: Int) -> Int {
    match b {
        0 => a.abs(),
        _ => gcd(b, a % b),
    }
}
