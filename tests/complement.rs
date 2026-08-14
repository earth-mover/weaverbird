//! Ported from `test/test_complement.py`.
//!
//! `test_complement_sympy` is gone: [`Int`](weaverbird::Int) is the only
//! integer here, so there is no symbolic extent to carry through. Its
//! substitution twin, `test_complement_sympy_substitution`, is concrete
//! and is ported in full — and the one symbolic `extend` case it does
//! not cover has a concrete stand-in below.
#![expect(clippy::unwrap_used, reason = "a test asserts the happy path")]

use std::cmp::Ordering;

use weaverbird::{HTuple, IntTuple, Layout, Stride, StrideScalar, e, ht, scaled_basis};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// `Layout(shape, stride)` over integer strides.
fn layout(shape: IntTuple, stride: IntTuple) -> Layout {
    Layout::from_base(shape, &stride.into()).unwrap()
}

/// `Layout(shape, stride)` over a stride that holds basis elements.
fn strided(shape: IntTuple, stride: &Stride) -> Layout {
    Layout::from_base(shape, stride).unwrap()
}

/// A stride leaf.
fn s(x: StrideScalar) -> Stride {
    HTuple::Leaf(x)
}

/// A stride mode. `ht!` cannot spell it, because a scaled basis element
/// is more than one token tree.
fn t(modes: Vec<Stride>) -> Stride {
    HTuple::Tuple(modes)
}

/// A codomain value read back as a coordinate. PyCuTe feeds `L(i)`
/// straight into another layout; [`Layout::call`] takes an [`IntTuple`],
/// so the conversion is spelled out.
fn crd(x: &StrideScalar) -> IntTuple {
    x.to_tuple()
}

/// PyCuTe's `postcondition_complement`.
///
/// The result is weakly congruent with the codomain, its own codomain
/// is ordered, and it meets the source's codomain nowhere.
fn postcondition_complement(source: &Layout) -> Layout {
    let result = source.complement().unwrap();

    // Post-condition: weak congruence with the codomain.
    assert!(source.coshape().unwrap().weakly_congruent(&result.shape), "{source} => {result}");

    // Post-condition: orderedness and disjointness of the codomains.
    let size_r = result.shape.size();
    let size_l = source.shape.size();
    for i in 1..10 + size_r {
        let (previous, current) =
            (result.eval(&ht!(i - 1)).unwrap(), result.eval(&ht!(i)).unwrap());
        assert_eq!(
            previous.try_cmp(&current).ok(),
            Some(Ordering::Less),
            "ordered: {source} => {result} at {i}"
        );
        for j in 0..size_l {
            assert_ne!(
                current,
                source.eval(&ht!(j)).unwrap(),
                "disjoint: {source} => {result} at ({i}, {j})"
            );
        }
    }

    result
}

/// PyCuTe's `postcondition_complement_strong`.
///
/// Beyond the weak post-condition, the source and its complement
/// together admit a right inverse, and that inverse is a generalized
/// reflexive inverse of the completed layout.
fn postcondition_complement_strong(source: &Layout) {
    let result = postcondition_complement(source);

    // Generalized inverse conditions.
    let completed = Layout::from_modes(vec![source.clone(), result]);
    let inverse = completed.right_inverse().unwrap();

    // Right inverse condition.
    for i in 0..inverse.shape.size() {
        let r = inverse.eval(&ht!(i)).unwrap();
        assert_eq!(
            inverse.eval(&crd(&completed.eval(&crd(&r)).unwrap())).unwrap(),
            r,
            "right inverse: {completed} => {inverse} at {i}"
        );
    }

    // Left inverse condition — the right inverse is a generalized
    // reflexive inverse.
    for i in 0..completed.shape.size() {
        let c = completed.eval(&ht!(i)).unwrap();
        assert_eq!(
            completed.eval(&crd(&inverse.eval(&crd(&c)).unwrap())).unwrap(),
            c,
            "reflexive: {completed} => {inverse} at {i}"
        );
    }
}

// ---------------------------------------------------------------------------
// complement
// ---------------------------------------------------------------------------

#[test]
fn complement_completes_the_codomain() {
    postcondition_complement_strong(&layout(ht!(1), ht!(0)));
    postcondition_complement_strong(&layout(ht!(1), ht!(1)));
    postcondition_complement_strong(&layout(ht!(1), ht!(2)));
    postcondition_complement_strong(&layout(ht!(1), ht!(4)));
    postcondition_complement_strong(&layout(ht!((1, 1)), ht!((0, 0))));
    postcondition_complement_strong(&layout(ht!((3, 7)), ht!((0, 0))));
    postcondition_complement_strong(&layout(ht!(5), ht!(1)));
    postcondition_complement_strong(&layout(ht!(5), ht!(3)));
    postcondition_complement_strong(&layout(ht!(4), ht!(0)));
    postcondition_complement_strong(&layout(ht!(4), ht!(1)));
    postcondition_complement_strong(&layout(ht!(4), ht!(2)));
    postcondition_complement_strong(&layout(ht!(4), ht!(4)));
    postcondition_complement_strong(&layout(ht!((2, 4)), ht!((1, 2))));
    postcondition_complement_strong(&layout(ht!((2, 3)), ht!((1, 2))));
    postcondition_complement_strong(&layout(ht!((2, 4)), ht!((1, 4))));
    postcondition_complement_strong(&layout(ht!((8, 4)), ht!((1, 8))));
    postcondition_complement_strong(&layout(ht!((8, 4)), ht!((4, 1))));
    postcondition_complement_strong(&layout(ht!((2, 4, 6)), ht!((1, 2, 8))));
    postcondition_complement_strong(&layout(ht!((2, 4, 6)), ht!((4, 1, 8))));
    postcondition_complement_strong(&layout(ht!((2, 4, 8)), ht!((8, 1, 64))));
    postcondition_complement_strong(&layout(ht!((2, 4, 8)), ht!((32, 0, 2))));
    postcondition_complement_strong(&layout(ht!((2, 4, 8)), ht!((2, 0, 32))));
    postcondition_complement_strong(&layout(ht!((2, 4, 4, 4, 2)), ht!((32, 0, 2, 0, 512))));
    postcondition_complement_strong(&layout(ht!(((2, 2), (2, 2))), ht!(((1, 4), (8, 32)))));
    postcondition_complement_strong(&layout(ht!((2, (3, 4))), ht!((3, (1, 6)))));
    postcondition_complement_strong(&layout(ht!((4, 2)), ht!((1, 16))));
}

#[test]
fn complement_completes_a_layout_with_gaps() {
    postcondition_complement(&layout(ht!((4, 2)), ht!((1, 6))));
    postcondition_complement(&layout(ht!((4, 2)), ht!((1, 5))));
    postcondition_complement(&layout(ht!((4, 2)), ht!((1, 10))));
    postcondition_complement(&layout(ht!((4, 2)), ht!((1, 11))));
    postcondition_complement(&layout(ht!((2, 4)), ht!((11, 1))));
}

#[test]
fn complement_completes_a_coordinate_codomain() {
    postcondition_complement_strong(&strided(ht!(3), &s(e!(0))));
    postcondition_complement_strong(&strided(ht!(3), &s(scaled_basis(4, &[2]))));
    postcondition_complement_strong(&strided(
        ht!((2, 5, 3)),
        &t(vec![s(scaled_basis(4, &[1])), s(scaled_basis(5, &[0])), s(scaled_basis(16, &[1]))]),
    ));
    postcondition_complement_strong(&strided(
        ht!((2, 3, 5)),
        &t(vec![s(scaled_basis(4, &[1])), s(scaled_basis(5, &[0])), s(scaled_basis(7, &[2, 1]))]),
    ));
    // A stride of integer 0 sits beside two basis strides. This case
    // caught the inverses projecting their accumulators ahead of the
    // stride-0 guard.
    postcondition_complement_strong(&strided(
        ht!((2, 3, 5)),
        &t(vec![s(scaled_basis(4, &[1])), s(StrideScalar::Int(0)), s(scaled_basis(7, &[2, 1]))]),
    ));
}

/// PyCuTe's `test_complement_sympy_substitution`. The symbolic
/// complement has to agree with the concrete one under any concrete
/// substitution, and each concrete result has to obey the post-condition.
#[test]
fn complement_agrees_under_substitution() {
    for n in [1, 2, 3, 5] {
        assert_eq!(
            layout(ht!((4, n)), ht!((1, 4))).complement().unwrap(),
            layout(ht!(1), ht!(4 * n))
        );
        assert_eq!(layout(ht!(n), ht!(1)).complement().unwrap(), layout(ht!(1), ht!(n)));
        assert_eq!(
            layout(ht!((n, 4)), ht!((1, n))).complement().unwrap(),
            layout(ht!(1), ht!(4 * n))
        );
        postcondition_complement(&layout(ht!((4, n)), ht!((1, 4))));
        postcondition_complement(&layout(ht!(n), ht!(1)));
        postcondition_complement(&layout(ht!((n, 4)), ht!((1, n))));
    }
}

// ---------------------------------------------------------------------------
// complement — extend
// ---------------------------------------------------------------------------

/// The concrete stand-in for the `extend` line of
/// `test_complement_sympy`, which reads
/// `Layout(256, 1)._complement(extend=(32,4,2,2,N)) == Layout((2,N), (256,512))`.
/// With a concrete trailing extent the two modes coalesce into one.
#[test]
fn complement_extends_to_cover_a_shape() {
    assert_eq!(
        layout(ht!(256), ht!(1)).complement_to(&ht!((32, 4, 2, 2, 3))).unwrap(),
        layout(ht!(6), ht!(256))
    );
    // `complement(Layout(4, 2)) == Layout((2, 1), (1, 8))`: the extend
    // grows that trailing `1` until the result spans the given shape.
    assert_eq!(
        layout(ht!(4), ht!(2)).complement_to(&ht!((64))).unwrap(),
        layout(ht!((2, 8)), ht!((1, 8)))
    );
    // An extend the complement already covers leaves only the gap mode.
    assert_eq!(layout(ht!(4), ht!(2)).complement_to(&ht!((2, 3))).unwrap(), layout(ht!(2), ht!(1)));
}
