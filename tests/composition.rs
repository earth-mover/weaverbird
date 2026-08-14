//! Ported from `test/test_composition.py`.
//!
//! `test_composition_sympy` and `test_composition_sympy_fails` are gone
//! for good: [`Int`](pinstripe::Int) is the only integer here, so there
//! is no symbolic extent whose divisibility could go unverified.
//!
//! Everything else is here, plus one test the Python file does not have.
//! `test_composition.py` only ever passes a `Layout` right-hand side, so
//! the rest of `_composition`'s dispatch head — an integer, a tuple, a
//! `None`, a per-mode `None` — is exercised by
//! [`the_dispatch_head_reads_every_tiler_form`]. Its expected values were
//! read off the reference implementation.
//!
//! The post-condition `R(i) == A(B(i))` needs one helper PyCuTe does not
//! write out, [`as_coord`]: `B(i)` may be an arithmetic tuple, and
//! PyCuTe's `idx2crd` pads its implicit trailing zeros on the way in.
#![expect(clippy::unwrap_used, reason = "a test asserts the happy path")]

use pinstripe::{
    HTuple, Int, IntTuple, Layout, OptTiler, Stride, StrideScalar, TilerLeaf, atuple::as_tuple,
    compatible, e, ht, size,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// The integer tuple as a stride, so the ported cases read like their
/// Python source.
fn as_stride(t: &IntTuple) -> Stride {
    t.transform_leaf(&|v: &Int| StrideScalar::Int(*v))
}

/// `Layout(shape, stride)` over integer strides.
fn layout(shape: IntTuple, stride: IntTuple) -> Layout {
    Layout::new(shape, &as_stride(&stride)).unwrap()
}

/// `Layout(shape)` — the compact, column-major default.
fn compact(shape: IntTuple) -> Layout {
    Layout::new(shape, &HTuple::Leaf(StrideScalar::Int(1))).unwrap()
}

/// `Layout(shape, stride)` over an already-built stride.
fn strided(shape: IntTuple, stride: Stride) -> Layout {
    Layout::new(shape, &stride).unwrap()
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

/// A layout as the right-hand side of a composition.
fn tiler(x: &Layout) -> OptTiler {
    HTuple::Leaf(Some(TilerLeaf::Layout(x.clone())))
}

/// An extent as the right-hand side of a composition. PyCuTe's `N`,
/// which the dispatch head promotes to `N:1`.
fn extent(n: Int) -> OptTiler {
    HTuple::Leaf(Some(TilerLeaf::Int(n)))
}

/// A by-mode right-hand side. `ht!` cannot spell it, because
/// `Some(TilerLeaf::Int(4))` is more than one token tree.
fn by_mode(modes: Vec<OptTiler>) -> OptTiler {
    HTuple::Tuple(modes)
}

/// The absent right-hand side. PyCuTe's `None`, whole or per-mode.
fn none() -> OptTiler {
    HTuple::Leaf(None)
}

/// An element of a layout's codomain, read back as a coordinate of
/// `shape`. PyCuTe's `ArithTuple._idx2crd`.
///
/// The post-conditions below evaluate `A(B(i))`, and `B(i)` may be an
/// arithmetic tuple whose trailing positions are implicitly zero. PyCuTe
/// pads those out inside `idx2crd`, which dispatches on the coordinate's
/// type; [`Layout::call`] takes a plain [`IntTuple`], so the padding
/// happens here instead.
fn as_coord(x: &StrideScalar, shape: &IntTuple) -> IntTuple {
    match (x, shape) {
        (StrideScalar::Arith(a), HTuple::Tuple(modes)) => HTuple::Tuple(
            modes
                .iter()
                .enumerate()
                .map(|(i, m)| as_coord(a.data().get(i).unwrap_or(&StrideScalar::Int(0)), m))
                .collect(),
        ),
        _ => as_tuple(x),
    }
}

/// PyCuTe's `postcondition_composition`.
///
/// The result takes `b`'s domain and this layout's values: `b` refines
/// the result's shape, and the result agrees with `a` after `b` at every
/// coordinate.
fn postcondition_composition(a: &Layout, b: &Layout) -> Layout {
    let r = a.composition(&tiler(b)).unwrap();

    // Post-condition: R is compatible with B.
    assert!(compatible(&b.shape, &r.shape), "{a} o {b} => {r}");

    // Post-condition: R(c) == A(B(c)) for all coordinates c in B.
    for i in 0..size(&r.shape, &[]).unwrap() {
        assert_eq!(
            r.call(&ht!(i)).unwrap(),
            a.call(&as_coord(&b.call(&ht!(i)).unwrap(), &a.shape))
                .unwrap(),
            "{a} o {b} => {r} disagree at {i}"
        );
    }
    r
}

// ---------------------------------------------------------------------------
// the exhaustive small cases
// ---------------------------------------------------------------------------

#[test]
fn composition_holds_over_every_small_shape_and_stride() {
    // Every combination with shapes and strides under 4.
    for a_stride in 0..4 {
        for b_stride in 0..4 {
            for a_shape in 1..4 {
                for b_shape in 1..4 {
                    postcondition_composition(
                        &layout(ht!(a_shape), ht!(a_stride)),
                        &layout(ht!(b_shape), ht!(b_stride)),
                    );
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// integer strides
// ---------------------------------------------------------------------------

#[test]
fn composition_holds_over_integer_strides() {
    for (a, b) in [
        (compact(ht!(12)), compact(ht!((4, 3)))),
        (layout(ht!(12), ht!(2)), compact(ht!((4, 3)))),
        (compact(ht!(12)), layout(ht!((4, 3)), ht!((3, 1)))),
        (layout(ht!(12), ht!(2)), layout(ht!((4, 3)), ht!((3, 1)))),
        (compact(ht!(12)), layout(ht!((2, 3)), ht!((2, 4)))),
        (compact(ht!((4, 3))), compact(ht!((4, 3)))),
        (compact(ht!((4, 3))), compact(ht!(12))),
        (compact(ht!((4, 3))), layout(ht!(6), ht!(2))),
        (compact(ht!((4, 3))), layout(ht!((6, 2)), ht!((2, 1)))),
        (layout(ht!((4, 3)), ht!((3, 1))), compact(ht!((4, 3)))),
        (layout(ht!((4, 3)), ht!((3, 1))), compact(ht!(12))),
        (layout(ht!((4, 3)), ht!((3, 1))), layout(ht!(6), ht!(2))),
        (
            layout(ht!((4, 3)), ht!((3, 1))),
            layout(ht!((6, 2)), ht!((2, 1))),
        ),
        (
            compact(ht!((8, 8))),
            layout(ht!(((2, 2, 2), (2, 2, 2))), ht!(((1, 16, 4), (8, 2, 32)))),
        ),
        (
            layout(ht!((8, 8)), ht!((8, 1))),
            layout(ht!(((2, 2, 2), (2, 2, 2))), ht!(((1, 16, 4), (8, 2, 32)))),
        ),
        (
            layout(ht!(((2, 2, 2), (2, 2, 2))), ht!(((1, 16, 4), (8, 2, 32)))),
            layout(ht!(8), ht!(4)),
        ),
        (
            layout(ht!((4, 2)), ht!((1, 16))),
            layout(ht!((4, 2)), ht!((2, 1))),
        ),
        (
            layout(ht!((2, 2)), ht!((2, 1))),
            layout(ht!((2, 2)), ht!((2, 1))),
        ),
        (
            compact(ht!((4, 8, 2))),
            layout(ht!((2, 2, 2)), ht!((2, 8, 1))),
        ),
        (
            layout(ht!((4, 8, 2)), ht!((2, 8, 1))),
            layout(ht!((2, 2, 2)), ht!((1, 8, 2))),
        ),
        (
            layout(ht!((4, 8, 2)), ht!((2, 8, 1))),
            layout(ht!((4, 2, 2)), ht!((2, 8, 1))),
        ),
        // Pre-coalesced LHS.
        (
            layout(ht!((4, 6, 8)), ht!((1, 4, 7))),
            layout(ht!(6), ht!(1)),
        ),
        // Mid-layout truncation.
        (
            layout(ht!((4, 6, 8, 10)), ht!((2, 3, 5, 7))),
            layout(ht!(6), ht!(12)),
        ),
        (
            layout(ht!((5, 126, 7)), ht!((1, 13, 0))),
            layout(ht!(21), ht!(30)),
        ),
        (layout(ht!((23, 5)), ht!((2, 120))), layout(ht!(7), ht!(3))),
        // Over the end.
        (
            layout(ht!((4, 6, 1)), ht!((2, 3, 0))),
            layout(ht!(30), ht!(4)),
        ),
        (layout(ht!((4, 6, 1)), ht!((1, 4, 0))), compact(ht!((6, 8)))),
        // Other.
        (layout(ht!((5, 5)), ht!((5, 5))), layout(ht!(5), ht!(5))),
        (layout(ht!(7), ht!(11)), layout(ht!(3), ht!(4))),
        (layout(ht!(7), ht!(11)), layout(ht!((3, 5)), ht!((6, 3)))),
    ] {
        postcondition_composition(&a, &b);
    }
}

#[test]
fn composition_names_the_divisibility_condition_it_violates() {
    // Violates the stride divisibility condition.
    let a = layout(ht!((5, 3)), ht!((7, 1)));
    assert!(a.composition(&tiler(&layout(ht!(2), ht!(3)))).is_err());
    // Violates the shape divisibility condition.
    assert!(a.composition(&tiler(&layout(ht!(7), ht!(1)))).is_err());
}

// ---------------------------------------------------------------------------
// basis strides
// ---------------------------------------------------------------------------

#[test]
fn composition_holds_over_a_basis_strided_lhs() {
    for (a, b) in [
        (strided(ht!(12), s(e(&[0]))), compact(ht!((4, 3)))),
        (strided(ht!(12), s(e(&[1]).scale(2))), compact(ht!((4, 3)))),
        (
            strided(ht!(12), s(e(&[0]))),
            layout(ht!((4, 3)), ht!((3, 1))),
        ),
        (
            strided(ht!(12), s(e(&[1]).scale(2))),
            layout(ht!((4, 3)), ht!((3, 1))),
        ),
        (
            strided(ht!(12), s(e(&[1, 1]))),
            layout(ht!((2, 3)), ht!((2, 4))),
        ),
        (
            strided(ht!((4, 3)), t(vec![s(e(&[0])), s(e(&[1]))])),
            compact(ht!((4, 3))),
        ),
        (
            strided(ht!((4, 3)), t(vec![s(e(&[0])), s(e(&[1]))])),
            compact(ht!(12)),
        ),
        (
            strided(ht!((4, 3)), t(vec![s(e(&[0])), s(e(&[1]))])),
            layout(ht!(6), ht!(2)),
        ),
        (
            strided(ht!((4, 3)), t(vec![s(e(&[0])), s(e(&[1]))])),
            layout(ht!((6, 2)), ht!((2, 1))),
        ),
        (
            strided(ht!((4, 3)), t(vec![s(e(&[1])), s(e(&[0]))])),
            compact(ht!((4, 3))),
        ),
        (
            strided(ht!((4, 3)), t(vec![s(e(&[1])), s(e(&[0]))])),
            compact(ht!(12)),
        ),
        (
            strided(ht!((4, 3)), t(vec![s(e(&[1])), s(e(&[0]))])),
            layout(ht!(6), ht!(2)),
        ),
        (
            strided(ht!((4, 3)), t(vec![s(e(&[1])), s(e(&[0]))])),
            layout(ht!((6, 2)), ht!((2, 1))),
        ),
        (
            strided(
                ht!((4, 3)),
                t(vec![s(e(&[1]).scale(6)), s(e(&[1]).scale(2))]),
            ),
            compact(ht!((4, 3))),
        ),
        (
            strided(
                ht!((4, 3)),
                t(vec![s(e(&[1]).scale(6)), s(e(&[1]).scale(2))]),
            ),
            compact(ht!(12)),
        ),
        (
            strided(
                ht!((4, 3)),
                t(vec![s(e(&[1]).scale(6)), s(e(&[1]).scale(2))]),
            ),
            layout(ht!(6), ht!(2)),
        ),
        (
            strided(
                ht!((4, 3)),
                t(vec![s(e(&[1]).scale(6)), s(e(&[1]).scale(2))]),
            ),
            layout(ht!((6, 2)), ht!((2, 1))),
        ),
        (
            strided(
                ht!((4, 4)),
                t(vec![
                    s(e(&[0]).add(&e(&[1])).unwrap()),
                    s(e(&[0]).scale(3).add(&e(&[1])).unwrap()),
                ]),
            ),
            layout(ht!((4, 2)), ht!((2, 1))),
        ),
        (
            strided(ht!((4, 6, 8)), t(vec![s(e(&[0])), s(e(&[1])), s(e(&[2]))])),
            compact(ht!((2, 2, 2))),
        ),
    ] {
        postcondition_composition(&a, &b);
    }
}

#[test]
fn composition_holds_over_a_basis_strided_rhs() {
    for (a, b) in [
        (
            layout(ht!((4, 4)), ht!((4, 1))),
            strided(ht!((4, 4)), t(vec![s(e(&[0])), s(e(&[1]))])),
        ),
        (
            layout(ht!((4, 4)), ht!((4, 1))),
            strided(ht!((4, 4)), t(vec![s(e(&[1])), s(e(&[0]))])),
        ),
        (
            layout(ht!((4, 5)), ht!((5, 1))),
            strided(ht!(30), s(e(&[0]))),
        ),
        (
            layout(ht!((4, 5)), ht!((5, 1))),
            strided(ht!(12), s(e(&[1]))),
        ),
        (
            layout(ht!((4, (4, 3), 1)), ht!((3, (12, 1), 0))),
            strided(ht!(12), s(e(&[1]))),
        ),
        (
            layout(ht!((4, (4, 3), 1)), ht!((3, (12, 1), 0))),
            strided(ht!(12), s(e(&[1, 0]))),
        ),
        (
            layout(ht!((4, (2, 3))), ht!((6, (3, 1)))),
            strided(ht!((2, 4)), t(vec![s(e(&[1, 1])), s(e(&[0]))])),
        ),
        (
            compact(ht!((4, 6, 8))),
            strided(ht!((2, 2, 2)), t(vec![s(e(&[0])), s(e(&[1])), s(e(&[2]))])),
        ),
        (
            compact(ht!((4, 6, 8))),
            strided(ht!((2, 2, 2)), t(vec![s(e(&[2])), s(e(&[0])), s(e(&[1]))])),
        ),
        (
            strided(ht!((4, 6, 8)), t(vec![s(e(&[0])), s(e(&[1])), s(e(&[2]))])),
            strided(ht!((2, 2, 2)), t(vec![s(e(&[0])), s(e(&[1])), s(e(&[2]))])),
        ),
        (
            strided(
                ht!((3, 5, 7, 11)),
                t(vec![s(e(&[0])), s(e(&[1])), s(e(&[2])), s(e(&[3]))]),
            ),
            strided(ht!(3), s(e(&[2]).scale(4))),
        ),
        (
            strided(
                ht!((3, 5, 7, 11)),
                t(vec![s(e(&[0])), s(e(&[1])), s(e(&[2])), s(e(&[3]))]),
            ),
            strided(ht!(3), s(e(&[2]).scale(4).add(&e(&[3]).scale(2)).unwrap())),
        ),
        (
            strided(
                ht!((3, 5, 7, 11)),
                t(vec![s(e(&[0])), s(e(&[1])), s(e(&[2])), s(e(&[3]))]),
            ),
            strided(ht!(3), s(e(&[0]).add(&e(&[3])).unwrap())),
        ),
        // Diag.
        (
            layout(ht!((4, 4)), ht!((3, 42))),
            strided(ht!(4), s(e(&[0]).add(&e(&[1])).unwrap())),
        ),
        // Skew diag.
        (
            layout(ht!((4, 8)), ht!((3, 42))),
            strided(ht!(4), s(e(&[0]).add(&e(&[1]).scale(2)).unwrap())),
        ),
    ] {
        postcondition_composition(&a, &b);
    }
}

// ---------------------------------------------------------------------------
// associativity
// ---------------------------------------------------------------------------

/// PyCuTe's `postcondition_associativity`.
///
/// Composition is associative wherever each intermediate result is
/// evaluated inside its guaranteed domain — every `c(i)` a coordinate of
/// `b`, and every `b(c(i))` a coordinate of `a`. The cases below are
/// nested that way, so both groupings agree on all of `c`'s domain.
fn postcondition_associativity(a: &Layout, b: &Layout, c: &Layout) {
    let ab_c = a
        .composition(&tiler(b))
        .unwrap()
        .composition(&tiler(c))
        .unwrap();
    let a_bc = a
        .composition(&tiler(&b.composition(&tiler(c)).unwrap()))
        .unwrap();

    for i in 0..size(&c.shape, &[]).unwrap() {
        let bc = b
            .call(&as_coord(&c.call(&ht!(i)).unwrap(), &b.shape))
            .unwrap();
        let abc = a.call(&as_coord(&bc, &a.shape)).unwrap();
        assert_eq!(ab_c.call(&ht!(i)).unwrap(), abc, "({a} o {b}) o {c} at {i}");
        assert_eq!(a_bc.call(&ht!(i)).unwrap(), abc, "{a} o ({b} o {c}) at {i}");
    }
}

#[test]
fn composition_is_associative() {
    for (a, b, c) in [
        (
            layout(ht!((4, 3)), ht!((3, 1))),
            layout(ht!((4, 3)), ht!((1, 4))),
            layout(ht!((4, 3)), ht!((1, 4))),
        ),
        (
            layout(ht!((4, 3)), ht!((3, 1))),
            layout(ht!((4, 3)), ht!((1, 4))),
            layout(ht!(6), ht!(2)),
        ),
        (
            layout(ht!((4, 3)), ht!((1, 4))),
            layout(ht!((4, 3)), ht!((3, 1))),
            layout(ht!((4, 3)), ht!((1, 4))),
        ),
        (
            layout(ht!((4, 3)), ht!((1, 4))),
            layout(ht!((4, 3)), ht!((3, 1))),
            layout(ht!(6), ht!(2)),
        ),
        (
            layout(ht!((6, 4)), ht!((4, 1))),
            layout(ht!((4, 3)), ht!((3, 1))),
            layout(ht!((4, 3)), ht!((1, 4))),
        ),
        (
            layout(ht!((6, 4)), ht!((4, 1))),
            layout(ht!((4, 3)), ht!((3, 1))),
            layout(ht!(6), ht!(2)),
        ),
        (
            layout(ht!(6), ht!(2)),
            layout(ht!((2, 3)), ht!((3, 1))),
            layout(ht!((2, 3)), ht!((1, 2))),
        ),
        (
            layout(ht!(6), ht!(2)),
            layout(ht!((2, 3)), ht!((1, 2))),
            layout(ht!((2, 3)), ht!((3, 1))),
        ),
        (
            layout(ht!((2, 3)), ht!((3, 1))),
            layout(ht!((2, 3)), ht!((1, 2))),
            layout(ht!((2, 3)), ht!((1, 2))),
        ),
        (
            layout(ht!((8, 8)), ht!((8, 1))),
            layout(ht!((6, 4)), ht!((4, 1))),
            layout(ht!(6), ht!(2)),
        ),
        (
            layout(ht!((8, 8)), ht!((8, 1))),
            layout(ht!((6, 4)), ht!((4, 1))),
            layout(ht!((2, 3)), ht!((1, 2))),
        ),
        (
            layout(ht!((4, 6)), ht!((1, 4))),
            layout(ht!((4, 3)), ht!((3, 1))),
            layout(ht!((4, 3)), ht!((1, 4))),
        ),
    ] {
        postcondition_associativity(&a, &b, &c);
    }
}

// ---------------------------------------------------------------------------
// the dispatch head
// ---------------------------------------------------------------------------

#[test]
fn the_dispatch_head_reads_every_tiler_form() {
    let a = layout(ht!((8, 8)), ht!((8, 1)));

    // RHS None, noop.
    assert_eq!(a.composition(&none()).unwrap(), a);
    // RHS int, A o N -> A o N:1.
    assert_eq!(a.composition(&extent(4)).unwrap(), layout(ht!(4), ht!(8)));
    // RHS tuple, by-mode.
    assert_eq!(
        a.composition(&by_mode(vec![extent(4), extent(2)])).unwrap(),
        layout(ht!((4, 2)), ht!((8, 1)))
    );
    assert_eq!(
        a.composition(&by_mode(vec![
            tiler(&layout(ht!(4), ht!(2))),
            tiler(&layout(ht!(2), ht!(1))),
        ]))
        .unwrap(),
        layout(ht!((4, 2)), ht!((16, 1)))
    );
    // A per-mode None leaves its mode alone.
    assert_eq!(
        a.composition(&by_mode(vec![none(), extent(4)])).unwrap(),
        layout(ht!((8, 4)), ht!((8, 1)))
    );
    assert_eq!(
        a.composition(&by_mode(vec![extent(4), none()])).unwrap(),
        layout(ht!((4, 8)), ht!((8, 1)))
    );
    assert_eq!(
        a.composition(&by_mode(vec![tiler(&layout(ht!(2), ht!(4))), none()]))
            .unwrap(),
        layout(ht!((2, 8)), ht!((32, 1)))
    );
    // A tiler tuple that outranks the mode it faces is rejected: mode 0
    // of `a` is rank 1, and `(2, 2)` is rank 2.
    assert!(
        a.composition(&by_mode(vec![
            by_mode(vec![extent(2), extent(2)]),
            extent(4),
        ]))
        .is_err()
    );
}
