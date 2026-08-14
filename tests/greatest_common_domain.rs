//! Ported from `test/test_greatest_common_domain.py`.
//!
//! One clause of the Python post-condition needs the rest of the
//! algebra and so waits for it:
//!
//! -- `size(greatest_common_domain(logical_divide(shape(A), R)[1],
//!    logical_divide(shape(B), R)[1])) == 1`.
//!
//! `logical_divide` is not ported yet.
//!
//! PyCuTe reads a shape off any operand, so its cases pass ints,
//! tuples, and `Layout`s interchangeably. Here the shape is the
//! argument, and a layout operand arrives as `&layout.shape`.
#![expect(clippy::unwrap_used, reason = "a test asserts the happy path")]

use pinstripe::{
    HTuple, Int, IntTuple, Layout, OptTiler, Stride, StrideScalar, Tiler, TilerLeaf, depth,
    greatest_common_domain, ht, size, tiler_to_layout,
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

/// A layout as the right-hand side of a composition.
fn tiler(x: &Layout) -> OptTiler {
    HTuple::Leaf(Some(TilerLeaf::Layout(x.clone())))
}

/// The tiler of a plain shape.
fn int_tiler(shape: &IntTuple) -> Tiler {
    shape.transform_leaf(&|v: &Int| TilerLeaf::Int(*v))
}

/// The domain size of a shape.
fn extent(shape: &IntTuple) -> Int {
    size(shape, &[]).unwrap()
}

/// The greatest common divisor, as the Python cases call `math.gcd`.
fn gcd(a: Int, b: Int) -> Int {
    match b {
        0 => a,
        _ => gcd(b, a % b),
    }
}

/// PyCuTe's `postcondition_greatest_common_domain`, less the clause
/// that needs `logical_divide`.
fn postcondition(a: &IntTuple, b: &IntTuple) -> Layout {
    let result = greatest_common_domain(a, b);

    // The result is flat.
    assert_eq!(depth(&result.shape, &[]).unwrap(), 1, "flat: {result}");

    // Its size divides both operands, and their gcd.
    let n = extent(&result.shape);
    assert_eq!(extent(a) % n, 0, "{result} does not divide {a:?}");
    assert_eq!(extent(b) % n, 0, "{result} does not divide {b:?}");
    assert_eq!(gcd(extent(a), extent(b)) % n, 0, "{result} exceeds the gcd");

    // It divides both operands: each admits a composition with it.
    for operand in [a, b] {
        assert!(
            tiler_to_layout(&int_tiler(operand), &StrideScalar::Int(1))
                .unwrap()
                .composition(&tiler(&result))
                .is_ok(),
            "{result} does not divide {operand:?}"
        );
    }

    // Symmetric in the two operands.
    assert_eq!(
        greatest_common_domain(b, a),
        result,
        "asymmetric: gcd({a:?}, {b:?})"
    );

    result
}

// ---------------------------------------------------------------------------
// Cases
// ---------------------------------------------------------------------------

#[test]
fn singleton() {
    // The empty / singleton case.
    assert_eq!(postcondition(&ht!(1), &ht!(1)), layout(ht!((1)), ht!((0))));
}

#[test]
fn int_vs_int() {
    // Same int -- full common domain.
    assert_eq!(extent(&postcondition(&ht!(10), &ht!(10)).shape), 10);

    // Coprime ints -- size 1.
    assert_eq!(extent(&postcondition(&ht!(7), &ht!(9)).shape), 1);

    // Shared prime factors.
    assert_eq!(extent(&postcondition(&ht!(12), &ht!(18)).shape), 6);
}

#[test]
fn int_vs_tuple_coerces_via_shape() {
    // `10` and `(10,)` have the same leaf sequence.
    assert_eq!(
        greatest_common_domain(&ht!(10), &ht!(10)),
        greatest_common_domain(&ht!((10)), &ht!((10)))
    );
}

#[test]
fn known_expected_results() {
    // Hand-checked expected results for canonical examples.
    let expected = vec![
        (ht!(1), ht!(1), layout(ht!((1)), ht!((0)))),
        (ht!(10), ht!(10), layout(ht!((10)), ht!((1)))),
        (
            ht!((16, 3)),
            ht!((16, 3)),
            layout(ht!((16, 3)), ht!((1, 16))),
        ),
        (ht!((5, 2)), ht!(10), layout(ht!((5, 2)), ht!((1, 5)))),
        (
            ht!((5, 3, 4)),
            ht!((10, 6)),
            layout(ht!((5, 2)), ht!((1, 30))),
        ),
        (
            ht!((5, 3, 3, 4)),
            ht!((10, 6, 3)),
            layout(ht!((5)), ht!((1))),
        ),
        (
            ht!((1, 5, 3, 3, 4)),
            ht!((10, 6, 3)),
            layout(ht!((5)), ht!((1))),
        ),
        (
            ht!((5, 3, 3, 4)),
            ht!((10, 6, 1, 3)),
            layout(ht!((5)), ht!((1))),
        ),
        (ht!((2, 21)), ht!((3, 14)), layout(ht!((7)), ht!((6)))),
        (
            ht!((6, 35)),
            ht!((15, 14)),
            layout(ht!((3, 7)), ht!((1, 30))),
        ),
        (
            ht!((16, 64)),
            ht!((4, 16, 16)),
            layout(ht!((4, 4, 4, 16)), ht!((1, 4, 16, 64))),
        ),
        (ht!((5, 3)), ht!((3, 5)), layout(ht!((1)), ht!((0)))),
        (ht!((5, 5, 3)), ht!((5, 3, 5)), layout(ht!((5)), ht!((1)))),
        (
            ht!((7, 2, 3, 2, 2, 3)),
            ht!((504)),
            layout(ht!((7, 2, 3, 2, 2, 3)), ht!((1, 7, 14, 42, 84, 168))),
        ),
        (
            ht!((7, 2, 3, 2, 2, 3, 2)),
            ht!((7, 3, 2, 2, 3, 2, 2)),
            layout(ht!((7, 2, 2)), ht!((1, 42, 504))),
        ),
        (
            ht!((7, 2, 2, 2, 2, 3, 5)),
            ht!((7, 3, 2, 2, 2, 2, 5)),
            layout(ht!((7, 5)), ht!((1, 336))),
        ),
    ];
    for (a, b, want) in &expected {
        assert_eq!(&postcondition(a, b), want, "gcd({a:?}, {b:?})");
    }
}

#[test]
fn coprime_leaves_yield_singleton() {
    // Order-aligned but pairwise coprime leaves yield the trivial
    // singleton.
    assert_eq!(
        postcondition(&ht!((5, 3)), &ht!((3, 5))),
        layout(ht!((1)), ht!((0)))
    );
    assert_eq!(
        postcondition(&ht!((7, 11)), &ht!((11, 7))),
        layout(ht!((1)), ht!((0)))
    );
}

#[test]
fn singleton_leaves_are_skipped() {
    // Leading / interior `1`s in either shape are transparent to the
    // walk.
    let r1 = greatest_common_domain(&ht!((5, 3, 3, 4)), &ht!((10, 6, 3)));
    let r2 = greatest_common_domain(&ht!((1, 5, 3, 3, 4)), &ht!((10, 6, 3)));
    let r3 = greatest_common_domain(&ht!((5, 3, 3, 4)), &ht!((10, 6, 1, 3)));
    let r4 = greatest_common_domain(&ht!((5, 1, 1, 3, 1, 3, 4)), &ht!((10, 6, 3)));
    assert_eq!(r1, r2);
    assert_eq!(r1, r3);
    assert_eq!(r1, r4);
}

#[test]
fn depends_only_on_shape() {
    // The result is determined by the shape alone; the strides beside
    // it are never read.
    let l1 = layout(ht!((5, 3, 4)), ht!((1, 5, 15)));
    let l2 = layout(ht!((5, 3, 4)), ht!((12, 4, 1)));
    let l3 = layout(ht!((5, 3, 4)), ht!((0, 0, 0)));
    let b = layout(ht!((10, 6)), ht!((1, 10)));

    let want = greatest_common_domain(&ht!((5, 3, 4)), &ht!((10, 6)));
    assert_eq!(greatest_common_domain(&l1.shape, &b.shape), want);
    assert_eq!(greatest_common_domain(&l2.shape, &b.shape), want);
    assert_eq!(greatest_common_domain(&l3.shape, &b.shape), want);
}

#[test]
fn nested() {
    // Nested shapes share the same leaf sequence.
    assert_eq!(
        greatest_common_domain(&ht!((2, (3, 4))), &ht!((6, 4))),
        greatest_common_domain(&ht!((2, 3, 4)), &ht!((24)))
    );
}
