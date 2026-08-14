//! Ported from `test/test_nullspace.py`.
//!
//! The Python post-condition is here in full. The two cases of
//! `test_nullspace_coord` that build their layout with `composition` —
//! the SM70 MMA 8x8x4 A TV and SM80 MMA 16x8 entries — wait on that
//! function.
#![expect(clippy::unwrap_used, reason = "a test asserts the happy path")]

use pinstripe::{
    HTuple, Int, IntTuple, Layout, Stride, StrideScalar, atuple::as_tuple, e, ht, size,
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

/// `Layout(shape, stride)` over an already-built stride.
fn strided(shape: IntTuple, stride: Stride) -> Layout {
    Layout::set(shape, stride)
}

/// A stride leaf.
fn s(x: StrideScalar) -> Stride {
    HTuple::Leaf(x)
}

/// A stride mode.
fn t(modes: Vec<Stride>) -> Stride {
    HTuple::Tuple(modes)
}

/// PyCuTe's `postcondition_nullspace`: every coordinate the nullspace
/// produces maps to zero.
fn postcondition_nullspace(source: &Layout) {
    let null = source.nullspace().unwrap();
    for i in 0..size(&null.shape, &[]).unwrap() {
        let n = null.call(&HTuple::Leaf(i)).unwrap();
        assert_eq!(
            source.call(&as_tuple(&n)).unwrap(),
            StrideScalar::Int(0),
            "{source} => {null} is non-zero at {i}"
        );
    }
}

// ---------------------------------------------------------------------------
// nullspace
// ---------------------------------------------------------------------------

#[test]
fn nullspace_maps_to_zero_over_integer_strides() {
    for source in [
        layout(ht!(1), ht!(0)),
        layout(ht!(1), ht!(1)),
        layout(ht!(1), ht!(2)),
        layout(ht!(1), ht!(4)),
        layout(ht!(4), ht!(0)),
        layout(ht!(4), ht!(1)),
        layout(ht!(4), ht!(2)),
        layout(ht!(4), ht!(4)),
        layout(ht!((1, 1)), ht!((0, 0))),
        layout(ht!((3, 7)), ht!((0, 0))),
        layout(ht!((2, 4)), ht!((0, 2))),
        layout(ht!((8, 4)), ht!((1, 8))),
        layout(ht!((8, 4)), ht!((4, 1))),
        layout(ht!((2, 4, 6)), ht!((1, 2, 0))),
        layout(ht!((2, 4, 6)), ht!((0, 1, 0))),
        layout(ht!((4, 2)), ht!((1, 16))),
        layout(ht!((4, 5, 6)), ht!((1, 1, 0))),
        layout(ht!((7, 5, 9)), ht!((2, 0, 1))),
        layout(ht!((7, 5, 9)), ht!((2, 0, 0))),
        layout(ht!((7, 5, 9)), ht!((0, 0, 0))),
    ] {
        postcondition_nullspace(&source);
    }
}

#[test]
fn nullspace_maps_to_zero_over_basis_strides() {
    for source in [
        strided(ht!((4, 5)), t(vec![s(e(&[0])), s(e(&[1]))])),
        strided(ht!((4, 5)), t(vec![s(StrideScalar::Int(0)), s(e(&[0]))])),
        strided(ht!((4, 5)), t(vec![s(StrideScalar::Int(0)), s(e(&[4, 1]))])),
        strided(
            ht!((4, 5)),
            t(vec![s(e(&[0]).scale(2)), s(StrideScalar::Int(0))]),
        ),
        // SM70 MMA 8x8x4 C TV inverse.
        strided(
            ht!(((2, 2, 2), (2, 2, 2))),
            t(vec![
                t(vec![s(e(&[0])), s(e(&[1]).scale(2)), s(e(&[0]).scale(4))]),
                t(vec![s(e(&[1])), s(e(&[0]).scale(2)), s(e(&[1]).scale(4))]),
            ]),
        ),
        strided(
            ht!(((2, 2, 2), (2, 2, 2))),
            t(vec![
                t(vec![s(e(&[0])), s(e(&[1]).scale(2)), s(e(&[0]).scale(5))]),
                t(vec![
                    s(e(&[1])),
                    s(StrideScalar::Int(0)),
                    s(e(&[1]).scale(5)),
                ]),
            ]),
        ),
        strided(
            ht!(((2, 2, 2), (2, 2, 2))),
            t(vec![
                t(vec![
                    s(e(&[0])),
                    s(StrideScalar::Int(0)),
                    s(e(&[0]).scale(5)),
                ]),
                t(vec![s(e(&[1])), s(e(&[0]).scale(2)), s(e(&[1]).scale(4))]),
            ]),
        ),
    ] {
        postcondition_nullspace(&source);
    }
}

#[test]
fn nullspace_is_trivial_without_a_stride_zero_mode() {
    assert_eq!(
        layout(ht!((8, 4)), ht!((4, 1))).nullspace().unwrap(),
        layout(ht!(1), ht!(0))
    );
    // The one stride-0 mode, at its position in the domain.
    assert_eq!(
        layout(ht!((2, 4, 6)), ht!((1, 2, 0))).nullspace().unwrap(),
        layout(ht!(6), ht!(8))
    );
}
