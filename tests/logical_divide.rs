//! Ported from `test/test_logical_divide.py`.
//!
//! The Python file is written around `zipped_divide`, which is not ported
//! yet — it promotes the tiler through `tiler_to_layout` before handing it
//! to `logical_divide`. Its `postcondition_zipped_divide` says as much:
//! the `(Tile, Grid)` post-conditions hold for any tiler under
//! `zipped_divide`, and for `logical_divide` only when `B` is a single
//! layout. Every case `test_zipped_divide` and `test_zipped_divide_coord`
//! pass *is* a single layout, so those two run here against
//! [`postcondition_logical_divide`], the same post-condition narrowed to
//! that case.
//!
//! What a later `zipped_divide` pass should revisit:
//!
//! -- `test_zipped_divide` / `test_zipped_divide_coord` under the tiler
//!    promotion, i.e. the post-condition as PyCuTe states it.
//! -- `test_zipped_divide_associativity` (the SM70 8x8x4 example), which
//!    is entirely about `zipped_divide` and has no `logical_divide` form.
//! -- The two halves of `test_logical_divide_tiler` that call
//!    `postcondition_zipped_divide` and that assert `logical_divide !=
//!    zipped_divide` on a multi-mode tiler. The by-mode half is ported.
//!
//! `test_logical_divide_types` is mostly the `algebra.py` facade —
//! promoting `A` from an int, a tuple or `None`, and the `Tensor`
//! overloads. Neither the facade nor `Tensor` is ported; the two lines
//! that exercise `_logical_divide`'s own dispatch head are here, in
//! [`the_dispatch_head_reads_every_tiler_form`], along with the rest of
//! that head. Its expected values were read off the reference
//! implementation.
//!
//! One post-condition of `postcondition_zipped_divide` is dropped:
//! `weakly_congruent(R[1], coprofile(tiler))`. It passes a `Layout` where
//! `weakly_congruent` wants a profile, and `is_tuple` reads a `Layout` as
//! a leaf — a leaf coarsens anything, so the assertion holds vacuously
//! for every case in the file. Restated over `shape(R[1])` it is false in
//! either direction: `logical_divide` extends the complement over
//! `shape(A)`, not over the tiler's codomain, so
//! `Layout((8, 8), (9, 1)) / Layout(4, E(0))` has a rank-2 Grid against a
//! rank-1 coprofile. The remaining three post-conditions are asserted.
#![expect(clippy::unwrap_used, reason = "a test asserts the happy path")]

use pinstripe::{
    HTuple, Int, IntTuple, Layout, OptTiler, Stride, StrideScalar, TilerLeaf, atuple::as_tuple,
    compatible, e, ht, make_layout, size, tiler_to_layout,
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

/// A layout as the right-hand side of a divide.
fn tiler(x: &Layout) -> OptTiler {
    HTuple::Leaf(Some(TilerLeaf::Layout(x.clone())))
}

/// An extent as the right-hand side of a divide. PyCuTe's `N`, which the
/// dispatch head promotes to `N:1`.
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
/// `shape`. PyCuTe's `ArithTuple._idx2crd`, as `tests/composition.rs`
/// writes it: `tiler(i)` may be an arithmetic tuple whose trailing
/// positions are implicitly zero, and [`Layout::call`] takes a plain
/// [`IntTuple`].
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

/// PyCuTe's `postcondition_zipped_divide`, narrowed to the single-layout
/// `B` on which `logical_divide` agrees with `zipped_divide`.
///
/// The result splits into a Tile mode and a Grid mode: the Tile is a
/// composition with the tiler, and between the two of them they still
/// reach every element of the source.
fn postcondition_logical_divide(a: &Layout, b: &Layout) {
    let r = a.logical_divide(&tiler(b)).unwrap();
    // PyCuTe's `tiler_to_layout(B)`. Over a layout leaf it rescales the
    // strides by the default `e = 1`, so the tiler is `b` itself.
    let tiler = tiler_to_layout(
        &HTuple::Leaf(TilerLeaf::Layout(b.clone())),
        &StrideScalar::Int(1),
    )
    .unwrap();

    // Post-condition: the rank is (Tile, Grid).
    assert_eq!(r.shape.rank(), 2, "{a} / {b} => {r}");

    // Post-condition: the Tile mode is composition with the tiler.
    assert!(compatible(&tiler.shape, &r.index(0).unwrap().shape));
    for i in 0..size(&tiler.shape, &[]).unwrap() {
        assert_eq!(
            r.call(&ht!((i, 0))).unwrap(),
            a.call(&as_coord(&tiler.call(&ht!(i)).unwrap(), &a.shape))
                .unwrap(),
            "tile: {a} / {b} => {r} at {i}"
        );
    }

    // Post-condition: every element of A appears in R as well.
    let size_r = size(&r.shape, &[]).unwrap();
    for i in 0..size(&a.shape, &[]).unwrap() {
        let value = a.call(&ht!(i)).unwrap();
        assert!(
            (0..size_r).any(|j| r.call(&ht!(j)).is_ok_and(|x| x == value)),
            "cover: {a} / {b} => {r} at {i}"
        );
    }
}

// ---------------------------------------------------------------------------
// integer strides
// ---------------------------------------------------------------------------

/// PyCuTe's `test_zipped_divide`, over the single-layout tilers on which
/// `logical_divide` meets the same post-condition.
#[test]
fn logical_divide_splits_into_a_tile_and_a_grid() {
    for (a, b) in [
        (layout(ht!(1), ht!(0)), layout(ht!(1), ht!(0))),
        (layout(ht!(1), ht!(0)), layout(ht!(1), ht!(1))),
        (layout(ht!(1), ht!(1)), layout(ht!(1), ht!(0))),
        (layout(ht!(1), ht!(1)), layout(ht!(1), ht!(1))),
        (layout(ht!(6), ht!(1)), layout(ht!(2), ht!(1))),
        (layout(ht!(6), ht!(1)), layout(ht!(2), ht!(3))),
        (layout(ht!(6), ht!(1)), layout(ht!((2, 3)), ht!((3, 1)))),
        (layout(ht!(6), ht!(2)), layout(ht!(2), ht!(1))),
        (layout(ht!(6), ht!(2)), layout(ht!(2), ht!(3))),
        (layout(ht!(6), ht!(2)), layout(ht!((2, 3)), ht!((3, 1)))),
        (
            layout(ht!((6, 6)), ht!((1, 12))),
            layout(ht!((6, 3)), ht!((3, 1))),
        ),
        (
            layout(ht!((6, 6)), ht!((12, 1))),
            layout(ht!((6, 3)), ht!((3, 1))),
        ),
        (layout(ht!(32), ht!(1)), layout(ht!(2), ht!(8))),
        (layout(ht!((4, 1)), ht!((1, 1))), layout(ht!(2), ht!(1))),
        (layout(ht!((4, 1)), ht!((1, 1))), layout(ht!(2), ht!(2))),
        (layout(ht!((8, 8)), ht!((1, 8))), layout(ht!(32), ht!(2))),
        (layout(ht!((8, 8)), ht!((8, 1))), layout(ht!(32), ht!(2))),
    ] {
        postcondition_logical_divide(&a, &b);
    }
}

// ---------------------------------------------------------------------------
// basis strides
// ---------------------------------------------------------------------------

/// PyCuTe's `test_zipped_divide_coord`, again over the single-layout
/// tilers.
#[test]
fn logical_divide_splits_a_coordinate_codomain() {
    let a = layout(ht!((8, 8)), ht!((9, 1)));
    for b in [
        strided(ht!(4), s(e(&[0]))),
        strided(ht!(4), s(e(&[1]))),
        strided(ht!(4), s(e(&[0]).scale(2))),
        strided(ht!(4), s(e(&[1]).scale(2))),
        strided(ht!((4, 4)), t(vec![s(e(&[1])), s(e(&[0]).scale(2))])),
        strided(
            ht!((5, 7)),
            t(vec![s(e(&[1]).scale(3)), s(e(&[0]).scale(2))]),
        ),
    ] {
        postcondition_logical_divide(&a, &b);
    }
}

// ---------------------------------------------------------------------------
// by-mode tilers
// ---------------------------------------------------------------------------

/// PyCuTe's `test_logical_divide_tiler`, less the two halves that need
/// `zipped_divide`: a multi-mode tiler makes `logical_divide` act exactly
/// by-mode.
#[test]
fn a_tiler_tuple_divides_by_mode() {
    for (a, modes) in [
        (
            layout(ht!((6, 4)), ht!((4, 1))),
            [layout(ht!(2), ht!(1)), layout(ht!(2), ht!(1))],
        ),
        (
            layout(ht!((8, 8)), ht!((8, 1))),
            [layout(ht!(2), ht!(1)), layout(ht!(4), ht!(1))],
        ),
    ] {
        let expected = make_layout(
            modes
                .iter()
                .enumerate()
                .map(|(i, m)| a.index(i).unwrap().logical_divide(&tiler(m)).unwrap())
                .collect(),
        );
        assert_eq!(
            a.logical_divide(&by_mode(modes.iter().map(tiler).collect()))
                .unwrap(),
            expected
        );
    }
}

// ---------------------------------------------------------------------------
// the dispatch head
// ---------------------------------------------------------------------------

#[test]
fn the_dispatch_head_reads_every_tiler_form() {
    let a = layout(ht!((8, 8)), ht!((8, 1)));

    // RHS None, noop. PyCuTe's `logical_divide(A, None) == A`.
    assert_eq!(a.logical_divide(&none()).unwrap(), a);
    // RHS int, A / N -> A / N:1. PyCuTe's
    // `logical_divide(A, 2) == logical_divide(A, Layout(2, 1))`.
    assert_eq!(
        a.logical_divide(&extent(2)).unwrap(),
        a.logical_divide(&tiler(&layout(ht!(2), ht!(1)))).unwrap()
    );
    assert_eq!(
        a.logical_divide(&extent(2)).unwrap(),
        layout(ht!((2, (4, 8))), ht!((8, (16, 1))))
    );
    // RHS tuple, by-mode.
    assert_eq!(
        a.logical_divide(&by_mode(vec![extent(2), extent(4)]))
            .unwrap(),
        layout(ht!(((2, 4), (4, 2))), ht!(((8, 16), (1, 4))))
    );
    // A per-mode None leaves its mode alone.
    assert_eq!(
        a.logical_divide(&by_mode(vec![tiler(&layout(ht!(2), ht!(1))), none()]))
            .unwrap(),
        layout(ht!(((2, 4), 8)), ht!(((8, 16), 1)))
    );
    assert_eq!(
        a.logical_divide(&by_mode(vec![none(), tiler(&layout(ht!(4), ht!(1)))]))
            .unwrap(),
        layout(ht!((8, (4, 2))), ht!((8, (1, 4))))
    );
    // A tiler the layout outranks runs out under `zip_longest`, and the
    // modes it does not reach take the no-op.
    assert_eq!(
        a.logical_divide(&by_mode(vec![tiler(&layout(ht!(2), ht!(1)))]))
            .unwrap(),
        layout(ht!(((2, 4), 8)), ht!(((8, 16), 1)))
    );
    // A tiler tuple that outranks the mode it faces is rejected: mode 0
    // of `a` is rank 1, and `(2, 2)` is rank 2.
    assert!(
        a.logical_divide(&by_mode(vec![
            by_mode(vec![extent(2), extent(2)]),
            extent(4),
        ]))
        .is_err()
    );
}
