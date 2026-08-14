//! Ported from `test/test_layout.py`, `test/test_make_layout.py`, and
//! `test/test_recast.py`.
//!
//! `test_zipped_divide_equals_logical_divide_with_tiler` needs
//! `zipped_divide` and `logical_divide`, which are not ported yet, so it
//! waits.
//!
//! The `test_sympy` and `test_sympy_substitution` cases are gone for
//! good: [`Int`] is the only integer here, so there is no symbolic
//! stride to order past the static ones.
//!
//! `test_call_with_packed_or_unpacked_coord` is likewise gone. It
//! asserts that `A(c)` and `A(*c)` agree, and Rust has no varargs — the
//! coordinate always arrives whole.
#![expect(clippy::unwrap_used, reason = "a test asserts the happy path")]
#![expect(clippy::panic, reason = "a test asserts the happy path")]

use std::{cmp::Ordering, collections::BTreeSet};

use pinstripe::{
    HTuple, Int, IntTuple, Layout, OptTiler, Scale, Stride, StrideScalar, Tiler, TilerLeaf,
    coprofile, coshape, e, ht, make_layout, make_layout_like, make_ordered_layout, recast,
    tiler_to_layout,
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

/// A slicing coordinate: `None` marks a retained mode. `ht!` cannot
/// spell it, because `Some(1)` is more than one token tree.
fn crd(modes: Vec<Option<Int>>) -> HTuple<Option<Int>> {
    HTuple::Tuple(modes.into_iter().map(HTuple::Leaf).collect())
}

/// `L(i)` as an integer.
fn offset(l: &Layout, crd: IntTuple) -> Int {
    match l.call(&crd).unwrap() {
        StrideScalar::Int(v) => v,
        other => panic!("{other:?} is not an integer offset"),
    }
}

/// The tiler of a plain shape.
fn int_tiler(shape: &IntTuple) -> Tiler {
    shape.transform_leaf(&|v: &Int| TilerLeaf::Int(*v))
}

/// The default `e` of `tiler_to_layout`.
fn one() -> StrideScalar {
    StrideScalar::Int(1)
}

/// The tiler as the right-hand side of a composition, which admits a
/// per-mode `None` the tiler itself has no leaf for.
fn opt(tiler: &Tiler) -> OptTiler {
    tiler.transform_leaf(&|leaf: &TilerLeaf| Some(leaf.clone()))
}

/// A layout as the right-hand side of a composition.
fn as_rhs(x: &Layout) -> OptTiler {
    HTuple::Leaf(Some(TilerLeaf::Layout(x.clone())))
}

// ---------------------------------------------------------------------------
// test_layout.py — TestLayout
// ---------------------------------------------------------------------------

#[test]
fn layout_indexes_evaluates_and_rebuilds() {
    let a = layout(ht!((3, (2, 4))), ht!((2, (1, 6))));

    assert_eq!(a.shape.product(), 24);
    assert_eq!(a.index(0).unwrap(), layout(ht!(3), ht!(2)));
    assert_eq!(a.index(1).unwrap(), layout(ht!((2, 4)), ht!((1, 6))));
    assert_eq!(
        a.index(1).unwrap().index(0).unwrap(),
        layout(ht!(2), ht!(1))
    );

    assert_eq!(
        a,
        make_layout(vec![
            layout(ht!(3), ht!(2)),
            make_layout(vec![layout(ht!(2), ht!(1)), layout(ht!(4), ht!(6))]),
        ])
    );

    let r = [
        0, 2, 4, 1, 3, 5, 6, 8, 10, 7, 9, 11, 12, 14, 16, 13, 15, 17, 18, 20, 22, 19, 21, 23,
    ];
    let rows = a.index(0).unwrap().shape.product();
    let cols = a.index(1).unwrap().shape.product();
    for i in 0..rows {
        for j in 0..cols {
            let flat = i + j * rows;
            assert_eq!(r[flat as usize], offset(&a, ht!(flat)));
            assert_eq!(r[flat as usize], offset(&a, ht!((i, j))));
        }
    }
}

#[test]
fn get_reads_the_sublayout_at_a_nested_mode() {
    let a = layout(ht!((3, (2, 4))), ht!((2, (1, 6))));
    assert_eq!(a.get(&[]).unwrap(), a);
    assert_eq!(a.get(&[1]).unwrap(), layout(ht!((2, 4)), ht!((1, 6))));
    assert_eq!(a.get(&[1, 1]).unwrap(), layout(ht!(4), ht!(6)));
    assert!(a.get(&[2]).is_err());
    assert!(a.index(2).is_err());
}

// ---------------------------------------------------------------------------
// test_layout.py — TestLayoutConstruction
// ---------------------------------------------------------------------------

#[test]
fn default_stride_is_column_major() {
    assert_eq!(compact(ht!(8)), layout(ht!(8), ht!(1)));
    assert_eq!(compact(ht!((4, 8))), layout(ht!((4, 8)), ht!((1, 4))));
    assert_eq!(
        compact(ht!((3, (2, 4)))),
        layout(ht!((3, (2, 4))), ht!((1, (3, 6))))
    );
}

#[test]
fn base_stride_scales() {
    let base = |shape: IntTuple, k: Int| Layout::new(shape, &HTuple::Leaf(StrideScalar::Int(k)));
    assert_eq!(
        base(ht!((4, 8)), 2).unwrap(),
        layout(ht!((4, 8)), ht!((2, 8)))
    );
    assert_eq!(
        base(ht!((3, (2, 4))), 5).unwrap(),
        layout(ht!((3, (2, 4))), ht!((5, (15, 30))))
    );
}

#[test]
fn explicit_stride_is_used_as_is() {
    assert_eq!(
        layout(ht!((4, 8)), ht!((8, 1))).stride,
        as_stride(&ht!((8, 1)))
    );
    assert_eq!(
        layout(ht!((3, (2, 4))), ht!((24, (1, 6)))).stride,
        as_stride(&ht!((24, (1, 6))))
    );
}

// ---------------------------------------------------------------------------
// test_layout.py — TestThreeCoordinateForms
// ---------------------------------------------------------------------------

#[test]
fn the_three_coordinate_forms_agree() {
    let a = layout(ht!((3, (2, 4))), ht!((2, (1, 6))));
    let rows = a.index(0).unwrap().shape.product();
    let inner = a.index(1).unwrap().index(0).unwrap().shape.product();
    for i in 0..a.shape.product() {
        let (c0, c1) = (i % rows, i / rows);
        let (n0, n1) = (c1 % inner, c1 / inner);
        assert_eq!(offset(&a, ht!(i)), offset(&a, ht!((c0, c1))));
        assert_eq!(offset(&a, ht!(i)), offset(&a, ht!((c0, (n0, n1)))));
    }
}

#[test]
fn an_out_of_bounds_integral_coordinate_leaves_the_image() {
    let a = layout(ht!((3, (2, 4))), ht!((2, (1, 6))));
    assert_eq!(offset(&a, ht!(100)), 99);
}

// ---------------------------------------------------------------------------
// test_layout.py — TestLayoutSlicing
// ---------------------------------------------------------------------------

#[test]
fn a_full_coordinate_slices_away_every_mode() {
    let a = layout(ht!((4, 4)), ht!((4, 1)));
    let (off, sub) = a.offset_and_slice(&crd(vec![Some(1), Some(2)])).unwrap();
    assert_eq!(off, a.call(&ht!((1, 2))).unwrap());
    assert_eq!(sub.shape.rank(), 0);
}

#[test]
fn a_partial_coordinate_returns_the_residual_layout() {
    let a = layout(ht!((4, 4)), ht!((4, 1)));

    let (off, sub) = a.offset_and_slice(&crd(vec![Some(1), None])).unwrap();
    assert_eq!(off, StrideScalar::Int(4));
    assert_eq!(sub, layout(ht!((4)), ht!((1))));

    let (off, sub) = a.offset_and_slice(&crd(vec![None, Some(2)])).unwrap();
    assert_eq!(off, StrideScalar::Int(2));
    assert_eq!(sub, layout(ht!((4)), ht!((4))));
}

#[test]
fn an_all_open_coordinate_returns_the_whole_layout() {
    let a = layout(ht!((4, 4)), ht!((4, 1)));
    let (off, sub) = a.offset_and_slice(&crd(vec![None, None])).unwrap();
    assert_eq!(off, StrideScalar::Int(0));
    assert_eq!(sub, a);
}

// ---------------------------------------------------------------------------
// test_layout.py — TestLayoutEquality
// ---------------------------------------------------------------------------

#[test]
fn equality_is_structural() {
    assert_eq!(compact(ht!((3, 4))), layout(ht!((3, 4)), ht!((1, 3))));
    assert_ne!(compact(ht!((3, 4))), compact(ht!((4, 3))));
    assert_ne!(
        layout(ht!((3, 4)), ht!((1, 3))),
        layout(ht!((3, 4)), ht!((4, 1)))
    );
}

#[test]
fn functionally_equivalent_layouts_may_differ_structurally() {
    let a = layout(ht!((4, 8)), ht!((1, 4)));
    let b = layout(ht!(32), ht!(1));
    assert_ne!(a, b);
    for i in 0..32 {
        assert_eq!(offset(&a, ht!(i)), offset(&b, ht!(i)));
    }
}

// ---------------------------------------------------------------------------
// test_layout.py — TestCoshape
// ---------------------------------------------------------------------------

#[test]
fn an_integer_stride_gives_an_integer_coshape() {
    let of = |l: Layout| coshape(&l, &[]).unwrap();
    assert_eq!(of(layout(ht!((4, 8)), ht!((1, 4)))), ht!(32));
    assert_eq!(of(layout(ht!((4, 8)), ht!((8, 1)))), ht!(32));
    assert_eq!(of(layout(ht!(8), ht!(1))), ht!(8));
    // Broadcast.
    assert_eq!(of(layout(ht!(8), ht!(0))), ht!(1));
}

#[test]
fn a_coordinate_stride_gives_a_tuple_coshape() {
    let of = |l: Layout| coshape(&l, &[]).unwrap();
    assert_eq!(
        of(strided(ht!((4, 8)), t(vec![s(e(&[0])), s(e(&[1]))]))),
        ht!((4, 8))
    );
    assert_eq!(
        of(strided(ht!((4, 8)), t(vec![s(e(&[1])), s(e(&[0]))]))),
        ht!((8, 4))
    );
}

#[test]
fn coprofile_matches_coshape() {
    let l = layout(ht!((4, 8)), ht!((1, 4)));
    assert_eq!(coprofile(&l, &[]).unwrap(), coshape(&l, &[]).unwrap());
}

// ---------------------------------------------------------------------------
// Printing — PyCuTe's `__str__` and `__repr__`, which its tests leave
// unasserted.
// ---------------------------------------------------------------------------

#[test]
fn a_layout_prints_as_shape_colon_stride() {
    let a = layout(ht!((3, (2, 4))), ht!((2, (1, 6))));
    assert_eq!(format!("{a}"), "(3,(2,4)):(2,(1,6))");
    assert_eq!(format!("{a:?}"), "Layout((3,(2,4)), (2,(1,6)))");
}

// ---------------------------------------------------------------------------
// test_make_layout.py — TestMakeLayout
// ---------------------------------------------------------------------------

#[test]
fn make_layout_concatenates_modes() {
    let a = layout(ht!(3), ht!(1));
    let b = layout(ht!(4), ht!(3));
    assert_eq!(make_layout(vec![a, b]), layout(ht!((3, 4)), ht!((1, 3))));
}

#[test]
fn a_nested_make_layout_is_hierarchical() {
    let l = make_layout(vec![
        layout(ht!(3), ht!(1)),
        make_layout(vec![layout(ht!(2), ht!(1)), layout(ht!(4), ht!(6))]),
        layout(ht!(2), ht!(42)),
    ]);
    assert_eq!(l, layout(ht!((3, (2, 4), 2)), ht!((1, (1, 6), 42))));
}

#[test]
fn a_layout_round_trips_through_index_and_make_layout() {
    let a = layout(ht!((3, (2, 4))), ht!((2, (1, 6))));
    let rebuilt = make_layout(vec![
        a.index(0).unwrap(),
        make_layout(vec![
            a.index(1).unwrap().index(0).unwrap(),
            a.index(1).unwrap().index(1).unwrap(),
        ]),
    ]);
    assert_eq!(a, rebuilt);
}

// ---------------------------------------------------------------------------
// test_make_layout.py — TestTilerToLayout
// ---------------------------------------------------------------------------

#[test]
fn an_integer_tiler_is_a_stride_one_layout() {
    assert_eq!(
        tiler_to_layout(&HTuple::Leaf(TilerLeaf::Int(3)), &one()).unwrap(),
        layout(ht!(3), ht!(1))
    );
}

#[test]
fn a_layout_tiler_is_itself() {
    let l = layout(ht!((7, 2)), ht!((3, 1)));
    assert_eq!(
        tiler_to_layout(&HTuple::Leaf(TilerLeaf::Layout(l.clone())), &one()).unwrap(),
        l
    );
}

#[test]
fn a_shape_tiler_becomes_a_coordinate_layout() {
    assert_eq!(
        tiler_to_layout(&int_tiler(&ht!((4, 5))), &one()).unwrap(),
        strided(ht!((4, 5)), t(vec![s(e(&[0])), s(e(&[1]))]))
    );
    assert_eq!(
        tiler_to_layout(&int_tiler(&ht!((2, 3, 5))), &one()).unwrap(),
        strided(ht!((2, 3, 5)), t(vec![s(e(&[0])), s(e(&[1])), s(e(&[2]))]))
    );
}

#[test]
fn a_tuple_of_layout_tilers_scales_each_by_its_basis() {
    let tiler = HTuple::Tuple(vec![
        HTuple::Leaf(TilerLeaf::Layout(layout(ht!(4), ht!(2)))),
        HTuple::Leaf(TilerLeaf::Layout(layout(ht!(5), ht!(3)))),
    ]);
    assert_eq!(
        tiler_to_layout(&tiler, &one()).unwrap(),
        strided(
            ht!((4, 5)),
            t(vec![s(e(&[0]).scale(2)), s(e(&[1]).scale(3))])
        )
    );
}

/// The defining post-condition of `tiler_to_layout`: composing with a
/// tiler equals composing with the layout it stands for.
#[test]
fn a_tiler_composes_as_its_layout_does() {
    let a = layout(ht!((12, (4, 8))), ht!((59, (13, 1))));
    let tilers = [
        int_tiler(&ht!((3, 8))),
        HTuple::Tuple(vec![
            HTuple::Leaf(TilerLeaf::Layout(layout(ht!(3), ht!(4)))),
            HTuple::Leaf(TilerLeaf::Layout(layout(ht!(8), ht!(1)))),
        ]),
    ];
    for tiler in &tilers {
        assert_eq!(
            a.composition(&opt(tiler)).unwrap(),
            a.composition(&as_rhs(&tiler_to_layout(tiler, &one()).unwrap()))
                .unwrap()
        );
    }
}

// ---------------------------------------------------------------------------
// test_make_layout.py — TestMakeLayoutLike
// ---------------------------------------------------------------------------

/// The structural post-conditions of `make_layout_like`. PyCuTe's
/// `postcondition_make_layout_like`.
fn postcondition_make_layout_like(l: &Layout) {
    let result = make_layout_like(l).unwrap();

    // The shape is preserved exactly, hierarchy included.
    assert_eq!(result.shape, l.shape);

    // Idempotence: the result is already in canonical form.
    assert_eq!(make_layout_like(&result).unwrap(), result);

    let src = l.stride.leaves();
    let dst = result.stride.leaves();
    let shp = l.shape.leaves();

    // A stride-0 mode carries no information and is pinned to stride 0;
    // every other mode is non-zero.
    for (sd, dd) in src.iter().zip(&dst) {
        assert_eq!(sd.is_zero(), dd.is_zero());
    }

    // The non-trivial modes form a compact layout whose strides are the
    // prefix products of the shapes taken in ascending order of the
    // *source* stride.
    let mut modes = shp
        .iter()
        .zip(src.iter())
        .zip(dst.iter())
        .map(|((&&s, &sd), &dd)| (s, sd, dd))
        .filter(|(s, sd, _)| *s != 1 && !sd.is_zero())
        .collect::<Vec<_>>();
    modes.sort_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal));
    let mut current = 1;
    for (s, _, dd) in modes {
        assert_eq!(*dd, StrideScalar::Int(current));
        current *= s;
    }

    // The codomain of the non-broadcast modes is exactly the contiguous
    // range [0, cosize) — no gaps, no overlaps.
    let image = (0..result.shape.product())
        .map(|i| offset(&result, ht!(i)))
        .collect::<BTreeSet<_>>();
    assert!(image.iter().copied().eq(0..image.len() as Int));
}

#[test]
fn a_compact_layout_is_returned_unchanged() {
    let like = |l: Layout| make_layout_like(&l).unwrap();
    assert_eq!(like(layout(ht!(8), ht!(1))), layout(ht!(8), ht!(1)));
    assert_eq!(
        like(layout(ht!((4, 8)), ht!((1, 4)))),
        layout(ht!((4, 8)), ht!((1, 4)))
    );
    assert_eq!(
        like(layout(ht!((2, 3, 4)), ht!((1, 2, 6)))),
        layout(ht!((2, 3, 4)), ht!((1, 2, 6)))
    );
    assert_eq!(
        like(layout(ht!((3, 4)), ht!((4, 1)))),
        layout(ht!((3, 4)), ht!((4, 1)))
    );
}

#[test]
fn non_compact_strides_are_repacked_in_order() {
    let like = |l: Layout| make_layout_like(&l).unwrap();
    assert_eq!(
        like(layout(ht!((4, 8)), ht!((100, 1)))),
        layout(ht!((4, 8)), ht!((8, 1)))
    );
    assert_eq!(
        like(layout(ht!((2, 3, 4)), ht!((1, 100, 20)))),
        layout(ht!((2, 3, 4)), ht!((1, 8, 2)))
    );
    assert_eq!(
        like(layout(ht!((3, 4, 5)), ht!((1, 100, 10)))),
        layout(ht!((3, 4, 5)), ht!((1, 15, 3)))
    );
}

#[test]
fn the_cute_documented_examples_of_make_layout_like() {
    let like = |l: Layout| make_layout_like(&l).unwrap();
    assert_eq!(
        like(layout(ht!((2, 2, 2, 2)), ht!((0, 2, 4, 1)))),
        layout(ht!((2, 2, 2, 2)), ht!((0, 2, 4, 1)))
    );
    assert_eq!(
        like(layout(ht!((2, 3, 4, 5)), ht!((0, 42, 1, 0)))),
        layout(ht!((2, 3, 4, 5)), ht!((0, 4, 1, 0)))
    );
}

#[test]
fn stride_zero_modes_stay_stride_zero() {
    let like = |l: Layout| make_layout_like(&l).unwrap();
    assert_eq!(like(layout(ht!(1), ht!(0))), layout(ht!(1), ht!(0)));
    assert_eq!(like(layout(ht!(8), ht!(0))), layout(ht!(8), ht!(0)));
    assert_eq!(
        like(layout(ht!((3, 7)), ht!((0, 0)))),
        layout(ht!((3, 7)), ht!((0, 0)))
    );
    assert_eq!(
        like(layout(ht!((8, 4)), ht!((0, 2)))),
        layout(ht!((8, 4)), ht!((0, 1)))
    );
    assert_eq!(
        like(layout(ht!((8, 4, 6)), ht!((1, 0, 2)))),
        layout(ht!((8, 4, 6)), ht!((1, 0, 8)))
    );
}

#[test]
fn size_one_modes_keep_their_slot_in_the_packing() {
    let like = |l: Layout| make_layout_like(&l).unwrap();
    assert_eq!(
        like(layout(ht!((1, 4)), ht!((7, 2)))),
        layout(ht!((1, 4)), ht!((4, 1)))
    );
    assert_eq!(
        like(layout(ht!((4, 1, 8)), ht!((1, 5, 4)))),
        layout(ht!((4, 1, 8)), ht!((1, 32, 4)))
    );
    assert_eq!(
        like(layout(ht!((1, 1)), ht!((5, 7)))),
        layout(ht!((1, 1)), ht!((1, 1)))
    );
}

#[test]
fn make_layout_like_packs_across_a_nested_shape() {
    let like = |l: Layout| make_layout_like(&l).unwrap();
    assert_eq!(
        like(layout(ht!(((2, 2), (2, 2))), ht!(((1, 4), (8, 32))))),
        layout(ht!(((2, 2), (2, 2))), ht!(((1, 2), (4, 8))))
    );
    assert_eq!(
        like(layout(ht!((2, (3, 4))), ht!((50, (1, 8))))),
        layout(ht!((2, (3, 4))), ht!((12, (1, 3))))
    );
}

#[test]
fn make_layout_like_postconditions() {
    let layouts = [
        layout(ht!(8), ht!(1)),
        layout(ht!(8), ht!(3)),
        layout(ht!((2, 4)), ht!((1, 2))),
        layout(ht!((2, 4)), ht!((4, 1))),
        layout(ht!((2, 4)), ht!((1, 4))),
        layout(ht!((8, 4)), ht!((1, 8))),
        layout(ht!((8, 4)), ht!((4, 1))),
        layout(ht!((2, 4, 6)), ht!((1, 2, 8))),
        layout(ht!((2, 4, 6)), ht!((4, 1, 8))),
        layout(ht!((2, 4, 8)), ht!((8, 1, 64))),
        layout(ht!((2, 4, 8)), ht!((32, 0, 2))),
        layout(ht!((2, 4, 8)), ht!((2, 0, 32))),
        layout(ht!((2, 4, 4, 4, 2)), ht!((32, 0, 2, 0, 512))),
        layout(ht!(((2, 2), (2, 2))), ht!(((1, 4), (8, 32)))),
        layout(ht!((2, (3, 4))), ht!((3, (1, 6)))),
        layout(ht!((4, 2)), ht!((1, 16))),
        layout(ht!((1, 4)), ht!((7, 2))),
        layout(ht!((3, 7)), ht!((0, 0))),
    ];
    layouts.iter().for_each(postcondition_make_layout_like);
}

#[test]
fn coordinate_strides_repack_in_basis_order() {
    let like = |l: Layout| make_layout_like(&l).unwrap();
    // Identity coordinate layout -> generalized column-major.
    assert_eq!(
        like(strided(
            ht!((2, 3, 4)),
            t(vec![s(e(&[0])), s(e(&[1])), s(e(&[2]))])
        )),
        layout(ht!((2, 3, 4)), ht!((1, 2, 6)))
    );
    // Reversed basis order -> generalized row-major.
    assert_eq!(
        like(strided(
            ht!((2, 3, 4)),
            t(vec![s(e(&[2])), s(e(&[1])), s(e(&[0]))])
        )),
        layout(ht!((2, 3, 4)), ht!((12, 4, 1)))
    );
    // An arbitrary basis permutation.
    assert_eq!(
        like(strided(
            ht!((2, 3, 4)),
            t(vec![s(e(&[1])), s(e(&[2])), s(e(&[0]))])
        )),
        layout(ht!((2, 3, 4)), ht!((4, 8, 1)))
    );
    // The basis position drives the order; a scale only breaks ties
    // between strides sharing a position.
    assert_eq!(
        like(strided(
            ht!((2, 3)),
            t(vec![s(e(&[0]).scale(2)), s(e(&[1]).scale(3))])
        )),
        layout(ht!((2, 3)), ht!((1, 2)))
    );
    assert_eq!(
        like(strided(
            ht!((4, 3)),
            t(vec![s(e(&[0]).scale(3)), s(e(&[0]))])
        )),
        layout(ht!((4, 3)), ht!((3, 1)))
    );
    // A rank-1 coordinate layout collapses to one contiguous mode.
    assert_eq!(like(strided(ht!(4), s(e(&[0])))), layout(ht!(4), ht!(1)));
    assert_eq!(like(strided(ht!(4), s(e(&[1])))), layout(ht!(4), ht!(1)));
}

#[test]
fn a_broadcast_mode_among_coordinate_strides_stays_zero() {
    let like = |l: Layout| make_layout_like(&l).unwrap();
    assert_eq!(
        like(strided(
            ht!((2, 3, 4)),
            t(vec![s(e(&[0])), s(StrideScalar::Int(0)), s(e(&[1]))])
        )),
        layout(ht!((2, 3, 4)), ht!((1, 0, 2)))
    );
    assert_eq!(
        like(strided(
            ht!((2, 3)),
            t(vec![s(StrideScalar::Int(0)), s(e(&[0]))])
        )),
        layout(ht!((2, 3)), ht!((0, 1)))
    );
}

#[test]
fn nested_coordinate_strides_pack_in_colex_order_of_their_paths() {
    let like = |l: Layout| make_layout_like(&l).unwrap();
    assert_eq!(
        like(strided(
            ht!((2, (3, 4))),
            t(vec![s(e(&[0])), t(vec![s(e(&[1])), s(e(&[2]))])])
        )),
        layout(ht!((2, (3, 4))), ht!((1, (2, 6))))
    );
    assert_eq!(
        like(strided(
            ht!((2, 3, 4)),
            t(vec![s(e(&[0, 0])), s(e(&[0, 1])), s(e(&[1]))])
        )),
        layout(ht!((2, 3, 4)), ht!((1, 2, 6)))
    );
}

#[test]
fn coordinate_strided_postconditions() {
    let layouts = [
        strided(ht!(4), s(e(&[0]))),
        strided(ht!((2, 3, 4)), t(vec![s(e(&[0])), s(e(&[1])), s(e(&[2]))])),
        strided(ht!((2, 3, 4)), t(vec![s(e(&[2])), s(e(&[1])), s(e(&[0]))])),
        strided(ht!((2, 3, 4)), t(vec![s(e(&[1])), s(e(&[2])), s(e(&[0]))])),
        strided(
            ht!((2, 3)),
            t(vec![s(e(&[0]).scale(2)), s(e(&[1]).scale(3))]),
        ),
        strided(ht!((4, 3)), t(vec![s(e(&[0]).scale(3)), s(e(&[0]))])),
        strided(
            ht!((2, 3, 4)),
            t(vec![s(e(&[0])), s(StrideScalar::Int(0)), s(e(&[1]))]),
        ),
        strided(
            ht!((2, (3, 4))),
            t(vec![s(e(&[0])), t(vec![s(e(&[1])), s(e(&[2]))])]),
        ),
        strided(
            ht!((2, 3, 4)),
            t(vec![s(e(&[0, 0])), s(e(&[0, 1])), s(e(&[1]))]),
        ),
    ];
    layouts.iter().for_each(postcondition_make_layout_like);
}

// ---------------------------------------------------------------------------
// test_make_layout.py — TestMakeOrderedLayout
// ---------------------------------------------------------------------------

/// The structural post-conditions of `make_ordered_layout`. PyCuTe's
/// `postcondition_make_ordered_layout`.
fn postcondition_make_ordered_layout(shape: &IntTuple, order: &IntTuple) {
    let result = make_ordered_layout(shape, order).unwrap();

    // The shape is preserved exactly, hierarchy included.
    assert_eq!(result.shape, *shape);

    // The result is compact: its codomain is exactly [0, size).
    let image = (0..result.shape.product())
        .map(|i| offset(&result, ht!(i)))
        .collect::<BTreeSet<_>>();
    assert!(image.iter().copied().eq(0..result.shape.product()));

    // The modes receive prefix-product strides taken in ascending order
    // of `order`, ties broken by left-to-right position.
    let shp = shape.leaves();
    let ord = order.leaves();
    let dst = result.stride.leaves();
    let mut modes = (0..shp.len()).collect::<Vec<_>>();
    modes.sort_by_key(|&i| ord[i]);
    let mut current = 1;
    for i in modes {
        assert_eq!(*dst[i], StrideScalar::Int(current));
        current *= shp[i];
    }
}

#[test]
fn an_ascending_order_is_column_major_and_a_descending_one_row_major() {
    let ordered = |shape: IntTuple, order: IntTuple| make_ordered_layout(&shape, &order).unwrap();
    assert_eq!(ordered(ht!(8), ht!(0)), layout(ht!(8), ht!(1)));
    assert_eq!(
        ordered(ht!((4, 8)), ht!((0, 1))),
        layout(ht!((4, 8)), ht!((1, 4)))
    );
    assert_eq!(
        ordered(ht!((4, 8)), ht!((1, 0))),
        layout(ht!((4, 8)), ht!((8, 1)))
    );
    assert_eq!(
        ordered(ht!((2, 3, 4)), ht!((0, 1, 2))),
        layout(ht!((2, 3, 4)), ht!((1, 2, 6)))
    );
    assert_eq!(
        ordered(ht!((2, 3, 4)), ht!((2, 1, 0))),
        layout(ht!((2, 3, 4)), ht!((12, 4, 1)))
    );
}

#[test]
fn the_cute_documented_example_of_make_ordered_layout() {
    assert_eq!(
        make_ordered_layout(&ht!((2, 2, 2, 2)), &ht!((0, 2, 3, 1))).unwrap(),
        layout(ht!((2, 2, 2, 2)), ht!((1, 4, 8, 2)))
    );
}

#[test]
fn an_order_may_be_any_permutation() {
    assert_eq!(
        make_ordered_layout(&ht!((2, 3, 4)), &ht!((2, 0, 1))).unwrap(),
        layout(ht!((2, 3, 4)), ht!((12, 1, 3)))
    );
}

#[test]
fn only_the_relative_ordering_matters() {
    assert_eq!(
        make_ordered_layout(&ht!((4, 8)), &ht!((5, 7))).unwrap(),
        make_ordered_layout(&ht!((4, 8)), &ht!((0, 1))).unwrap()
    );
    assert_eq!(
        make_ordered_layout(&ht!((2, 3, 4, 5)), &ht!((2, 67, 42, 50))).unwrap(),
        layout(ht!((2, 3, 4, 5)), ht!((1, 40, 2, 8)))
    );
}

#[test]
fn ties_break_by_position() {
    let ordered = |shape: IntTuple, order: IntTuple| make_ordered_layout(&shape, &order).unwrap();
    assert_eq!(
        ordered(ht!((4, 8)), ht!((5, 5))),
        layout(ht!((4, 8)), ht!((1, 4)))
    );
    assert_eq!(
        ordered(ht!((8, 4)), ht!((5, 5))),
        layout(ht!((8, 4)), ht!((1, 8)))
    );
    assert_eq!(
        ordered(ht!((2, 3, 4, 2)), ht!((0, 2, 3, 0))),
        layout(ht!((2, 3, 4, 2)), ht!((1, 4, 12, 2)))
    );
}

#[test]
fn make_ordered_layout_packs_across_a_nested_shape() {
    let ordered = |shape: IntTuple, order: IntTuple| make_ordered_layout(&shape, &order).unwrap();
    assert_eq!(
        ordered(ht!(((2, 2), (2, 2))), ht!(((0, 1), (2, 3)))),
        layout(ht!(((2, 2), (2, 2))), ht!(((1, 2), (4, 8))))
    );
    assert_eq!(
        ordered(ht!((2, (3, 4))), ht!((0, (1, 2)))),
        layout(ht!((2, (3, 4))), ht!((1, (2, 6))))
    );
    assert_eq!(
        ordered(ht!((2, (3, 4))), ht!((2, (1, 0)))),
        layout(ht!((2, (3, 4))), ht!((12, (4, 1))))
    );
}

#[test]
fn a_size_one_mode_leaves_the_running_product_alone() {
    let ordered = |shape: IntTuple, order: IntTuple| make_ordered_layout(&shape, &order).unwrap();
    assert_eq!(
        ordered(ht!((1, 4)), ht!((0, 1))),
        layout(ht!((1, 4)), ht!((1, 1)))
    );
    assert_eq!(
        ordered(ht!((4, 1, 8)), ht!((0, 1, 2))),
        layout(ht!((4, 1, 8)), ht!((1, 4, 4)))
    );
}

#[test]
fn an_incongruent_order_is_rejected() {
    assert!(make_ordered_layout(&ht!((4, 8)), &ht!((0, 1, 2))).is_err());
    assert!(make_ordered_layout(&ht!((4, (8, 2))), &ht!((0, 1))).is_err());
    assert!(make_ordered_layout(&ht!(8), &ht!((0, 1))).is_err());
}

#[test]
fn make_ordered_layout_postconditions() {
    let cases = [
        (ht!(8), ht!(0)),
        (ht!((4, 8)), ht!((0, 1))),
        (ht!((4, 8)), ht!((1, 0))),
        (ht!((2, 3, 4)), ht!((0, 1, 2))),
        (ht!((2, 3, 4)), ht!((2, 1, 0))),
        (ht!((2, 3, 4)), ht!((2, 0, 1))),
        (ht!((2, 2, 2, 2)), ht!((0, 2, 3, 1))),
        (ht!((2, 3, 4, 5)), ht!((2, 67, 42, 50))),
        (ht!((4, 8)), ht!((5, 5))),
        (ht!((8, 4)), ht!((5, 5))),
        (ht!(((2, 2), (2, 2))), ht!(((0, 1), (2, 3)))),
        (ht!((2, (3, 4))), ht!((0, (1, 2)))),
        (ht!((2, (3, 4))), ht!((2, (1, 0)))),
        (ht!((1, 4)), ht!((0, 1))),
        (ht!((4, 1, 8)), ht!((0, 1, 2))),
    ];
    for (shape, order) in &cases {
        postcondition_make_ordered_layout(shape, order);
    }
}

// ---------------------------------------------------------------------------
// test_recast.py
// ---------------------------------------------------------------------------

/// Every scale the recast cases sweep, in their Python order.
fn scales() -> [Scale; 9] {
    [
        Scale::whole(8).unwrap(),
        Scale::whole(6).unwrap(),
        Scale::whole(4).unwrap(),
        Scale::whole(2).unwrap(),
        Scale::whole(1).unwrap(),
        Scale::fraction(1, 2).unwrap(),
        Scale::fraction(1, 4).unwrap(),
        Scale::fraction(1, 6).unwrap(),
        Scale::fraction(1, 8).unwrap(),
    ]
}

/// One recast case: `recast(l, scale) == expected`.
fn recast_is(l: &Layout, scale: Scale, expected: Layout) {
    assert_eq!(recast(l, scale).unwrap(), expected);
}

#[test]
fn recast_a_1d_contiguous_layout() {
    let l = layout(ht!(24), ht!(1));
    let case = |scale: Scale, extent| recast_is(&l, scale, layout(ht!(extent), ht!(1)));
    let [s8, s6, s4, s2, s1, h2, h4, h6, h8] = scales();
    case(s8, 3);
    case(s6, 4);
    case(s4, 6);
    case(s2, 12);
    case(s1, 24);
    case(h2, 48);
    case(h4, 96);
    case(h6, 144);
    case(h8, 192);
}

#[test]
fn recast_a_1d_layout_of_stride_two() {
    let l = layout(ht!(24), ht!(2));
    let case =
        |scale: Scale, extent, stride| recast_is(&l, scale, layout(ht!(extent), ht!(stride)));
    let [s8, s6, s4, s2, s1, h2, h4, h6, h8] = scales();
    case(s8, 6, 1);
    case(s6, 8, 1);
    case(s4, 12, 1);
    case(s2, 24, 1);
    case(s1, 24, 2);
    case(h2, 24, 4);
    case(h4, 24, 8);
    case(h6, 24, 12);
    case(h8, 24, 16);
}

#[test]
fn recast_a_2d_column_major_layout() {
    let l = layout(ht!((24, 24)), ht!((24, 1)));
    let case = |scale: Scale, n, d| recast_is(&l, scale, layout(ht!((24, n)), ht!((d, 1))));
    let [s8, s6, s4, s2, s1, h2, h4, h6, h8] = scales();
    case(s8, 3, 3);
    case(s6, 4, 4);
    case(s4, 6, 6);
    case(s2, 12, 12);
    case(s1, 24, 24);
    case(h2, 48, 48);
    case(h4, 96, 96);
    case(h6, 144, 144);
    case(h8, 192, 192);
}

#[test]
fn recast_a_small_2d_column_major_layout() {
    let l = layout(ht!((4, 6)), ht!((6, 1)));
    let case = |scale: Scale, n, d| recast_is(&l, scale, layout(ht!((4, n)), ht!((d, 1))));
    let [_, s6, _, s2, s1, h2, h4, h6, h8] = scales();
    case(s6, 1, 1);
    case(s2, 3, 3);
    case(s1, 6, 6);
    case(h2, 12, 12);
    case(h4, 24, 24);
    case(h6, 36, 36);
    case(h8, 48, 48);
}

#[test]
fn recast_a_2d_row_major_layout() {
    let l = layout(ht!((4, 4)), ht!((4, 1)));
    let case = |scale: Scale, m, n, d| recast_is(&l, scale, layout(ht!((m, n)), ht!((d, 1))));
    let [s8, _, s4, s2, s1, h2, h4, h6, h8] = scales();
    case(s8, 2, 1, 1);
    case(s4, 4, 1, 1);
    case(s2, 4, 2, 2);
    case(s1, 4, 4, 4);
    case(h2, 4, 8, 8);
    case(h4, 4, 16, 16);
    case(h6, 4, 24, 24);
    case(h8, 4, 32, 32);
}

#[test]
fn a_stride_zero_layout_is_unchanged_by_recast() {
    let l = layout(ht!(8), ht!(0));
    for scale in scales() {
        recast_is(&l, scale, layout(ht!(8), ht!(0)));
    }
}

#[test]
fn recast_a_2d_layout_with_a_stride_zero_mode() {
    let l = layout(ht!((8, 4)), ht!((0, 2)));
    let case = |scale: Scale, n, d| recast_is(&l, scale, layout(ht!((8, n)), ht!((0, d))));
    let [s8, _, s4, s2, s1, h2, h4, h6, h8] = scales();
    case(s8, 1, 1);
    case(s4, 2, 1);
    case(s2, 4, 1);
    case(s1, 4, 2);
    case(h2, 4, 4);
    case(h4, 4, 8);
    case(h6, 4, 12);
    case(h8, 4, 16);
}

#[test]
fn recast_a_3d_layout_with_an_interior_stride_zero_mode() {
    let l = layout(ht!((8, 4, 6)), ht!((1, 0, 2)));
    let case = |scale: Scale, m, n, d| recast_is(&l, scale, layout(ht!((m, 4, n)), ht!((1, 0, d))));
    let [s8, _, s4, s2, s1, h2, h4, h6, h8] = scales();
    case(s8, 1, 2, 1);
    case(s4, 2, 3, 1);
    case(s2, 4, 6, 1);
    case(s1, 8, 6, 2);
    case(h2, 16, 6, 4);
    case(h4, 32, 6, 8);
    case(h6, 48, 6, 12);
    case(h8, 64, 6, 16);
}

#[test]
fn recast_a_nested_shape() {
    let l = layout(ht!(((4, 6), 8)), ht!(((1, 4), 24)));
    let case = |scale: Scale, m, n, d| {
        recast_is(&l, scale, layout(ht!(((m, n), 8)), ht!(((1, m), d))));
    };
    let [s8, _, s4, s2, s1, h2, h4, h6, h8] = scales();
    case(s8, 1, 3, 3);
    case(s4, 1, 6, 6);
    case(s2, 2, 6, 12);
    case(s1, 4, 6, 24);
    case(h2, 8, 6, 48);
    case(h4, 16, 6, 96);
    case(h6, 24, 6, 144);
    case(h8, 32, 6, 192);
}

#[test]
fn recast_a_nested_shape_at_every_scale() {
    let l = layout(ht!(((24, 24), 24)), ht!((1, 24)));
    let case = |scale: Scale, m, d| {
        recast_is(&l, scale, layout(ht!(((m, 24), 24)), ht!(((1, d), d))));
    };
    let [s8, s6, s4, s2, s1, h2, h4, h6, h8] = scales();
    case(s8, 3, 3);
    case(s6, 4, 4);
    case(s4, 6, 6);
    case(s2, 12, 12);
    case(s1, 24, 24);
    case(h2, 48, 48);
    case(h4, 96, 96);
    case(h6, 144, 144);
    case(h8, 192, 192);
}

#[test]
fn recast_at_scale_one_is_the_identity() {
    let layouts = [
        layout(ht!(24), ht!(1)),
        layout(ht!(24), ht!(2)),
        layout(ht!((4, 6)), ht!((6, 1))),
        layout(ht!((4, 4)), ht!((4, 1))),
        layout(ht!((8, 4, 6)), ht!((1, 0, 2))),
        layout(ht!(8), ht!(0)),
    ];
    for l in &layouts {
        assert_eq!(recast(l, Scale::whole(1).unwrap()).unwrap(), *l);
    }
}

#[test]
fn recast_rejects_a_leaf_that_divides_unevenly() {
    // 3 and 2 divide neither way, so the leaf has no rescaling.
    assert!(recast(&layout(ht!(24), ht!(3)), Scale::whole(2).unwrap()).is_err());
    assert!(recast(&layout(ht!(24), ht!(4)), Scale::fraction(1, 3).unwrap()).is_ok());
}
