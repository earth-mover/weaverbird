//! Ported from `test/test_logical_product.py`.
//!
//! `test_logical_product` and `test_logical_product_tiler` are here in
//! full. `test_logical_product_types` is mostly the `algebra.py` facade —
//! promoting `A` from an int or a tuple through `tiler_to_layout`, the
//! `TypeError` for `A = None`, and the `Tensor` overload. Neither the
//! facade nor `Tensor` is ported, so what is left of it is the dispatch
//! head, in [`the_dispatch_head_reads_every_tiler_form`], whose expected
//! values were read off the reference implementation.
//!
//! Nothing here waits on `blocked_product` or `raked_product`: the Python
//! file does not test them.
#![expect(clippy::unwrap_used, reason = "a test asserts the happy path")]

use weaverbird::{IntTuple, Layout, Stride, Tiler, ht};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// `Layout(shape, stride)` over integer strides.
fn layout(shape: IntTuple, stride: IntTuple) -> Layout {
    Layout::from_base(shape, &Stride::from(stride)).unwrap()
}

/// PyCuTe's `postcondition_logical_product`.
///
/// The result is a pair: mode 0 is the source untouched, and mode 1 is
/// compatible with the layout it is reproduced over.
fn postcondition_logical_product(a: &Layout, b: &Layout) {
    let r = a.logical_product(b).unwrap();

    assert_eq!(r.shape.rank(), 2, "{a} x {b} => {r}");
    assert_eq!(&r.mode(0).unwrap(), a, "{a} x {b} => {r}");
    assert!(b.shape.compatible_with(&r.mode(1).unwrap().shape), "{a} x {b} => {r}");
}

// ---------------------------------------------------------------------------
// logical_product
// ---------------------------------------------------------------------------

#[test]
fn logical_product_reproduces_a_layout_over_another() {
    for (a, b) in [
        (layout(ht!(1), ht!(0)), layout(ht!(1), ht!(0))),
        (layout(ht!(1), ht!(1)), layout(ht!(1), ht!(0))),
        (layout(ht!(1), ht!(0)), layout(ht!(1), ht!(1))),
        (layout(ht!(1), ht!(1)), layout(ht!(1), ht!(1))),
        (layout(ht!(3), ht!(1)), layout(ht!(4), ht!(1))),
        (layout(ht!(3), ht!(1)), layout(ht!(4), ht!(0))),
        (layout(ht!(3), ht!(1)), layout(ht!((2, 4)), ht!((1, 2)))),
        (layout(ht!((2, 4)), ht!((1, 2))), layout(ht!(3), ht!(1))),
        (Layout::compact(ht!((8, (2, 2)))), layout(ht!(4), ht!(2))),
        (Layout::compact(ht!((2, 2))), layout(ht!((3, 3)), ht!((3, 1)))),
        (layout(ht!(3), ht!(32)), layout(ht!(32), ht!(1))),
        (layout(ht!(3), ht!(2)), layout(ht!(4), ht!(1))),
        (layout(ht!(3), ht!(32)), layout(ht!(128), ht!(1))),
        (layout(ht!(3), ht!(32)), Layout::compact(ht!((8, 8)))),
        (layout(ht!(3), ht!(32)), layout(ht!((8, 8)), ht!((8, 1)))),
        (layout(ht!((4, 2)), ht!((1, 16))), Layout::compact(ht!((4, 4)))),
        (layout(ht!((4, 2)), ht!((1, 16))), layout(ht!((4, 2)), ht!((2, 1)))),
        (layout(ht!(((2, 2), (2, 2))), ht!(((1, 4), (8, 32)))), layout(ht!((2, 2)), ht!((1, 2)))),
        (layout(ht!(((2, 2), (2, 2))), ht!(((1, 4), (8, 32)))), layout(ht!((2, 2)), ht!((2, 1)))),
        (layout(ht!((4, 6)), ht!((1, 6))), layout(ht!(3), ht!(1))),
    ] {
        postcondition_logical_product(&a, &b);
    }
}

/// The two worked examples from `algebra.logical_product`'s docstring.
#[test]
fn logical_product_matches_the_worked_examples() {
    assert_eq!(
        layout(ht!((2, 2)), ht!((4, 1))).logical_product(layout(ht!(6), ht!(1))).unwrap(),
        layout(ht!(((2, 2), (2, 3))), ht!(((4, 1), (2, 8))))
    );
    assert_eq!(
        layout(ht!(3), ht!(1)).logical_product(layout(ht!(4), ht!(1))).unwrap(),
        layout(ht!((3, 4)), ht!((1, 3)))
    );
}

// ---------------------------------------------------------------------------
// by-mode tilers
// ---------------------------------------------------------------------------

/// PyCuTe's `test_logical_product_tiler`: a multi-mode tiler makes
/// `logical_product` act by-mode.
#[test]
fn a_tiler_tuple_multiplies_by_mode() {
    let a = layout(ht!((6, 4)), ht!((4, 1)));
    let modes = [layout(ht!(2), ht!(1)), layout(ht!(2), ht!(1))];
    let expected = Layout::from_modes(
        modes
            .iter()
            .enumerate()
            .map(|(i, m)| a.mode(i).unwrap().logical_product(m).unwrap())
            .collect(),
    );
    assert_eq!(
        a.logical_product(Tiler::ByMode(modes.iter().map(Tiler::from).collect())).unwrap(),
        expected
    );
}

// ---------------------------------------------------------------------------
// the dispatch head
// ---------------------------------------------------------------------------

#[test]
fn the_dispatch_head_reads_every_tiler_form() {
    let a = layout(ht!((8, 8)), ht!((8, 1)));

    // RHS None, noop. PyCuTe's `logical_product(A, None) == A`.
    assert_eq!(a.logical_product(&Tiler::Skip).unwrap(), a);
    // RHS int, A x N -> A x N:1. PyCuTe's
    // `logical_product(A, 2) == logical_product(A, Layout(2, 1))`.
    assert_eq!(
        a.logical_product(Tiler::Extent(2)).unwrap(),
        a.logical_product(layout(ht!(2), ht!(1))).unwrap()
    );
    assert_eq!(
        a.logical_product(Tiler::Extent(2)).unwrap(),
        layout(ht!(((8, 8), 2)), ht!(((8, 1), 64)))
    );
    // RHS tuple, by-mode.
    assert_eq!(
        a.logical_product(Tiler::ByMode(vec![Tiler::Extent(2), Tiler::Extent(4)])).unwrap(),
        layout(ht!(((8, 2), (8, 4))), ht!(((8, 1), (1, 8))))
    );
    // A per-mode None leaves its mode alone.
    assert_eq!(
        a.logical_product(Tiler::ByMode(vec![Tiler::from(&layout(ht!(2), ht!(1))), Tiler::Skip]))
            .unwrap(),
        layout(ht!(((8, 2), 8)), ht!(((8, 1), 1)))
    );
    assert_eq!(
        a.logical_product(Tiler::ByMode(vec![Tiler::Skip, Tiler::from(&layout(ht!(4), ht!(1)))]))
            .unwrap(),
        layout(ht!((8, (8, 4))), ht!((8, (1, 8))))
    );
    // A tiler the layout outranks runs out under `zip_longest`, and the
    // modes it does not reach take the no-op.
    assert_eq!(
        a.logical_product(Tiler::ByMode(vec![Tiler::from(&layout(ht!(2), ht!(1)))])).unwrap(),
        layout(ht!(((8, 2), 8)), ht!(((8, 1), 1)))
    );
    // PyCuTe's `logical_product(6, (Layout(2, 1), Layout(2, 1)))` raises:
    // a rank-1 layout cannot take a rank-2 tiler.
    assert!(
        layout(ht!(6), ht!(1))
            .logical_product(Tiler::ByMode(vec![
                Tiler::from(&layout(ht!(2), ht!(1))),
                Tiler::from(&layout(ht!(2), ht!(1))),
            ]))
            .is_err()
    );
}
