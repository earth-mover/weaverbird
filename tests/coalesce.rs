//! Ported from `test/test_coalesce.py` and `test/test_coalesce_z.py`.
//!
//! The `sympy` cases are gone for good: [`Int`] is the only integer
//! here, so there is no symbolic extent to carry through the fold.
#![expect(clippy::unwrap_used, reason = "a test asserts the happy path")]

use weaverbird::{HTuple, IntTuple, Layout, Profile, Stride, StrideScalar, e, ht};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// `Layout(shape, stride)` over integer strides.
fn layout(shape: IntTuple, stride: IntTuple) -> Layout {
    Layout::from_base(shape, &Stride::from(stride)).unwrap()
}

/// `Layout(shape, stride)` over an already-built stride.
fn strided(shape: IntTuple, stride: &Stride) -> Layout {
    Layout::from_base(shape, stride).unwrap()
}

/// A stride leaf.
fn s(x: StrideScalar) -> Stride {
    HTuple::Leaf(x)
}

/// A stride mode.
fn t(modes: Vec<Stride>) -> Stride {
    HTuple::Tuple(modes)
}

/// PyCuTe's `postcondition_coalesce`.
///
/// The result is flat, keeps the domain and evaluates as the
/// source does at every coordinate. Coalescing it again changes
/// nothing, and composing the source with its own flat domain lands on
/// the same layout.
fn postcondition_coalesce(source: &Layout, expected: Option<Layout>) {
    let result = source.coalesced().unwrap();
    if let Some(expected) = expected {
        assert_eq!(result, expected, "coalesce({source})");
    }
    assert!(result.shape.depth() <= 1, "flat: {result}");
    assert_eq!(result.shape.size(), source.shape.size(), "size held: {source} => {result}");
    for i in 0..source.shape.size() {
        assert_eq!(
            result.eval(&HTuple::Leaf(i)).unwrap(),
            source.eval(&HTuple::Leaf(i)).unwrap(),
            "{source} => {result} disagree at {i}"
        );
    }
    assert_eq!(result.coalesced().unwrap(), result, "idempotent: {result}");

    // Composition-is-coalesced.
    let flat = layout(ht!(source.shape.size()), ht!(1));
    assert_eq!(source.compose(&flat).unwrap(), result, "composed: {source} o {flat} => {result}");
}

/// PyCuTe's `postcondition_coalesce_z`. The `_z` variant keeps the
/// size-1 modes, so it holds the rank the plain one trims.
fn postcondition_coalesce_z(source: &Layout, expected: Option<Layout>) {
    let result = source.coalesced_z().unwrap();
    if let Some(expected) = expected {
        assert_eq!(result, expected, "coalesce_z({source})");
    }
    assert_eq!(result.shape.size(), source.shape.size(), "size held: {source} => {result}");
    for i in 0..source.shape.size() {
        assert_eq!(
            result.eval(&HTuple::Leaf(i)).unwrap(),
            source.eval(&HTuple::Leaf(i)).unwrap(),
            "{source} => {result} disagree at {i}"
        );
    }
    assert_eq!(result.coalesced_z().unwrap(), result, "idempotent: {result}");
}

// ---------------------------------------------------------------------------
// coalesce — integer strides
// ---------------------------------------------------------------------------

#[test]
fn coalesce_holds_the_map_over_integer_strides() {
    for source in [
        layout(ht!(1), ht!(0)),
        layout(ht!(1), ht!(1)),
        layout(ht!((1, 1)), ht!((5, 7))),
        layout(ht!((2, 4)), ht!((1, 2))),
        layout(ht!((2, 4)), ht!((4, 1))),
        layout(ht!((2, 4, 6)), ht!((1, 2, 8))),
        layout(ht!((2, 4, 6)), ht!((24, 6, 1))),
        layout(ht!((2, (4, 6))), ht!((1, (2, 8)))),
        layout(ht!((2, 4, 6)), ht!((1, 6, 2))),
        layout(ht!((2, 1, 6)), ht!((1, 7, 2))),
        layout(ht!((2, 1, 6)), ht!((4, 7, 8))),
        layout(ht!((2, 1, 6, 1)), ht!((4, 7, 8, 0))),
        layout(ht!((2, 1, 6, 1)), ht!((4, 7, 8, 57))),
        layout(ht!((2, 1, 6, 1)), ht!((1, 7, 8, 0))),
        layout(ht!((2, 1, 6, 1)), ht!((1, 7, 8, 57))),
        layout(ht!((2, 1, 6, 1)), ht!((1, 7, 8, 48))),
        layout(ht!((2, 1, 3)), ht!((2, 4, 4))),
        layout(ht!(((2, 2), (2, 2))), ht!(((1, 4), (8, 32)))),
    ] {
        postcondition_coalesce(&source, None);
    }
}

#[test]
fn coalesce_merges_the_adjacent_modes_that_are_linear() {
    // (2, 4):(1, 2) walks 0..8 in step: one mode of 8.
    postcondition_coalesce(&layout(ht!((2, 4)), ht!((1, 2))), Some(layout(ht!(8), ht!(1))));
    // ((2,2),(2,2)):((1,4),(8,32)) merges only the middle pair. 2:1 and
    // 2:4 do not meet (2*1 != 4); 2:4 and 2:8 do, giving 4:4; 4:4 and
    // 2:32 do not (4*4 != 32).
    postcondition_coalesce(
        &layout(ht!(((2, 2), (2, 2))), ht!(((1, 4), (8, 32)))),
        Some(layout(ht!((2, 4, 2)), ht!((1, 4, 32)))),
    );
}

#[test]
fn coalesce_drops_the_size_one_modes() {
    // The 1:7 in the middle carries nothing, so 2:1 and 6:2 meet across
    // it.
    postcondition_coalesce(&layout(ht!((2, 1, 6)), ht!((1, 7, 2))), Some(layout(ht!(12), ht!(1))));
    // An all-size-1 layout keeps one mode. The fold drops the first as
    // trailing, then appends the second, so the surviving stride is the
    // last one — not zero.
    postcondition_coalesce(&layout(ht!((1, 1)), ht!((5, 7))), Some(layout(ht!(1), ht!(7))));
}

// ---------------------------------------------------------------------------
// coalesce — basis strides
// ---------------------------------------------------------------------------

#[test]
fn coalesce_holds_the_map_over_basis_strides() {
    for source in [
        strided(ht!(1), &s(e(&[0]))),
        strided(ht!(1), &s(e(&[1]))),
        strided(ht!((1, 1)), &t(vec![s(e(&[0])), s(e(&[1]))])),
        strided(ht!((2, 4)), &t(vec![s(e(&[0])), s(e(&[1]))])),
        strided(ht!((2, 4)), &t(vec![s(e(&[1])), s(e(&[1]).scale(2))])),
        strided(
            ht!((2, 1, 6, 1)),
            &t(vec![s(e(&[1, 1])), s(e(&[2, 3])), s(e(&[1, 1]).scale(2)), s(e(&[2, 3]))]),
        ),
        strided(
            ht!((2, 1, 6, 1)),
            &t(vec![s(e(&[1, 1])), s(e(&[2, 3])), s(e(&[1, 0]).scale(2)), s(e(&[2, 3]))]),
        ),
        strided(
            ht!((2, 1, 6, 1)),
            &t(vec![s(e(&[1, 0])), s(e(&[2, 3])), s(e(&[1, 0]).scale(2)), s(e(&[1, 0]).scale(12))]),
        ),
        strided(
            ht!(((2, 2), (2, 2))),
            &t(vec![
                t(vec![s(e(&[0])), s(e(&[1]))]),
                t(vec![s(e(&[1]).scale(2)), s(e(&[0]).scale(2))]),
            ]),
        ),
    ] {
        postcondition_coalesce(&source, None);
    }
}

// ---------------------------------------------------------------------------
// coalesce_z — the size-1-preserving variant
// ---------------------------------------------------------------------------

#[test]
fn coalesce_z_holds_the_map() {
    for source in [
        layout(ht!(1), ht!(0)),
        layout(ht!((1, 1)), ht!((5, 7))),
        layout(ht!((2, 4)), ht!((1, 2))),
        layout(ht!((2, 4, 6)), ht!((1, 6, 2))),
        layout(ht!((2, 1, 6)), ht!((1, 7, 2))),
        layout(ht!((2, 1, 6, 1)), ht!((1, 7, 8, 48))),
        layout(ht!(((2, 2), (2, 2))), ht!(((1, 4), (8, 32)))),
    ] {
        postcondition_coalesce_z(&source, None);
    }
}

#[test]
fn coalesce_z_keeps_a_trailing_size_one_mode() {
    // The `_z` variant is exactly `coalesce` without the final trim, so
    // the trailing 1 that `coalesce` drops survives here.
    let source = layout(ht!((2, 4, 1)), ht!((1, 2, 0)));
    assert_eq!(source.coalesced_z().unwrap(), layout(ht!((8, 1)), ht!((1, 0))));
    assert_eq!(source.coalesced().unwrap(), layout(ht!(8), ht!(1)));
}

// ---------------------------------------------------------------------------
// the profile
// ---------------------------------------------------------------------------

#[test]
fn a_none_profile_is_the_no_op() {
    let source = layout(ht!((2, 4)), ht!((1, 2)));
    assert_eq!(source.coalesce(&Profile::Keep).unwrap(), source);
    assert_eq!(source.coalesce_z(&Profile::Keep).unwrap(), source);
}

#[test]
fn a_tuple_profile_coalesces_by_mode() {
    // Each mode folds on its own, so the two-mode rank survives a fold
    // that would otherwise merge everything into one.
    let source = layout(ht!(((2, 4), (3, 5))), ht!(((1, 2), (8, 24))));
    assert_eq!(
        source.coalesce(&Profile::ByMode(vec![Profile::Merge, Profile::Merge])).unwrap(),
        layout(ht!((8, 15)), ht!((1, 8)))
    );
}

#[test]
fn a_profile_that_outranks_the_layout_is_rejected() {
    let source = layout(ht!((2, 4)), ht!((1, 2)));
    assert!(
        source
            .coalesce(&Profile::ByMode(vec![Profile::Merge, Profile::Merge, Profile::Merge]))
            .is_err()
    );
    assert!(
        source
            .coalesce_z(&Profile::ByMode(vec![Profile::Merge, Profile::Merge, Profile::Merge]))
            .is_err()
    );
}
