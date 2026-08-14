//! The `pycute/algebra.py` facade: `coalesce_z`, `coalesce`,
//! `composition`, the two inverses, `complement`, `nullspace`.
//!
//! There is no PyCuTe test file for these. Each function is a dispatch
//! head over a `Layout` method, and the Python suite exercises that head
//! incidentally, from every call site that passes an int or a tuple. The
//! tests here take it head-on: each wrapper reads every argument form
//! PyCuTe reads — a `Layout`, an integer, a tuple, `None` — and agrees
//! with its method on a `Layout`.
//!
//! The algorithms themselves are tested in `tests/coalesce.rs`,
//! `tests/composition.rs`, `tests/inverses.rs`, `tests/complement.rs`
//! and `tests/nullspace.rs`; the handful of concrete values below are
//! the docstring examples, kept as anchors so the wrappers are not
//! checked against the methods alone.
#![expect(clippy::unwrap_used, reason = "a test asserts the happy path")]

use pinstripe::{
    HTuple, Int, IntTuple, Layout, OptTiler, Profile, Result, Stride, StrideScalar, TilerLeaf,
    coalesce, coalesce_z, complement, composition, ht, left_inverse, nullspace, right_inverse,
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

/// A layout as a facade argument.
fn tiler(x: &Layout) -> OptTiler {
    HTuple::Leaf(Some(TilerLeaf::Layout(x.clone())))
}

/// An extent as a facade argument. PyCuTe's bare `int`.
fn extent(n: Int) -> OptTiler {
    HTuple::Leaf(Some(TilerLeaf::Int(n)))
}

/// A by-mode facade argument. `ht!` cannot spell it, because
/// `Some(TilerLeaf::Int(4))` is more than one token tree.
fn by_mode(modes: Vec<OptTiler>) -> OptTiler {
    HTuple::Tuple(modes)
}

/// The absent argument. PyCuTe's `None`, whole or per-mode.
fn none() -> OptTiler {
    HTuple::Leaf(None)
}

/// The profile PyCuTe spells `1` — coalesce every mode.
fn all() -> Profile {
    HTuple::Leaf(Some(1))
}

/// A by-mode coalesce profile.
fn profile(modes: Vec<Option<Int>>) -> Profile {
    HTuple::Tuple(modes.into_iter().map(HTuple::Leaf).collect())
}

/// The tuple of leaves as a facade argument.
fn as_argument(modes: &[TilerLeaf]) -> OptTiler {
    HTuple::Tuple(
        modes
            .iter()
            .map(|m| HTuple::Leaf(Some(m.clone())))
            .collect(),
    )
}

/// The tuple of leaves as the layout PyCuTe promotes it to.
/// `tiler_to_layout(A)`, whose `e` defaults to `1`.
fn as_promoted(modes: &[TilerLeaf]) -> Layout {
    tiler_to_layout(
        &HTuple::Tuple(modes.iter().cloned().map(HTuple::Leaf).collect()),
        &StrideScalar::Int(1),
    )
    .unwrap()
}

// ---------------------------------------------------------------------------
// The seven wrappers, paired with the methods they dispatch to.
//
// Each pair is `(name, wrapper, method)`, with the extra arguments —
// the coalesce profile, composition's right-hand side, complement's
// `extend` — pinned to the same value on both sides, so that the two
// halves differ only in how the first argument arrives.
// ---------------------------------------------------------------------------

type Facade = (
    &'static str,
    fn(&OptTiler) -> Result<Option<Layout>>,
    fn(&Layout) -> Result<Layout>,
);

fn facades() -> Vec<Facade> {
    vec![
        (
            "coalesce_z",
            |a| coalesce_z(a, &all()),
            |l| l.coalesce_z(&all()),
        ),
        ("coalesce", |a| coalesce(a, &all()), |l| l.coalesce(&all())),
        // An absent right-hand side is composition's no-op, so this pair
        // isolates the left-hand dispatch.
        (
            "composition",
            |a| composition(a, &none()),
            |l| l.composition(&none()),
        ),
        ("right_inverse", right_inverse, Layout::right_inverse),
        ("left_inverse", left_inverse, Layout::left_inverse),
        (
            "complement",
            |a| complement(a, None),
            |l| l.complement(None),
        ),
        ("nullspace", nullspace, Layout::nullspace),
    ]
}

/// Layouts every one of the seven accepts. `left_inverse` is the
/// narrowest of them — it rejects a stride chain with a gap — so these
/// all carry ordered, divisible strides.
fn examples() -> Vec<Layout> {
    vec![
        compact(ht!(12)),
        layout(ht!(12), ht!(2)),
        layout(ht!((4, 8)), ht!((1, 4))),
        layout(ht!((4, 8)), ht!((8, 1))),
        layout(ht!((2, 1, 6, 1)), ht!((1, 7, 8, 0))),
        layout(ht!((2, 4, 6)), ht!((1, 2, 0))),
        layout(ht!((2, (1, 6))), ht!((1, (6, 2)))),
    ]
}

// ---------------------------------------------------------------------------
// The argument forms
// ---------------------------------------------------------------------------

#[test]
fn an_absent_argument_answers_absent() {
    // PyCuTe's `if A is None: return None`.
    for (name, wrapper, _) in facades() {
        assert_eq!(wrapper(&none()).unwrap(), None, "{name}(None)");
    }
}

#[test]
fn a_layout_argument_agrees_with_the_method() {
    for source in examples() {
        for (name, wrapper, method) in facades() {
            assert_eq!(
                wrapper(&tiler(&source)).unwrap(),
                Some(method(&source).unwrap()),
                "{name}({source})"
            );
        }
    }
}

#[test]
fn an_integer_argument_promotes_to_n_stride_1() {
    // PyCuTe's `tiler_to_layout(3) == Layout(3, 1)`.
    for n in [1, 4, 12] {
        for (name, wrapper, method) in facades() {
            assert_eq!(
                wrapper(&extent(n)).unwrap(),
                Some(method(&compact(ht!(n))).unwrap()),
                "{name}({n})"
            );
        }
    }
}

#[test]
fn a_tuple_argument_promotes_through_tiler_to_layout() {
    // `(4, 5)` and `(Layout(4, 2), Layout(5, 3))`: the two tuple forms
    // `tiler_to_layout` reads, one of extents and one of layouts.
    for modes in [
        vec![TilerLeaf::Int(4), TilerLeaf::Int(5)],
        vec![
            TilerLeaf::Layout(layout(ht!(4), ht!(2))),
            TilerLeaf::Layout(layout(ht!(5), ht!(3))),
        ],
    ] {
        let expected = as_promoted(&modes);
        for (name, wrapper, method) in facades() {
            assert_eq!(
                wrapper(&as_argument(&modes)).unwrap(),
                Some(method(&expected).unwrap()),
                "{name}({expected})"
            );
        }
    }
}

#[test]
fn a_tuple_with_an_absent_mode_is_rejected() {
    // PyCuTe's per-mode `None` is an argument to the *methods* that
    // honour it, never to `tiler_to_layout`, which walks off the end of
    // its own dispatch on a `None` leaf.
    for (name, wrapper, _) in facades() {
        assert!(
            wrapper(&by_mode(vec![extent(4), none()])).is_err(),
            "{name}((4, None))"
        );
    }
}

// ---------------------------------------------------------------------------
// The arguments the wrappers carry past the dispatch head
// ---------------------------------------------------------------------------

#[test]
fn the_coalesce_profile_reaches_the_method() {
    // The by-mode profile keeps the rank the whole-layout one folds
    // away.
    let source = layout(ht!((2, (1, 6))), ht!((1, (6, 2))));
    assert_eq!(
        coalesce(&tiler(&source), &all()).unwrap(),
        Some(layout(ht!(12), ht!(1)))
    );
    assert_eq!(
        coalesce(&tiler(&source), &profile(vec![Some(1), Some(1)])).unwrap(),
        Some(layout(ht!((2, 6)), ht!((1, 2))))
    );
    // A `None` profile is the no-op, at both depths.
    assert_eq!(
        coalesce(&tiler(&source), &none_profile()).unwrap(),
        Some(source.clone())
    );
    assert_eq!(
        coalesce_z(&tiler(&source), &none_profile()).unwrap(),
        Some(source)
    );
}

/// The absent profile. PyCuTe's `profile=None`.
fn none_profile() -> Profile {
    HTuple::Leaf(None)
}

#[test]
fn the_complement_extend_reaches_the_method() {
    let source = layout(ht!(4), ht!(2));
    assert_eq!(
        complement(&tiler(&source), None).unwrap(),
        Some(layout(ht!((2, 1)), ht!((1, 8))))
    );
    assert_eq!(
        complement(&tiler(&source), Some(&compact(ht!(20)).shape)).unwrap(),
        Some(layout(ht!((2, 3)), ht!((1, 8))))
    );
}

#[test]
fn the_composition_right_hand_side_reaches_the_method() {
    // Every form of the right-hand side, over a promoted left one:
    // `Layout(12) o (4, 3) == Layout((4, 3), (1, 4))`.
    assert_eq!(
        composition(&extent(12), &tiler(&compact(ht!((4, 3))))).unwrap(),
        Some(layout(ht!((4, 3)), ht!((1, 4))))
    );
    assert_eq!(
        composition(&extent(12), &extent(4)).unwrap(),
        Some(layout(ht!(4), ht!(1)))
    );
    assert_eq!(
        composition(
            &tiler(&layout(ht!((8, 8)), ht!((8, 1)))),
            &by_mode(vec![extent(4), none()])
        )
        .unwrap(),
        Some(layout(ht!((4, 8)), ht!((8, 1))))
    );
}

#[test]
fn an_absent_left_operand_returns_the_right() {
    // PyCuTe's `if A is None: return B`, read as a layout.
    let b = layout(ht!((4, 3)), ht!((3, 1)));
    assert_eq!(composition(&none(), &tiler(&b)).unwrap(), Some(b));
    assert_eq!(
        composition(&none(), &extent(12)).unwrap(),
        Some(compact(ht!(12)))
    );
    assert_eq!(composition(&none(), &none()).unwrap(), None);
}

// ---------------------------------------------------------------------------
// The docstring examples, as anchors
// ---------------------------------------------------------------------------

#[test]
fn the_docstring_examples() {
    let source = layout(ht!((2, 1, 6, 1)), ht!((1, 7, 8, 0)));
    assert_eq!(
        coalesce_z(&tiler(&source), &all()).unwrap(),
        Some(layout(ht!((2, 6, 1)), ht!((1, 8, 0))))
    );
    assert_eq!(
        coalesce(&tiler(&source), &all()).unwrap(),
        Some(layout(ht!((2, 6)), ht!((1, 8))))
    );
    assert_eq!(
        composition(
            &tiler(&layout(ht!((6, 2)), ht!((8, 2)))),
            &tiler(&layout(ht!((4, 3)), ht!((3, 1))))
        )
        .unwrap(),
        Some(layout(ht!(((2, 2), 3)), ht!(((24, 2), 8))))
    );
    assert_eq!(
        right_inverse(&tiler(&layout(ht!((4, 8)), ht!((1, 4))))).unwrap(),
        Some(layout(ht!(32), ht!(1)))
    );
    assert_eq!(
        right_inverse(&tiler(&layout(ht!((4, 8)), ht!((1, 5))))).unwrap(),
        Some(layout(ht!(4), ht!(1)))
    );
    assert_eq!(
        left_inverse(&tiler(&layout(ht!((4, 8)), ht!((1, 5))))).unwrap(),
        Some(layout(ht!((5, 8)), ht!((1, 4))))
    );
    assert_eq!(
        complement(&tiler(&layout(ht!((2, 2)), ht!((1, 6)))), None).unwrap(),
        Some(layout(ht!((3, 1)), ht!((2, 12))))
    );
    assert_eq!(
        nullspace(&tiler(&layout(ht!((2, 4, 6)), ht!((1, 2, 0))))).unwrap(),
        Some(layout(ht!(6), ht!(8)))
    );
}
