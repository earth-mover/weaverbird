//! The [`Tiler`] argument the by-mode operations share.
//!
//! `compose`, `logical_divide` and `logical_product` all open with the
//! same dispatch: a skip passes the operand through, an extent promotes
//! to `n:1`, a layout works as itself, and a by-mode tiler dispatches per
//! mode. These cases pin that head down once. The algorithms behind it
//! are tested in `composition.rs`, `logical_divide.rs` and
//! `logical_product.rs`.
//!
//! PyCuTe carries the same forms through the `algebra.py` facade, which
//! this crate has no counterpart for: a caller holding a layout calls the
//! method.
#![expect(clippy::expect_used, reason = "a test asserts the happy path")]

use weaverbird::{Layout, Profile, Tiler, ht, layout};

/// The operations that share the dispatch head.
type ByMode = (&'static str, fn(&Layout, Tiler) -> Layout);

fn operations() -> Vec<ByMode> {
    vec![
        ("compose", |a, b| a.compose(b).expect("compose")),
        ("logical_divide", |a, b| a.logical_divide(b).expect("logical_divide")),
        ("logical_product", |a, b| a.logical_product(b).expect("logical_product")),
    ]
}

/// A source every operation above accepts, whatever the tiler form.
fn source() -> Layout {
    layout!((8, 8):(8, 1))
}

#[test]
fn a_skip_passes_the_operand_through() {
    for (name, op) in operations() {
        assert_eq!(op(&source(), Tiler::Skip), source(), "{name}(A, Skip)");
    }
}

#[test]
fn an_extent_promotes_to_stride_one() {
    for (name, op) in operations() {
        for n in [1, 2, 4] {
            assert_eq!(
                op(&source(), Tiler::Extent(n)),
                op(&source(), Tiler::Layout(Layout::compact(ht!(n)))),
                "{name}(A, {n})"
            );
        }
    }
}

#[test]
fn a_layout_converts_by_value_and_by_reference() {
    let b = layout!(4:2);
    for (name, op) in operations() {
        assert_eq!(op(&source(), b.clone().into()), op(&source(), (&b).into()), "{name}");
    }
}

#[test]
fn a_by_mode_tiler_dispatches_per_mode() {
    let a = layout!((8, 8):(8, 1));
    // Mode 0 takes the extent, and mode 1 passes through.
    assert_eq!(a.compose(vec![Tiler::Extent(4), Tiler::Skip]).unwrap(), layout!((4, 8):(8, 1)));
    // A tiler shorter than the layout leaves the rest alone.
    assert_eq!(a.compose(vec![Tiler::Extent(4)]).unwrap(), layout!((4, 8):(8, 1)));
}

#[test]
fn a_by_mode_tiler_may_not_outrank_the_layout() {
    let a = layout!((8, 8):(8, 1));
    let too_deep = vec![Tiler::Extent(2), Tiler::Extent(2), Tiler::Extent(2)];
    assert!(a.compose(too_deep.clone()).is_err());
    assert!(a.logical_divide(too_deep.clone()).is_err());
    assert!(a.logical_product(too_deep).is_err());
}

#[test]
fn to_layout_promotes_every_form() {
    assert_eq!(Tiler::Extent(3).to_layout().unwrap(), layout!(3:1));
    assert_eq!(Tiler::Layout(layout!(4:2)).to_layout().unwrap(), layout!(4:2));
    // Each mode of a tuple takes its own basis element, so the modes land
    // in separate positions of the codomain.
    let promoted = Tiler::ByMode(vec![Tiler::Extent(4), Tiler::Extent(5)]).to_layout().unwrap();
    assert_eq!(promoted.shape, ht!((4, 5)));
    assert_eq!(format!("{promoted}"), "(4,5):(1@0,1@1)");
}

#[test]
fn to_layout_rejects_a_skip() {
    assert!(Tiler::Skip.to_layout().is_err());
    assert!(Tiler::ByMode(vec![Tiler::Extent(4), Tiler::Skip]).to_layout().is_err());
}

#[test]
fn a_coalesce_profile_dispatches_the_same_way() {
    let a = layout!((2, (1, 6)):(1, (6, 2)));
    // Merge folds the whole layout, and a by-mode profile keeps the rank.
    assert_eq!(a.coalesce(&Profile::Merge).unwrap(), layout!(12:1));
    assert_eq!(
        a.coalesce(&Profile::ByMode(vec![Profile::Merge, Profile::Merge])).unwrap(),
        layout!((2, 6):(1, 2))
    );
    // Keep is the no-op, in both folds.
    assert_eq!(a.coalesce(&Profile::Keep).unwrap(), a);
    assert_eq!(a.coalesce_z(&Profile::Keep).unwrap(), a);
}

#[test]
fn a_coalesce_profile_may_not_outrank_the_layout() {
    let a = layout!((2, 6):(1, 2));
    let too_deep = Profile::ByMode(vec![Profile::Merge, Profile::Merge, Profile::Merge]);
    assert!(a.coalesce(&too_deep).is_err());
    assert!(a.coalesce_z(&too_deep).is_err());
}
