//! Ported from `test/test_inverse_right.py` and `test/test_inverse_left.py`.
//!
//! Both Python post-conditions evaluate the inverse against its source at
//! every coordinate, and both are here in full.
//!
//! The `sympy` cases are gone for good, as they are in the coalesce port:
//! [`Int`](pinstripe::Int) is the only integer here, so
//! `test_right_inverse_sympy` has no symbolic extent to carry through the
//! chain, and `test_right_inverse_sympy_substitution` — which only checks
//! that a substituted symbolic result is still an inverse — has nothing
//! left to substitute.
#![expect(clippy::unwrap_used, reason = "a test asserts the happy path")]

use pinstripe::{
    HTuple, Int, IntTuple, Layout, OptTiler, Stride, StrideScalar, Tiler, TilerLeaf,
    atuple::as_tuple, coprofile, e, ht, htuple::weakly_congruent, make_layout, size,
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

/// A codomain value read back as a coordinate.
///
/// PyCuTe feeds `L(i)` straight into another layout: an integer is an
/// integral coordinate, and an arithmetic tuple is a natural one, because
/// it indexes like a tuple. Rust's `call` takes an [`IntTuple`], so the
/// conversion is spelled out.
fn crd(x: &StrideScalar) -> IntTuple {
    as_tuple(x)
}

/// A layout as the right-hand side of a composition.
fn tiler(x: &Layout) -> OptTiler {
    HTuple::Leaf(Some(TilerLeaf::Layout(x.clone())))
}

/// The tiler of a plain shape.
fn int_tiler(shape: &IntTuple) -> Tiler {
    shape.transform_leaf(&|v: &Int| TilerLeaf::Int(*v))
}

/// The default `e` of [`tiler_to_layout`].
fn one() -> StrideScalar {
    StrideScalar::Int(1)
}

/// `composition(tiler_to_layout(shape), inner)`, the shape PyCuTe's MMA
/// TV cases are built in.
fn tv(shape: IntTuple, inner: &Layout) -> Layout {
    tiler_to_layout(&int_tiler(&shape), &one())
        .unwrap()
        .composition(&tiler(inner))
        .unwrap()
}

/// The SM70 MMA 8x8x4 C TV layout, over the two trailing coefficients
/// PyCuTe varies.
fn sm70_c_tv(a: Int, b: Int) -> Layout {
    strided(
        ht!(((2, 2, 2), (2, 2, 2))),
        t(vec![
            t(vec![s(e(&[0])), s(e(&[1]).scale(2)), s(e(&[0]).scale(a))]),
            t(vec![s(e(&[1])), s(e(&[0]).scale(2)), s(e(&[1]).scale(b))]),
        ]),
    )
}

/// PyCuTe's `postcondition_right_inverse`.
///
/// The result's shape coarsens the codomain's profile, and the
/// generalized right inverse condition holds at every coordinate. When
/// the codomain is `Z`, so does the canonical one.
fn postcondition_right_inverse(source: &Layout) {
    let inv = source.right_inverse().unwrap();
    assert!(
        weakly_congruent(&coprofile(source, &[]).unwrap(), &inv.shape),
        "{source} => {inv}"
    );

    // Generalized right inverse condition.
    for i in 0..size(&inv.shape, &[]).unwrap() {
        let r = inv.call(&HTuple::Leaf(i)).unwrap();
        let l_r = source.call(&crd(&r)).unwrap();
        assert_eq!(
            inv.call(&crd(&l_r)).unwrap(),
            r,
            "{source} => {inv} disagree at {i}"
        );
    }

    // Canonical right inverse post-condition, over a codomain of `Z`.
    if matches!(source.call(&HTuple::Leaf(0)).unwrap(), StrideScalar::Int(_)) {
        for i in 0..size(&inv.shape, &[]).unwrap() {
            let r = inv.call(&HTuple::Leaf(i)).unwrap();
            assert_eq!(
                source.call(&crd(&r)).unwrap(),
                StrideScalar::Int(i),
                "{source} => {inv} is not canonical at {i}"
            );
        }
    }
}

/// PyCuTe's `postcondition_left_inverse`.
fn postcondition_left_inverse(source: &Layout) {
    let inv = source.left_inverse().unwrap();
    assert!(
        weakly_congruent(&coprofile(source, &[]).unwrap(), &inv.shape),
        "{source} => {inv}"
    );

    // Generalized left inverse condition.
    for i in 0..size(&source.shape, &[]).unwrap() {
        let l = source.call(&HTuple::Leaf(i)).unwrap();
        let inv_l = inv.call(&crd(&l)).unwrap();
        assert_eq!(
            source.call(&crd(&inv_l)).unwrap(),
            l,
            "{source} => {inv} disagree at {i}"
        );
    }
}

// ---------------------------------------------------------------------------
// right_inverse
// ---------------------------------------------------------------------------

#[test]
fn right_inverse_inverts_over_integer_strides() {
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
        layout(ht!((2, 4, 6)), ht!((1, 2, 8))),
        layout(ht!((2, 4, 6)), ht!((4, 1, 8))),
        layout(ht!((4, 2)), ht!((1, 16))),
    ] {
        postcondition_right_inverse(&source);
    }
}

#[test]
fn right_inverse_handles_a_non_injective_layout() {
    // The largest right inverse is still defined: the modes that break
    // the stride chain are dropped rather than rejected.
    for source in [
        layout(ht!((4, 5, 6)), ht!((1, 1, 4))),
        layout(ht!((7, 5, 9)), ht!((2, 0, 1))),
    ] {
        postcondition_right_inverse(&source);
    }
}

#[test]
fn right_inverse_inverts_over_basis_strides() {
    for source in [
        strided(ht!((4, 5)), t(vec![s(e(&[0])), s(e(&[1]))])),
        strided(ht!((4, 5)), t(vec![s(e(&[1])), s(e(&[0]))])),
        strided(ht!((4, 5)), t(vec![s(e(&[1])), s(e(&[4, 1]))])),
        strided(
            ht!((4, 5)),
            t(vec![s(e(&[0]).scale(2)), s(e(&[1]).scale(2))]),
        ),
        // SM70 MMA 8x8x4 C TV inverse.
        sm70_c_tv(4, 4),
        sm70_c_tv(5, 5),
        sm70_c_tv(5, 4),
        // SM70 MMA 8x8x4 A TV inverse.
        tv(ht!((8, 4)), &layout(ht!(((4, 2), 4)), ht!(((8, 4), 1)))),
        // SM80 MMA 16x8 TV inverse.
        tv(
            ht!((16, 8)),
            &layout(ht!(((4, 8), (2, 2))), ht!(((32, 1), (16, 8)))),
        ),
    ] {
        postcondition_right_inverse(&source);
    }
}

#[test]
fn right_inverse_reads_off_the_worked_cases() {
    assert_eq!(
        layout(ht!((8, 4)), ht!((4, 1))).right_inverse().unwrap(),
        layout(ht!((4, 8)), ht!((8, 1)))
    );
    // A strided layout is not surjective, so only the trivial inverse
    // survives.
    assert_eq!(
        layout(ht!(4), ht!(2)).right_inverse().unwrap(),
        layout(ht!(1), ht!(0))
    );
}

// ---------------------------------------------------------------------------
// left_inverse
// ---------------------------------------------------------------------------

#[test]
fn left_inverse_inverts_over_integer_strides() {
    for source in [
        layout(ht!(1), ht!(0)),
        layout(ht!(1), ht!(1)),
        layout(ht!(1), ht!(2)),
        layout(ht!(1), ht!(4)),
        layout(ht!((1, 1)), ht!((0, 0))),
        layout(ht!((3, 7)), ht!((0, 0))),
        layout(ht!(4), ht!(0)),
        layout(ht!(4), ht!(1)),
        layout(ht!(4), ht!(2)),
        layout(ht!(4), ht!(4)),
        layout(ht!((8, 4)), ht!((1, 8))),
        layout(ht!((8, 4)), ht!((4, 1))),
        layout(ht!((2, 4, 6)), ht!((1, 2, 8))),
        layout(ht!((2, 4, 6)), ht!((4, 1, 8))),
        layout(ht!((2, 4, 8)), ht!((32, 0, 2))),
        layout(ht!((2, 4, 8)), ht!((2, 0, 32))),
        layout(ht!((2, 4, 4, 4, 2)), ht!((32, 0, 2, 0, 512))),
        layout(ht!((4, 2)), ht!((1, 16))),
        layout(ht!((4, 2)), ht!((1, 5))),
        layout(ht!((4, 2)), ht!((1, 10))),
        layout(ht!((4, 2)), ht!((1, 11))),
    ] {
        postcondition_left_inverse(&source);
    }
}

#[test]
fn left_inverse_inverts_the_tmem_layouts() {
    for source in [
        layout(ht!((32, 8)), ht!((65536, 1))),
        layout(ht!((32, 12)), ht!((65536, 1))),
        layout(ht!((32, 3, 8)), ht!((65536, 512, 1))),
        layout(ht!((32, 8)), ht!((131072, 2))),
        layout(
            ht!((((((2, 4), 1), (2, 2)), 4), 1, (2, 2), 2)),
            ht!((((((262144, 4), 0), (0, 1)), 8388608), 0, (2, 16), 32)),
        ),
    ] {
        postcondition_left_inverse(&source);
    }
}

#[test]
fn left_inverse_rejects_an_unordered_chain() {
    // Coprime (non-divisible) strides are injective but unordered, and
    // are rejected as a deliberate simplification even though a layout
    // left inverse exists.
    for source in [
        layout(ht!((2, 2)), ht!((2, 3))),
        layout(ht!((2, 2)), ht!((2, 5))),
    ] {
        assert!(
            matches!(
                source.left_inverse(),
                Err(pinstripe::Error::Divisibility { .. })
            ),
            "{source} should not form an ordered chain"
        );
    }
}

#[test]
fn left_inverse_rejects_a_non_injective_layout() {
    // Overlapping / repeated non-zero strides.
    for source in [
        layout(ht!((63, 2)), ht!((1, 1))),
        layout(ht!((2, 2)), ht!((1, 1))),
        layout(ht!((2, 3)), ht!((2, 1))),
    ] {
        assert!(
            matches!(
                source.left_inverse(),
                Err(pinstripe::Error::NonInjective { .. })
            ),
            "{source} should be non-injective"
        );
    }
}

#[test]
fn left_inverse_inverts_over_basis_strides() {
    for source in [
        strided(ht!((4, 5)), t(vec![s(e(&[0])), s(e(&[1]))])),
        strided(ht!((4, 5)), t(vec![s(e(&[1])), s(e(&[0]))])),
        strided(ht!((4, 5)), t(vec![s(e(&[1])), s(e(&[4, 1]))])),
        strided(
            ht!((4, 5)),
            t(vec![s(e(&[0]).scale(2)), s(e(&[1]).scale(2))]),
        ),
        strided(
            ht!((3, (2, 2))),
            t(vec![
                s(e(&[0]).scale(34)),
                t(vec![s(e(&[0]).scale(2)), s(e(&[1]).scale(2))]),
            ]),
        ),
        // SM70 MMA 8x8x4 C TV inverse.
        sm70_c_tv(4, 4),
        sm70_c_tv(6, 6),
        sm70_c_tv(6, 4),
        // SM70 MMA 8x8x4 A TV inverse.
        tv(ht!((8, 4)), &layout(ht!(((4, 2), 4)), ht!(((8, 4), 1)))),
        // SM80 MMA 16x8 TV inverse.
        tv(
            ht!((16, 8)),
            &layout(ht!(((4, 8), (2, 2))), ht!(((32, 1), (16, 8)))),
        ),
    ] {
        postcondition_left_inverse(&source);
    }
}

/// PyCuTe's `test_left_inverse_app`: a common cotiling failure.
///
/// The data layout's inverse takes an address back to a data coordinate,
/// so composing it with the atom's (tid, vid) layout gives (tid, vid) ->
/// data coordinate. Mapping that back through the data layout has to
/// return the atom: `D o (Di o TV) == TV`.
#[test]
fn left_inverse_recovers_a_tv_layout_through_its_data_layout() {
    let atom_tv = layout(ht!(((32, 4), (16, 32))), ht!(((0, 2097152), (1, 65536))));
    let data = layout(ht!((128, 16)), ht!((65536, 1)));

    // data addr -> data coord. The appended `1:0` gives the
    // off-the-ends the stride-0.
    let inv_data = make_layout(vec![data.left_inverse().unwrap(), layout(ht!(1), ht!(0))]);
    // (tid, vid) -> data coord.
    let tv_data = inv_data.composition(&tiler(&atom_tv)).unwrap();

    let all = HTuple::Leaf(Some(1));
    assert_eq!(
        data.composition(&tiler(&tv_data))
            .unwrap()
            .coalesce(&all)
            .unwrap(),
        atom_tv.coalesce(&all).unwrap()
    );
}

#[test]
fn left_inverse_reads_off_the_worked_cases() {
    assert_eq!(
        layout(ht!((8, 4)), ht!((4, 1))).left_inverse().unwrap(),
        layout(ht!((4, 8)), ht!((8, 1)))
    );
    // The 4:1 mode is padded out to the next stride, so the holes
    // between 4 and 16 become part of the domain.
    assert_eq!(
        layout(ht!((4, 2)), ht!((1, 16))).left_inverse().unwrap(),
        layout(ht!((16, 2)), ht!((1, 4)))
    );
}
