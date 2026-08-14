//! Ported from `pycute/stride.py`'s docstring examples.
//!
//! `test/test_coalesce_z.py` drives the `Layout`-level `coalesce_z` — it
//! builds `Layout`s and evaluates them — so it waits for that module.
//! The stride-level fold is exercised here directly instead.
#![expect(clippy::unwrap_used, reason = "a test asserts the happy path")]

use pinstripe::{
    HTuple, Int, IntTuple, Stride, StrideScalar, coprofile, coshape, e, ht, inner_product,
    prefix_product, stride,
    stride::{Coshape, coalesce_z},
};

/// The integer tuple as a stride, so the ported cases read like their
/// Python source.
fn as_stride(t: &IntTuple) -> Stride {
    t.transform_leaf(&|v: &Int| StrideScalar::Int(*v))
}

/// The default `init` of [`prefix_product`].
fn one() -> Stride {
    HTuple::Leaf(StrideScalar::Int(1))
}

#[test]
fn stride_reads_the_sub_stride_at_a_mode() {
    let s = as_stride(&ht!((1, (3, 6))));
    assert_eq!(stride(&s, &[]).unwrap(), &s);
    assert_eq!(stride(&s, &[1]).unwrap(), &as_stride(&ht!((3, 6))));
    assert_eq!(
        stride(&s, &[1, 0]).unwrap(),
        &HTuple::Leaf(StrideScalar::Int(3))
    );
    assert!(stride(&s, &[2]).is_err());
    assert!(stride(&s, &[0, 0]).is_err());
}

#[test]
fn inner_product_sums_the_leaf_wise_products() {
    let product = |a: IntTuple, b: IntTuple| inner_product(&a, &as_stride(&b)).unwrap();
    assert_eq!(
        product(ht!((1, 0, 1)), ht!((1, 3, 6))),
        StrideScalar::Int(7)
    );
    assert_eq!(product(ht!((2, 3)), ht!((1, 4))), StrideScalar::Int(14));
    assert_eq!(
        product(ht!((1, (2, 3))), ht!((1, (10, 100)))),
        StrideScalar::Int(321)
    );
}

#[test]
fn inner_product_over_basis_strides_lands_in_the_codomain() {
    let basis = HTuple::Tuple(vec![HTuple::Leaf(e(&[0])), HTuple::Leaf(e(&[1]))]);
    assert_eq!(
        inner_product(&ht!((3, 5)), &basis).unwrap(),
        e(&[0]).scale(3).add(&e(&[1]).scale(5)).unwrap()
    );
}

#[test]
fn inner_product_rejects_an_incongruent_pair() {
    assert!(inner_product(&ht!((1, 2)), &as_stride(&ht!((1, 2, 3)))).is_err());
    assert!(inner_product(&ht!((1, (2, 3))), &as_stride(&ht!((1, 2)))).is_err());
}

#[test]
fn prefix_product_is_the_exclusive_product_of_the_leaves() {
    let prefix = |a: IntTuple| prefix_product(&a, &one()).unwrap();
    assert_eq!(prefix(ht!((3, 2, 4))), as_stride(&ht!((1, 3, 6))));
    assert_eq!(prefix(ht!((3, (2, 4)))), as_stride(&ht!((1, (3, 6)))));
    assert_eq!(prefix(ht!(7)), as_stride(&ht!(1)));
}

#[test]
fn prefix_product_seeds_the_running_product_with_init() {
    assert_eq!(
        prefix_product(&ht!((4, 8)), &HTuple::Leaf(StrideScalar::Int(2))).unwrap(),
        as_stride(&ht!((2, 8)))
    );
}

#[test]
fn prefix_product_takes_one_base_per_mode() {
    assert_eq!(
        prefix_product(&ht!(((2, 3), (4, 5))), &as_stride(&ht!((1, 100)))).unwrap(),
        as_stride(&ht!(((1, 2), (100, 400))))
    );
}

#[test]
fn prefix_product_rejects_an_init_that_does_not_coarsen_the_shape() {
    assert!(prefix_product(&ht!((3, 2)), &as_stride(&ht!((1, 2, 3)))).is_err());
    assert!(prefix_product(&ht!(6), &as_stride(&ht!((1, 2)))).is_err());
}

/// A stand-in for the `Layout` that will implement [`Coshape`].
struct Codomain(IntTuple);

impl Coshape for Codomain {
    fn coshape(&self) -> IntTuple {
        self.0.clone()
    }
}

#[test]
fn coshape_reads_the_codomain_at_a_mode() {
    let obj = Codomain(ht!((3, (2, 4))));
    assert_eq!(coshape(&obj, &[]).unwrap(), ht!((3, (2, 4))));
    assert_eq!(coshape(&obj, &[1]).unwrap(), ht!((2, 4)));
    assert_eq!(coshape(&obj, &[1, 1]).unwrap(), ht!(4));
    assert!(coshape(&obj, &[2]).is_err());
}

#[test]
fn coprofile_agrees_with_coshape() {
    let obj = Codomain(ht!((3, (2, 4))));
    assert_eq!(coprofile(&obj, &[1]).unwrap(), coshape(&obj, &[1]).unwrap());
}

#[test]
fn coalesce_z_merges_adjacent_modes() {
    let fold = |s: IntTuple, d: IntTuple| coalesce_z(&s, &as_stride(&d)).unwrap();
    // (2,4):(1,2) is one contiguous run of 8.
    assert_eq!(
        fold(ht!((2, 4)), ht!((1, 2))),
        (ht!((8)), as_stride(&ht!((1))))
    );
    // (2,4,6):(1,2,8) folds all the way down.
    assert_eq!(
        fold(ht!((2, 4, 6)), ht!((1, 2, 8))),
        (ht!((48)), as_stride(&ht!((1))))
    );
    // A stride that skips blocks the merge.
    assert_eq!(
        fold(ht!((2, 4)), ht!((1, 4))),
        (ht!((2, 4)), as_stride(&ht!((1, 4))))
    );
}

#[test]
fn coalesce_z_drops_size_one_modes_it_folds_across() {
    let fold = |s: IntTuple, d: IntTuple| coalesce_z(&s, &as_stride(&d)).unwrap();
    assert_eq!(
        fold(ht!((2, 1, 6)), ht!((1, 7, 2))),
        (ht!((12)), as_stride(&ht!((1))))
    );
    // A trailing size-1 mode has nothing after it, so it stays — this is
    // what the "_z" fold preserves.
    assert_eq!(
        fold(ht!((2, 1)), ht!((1, 7))),
        (ht!((2, 1)), as_stride(&ht!((1, 7))))
    );
}

#[test]
fn coalesce_z_folds_basis_strides() {
    let basis = HTuple::Tuple(vec![HTuple::Leaf(e(&[0])), HTuple::Leaf(e(&[0]).scale(2))]);
    assert_eq!(
        coalesce_z(&ht!((2, 4)), &basis).unwrap(),
        (ht!((8)), HTuple::Tuple(vec![HTuple::Leaf(e(&[0]))]))
    );
    // Distinct basis directions never fold together.
    let mixed = HTuple::Tuple(vec![HTuple::Leaf(e(&[0])), HTuple::Leaf(e(&[1]))]);
    assert_eq!(
        coalesce_z(&ht!((2, 4)), &mixed).unwrap(),
        (ht!((2, 4)), mixed)
    );
}

#[test]
fn coalesce_z_is_idempotent() {
    let cases = [
        (ht!((2, 4)), ht!((1, 2))),
        (ht!((2, 1, 6)), ht!((1, 7, 2))),
        (ht!((2, 4, 6)), ht!((1, 6, 2))),
        (ht!(((2, 2), (2, 2))), ht!(((1, 4), (8, 32)))),
    ];
    for (s, d) in cases {
        let once = coalesce_z(&s, &as_stride(&d)).unwrap();
        assert_eq!(coalesce_z(&once.0, &once.1).unwrap(), once);
    }
}

#[test]
fn coalesce_z_preserves_the_layout_map() {
    // Every coordinate of the folded shape sends to the same offset as
    // the coordinate it came from, which is `crd2idx` in miniature.
    let cases = [
        (ht!((2, 4)), ht!((1, 2))),
        (ht!((2, 1, 6)), ht!((1, 7, 2))),
        (ht!((2, 4, 6)), ht!((1, 6, 2))),
        (ht!((2, 1, 3)), ht!((2, 4, 4))),
        (ht!(((2, 2), (2, 2))), ht!(((1, 4), (8, 32)))),
    ];
    for (s, d) in cases {
        let d = as_stride(&d);
        let (folded_s, folded_d) = coalesce_z(&s, &d).unwrap();
        assert_eq!(folded_s.product(), s.product());
        for i in 0..s.product() {
            let at = |shape: &IntTuple, stride: &Stride| {
                inner_product(&idx2crd(i, shape), stride).unwrap()
            };
            assert_eq!(at(&folded_s, &folded_d), at(&s, &d));
        }
    }
}

/// The colexicographic decomposition of `idx` over `shape`.
///
// TODO(merge): replaced by crate::shape at merge time
/// A stand-in for `shape.py`'s `idx2crd`, which is being ported
/// elsewhere.
fn idx2crd(idx: Int, shape: &IntTuple) -> IntTuple {
    let mut rest = idx;
    let mut crds = shape
        .leaves()
        .into_iter()
        .map(|&s| {
            let crd = rest % s;
            rest /= s;
            crd
        })
        .collect::<Vec<_>>()
        .into_iter();
    HTuple::unflatten(&mut crds, shape).unwrap()
}
