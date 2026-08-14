//! Unit tests for the stride helpers.
//!
//! These cases reach `coalesce_modes` and the stride helpers, which the
//! crate keeps private. The `Layout`-level fold is tested in
//! `tests/coalesce.rs`.

use crate::{HTuple, IntTuple, Stride, StrideScalar, e, ht, stride::coalesce_modes};

/// The default `init` of [`IntTuple::prefix_product`].
fn one() -> Stride {
    HTuple::Leaf(StrideScalar::Int(1))
}

#[test]
fn get_reads_the_sub_stride_at_a_mode() {
    let s = Stride::from(ht!((1, (3, 6))));
    assert_eq!(s.get(&[]), Some(&s));
    assert_eq!(s.get(&[1]), Some(&Stride::from(ht!((3, 6)))));
    assert_eq!(s.get(&[1, 0]), Some(&HTuple::Leaf(StrideScalar::Int(3))));
    assert_eq!(s.get(&[2]), None);
    assert_eq!(s.get(&[0, 0]), None);
}

#[test]
fn inner_product_sums_the_leaf_wise_products() {
    let product = |a: IntTuple, b: IntTuple| a.inner_product(&Stride::from(b)).unwrap();
    assert_eq!(product(ht!((1, 0, 1)), ht!((1, 3, 6))), StrideScalar::Int(7));
    assert_eq!(product(ht!((2, 3)), ht!((1, 4))), StrideScalar::Int(14));
    assert_eq!(product(ht!((1, (2, 3))), ht!((1, (10, 100)))), StrideScalar::Int(321));
}

#[test]
fn inner_product_over_basis_strides_lands_in_the_codomain() {
    let basis = HTuple::Tuple(vec![HTuple::Leaf(e(&[0])), HTuple::Leaf(e(&[1]))]);
    assert_eq!(
        ht!((3, 5)).inner_product(&basis).unwrap(),
        e(&[0]).scale(3).add(&e(&[1]).scale(5)).unwrap()
    );
}

#[test]
fn inner_product_rejects_an_incongruent_pair() {
    assert!(ht!((1, 2)).inner_product(&Stride::from(ht!((1, 2, 3)))).is_err());
    assert!(ht!((1, (2, 3))).inner_product(&Stride::from(ht!((1, 2)))).is_err());
}

#[test]
fn prefix_product_is_the_exclusive_product_of_the_leaves() {
    let prefix = |a: IntTuple| a.prefix_product(&one()).unwrap();
    assert_eq!(prefix(ht!((3, 2, 4))), Stride::from(ht!((1, 3, 6))));
    assert_eq!(prefix(ht!((3, (2, 4)))), Stride::from(ht!((1, (3, 6)))));
    assert_eq!(prefix(ht!(7)), Stride::from(ht!(1)));
}

#[test]
fn prefix_product_seeds_the_running_product_with_init() {
    assert_eq!(
        ht!((4, 8)).prefix_product(&HTuple::Leaf(StrideScalar::Int(2))).unwrap(),
        Stride::from(ht!((2, 8)))
    );
}

#[test]
fn prefix_product_takes_one_base_per_mode() {
    assert_eq!(
        ht!(((2, 3), (4, 5))).prefix_product(&Stride::from(ht!((1, 100)))).unwrap(),
        Stride::from(ht!(((1, 2), (100, 400))))
    );
}

#[test]
fn prefix_product_rejects_an_init_that_does_not_coarsen_the_shape() {
    assert!(ht!((3, 2)).prefix_product(&Stride::from(ht!((1, 2, 3)))).is_err());
    assert!(ht!(6).prefix_product(&Stride::from(ht!((1, 2)))).is_err());
}

#[test]
fn coalesce_modes_merges_adjacent_modes() {
    let fold = |s: IntTuple, d: IntTuple| coalesce_modes(&s, &Stride::from(d)).unwrap();
    // (2,4):(1,2) is one contiguous run of 8.
    assert_eq!(fold(ht!((2, 4)), ht!((1, 2))), (ht!((8)), Stride::from(ht!((1)))));
    // (2,4,6):(1,2,8) folds all the way down.
    assert_eq!(fold(ht!((2, 4, 6)), ht!((1, 2, 8))), (ht!((48)), Stride::from(ht!((1)))));
    // A stride that skips blocks the merge.
    assert_eq!(fold(ht!((2, 4)), ht!((1, 4))), (ht!((2, 4)), Stride::from(ht!((1, 4)))));
}

#[test]
fn coalesce_modes_drops_size_one_modes_it_folds_across() {
    let fold = |s: IntTuple, d: IntTuple| coalesce_modes(&s, &Stride::from(d)).unwrap();
    assert_eq!(fold(ht!((2, 1, 6)), ht!((1, 7, 2))), (ht!((12)), Stride::from(ht!((1)))));
    // A trailing size-1 mode has nothing after it, so it stays — this is
    // what the "_z" fold preserves.
    assert_eq!(fold(ht!((2, 1)), ht!((1, 7))), (ht!((2, 1)), Stride::from(ht!((1, 7)))));
}

#[test]
fn coalesce_modes_folds_basis_strides() {
    let basis = HTuple::Tuple(vec![HTuple::Leaf(e(&[0])), HTuple::Leaf(e(&[0]).scale(2))]);
    assert_eq!(
        coalesce_modes(&ht!((2, 4)), &basis).unwrap(),
        (ht!((8)), HTuple::Tuple(vec![HTuple::Leaf(e(&[0]))]))
    );
    // Distinct basis directions never fold together.
    let mixed = HTuple::Tuple(vec![HTuple::Leaf(e(&[0])), HTuple::Leaf(e(&[1]))]);
    assert_eq!(coalesce_modes(&ht!((2, 4)), &mixed).unwrap(), (ht!((2, 4)), mixed));
}

#[test]
fn coalesce_modes_is_idempotent() {
    let cases = [
        (ht!((2, 4)), ht!((1, 2))),
        (ht!((2, 1, 6)), ht!((1, 7, 2))),
        (ht!((2, 4, 6)), ht!((1, 6, 2))),
        (ht!(((2, 2), (2, 2))), ht!(((1, 4), (8, 32)))),
    ];
    for (s, d) in cases {
        let once = coalesce_modes(&s, &Stride::from(d)).unwrap();
        assert_eq!(coalesce_modes(&once.0, &once.1).unwrap(), once);
    }
}

#[test]
fn coalesce_modes_preserves_the_layout_map() {
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
        let d = Stride::from(d);
        let (folded_s, folded_d) = coalesce_modes(&s, &d).unwrap();
        assert_eq!(folded_s.size(), s.size());
        for i in 0..s.size() {
            let at = |shape: &IntTuple, stride: &Stride| {
                shape.idx2crd(&HTuple::Leaf(i)).unwrap().inner_product(stride).unwrap()
            };
            assert_eq!(at(&folded_s, &folded_d), at(&s, &d));
        }
    }
}
