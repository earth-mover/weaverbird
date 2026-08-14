//! Ported from `test/test_htuple.py`.
//!
//! The cases that need `Layout`, `inner_product`, `prefix_product`, or
//! `idx2crd` wait for those modules.
#![expect(clippy::unwrap_used, reason = "a test asserts the happy path")]

use pinstripe::{
    HTuple, ht,
    htuple::{
        congruent, dice_, fold_leaf, slice_, transform_apply_leaf, transform_apply_leaf2,
        transform_apply_leaf4, weakly_congruent, zip_leaves, zip_transform_leaf,
        zip3_transform_leaf,
    },
};

#[test]
fn is_tuple_separates_a_node_from_a_leaf() {
    assert!(ht!((1, 2, 3)).is_tuple());
    assert!(HTuple::<i64>::Tuple(vec![]).is_tuple());
    assert!(!ht!(7).is_tuple());
}

#[test]
fn wrap_makes_a_one_tuple_and_unwrap_strips_them() {
    assert_eq!(ht!(7).wrap(), ht!((7)));
    assert_eq!(ht!((7)).wrap(), ht!((7)));
    assert_eq!(ht!((7, 8)).wrap(), ht!((7, 8)));

    assert_eq!(ht!(42).unwrap(), &ht!(42));
    assert_eq!(ht!((42)).unwrap(), &ht!(42));
    assert_eq!(ht!((42)).unwrap(), &ht!(42));
    assert_eq!(ht!((1, 2)).unwrap(), &ht!((1, 2)));
}

#[test]
fn front_and_back_read_the_ends_or_the_leaf() {
    assert_eq!(ht!((1, 2, 3)).front(), &ht!(1));
    assert_eq!(ht!((1, 2, 3)).back(), &ht!(3));
    assert_eq!(ht!(7).front(), &ht!(7));
    assert_eq!(ht!(7).back(), &ht!(7));
}

#[test]
fn replace_writes_the_ends_and_a_leaf_becomes_the_value() {
    assert_eq!(ht!((1, 2, 3)).replace_front(ht!(9)), ht!((9, 2, 3)));
    assert_eq!(ht!((1, 2, 3)).replace_back(ht!(9)), ht!((1, 2, 9)));
    assert_eq!(ht!(0).replace_front(ht!(9)), ht!(9));
    assert_eq!(ht!(0).replace_back(ht!(9)), ht!(9));
}

#[test]
fn product_multiplies_every_leaf() {
    assert_eq!(ht!(2).product(), 2);
    assert_eq!(ht!((3, 2)).product(), 6);
    assert_eq!(ht!(((2, 3), 4)).product(), 24);
}

#[test]
fn product_each_collapses_each_mode_and_keeps_the_rank() {
    assert_eq!(ht!((2, 3)).product_each(), ht!((2, 3)));
    assert_eq!(ht!((2, 3, 4)).product_each(), ht!((2, 3, 4)));
    assert_eq!(ht!(((2, 3), 4)).product_each(), ht!((6, 4)));
    assert_eq!(
        ht!(((2, (3, 4)), (5, 6), 7)).product_each(),
        ht!((24, 30, 7))
    );
}

#[test]
fn get_walks_a_path_and_an_empty_path_is_the_identity() {
    assert_eq!(ht!(((0, 0, (0, 0, 0, 42)))).get(&[0, 2, 3]), Some(&ht!(42)));
    assert_eq!(ht!((1, (2, (3, 4)))).get(&[1, 1, 0]), Some(&ht!(3)));
    assert_eq!(ht!(7).get(&[]), Some(&ht!(7)));
    assert_eq!(ht!(7).get(&[0]), None);
}

#[test]
fn lift_and_get_invert_each_other() {
    for path in [vec![], vec![0], vec![1], vec![0, 2, 3], vec![1, 1, 0]] {
        assert_eq!(HTuple::lift(99, &path).get(&path), Some(&ht!(99)));
    }
}

#[test]
fn lift_pads_with_zeros() {
    assert_eq!(HTuple::lift(42, &[0, 2, 3]), ht!(((0, 0, (0, 0, 0, 42)))));
}

#[test]
fn select_picks_top_level_modes() {
    let t = ht!((1, 2, 3, 4));
    assert_eq!(t.select(&[0, 2]).unwrap(), ht!((1, 3)));
    assert_eq!(t.select(&[2]).unwrap(), ht!((3)));
    assert_eq!(t.select(&[0, 1, 3]).unwrap(), ht!((1, 2, 4)));
}

#[test]
fn take_picks_a_half_open_range() {
    let t = ht!((1, 2, 3, 4));
    assert_eq!(t.take(1, 3).unwrap(), ht!((2, 3)));
    assert_eq!(t.take(0, 4).unwrap(), ht!((1, 2, 3, 4)));
    assert_eq!(t.take(1, 1).unwrap(), HTuple::Tuple(vec![]));
}

#[test]
fn take_rejects_a_reversed_range() {
    assert!(ht!((1, 2, 3, 4)).take(3, 1).is_err());
}

#[test]
fn transform_leaf_maps_every_leaf_and_keeps_the_profile() {
    assert_eq!(
        ht!(((1, (2, 3)), 4)).transform_leaf(&|a| a * 2),
        ht!(((2, (4, 6)), 8))
    );
}

#[test]
fn zip_transform_leaf_maps_pairs_of_leaves() {
    let sum = zip_transform_leaf(
        &|a, b| a + b,
        &ht!(((1, (2, 3)), 4)),
        &ht!(((10, (20, 30)), 40)),
    );
    assert_eq!(sum.unwrap(), ht!(((11, (22, 33)), 44)));
}

#[test]
fn zip_transform_leaf_rejects_a_profile_mismatch() {
    assert!(zip_transform_leaf(&|a, b| a + b, &ht!((1, 2)), &ht!((1, (2, 3)))).is_err());
}

#[test]
fn flatten_collects_the_leaves() {
    assert_eq!(ht!(((1, (2, 3)), 4)).flatten(), ht!((1, 2, 3, 4)));
    assert_eq!(ht!(7).flatten(), ht!((7)));
}

#[test]
fn leaves_reads_them_in_pre_order() {
    assert_eq!(ht!(((1, (2, 3)), 4)).leaves(), vec![&1, &2, &3, &4]);
}

#[test]
fn leaf_paths_inverts_get() {
    let t = ht!(((1, (2, 3)), 4));
    assert_eq!(
        t.leaf_paths(),
        vec![vec![0, 0], vec![0, 1, 0], vec![0, 1, 1], vec![1]]
    );
    for (path, leaf) in t.leaf_paths().into_iter().zip(t.leaves()) {
        assert_eq!(t.get(&path), Some(&HTuple::Leaf(*leaf)));
    }
}

#[test]
fn unflatten_rebuilds_along_a_profile() {
    let rebuilt = HTuple::unflatten(&mut [10, 20, 30, 40].into_iter(), &ht!(((1, (2, 3)), 4)));
    assert_eq!(rebuilt, Some(ht!(((10, (20, 30)), 40))));
}

#[test]
fn flatten_and_unflatten_round_trip() {
    for t in [ht!(7), ht!((3, 4)), ht!(((1, 2), 3)), ht!(((1, (2, 3)), 4))] {
        let mut leaves = t.leaves().into_iter().copied();
        assert_eq!(HTuple::unflatten(&mut leaves, &t), Some(t.clone()));
    }
}

#[test]
fn repeat_like_fills_a_profile() {
    assert_eq!(
        HTuple::repeat_like(&0, &ht!(((1, (2, 3)), 4))),
        ht!(((0, (0, 0)), 0))
    );
    assert_eq!(HTuple::repeat_like(&7, &ht!(1)), ht!(7));
}

#[test]
fn congruent_compares_profiles_and_ignores_values() {
    assert!(congruent(&ht!((4, 8)), &ht!((5, 7))));
    assert!(congruent(&ht!(31), &ht!(42)));
    assert!(!congruent(&ht!((4, 8)), &ht!((4, (2, 4)))));
    assert!(!congruent(&ht!(31), &ht!((4, 8))));
    assert!(!congruent(&ht!((1, 1, 1)), &ht!((1, 1))));
}

#[test]
fn weakly_congruent_is_the_coarsening_order() {
    assert!(weakly_congruent(&ht!(30), &ht!((3, 4))));
    assert!(weakly_congruent(&ht!(30), &ht!(((3, 4), 5))));
    assert!(!weakly_congruent(&ht!((3, 4)), &ht!(30)));
    assert!(weakly_congruent(&ht!((3, 4)), &ht!((5, (6, 7)))));
    assert!(!weakly_congruent(&ht!((3, (4, 5))), &ht!((5, 6))));
    assert!(!weakly_congruent(&ht!((1, 2, 3)), &ht!((1, 2))));
}

/// The product of every leaf, counting an absent tuple as the empty
/// product. The leaf function of the `transform_apply_leaf` cases.
fn prod(t: Option<&HTuple<i64>>) -> i64 {
    t.map_or(1, |x| x.leaves().into_iter().product())
}

#[test]
fn zip3_transform_leaf_maps_triples_of_leaves() {
    let sum = zip3_transform_leaf(
        &|a, b, c| Ok(a + b + c),
        &ht!(((1, (2, 3)), 4)),
        &ht!(((10, (20, 30)), 40)),
        &ht!(((100, (200, 300)), 400)),
    );
    assert_eq!(sum.unwrap(), ht!(((111, (222, 333)), 444)));
}

#[test]
fn zip3_transform_leaf_rejects_a_profile_mismatch() {
    let f = |a: &i64, b: &i64, c: &i64| Ok(a + b + c);
    assert!(zip3_transform_leaf(&f, &ht!((1, 2)), &ht!((1, (2, 3))), &ht!((1, 2))).is_err());
}

#[test]
fn transform_apply_leaf_faces_each_leaf_of_the_driver() {
    let total = transform_apply_leaf(
        &|parts: Vec<i64>| parts.into_iter().sum(),
        &|a, b| Ok(prod(a) * prod(b)),
        Some(&ht!((1, 2))),
        Some(&ht!(((10, 20), 30))),
    );
    assert_eq!(total.unwrap(), 200 + 60);
}

#[test]
fn transform_apply_leaf_pads_the_shorter_tuple() {
    let total = transform_apply_leaf(
        &|parts: Vec<i64>| parts.into_iter().sum(),
        &|a, b| Ok(prod(a) * prod(b)),
        Some(&ht!((1, 2))),
        Some(&ht!((10))),
    );
    assert_eq!(total.unwrap(), 10 + 2);
}

#[test]
fn transform_apply_leaf_applies_f_to_a_leaf_driver() {
    let total = transform_apply_leaf(
        &|parts: Vec<i64>| parts.into_iter().sum(),
        &|a, b| Ok(prod(a) * prod(b)),
        Some(&ht!(7)),
        Some(&ht!((2, 3))),
    );
    assert_eq!(total.unwrap(), 42);
}

#[test]
fn transform_apply_leaf_rejects_a_leaf_where_it_must_zip() {
    let total = transform_apply_leaf(
        &|parts: Vec<i64>| parts.into_iter().sum(),
        &|a, b| Ok(prod(a) * prod(b)),
        Some(&ht!((1, 2))),
        Some(&ht!(5)),
    );
    assert!(total.is_err());
}

#[test]
fn transform_apply_leaf2_zips_two_extra_tuples() {
    let total = transform_apply_leaf2(
        &|parts: Vec<i64>| parts.into_iter().sum(),
        &|a, b, c| Ok(prod(a) * prod(b) * prod(c)),
        Some(&ht!((1, 2))),
        Some(&ht!((10, 20))),
        Some(&ht!((100, 200))),
    );
    assert_eq!(total.unwrap(), 1000 + 8000);
}

#[test]
fn transform_apply_leaf4_zips_four_extra_tuples() {
    let total = transform_apply_leaf4(
        &|parts: Vec<i64>| parts.into_iter().sum(),
        &|a, b, c, d, e| Ok(prod(a) * prod(b) * prod(c) * prod(d) * prod(e)),
        Some(&ht!((1, 2))),
        Some(&ht!((3, 4))),
        Some(&ht!((5, 6))),
        Some(&ht!((7, 8))),
        Some(&ht!((9, 10))),
    );
    assert_eq!(total.unwrap(), 945 + 3840);
}

#[test]
fn zip_leaves_faces_a_leaf_with_a_sub_tuple() {
    let a = ht!((1, 2));
    let b = ht!(((10, 20), 30));
    assert_eq!(
        zip_leaves(&a, &b).unwrap(),
        vec![(&1, &ht!((10, 20))), (&2, &ht!(30))]
    );
}

#[test]
fn zip_leaves_descends_into_matching_profiles() {
    let a = ht!(((1, 2), 3));
    let b = ht!(((10, 20), 30));
    assert_eq!(
        zip_leaves(&a, &b).unwrap(),
        vec![(&1, &ht!(10)), (&2, &ht!(20)), (&3, &ht!(30))]
    );
}

#[test]
fn zip_leaves_rejects_a_finer_first_tuple() {
    assert!(zip_leaves(&ht!(((1, 2), 3)), &ht!((10, 30))).is_err());
}

#[test]
fn fold_leaf_runs_over_the_pairs_left_to_right() {
    let total = fold_leaf(
        &|acc, a: &i64, b: &HTuple<i64>| acc + a * b.product(),
        0,
        &ht!((1, 2)),
        &ht!(((10, 20), 30)),
    );
    assert_eq!(total.unwrap(), 200 + 60);
}

#[test]
fn slice_and_dice_split_on_the_open_modes() {
    let crd = HTuple::Tuple(vec![HTuple::Leaf(None), HTuple::Leaf(Some(3_i64))]);
    let shape = ht!(((2, 3), 4));
    assert_eq!(slice_(&crd, &shape).unwrap(), ht!(((2, 3))));
    assert_eq!(dice_(&crd, &shape).unwrap(), ht!((4)));
}

#[test]
fn slice_and_dice_flatten_one_level() {
    let crd = HTuple::Tuple(vec![
        HTuple::Tuple(vec![HTuple::Leaf(None), HTuple::Leaf(Some(1_i64))]),
        HTuple::Leaf(None),
    ]);
    let shape = ht!(((2, 3), 4));
    assert_eq!(slice_(&crd, &shape).unwrap(), ht!((2, 4)));
    assert_eq!(dice_(&crd, &shape).unwrap(), ht!((3)));
}
