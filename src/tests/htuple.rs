//! Ported from `test/test_htuple.py`.
//!
//! `test_inner_product` and `test_prefix_product` live in `stride.rs`,
//! and `test_idx2crd` and `TestModeOpDecorator` in `shape.rs`: this port
//! follows PyCuTe's module for each function, and those four sit outside
//! `htuple`. The `Layout` cases are here.

use crate::{HTuple, IntTuple, Layout, Stride};

/// `Layout(shape, stride)` over integer strides.
fn layout(shape: IntTuple, stride: IntTuple) -> Layout {
    Layout::from_base(shape, &Stride::from(stride)).unwrap()
}

#[test]
fn is_tuple_separates_a_node_from_a_leaf() {
    assert!(ht!((1, 2, 3)).is_tuple());
    assert!(HTuple::<i64>::Tuple(vec![]).is_tuple());
    assert!(!ht!(7).is_tuple());
}

#[test]
fn wrap_makes_a_one_tuple_and_strip_singletons_undoes_it() {
    assert_eq!(ht!(7).wrap(), ht!((7)));
    assert_eq!(ht!((7)).wrap(), ht!((7)));
    assert_eq!(ht!((7, 8)).wrap(), ht!((7, 8)));

    assert_eq!(ht!(42).strip_singletons(), &ht!(42));
    assert_eq!(ht!((42)).strip_singletons(), &ht!(42));
    assert_eq!(ht!((42)).strip_singletons(), &ht!(42));
    assert_eq!(ht!((1, 2)).strip_singletons(), &ht!((1, 2)));
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
    assert_eq!(ht!(2).size(), 2);
    assert_eq!(ht!((3, 2)).size(), 6);
    assert_eq!(ht!(((2, 3), 4)).size(), 24);
}

#[test]
fn product_each_collapses_each_mode_and_keeps_the_rank() {
    assert_eq!(ht!((2, 3)).size_each(), ht!((2, 3)));
    assert_eq!(ht!((2, 3, 4)).size_each(), ht!((2, 3, 4)));
    assert_eq!(ht!(((2, 3), 4)).size_each(), ht!((6, 4)));
    assert_eq!(ht!(((2, (3, 4)), (5, 6), 7)).size_each(), ht!((24, 30, 7)));
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

/// One sub-layout per named mode, which is [`Layout::mode`] here. A
/// layout is not an [`HTuple`], so it has no `select` of its own.
#[test]
fn select_over_a_layout_reads_one_sub_layout_per_mode() {
    let a = layout(ht!((2, 3, 5, 7)), ht!((1, 2, 6, 30)));
    let select = |modes: &[usize]| modes.iter().map(|&i| a.mode(i).unwrap()).collect::<Vec<_>>();
    assert_eq!(select(&[1, 3]), vec![layout(ht!(3), ht!(2)), layout(ht!(7), ht!(30))]);
    assert_eq!(
        select(&[0, 1, 3]),
        vec![layout(ht!(2), ht!(1)), layout(ht!(3), ht!(2)), layout(ht!(7), ht!(30)),]
    );
    assert_eq!(select(&[2]), vec![layout(ht!(5), ht!(6))]);

    // `Layout::from_modes(select[I...](A))` is the C++-style `cute::select`.
    assert_eq!(Layout::from_modes(select(&[1, 3])), layout(ht!((3, 7)), ht!((2, 30))));
    assert_eq!(Layout::from_modes(select(&[0, 1, 3])), layout(ht!((2, 3, 7)), ht!((1, 2, 30))));
}

/// PyCuTe's `take[begin, end](A)` over a layout: the consecutive modes
/// of the range.
#[test]
fn take_over_a_layout_reads_a_run_of_sub_layouts() {
    let a = layout(ht!((2, 3, 5, 7)), ht!((1, 2, 6, 30)));
    let take = |begin, end| (begin..end).map(|i| a.mode(i).unwrap()).collect::<Vec<_>>();
    assert_eq!(take(1, 3), vec![layout(ht!(3), ht!(2)), layout(ht!(5), ht!(6))]);
    assert_eq!(
        take(1, 4),
        vec![layout(ht!(3), ht!(2)), layout(ht!(5), ht!(6)), layout(ht!(7), ht!(30)),]
    );
}

#[test]
fn transform_leaf_maps_every_leaf_and_keeps_the_profile() {
    assert_eq!(ht!(((1, (2, 3)), 4)).transform_leaf(&|a| a * 2), ht!(((2, (4, 6)), 8)));
}

#[test]
fn zip_transform_maps_pairs_of_leaves() {
    let sum = ht!(((1, (2, 3)), 4)).zip_transform(&ht!(((10, (20, 30)), 40)), &|a, b| a + b);
    assert_eq!(sum.unwrap(), ht!(((11, (22, 33)), 44)));
}

#[test]
fn zip_transform_rejects_a_profile_mismatch() {
    assert!(ht!((1, 2)).zip_transform(&ht!((1, (2, 3))), &|a, b| a + b).is_err());
}

#[test]
fn flatten_collects_the_leaves() {
    assert_eq!(ht!(((1, (2, 3)), 4)).flatten(), ht!((1, 2, 3, 4)));
    assert_eq!(ht!(7).flatten(), ht!((7)));
}

#[test]
fn leaves_reads_them_in_pre_order() {
    assert_eq!(ht!(((1, (2, 3)), 4)).leaves().collect::<Vec<_>>(), vec![&1, &2, &3, &4]);
}

#[test]
fn leaf_paths_inverts_get() {
    let t = ht!(((1, (2, 3)), 4));
    assert_eq!(t.leaf_paths(), vec![vec![0, 0], vec![0, 1, 0], vec![0, 1, 1], vec![1]]);
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
        let mut leaves = t.leaves().copied().collect::<Vec<_>>().into_iter();
        assert_eq!(HTuple::unflatten(&mut leaves, &t), Some(t.clone()));
    }
}

#[test]
fn repeat_like_fills_a_profile() {
    assert_eq!(HTuple::repeat_like(&0, &ht!(((1, (2, 3)), 4))), ht!(((0, (0, 0)), 0)));
    assert_eq!(HTuple::repeat_like(&7, &ht!(1)), ht!(7));
}

#[test]
fn congruent_compares_profiles_and_ignores_values() {
    assert!(ht!((4, 8)).congruent(&ht!((5, 7))));
    assert!(ht!(31).congruent(&ht!(42)));
    assert!(!ht!((4, 8)).congruent(&ht!((4, (2, 4)))));
    assert!(!ht!(31).congruent(&ht!((4, 8))));
    assert!(!ht!((1, 1, 1)).congruent(&ht!((1, 1))));
}

#[test]
fn weakly_congruent_is_the_coarsening_order() {
    assert!(ht!(30).weakly_congruent(&ht!((3, 4))));
    assert!(ht!(30).weakly_congruent(&ht!(((3, 4), 5))));
    assert!(!ht!((3, 4)).weakly_congruent(&ht!(30)));
    assert!(ht!((3, 4)).weakly_congruent(&ht!((5, (6, 7)))));
    assert!(!ht!((3, (4, 5))).weakly_congruent(&ht!((5, 6))));
    assert!(!ht!((1, 2, 3)).weakly_congruent(&ht!((1, 2))));
}

#[test]
fn zip_leaves_faces_a_leaf_with_a_sub_tuple() {
    let a = ht!((1, 2));
    let b = ht!(((10, 20), 30));
    assert_eq!(a.zip_leaves(&b).unwrap(), vec![(&1, &ht!((10, 20))), (&2, &ht!(30))]);
}

#[test]
fn zip_leaves_descends_into_matching_profiles() {
    let a = ht!(((1, 2), 3));
    let b = ht!(((10, 20), 30));
    assert_eq!(a.zip_leaves(&b).unwrap(), vec![(&1, &ht!(10)), (&2, &ht!(20)), (&3, &ht!(30))]);
}

#[test]
fn zip_leaves_rejects_a_finer_first_tuple() {
    assert!(ht!(((1, 2), 3)).zip_leaves(&ht!((10, 30))).is_err());
}

#[test]
fn fold_leaves_runs_over_the_pairs_left_to_right() {
    let total =
        ht!((1, 2)).fold_leaves(&ht!(((10, 20), 30)), 0, &|acc, a: &i64, b: &HTuple<i64>| {
            acc + a * b.size()
        });
    assert_eq!(total.unwrap(), 200 + 60);
}

#[test]
fn slice_and_dice_split_on_the_open_modes() {
    let crd = HTuple::Tuple(vec![HTuple::Leaf(None), HTuple::Leaf(Some(3_i64))]);
    let shape = ht!(((2, 3), 4));
    assert_eq!(crd.slice(&shape).unwrap(), ht!(((2, 3))));
    assert_eq!(crd.dice(&shape).unwrap(), ht!((4)));
}

#[test]
fn slice_and_dice_flatten_one_level() {
    let crd = HTuple::Tuple(vec![
        HTuple::Tuple(vec![HTuple::Leaf(None), HTuple::Leaf(Some(1_i64))]),
        HTuple::Leaf(None),
    ]);
    let shape = ht!(((2, 3), 4));
    assert_eq!(crd.slice(&shape).unwrap(), ht!((2, 4)));
    assert_eq!(crd.dice(&shape).unwrap(), ht!((3)));
}
