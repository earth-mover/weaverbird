//! Ported from `test/test_compatibility.py`, plus the worked examples in
//! the docstrings of `pycute/shape.py`.
//!
//! PyCuTe reads a shape off any operand, so `test_accepts_layouts`
//! passes `Layout`s where the other cases pass tuples. Here the shape is
//! the argument, and a layout operand arrives as `&layout.shape`.
//!
//! `test/test_typing.py` holds no shape assertions to port: it exercises
//! Python's ABC registration, `typing.get_type_hints`, and module
//! layout, none of which survive the translation to Rust types.
#![expect(clippy::unwrap_used, reason = "a test asserts the happy path")]

use weaverbird::{HTuple, IntTuple, Layout, ht};

/// `Layout(shape)` — the compact, column-major default.
fn compact(shape: IntTuple) -> Layout {
    Layout::compact(shape)
}

// ---------------------------------------------------------------------------
// congruent — same hierarchical profile
// ---------------------------------------------------------------------------

#[test]
fn congruent_ignores_the_leaf_values() {
    assert!(ht!((4, 8)).congruent(&ht!((5, 7))));
    assert!(ht!(31).congruent(&ht!(42)));
    assert!(ht!(((4, 6), (3, (2, 2), 8))).congruent(&ht!(((1, 1), (1, (1, 1), 1)))));
}

#[test]
fn congruent_separates_different_profiles() {
    assert!(!ht!((4, 8)).congruent(&ht!((4, (2, 4)))));
    assert!(!ht!(31).congruent(&ht!((4, 8))));
    assert!(!ht!((1, 1, 1)).congruent(&ht!((1, 1))));
}

#[test]
fn congruent_is_reflexive() {
    for x in reflexive_cases() {
        assert!(x.congruent(&x));
    }
}

// ---------------------------------------------------------------------------
// weakly_congruent — the coarsening order on profiles
// ---------------------------------------------------------------------------

#[test]
fn weakly_congruent_lets_an_int_coarsen_anything() {
    assert!(ht!(30).weakly_congruent(&ht!((3, 4))));
    assert!(ht!(30).weakly_congruent(&ht!(((3, 4), 5))));
    assert!(ht!(30).weakly_congruent(&ht!((((3, 4), 5), (6, 7)))));
}

#[test]
fn weakly_congruent_never_lets_a_tuple_coarsen_an_int() {
    assert!(!ht!((3, 4)).weakly_congruent(&ht!(30)));
}

#[test]
fn weakly_congruent_needs_the_same_top_level_rank() {
    assert!(!ht!((1, 2, 3)).weakly_congruent(&ht!((1, 2))));
}

#[test]
fn weakly_congruent_recurses_into_the_modes() {
    assert!(ht!((3, 4)).weakly_congruent(&ht!((5, (6, 7)))));
    assert!(!ht!((3, (4, 5))).weakly_congruent(&ht!((5, 6))));
}

#[test]
fn weakly_congruent_is_reflexive() {
    for x in reflexive_cases() {
        assert!(x.weakly_congruent(&x));
    }
}

// ---------------------------------------------------------------------------
// compatible — the coarsening order on shapes
// ---------------------------------------------------------------------------

#[test]
fn compatible_walks_the_whitepaper_chains() {
    // 30 ≼ (2, 15) ≼ (2, (3, 5))
    assert!(ht!(30).compatible_with(&ht!((2, 15))));
    assert!(ht!((2, 15)).compatible_with(&ht!((2, (3, 5)))));
    assert!(ht!(30).compatible_with(&ht!((2, (3, 5)))));

    // 30 ≼ (6, 5) ≼ ((3, 2), 5)
    assert!(ht!(30).compatible_with(&ht!((6, 5))));
    assert!(ht!((6, 5)).compatible_with(&ht!(((3, 2), 5))));
    assert!(ht!(30).compatible_with(&ht!(((3, 2), 5))));

    // The two refinements of 30 are incompatible with each other.
    assert!(!ht!((2, (3, 5))).compatible_with(&ht!(((3, 2), 5))));
    assert!(!ht!(((3, 2), 5)).compatible_with(&ht!((2, (3, 5)))));
}

#[test]
fn compatible_needs_the_sizes_to_match() {
    assert!(!ht!(24).compatible_with(&ht!(32)));
    assert!(!ht!(24).compatible_with(&ht!((4, 8))));
}

#[test]
fn compatible_puts_an_int_below_every_shape_of_its_size() {
    assert!(ht!(24).compatible_with(&ht!((4, 6))));
    assert!(ht!(24).compatible_with(&ht!(((2, 2), 6))));
    assert!(ht!(24).compatible_with(&ht!(((2, 2), (3, 2)))));
    assert!(ht!(24).compatible_with(&ht!(((2, 3), 4))));
}

#[test]
fn compatible_separates_an_int_from_a_one_tuple() {
    assert!(ht!(24).compatible_with(&ht!((24))));
    assert!(!ht!((24)).compatible_with(&ht!(24)));
    assert!(!ht!((24)).compatible_with(&ht!((4, 6))));
}

#[test]
fn compatible_is_reflexive() {
    for x in reflexive_cases() {
        assert!(x.compatible_with(&x));
    }
}

#[test]
fn compatible_implies_weakly_congruent() {
    let pairs = [
        (ht!(24), ht!((4, 6))),
        (ht!(24), ht!(((2, 2), (3, 2)))),
        (ht!((4, 6)), ht!(((2, 2), 6))),
        (ht!((6, 5)), ht!(((3, 2), 5))),
    ];
    for (a, b) in pairs {
        assert!(a.compatible_with(&b));
        assert!(a.weakly_congruent(&b));
    }
}

#[test]
fn weak_congruence_does_not_imply_compatibility() {
    assert!(ht!(30).weakly_congruent(&ht!((3, 4))));
    assert!(!ht!(30).compatible_with(&ht!((3, 4))));
}

// ---------------------------------------------------------------------------
// common_refinement — the join
// ---------------------------------------------------------------------------

#[test]
fn common_refinement_of_equal_ints_is_that_int() {
    assert_refinement(&ht!(1), &ht!(1), &ht!(1));
    assert_refinement(&ht!(30), &ht!(30), &ht!(30));
}

#[test]
fn common_refinement_of_an_int_and_a_tuple_of_its_size_is_the_tuple() {
    assert_refinement(&ht!(10), &ht!((2, 5)), &ht!((2, 5)));
    assert_refinement(&ht!(30), &ht!((2, (3, 5))), &ht!((2, (3, 5))));
    assert_refinement(&ht!(24), &ht!(((2, 2), 6)), &ht!(((2, 2), 6)));
    // A one-tuple is strictly more refined than the int.
    assert_refinement(&ht!(10), &ht!((10)), &ht!((10)));
}

#[test]
fn common_refinement_is_reflexive() {
    for x in reflexive_cases() {
        assert_eq!(x.common_refinement(&x).unwrap(), x);
    }
}

#[test]
fn common_refinement_walks_the_whitepaper_chains() {
    assert_refinement(&ht!(30), &ht!((2, 15)), &ht!((2, 15)));
    assert_refinement(&ht!((2, 15)), &ht!((2, (3, 5))), &ht!((2, (3, 5))));
    assert_refinement(&ht!(30), &ht!((2, (3, 5))), &ht!((2, (3, 5))));

    assert_refinement(&ht!(30), &ht!((6, 5)), &ht!((6, 5)));
    assert_refinement(&ht!((6, 5)), &ht!(((3, 2), 5)), &ht!(((3, 2), 5)));
}

#[test]
fn common_refinement_refines_each_mode_on_its_own() {
    assert_refinement(&ht!((4, (3, 5))), &ht!(((2, 2), 15)), &ht!(((2, 2), (3, 5))));
    assert_refinement(&ht!((2, (6, 5))), &ht!((2, ((3, 2), 5))), &ht!((2, ((3, 2), 5))));
}

#[test]
fn common_refinement_picks_the_most_refined_side_of_each_mode() {
    assert_refinement(&ht!(((2, 3), 20)), &ht!((6, (4, 5))), &ht!(((2, 3), (4, 5))));
}

#[test]
fn common_refinement_rejects_unequal_ints() {
    assert!(ht!(3).common_refinement(&ht!(4)).is_err());
}

#[test]
fn common_refinement_rejects_an_int_beside_a_tuple_of_another_size() {
    assert!(ht!(7).common_refinement(&ht!((2, 3))).is_err());
    assert!(ht!((2, 3)).common_refinement(&ht!(7)).is_err());
}

#[test]
fn common_refinement_rejects_a_rank_mismatch() {
    assert!(ht!((2, 3)).common_refinement(&ht!((2, 3, 1))).is_err());
    assert!(ht!((6)).common_refinement(&ht!((2, 3))).is_err());
}

#[test]
fn common_refinement_rejects_incompatible_profiles_of_equal_size() {
    assert!(ht!((2, (3, 5))).common_refinement(&ht!(((3, 2), 5))).is_err());
    assert!(ht!(((3, 2), 5)).common_refinement(&ht!((2, (3, 5)))).is_err());
}

#[test]
fn common_refinement_rejects_a_single_bad_mode() {
    assert!(ht!((2, 15)).common_refinement(&ht!((3, 10))).is_err());
    assert!(ht!((4, (3, 5))).common_refinement(&ht!((4, (2, 5)))).is_err());
}

#[test]
fn common_refinement_reads_the_shape_off_a_layout() {
    let l1 = compact(ht!((2, 15)));
    let l2 = compact(ht!((2, (3, 5))));
    assert_eq!(l1.shape.common_refinement(&l2.shape).unwrap(), ht!((2, (3, 5))));
    assert_eq!(l1.shape.common_refinement(&ht!((2, (3, 5)))).unwrap(), ht!((2, (3, 5))));
    assert_eq!(ht!(30).common_refinement(&l1.shape).unwrap(), ht!((2, 15)));
}

// ---------------------------------------------------------------------------
// common_coarsening — the meet
// ---------------------------------------------------------------------------

#[test]
fn common_coarsening_of_equal_ints_is_that_int() {
    assert_coarsening(&ht!(1), &ht!(1), &ht!(1));
    assert_coarsening(&ht!(30), &ht!(30), &ht!(30));
}

#[test]
fn common_coarsening_with_an_int_is_that_int() {
    assert_coarsening(&ht!(10), &ht!((2, 5)), &ht!(10));
    assert_coarsening(&ht!(30), &ht!((2, (3, 5))), &ht!(30));
    assert_coarsening(&ht!(24), &ht!(((2, 2), 6)), &ht!(24));
    // (10,) is strictly more refined than 10, so the meet is the int.
    assert_coarsening(&ht!(10), &ht!((10)), &ht!(10));
}

#[test]
fn common_coarsening_is_reflexive() {
    for x in reflexive_cases() {
        assert_eq!(x.common_coarsening(&x).unwrap(), x);
    }
}

#[test]
fn common_coarsening_walks_the_whitepaper_chains() {
    assert_coarsening(&ht!((2, 15)), &ht!((2, (3, 5))), &ht!((2, 15)));
    assert_coarsening(&ht!(30), &ht!((2, (3, 5))), &ht!(30));
    assert_coarsening(&ht!((2, (3, 5))), &ht!(((3, 2), 5)), &ht!(30));
    assert_coarsening(&ht!(((3, 2), 5)), &ht!((2, (3, 5))), &ht!(30));
}

#[test]
fn common_coarsening_falls_back_to_the_int_on_a_mode_mismatch() {
    assert_coarsening(&ht!((6, 5)), &ht!((2, 15)), &ht!(30));
    assert_coarsening(&ht!((6, 5)), &ht!((3, 10)), &ht!(30));
    assert_coarsening(&ht!((4, 6)), &ht!((3, 8)), &ht!(24));
}

#[test]
fn common_coarsening_meets_each_mode_on_its_own() {
    assert_coarsening(&ht!((4, (3, 5))), &ht!(((2, 2), 15)), &ht!((4, 15)));
    // Mode 0 falls back to its size, mode 1 meets outright.
    assert_coarsening(&ht!(((2, 3), 4)), &ht!(((3, 2), 4)), &ht!((6, 4)));
}

#[test]
fn common_coarsening_falls_back_to_the_int_on_a_rank_mismatch() {
    assert_coarsening(&ht!((2, 3)), &ht!((2, 3, 1)), &ht!(6));
    assert_coarsening(&ht!((6)), &ht!(6), &ht!(6));
    assert_coarsening(&ht!((5, 3, 4)), &ht!((10, 6)), &ht!(60));
}

#[test]
fn common_coarsening_rejects_unequal_ints() {
    assert!(ht!(3).common_coarsening(&ht!(4)).is_err());
}

#[test]
fn common_coarsening_rejects_an_int_beside_a_tuple_of_another_size() {
    assert!(ht!(7).common_coarsening(&ht!((2, 3))).is_err());
    assert!(ht!((2, 3)).common_coarsening(&ht!(7)).is_err());
}

#[test]
fn common_coarsening_rejects_a_total_size_mismatch() {
    assert!(ht!((2, 3)).common_coarsening(&ht!((4, 5))).is_err());
    assert!(ht!((2, 3, 4)).common_coarsening(&ht!((6, 5))).is_err());
}

#[test]
fn common_coarsening_reads_the_shape_off_a_layout() {
    let l1 = compact(ht!((2, 15)));
    let l2 = compact(ht!((2, (3, 5))));
    assert_eq!(l1.shape.common_coarsening(&l2.shape).unwrap(), ht!((2, 15)));
    assert_eq!(l1.shape.common_coarsening(&ht!((2, (3, 5)))).unwrap(), ht!((2, 15)));
    assert_eq!(ht!(30).common_coarsening(&l1.shape).unwrap(), ht!(30));
}

// ---------------------------------------------------------------------------
// The lattice, read from both ends
// ---------------------------------------------------------------------------

#[test]
fn the_meet_sits_below_the_join() {
    let pairs = [
        (ht!(30), ht!((2, 15))),
        (ht!((2, 15)), ht!((2, (3, 5)))),
        (ht!((4, (3, 5))), ht!(((2, 2), 15))),
        (ht!(((2, 3), 20)), ht!((6, (4, 5)))),
    ];
    for (a, b) in pairs {
        let join = a.common_refinement(&b).unwrap();
        let meet = a.common_coarsening(&b).unwrap();
        assert!(meet.compatible_with(&join));
        assert!(meet.compatible_with(&a));
        assert!(meet.compatible_with(&b));
        assert!(a.compatible_with(&join));
        assert!(b.compatible_with(&join));
    }
}

#[test]
fn a_join_guarantees_a_meet() {
    let pairs = [
        (ht!(30), ht!((2, 15))),
        (ht!((2, 15)), ht!((2, (3, 5)))),
        (ht!((4, (3, 5))), ht!(((2, 2), 15))),
    ];
    for (a, b) in pairs {
        assert!(a.common_refinement(&b).is_ok());
        assert!(a.common_coarsening(&b).is_ok());
    }
}

#[test]
fn a_meet_does_not_guarantee_a_join() {
    assert!(ht!((2, (3, 5))).common_refinement(&ht!(((3, 2), 5))).is_err());
    assert_eq!(ht!((2, (3, 5))).common_coarsening(&ht!(((3, 2), 5))).unwrap(), ht!(30));
}

// ---------------------------------------------------------------------------
// The accessors, from the `pycute/shape.py` docstrings
// ---------------------------------------------------------------------------

#[test]
fn get_projects_a_mode_path() {
    let s = ht!((3, (2, 4)));
    assert_eq!(s.get(&[]), Some(&s));
    assert_eq!(s.get(&[1]), Some(&ht!((2, 4))));
    assert_eq!(s.get(&[1, 0]), Some(&ht!(2)));
    assert_eq!(s.get(&[2]), None);
}

#[test]
fn size_multiplies_the_extents_of_a_mode() {
    let s = ht!((3, (2, 4)));
    assert_eq!(s.size(), 24);
    assert_eq!(s.get(&[1]).unwrap().size(), 8);
    assert_eq!(ht!(14).size(), 14);
}

#[test]
fn rank_counts_the_top_level_modes() {
    assert_eq!(ht!((3, (2, 4))).rank(), 2);
    assert_eq!(ht!((3, (2, 4))).get(&[1]).unwrap().rank(), 2);
    assert_eq!(ht!((3, (2, 4))).get(&[0]).unwrap().rank(), 1);
    assert_eq!(ht!(14).rank(), 1);
}

#[test]
fn depth_measures_the_longest_path_to_a_leaf() {
    assert_eq!(ht!(14).depth(), 0);
    assert_eq!(ht!((3, 2, 4)).depth(), 1);
    assert_eq!(ht!((3, (2, 4))).depth(), 2);
    assert_eq!(ht!((3, (2, (4, 5)))).depth(), 3);
    assert_eq!(ht!((3, (2, 4))).get(&[0]).unwrap().depth(), 0);
}

// ---------------------------------------------------------------------------
// idx2crd and coordinates
// ---------------------------------------------------------------------------

#[test]
fn idx2crd_decomposes_colexicographically() {
    assert_eq!(ht!(14).idx2crd(&ht!(7)).unwrap(), ht!(7));
    assert_eq!(ht!((3, 2, 4)).idx2crd(&ht!(7)).unwrap(), ht!((1, 0, 1)));
    assert_eq!(ht!((3, (2, 4))).idx2crd(&ht!(7)).unwrap(), ht!((1, (0, 1))));
    assert_eq!(ht!(((3, 2), 4)).idx2crd(&ht!(7)).unwrap(), ht!(((1, 0), 1)));
}

#[test]
fn idx2crd_lets_the_last_leaf_absorb_the_excess() {
    assert_eq!(ht!((3, 7, 2)).idx2crd(&ht!(42)).unwrap(), ht!((0, 0, 2)));
}

#[test]
fn idx2crd_recurses_through_a_congruent_coordinate() {
    assert_eq!(ht!((3, (2, 4))).idx2crd(&ht!((1, 5))).unwrap(), ht!((1, (1, 2))));
}

#[test]
fn idx2crd_rejects_a_coordinate_that_does_not_index_the_shape() {
    assert!(ht!((3, 2)).idx2crd(&ht!((1, 2, 3))).is_err());
    assert!(ht!(6).idx2crd(&ht!((1, 2))).is_err());
}

#[test]
fn idx2crd_is_congruent_with_the_shape() {
    for s in [ht!(6), ht!((3, 2)), ht!((2, (2, 2))), ht!(((3, 2), 4))] {
        for i in 0..s.size() {
            assert!(s.idx2crd(&HTuple::Leaf(i)).unwrap().congruent(&s));
        }
    }
}

#[test]
fn coordinates_enumerates_colexicographically() {
    assert_eq!(
        ht!(6).coordinates().collect::<Vec<_>>(),
        vec![ht!(0), ht!(1), ht!(2), ht!(3), ht!(4), ht!(5)]
    );
    assert_eq!(
        ht!((3, 2)).coordinates().collect::<Vec<_>>(),
        vec![ht!((0, 0)), ht!((1, 0)), ht!((2, 0)), ht!((0, 1)), ht!((1, 1)), ht!((2, 1)),]
    );
    assert_eq!(
        ht!((2, (2, 2))).coordinates().collect::<Vec<_>>(),
        vec![
            ht!((0, (0, 0))),
            ht!((1, (0, 0))),
            ht!((0, (1, 0))),
            ht!((1, (1, 0))),
            ht!((0, (0, 1))),
            ht!((1, (0, 1))),
            ht!((0, (1, 1))),
            ht!((1, (1, 1))),
        ]
    );
}

#[test]
fn crd2idx_recomposes_colexicographically() {
    assert_eq!(ht!((3, 2, 4)).crd2idx(&ht!((1, 0, 1))).unwrap(), 7.into());
    assert_eq!(ht!((3, (2, 4))).crd2idx(&ht!((1, (0, 1)))).unwrap(), 7.into());
}

#[test]
fn crd2idx_passes_an_integral_coordinate_through() {
    assert_eq!(ht!((3, (2, 4))).crd2idx(&ht!(7)).unwrap(), 7.into());
}

#[test]
fn crd2idx_takes_a_flat_coordinate_of_a_nested_shape() {
    // The leaf 5 stands for the sub-shape (2, 3), which contributes its
    // size: 2 + 5 * 3 == 17.
    assert_eq!(ht!((3, (2, 3))).crd2idx(&ht!((2, 5))).unwrap(), 17.into());
}

#[test]
fn crd2idx_rejects_a_coordinate_that_does_not_coarsen_the_shape() {
    assert!(ht!((3, 2)).crd2idx(&ht!((1, 2, 3))).is_err());
}

#[test]
fn crd2idx_inverts_idx2crd() {
    for s in [ht!(6), ht!((3, 2)), ht!((2, (2, 2))), ht!(((3, 2), 4))] {
        for i in 0..s.size() {
            let crd = s.idx2crd(&HTuple::Leaf(i)).unwrap();
            assert_eq!(s.crd2idx(&crd).unwrap(), i.into());
        }
    }
}

#[test]
fn crd2idx_walks_the_coordinates_in_order() {
    for s in [ht!(6), ht!((3, 2)), ht!((2, (2, 2))), ht!(((3, 2), 4))] {
        let walked = s
            .coordinates()
            .collect::<Vec<_>>()
            .iter()
            .map(|c| s.crd2idx(c).unwrap())
            .collect::<Vec<_>>();
        let expected = (0..s.size()).map(Into::into).collect::<Vec<_>>();
        assert_eq!(walked, expected);
    }
}

#[test]
fn coordinates_agree_with_idx2crd() {
    for s in [ht!(6), ht!((3, 2)), ht!((2, (2, 2))), ht!(((3, 2), 4))] {
        let walked =
            (0..s.size()).map(|i| s.idx2crd(&HTuple::Leaf(i)).unwrap()).collect::<Vec<_>>();
        assert_eq!(s.coordinates().collect::<Vec<_>>(), walked);
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// The shapes the reflexivity cases sweep.
fn reflexive_cases() -> Vec<IntTuple> {
    vec![ht!(7), ht!((3, 4)), ht!(((1, 2), 3)), ht!(((1, (2, 3)), 4)), ht!((2, (3, 5)))]
}

/// The join is `expected`, it refines both sides, and it is symmetric.
fn assert_refinement(a: &IntTuple, b: &IntTuple, expected: &IntTuple) {
    let join = a.common_refinement(b).unwrap();
    assert_eq!(&join, expected);
    assert!(a.compatible_with(&join), "expected {a:?} ≼ {join:?}");
    assert!(b.compatible_with(&join), "expected {b:?} ≼ {join:?}");
    assert_eq!(&b.common_refinement(a).unwrap(), expected);
}

/// The meet is `expected`, it coarsens both sides, and it is symmetric.
fn assert_coarsening(a: &IntTuple, b: &IntTuple, expected: &IntTuple) {
    let meet = a.common_coarsening(b).unwrap();
    assert_eq!(&meet, expected);
    assert!(meet.compatible_with(a), "expected {meet:?} ≼ {a:?}");
    assert!(meet.compatible_with(b), "expected {meet:?} ≼ {b:?}");
    assert_eq!(&b.common_coarsening(a).unwrap(), expected);
}
