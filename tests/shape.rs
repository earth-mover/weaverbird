//! Ported from `test/test_compatibility.py`, plus the worked examples in
//! the docstrings of `pycute/shape.py`.
//!
//! The cases that need `Layout` — `test_accepts_layouts` — wait for that
//! module, as does anything reached through `crd2idx`.
//!
//! `test/test_typing.py` holds no shape assertions to port: it exercises
//! Python's ABC registration, `typing.get_type_hints`, and module
//! layout, none of which survive the translation to Rust types.
#![expect(clippy::unwrap_used, reason = "a test asserts the happy path")]

use pinstripe::{
    HTuple, IntTuple, common_coarsening, common_refinement, compatible, coordinates, depth, ht,
    htuple::{congruent, weakly_congruent},
    idx2crd, rank, shape, size,
};

// ---------------------------------------------------------------------------
// congruent — same hierarchical profile
// ---------------------------------------------------------------------------

#[test]
fn congruent_ignores_the_leaf_values() {
    assert!(congruent(&ht!((4, 8)), &ht!((5, 7))));
    assert!(congruent(&ht!(31), &ht!(42)));
    assert!(congruent(
        &ht!(((4, 6), (3, (2, 2), 8))),
        &ht!(((1, 1), (1, (1, 1), 1)))
    ));
}

#[test]
fn congruent_separates_different_profiles() {
    assert!(!congruent(&ht!((4, 8)), &ht!((4, (2, 4)))));
    assert!(!congruent(&ht!(31), &ht!((4, 8))));
    assert!(!congruent(&ht!((1, 1, 1)), &ht!((1, 1))));
}

#[test]
fn congruent_is_reflexive() {
    for x in reflexive_cases() {
        assert!(congruent(&x, &x));
    }
}

// ---------------------------------------------------------------------------
// weakly_congruent — the coarsening order on profiles
// ---------------------------------------------------------------------------

#[test]
fn weakly_congruent_lets_an_int_coarsen_anything() {
    assert!(weakly_congruent(&ht!(30), &ht!((3, 4))));
    assert!(weakly_congruent(&ht!(30), &ht!(((3, 4), 5))));
    assert!(weakly_congruent(&ht!(30), &ht!((((3, 4), 5), (6, 7)))));
}

#[test]
fn weakly_congruent_never_lets_a_tuple_coarsen_an_int() {
    assert!(!weakly_congruent(&ht!((3, 4)), &ht!(30)));
}

#[test]
fn weakly_congruent_needs_the_same_top_level_rank() {
    assert!(!weakly_congruent(&ht!((1, 2, 3)), &ht!((1, 2))));
}

#[test]
fn weakly_congruent_recurses_into_the_modes() {
    assert!(weakly_congruent(&ht!((3, 4)), &ht!((5, (6, 7)))));
    assert!(!weakly_congruent(&ht!((3, (4, 5))), &ht!((5, 6))));
}

#[test]
fn weakly_congruent_is_reflexive() {
    for x in reflexive_cases() {
        assert!(weakly_congruent(&x, &x));
    }
}

// ---------------------------------------------------------------------------
// compatible — the coarsening order on shapes
// ---------------------------------------------------------------------------

#[test]
fn compatible_walks_the_whitepaper_chains() {
    // 30 ≼ (2, 15) ≼ (2, (3, 5))
    assert!(compatible(&ht!(30), &ht!((2, 15))));
    assert!(compatible(&ht!((2, 15)), &ht!((2, (3, 5)))));
    assert!(compatible(&ht!(30), &ht!((2, (3, 5)))));

    // 30 ≼ (6, 5) ≼ ((3, 2), 5)
    assert!(compatible(&ht!(30), &ht!((6, 5))));
    assert!(compatible(&ht!((6, 5)), &ht!(((3, 2), 5))));
    assert!(compatible(&ht!(30), &ht!(((3, 2), 5))));

    // The two refinements of 30 are incompatible with each other.
    assert!(!compatible(&ht!((2, (3, 5))), &ht!(((3, 2), 5))));
    assert!(!compatible(&ht!(((3, 2), 5)), &ht!((2, (3, 5)))));
}

#[test]
fn compatible_needs_the_sizes_to_match() {
    assert!(!compatible(&ht!(24), &ht!(32)));
    assert!(!compatible(&ht!(24), &ht!((4, 8))));
}

#[test]
fn compatible_puts_an_int_below_every_shape_of_its_size() {
    assert!(compatible(&ht!(24), &ht!((4, 6))));
    assert!(compatible(&ht!(24), &ht!(((2, 2), 6))));
    assert!(compatible(&ht!(24), &ht!(((2, 2), (3, 2)))));
    assert!(compatible(&ht!(24), &ht!(((2, 3), 4))));
}

#[test]
fn compatible_separates_an_int_from_a_one_tuple() {
    assert!(compatible(&ht!(24), &ht!((24))));
    assert!(!compatible(&ht!((24)), &ht!(24)));
    assert!(!compatible(&ht!((24)), &ht!((4, 6))));
}

#[test]
fn compatible_is_reflexive() {
    for x in reflexive_cases() {
        assert!(compatible(&x, &x));
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
        assert!(compatible(&a, &b));
        assert!(weakly_congruent(&a, &b));
    }
}

#[test]
fn weak_congruence_does_not_imply_compatibility() {
    assert!(weakly_congruent(&ht!(30), &ht!((3, 4))));
    assert!(!compatible(&ht!(30), &ht!((3, 4))));
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
        assert_eq!(common_refinement(&x, &x).unwrap(), x);
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
    assert_refinement(
        &ht!((4, (3, 5))),
        &ht!(((2, 2), 15)),
        &ht!(((2, 2), (3, 5))),
    );
    assert_refinement(
        &ht!((2, (6, 5))),
        &ht!((2, ((3, 2), 5))),
        &ht!((2, ((3, 2), 5))),
    );
}

#[test]
fn common_refinement_picks_the_most_refined_side_of_each_mode() {
    assert_refinement(
        &ht!(((2, 3), 20)),
        &ht!((6, (4, 5))),
        &ht!(((2, 3), (4, 5))),
    );
}

#[test]
fn common_refinement_rejects_unequal_ints() {
    assert!(common_refinement(&ht!(3), &ht!(4)).is_err());
}

#[test]
fn common_refinement_rejects_an_int_beside_a_tuple_of_another_size() {
    assert!(common_refinement(&ht!(7), &ht!((2, 3))).is_err());
    assert!(common_refinement(&ht!((2, 3)), &ht!(7)).is_err());
}

#[test]
fn common_refinement_rejects_a_rank_mismatch() {
    assert!(common_refinement(&ht!((2, 3)), &ht!((2, 3, 1))).is_err());
    assert!(common_refinement(&ht!((6)), &ht!((2, 3))).is_err());
}

#[test]
fn common_refinement_rejects_incompatible_profiles_of_equal_size() {
    assert!(common_refinement(&ht!((2, (3, 5))), &ht!(((3, 2), 5))).is_err());
    assert!(common_refinement(&ht!(((3, 2), 5)), &ht!((2, (3, 5)))).is_err());
}

#[test]
fn common_refinement_rejects_a_single_bad_mode() {
    assert!(common_refinement(&ht!((2, 15)), &ht!((3, 10))).is_err());
    assert!(common_refinement(&ht!((4, (3, 5))), &ht!((4, (2, 5)))).is_err());
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
        assert_eq!(common_coarsening(&x, &x).unwrap(), x);
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
    assert!(common_coarsening(&ht!(3), &ht!(4)).is_err());
}

#[test]
fn common_coarsening_rejects_an_int_beside_a_tuple_of_another_size() {
    assert!(common_coarsening(&ht!(7), &ht!((2, 3))).is_err());
    assert!(common_coarsening(&ht!((2, 3)), &ht!(7)).is_err());
}

#[test]
fn common_coarsening_rejects_a_total_size_mismatch() {
    assert!(common_coarsening(&ht!((2, 3)), &ht!((4, 5))).is_err());
    assert!(common_coarsening(&ht!((2, 3, 4)), &ht!((6, 5))).is_err());
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
        let join = common_refinement(&a, &b).unwrap();
        let meet = common_coarsening(&a, &b).unwrap();
        assert!(compatible(&meet, &join));
        assert!(compatible(&meet, &a));
        assert!(compatible(&meet, &b));
        assert!(compatible(&a, &join));
        assert!(compatible(&b, &join));
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
        assert!(common_refinement(&a, &b).is_ok());
        assert!(common_coarsening(&a, &b).is_ok());
    }
}

#[test]
fn a_meet_does_not_guarantee_a_join() {
    assert!(common_refinement(&ht!((2, (3, 5))), &ht!(((3, 2), 5))).is_err());
    assert_eq!(
        common_coarsening(&ht!((2, (3, 5))), &ht!(((3, 2), 5))).unwrap(),
        ht!(30)
    );
}

// ---------------------------------------------------------------------------
// The accessors, from the `pycute/shape.py` docstrings
// ---------------------------------------------------------------------------

#[test]
fn shape_projects_a_mode_path() {
    let s = ht!((3, (2, 4)));
    assert_eq!(shape(&s, &[]).unwrap(), &s);
    assert_eq!(shape(&s, &[1]).unwrap(), &ht!((2, 4)));
    assert_eq!(shape(&s, &[1, 0]).unwrap(), &ht!(2));
    assert!(shape(&s, &[2]).is_err());
}

#[test]
fn size_multiplies_the_extents_of_a_mode() {
    let s = ht!((3, (2, 4)));
    assert_eq!(size(&s, &[]).unwrap(), 24);
    assert_eq!(size(&s, &[1]).unwrap(), 8);
    assert_eq!(size(&ht!(14), &[]).unwrap(), 14);
}

#[test]
fn rank_counts_the_top_level_modes() {
    assert_eq!(rank(&ht!((3, (2, 4))), &[]).unwrap(), 2);
    assert_eq!(rank(&ht!((3, (2, 4))), &[1]).unwrap(), 2);
    assert_eq!(rank(&ht!((3, (2, 4))), &[0]).unwrap(), 1);
    assert_eq!(rank(&ht!(14), &[]).unwrap(), 1);
}

#[test]
fn depth_measures_the_longest_path_to_a_leaf() {
    assert_eq!(depth(&ht!(14), &[]).unwrap(), 0);
    assert_eq!(depth(&ht!((3, 2, 4)), &[]).unwrap(), 1);
    assert_eq!(depth(&ht!((3, (2, 4))), &[]).unwrap(), 2);
    assert_eq!(depth(&ht!((3, (2, (4, 5)))), &[]).unwrap(), 3);
    assert_eq!(depth(&ht!((3, (2, 4))), &[0]).unwrap(), 0);
}

// ---------------------------------------------------------------------------
// idx2crd and coordinates
// ---------------------------------------------------------------------------

#[test]
fn idx2crd_decomposes_colexicographically() {
    assert_eq!(idx2crd(&ht!(7), &ht!(14)).unwrap(), ht!(7));
    assert_eq!(idx2crd(&ht!(7), &ht!((3, 2, 4))).unwrap(), ht!((1, 0, 1)));
    assert_eq!(
        idx2crd(&ht!(7), &ht!((3, (2, 4)))).unwrap(),
        ht!((1, (0, 1)))
    );
    assert_eq!(
        idx2crd(&ht!(7), &ht!(((3, 2), 4))).unwrap(),
        ht!(((1, 0), 1))
    );
}

#[test]
fn idx2crd_lets_the_last_leaf_absorb_the_excess() {
    assert_eq!(idx2crd(&ht!(42), &ht!((3, 7, 2))).unwrap(), ht!((0, 0, 2)));
}

#[test]
fn idx2crd_recurses_through_a_congruent_coordinate() {
    assert_eq!(
        idx2crd(&ht!((1, 5)), &ht!((3, (2, 4)))).unwrap(),
        ht!((1, (1, 2)))
    );
}

#[test]
fn idx2crd_rejects_a_coordinate_that_does_not_index_the_shape() {
    assert!(idx2crd(&ht!((1, 2, 3)), &ht!((3, 2))).is_err());
    assert!(idx2crd(&ht!((1, 2)), &ht!(6)).is_err());
}

#[test]
fn idx2crd_is_congruent_with_the_shape() {
    for s in [ht!(6), ht!((3, 2)), ht!((2, (2, 2))), ht!(((3, 2), 4))] {
        for i in 0..s.product() {
            assert!(congruent(&idx2crd(&HTuple::Leaf(i), &s).unwrap(), &s));
        }
    }
}

#[test]
fn coordinates_enumerates_colexicographically() {
    assert_eq!(
        coordinates(&ht!(6)),
        vec![ht!(0), ht!(1), ht!(2), ht!(3), ht!(4), ht!(5)]
    );
    assert_eq!(
        coordinates(&ht!((3, 2))),
        vec![
            ht!((0, 0)),
            ht!((1, 0)),
            ht!((2, 0)),
            ht!((0, 1)),
            ht!((1, 1)),
            ht!((2, 1)),
        ]
    );
    assert_eq!(
        coordinates(&ht!((2, (2, 2)))),
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
fn coordinates_agree_with_idx2crd() {
    for s in [ht!(6), ht!((3, 2)), ht!((2, (2, 2))), ht!(((3, 2), 4))] {
        let walked = (0..s.product())
            .map(|i| idx2crd(&HTuple::Leaf(i), &s).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(coordinates(&s), walked);
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// The shapes the reflexivity cases sweep.
fn reflexive_cases() -> Vec<IntTuple> {
    vec![
        ht!(7),
        ht!((3, 4)),
        ht!(((1, 2), 3)),
        ht!(((1, (2, 3)), 4)),
        ht!((2, (3, 5))),
    ]
}

/// The join is `expected`, it refines both sides, and it is symmetric.
fn assert_refinement(a: &IntTuple, b: &IntTuple, expected: &IntTuple) {
    let join = common_refinement(a, b).unwrap();
    assert_eq!(&join, expected);
    assert!(compatible(a, &join), "expected {a:?} ≼ {join:?}");
    assert!(compatible(b, &join), "expected {b:?} ≼ {join:?}");
    assert_eq!(&common_refinement(b, a).unwrap(), expected);
}

/// The meet is `expected`, it coarsens both sides, and it is symmetric.
fn assert_coarsening(a: &IntTuple, b: &IntTuple, expected: &IntTuple) {
    let meet = common_coarsening(a, b).unwrap();
    assert_eq!(&meet, expected);
    assert!(compatible(&meet, a), "expected {meet:?} ≼ {a:?}");
    assert!(compatible(&meet, b), "expected {meet:?} ≼ {b:?}");
    assert_eq!(&common_coarsening(b, a).unwrap(), expected);
}
