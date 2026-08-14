//! Ported from `test/test_atuple.py`.
//!
//! `test_idx2crd_is_consistent_across_constructions` and
//! `TestWeaklyCongruentImplicitZero` are gone for good: both read an
//! [`ArithTuple`] as the *index* argument, and this crate types that
//! argument [`IntTuple`] — `idx2crd` and `weakly_congruent` never see a
//! stride scalar, so the Python dispatch they exercise has no
//! counterpart.

use std::cmp::Ordering;

use crate::{
    ArithTuple, HTuple, Int, IntTuple, Layout, Stride, StrideScalar,
    atuple::{basis_repr, is_basis, make_basis_like, proj, proj_tuple, proj_tuple_mut, unit},
    e, ht, scaled_basis,
};

/// Builds an [`ArithTuple`] from nested parentheses, so the ported cases
/// read like their Python source. A bare value is the integer leaf.
macro_rules! at {
    (( $($inner:tt),+ $(,)? )) => {
        StrideScalar::Arith(ArithTuple::from_data(vec![ $(at!($inner)),+ ]))
    };
    ($value:expr) => { StrideScalar::Int($value) };
}

/// The integer tuple as an arithmetic tuple. PyCuTe's
/// `ArithTuple(crd)`, which lifts a coordinate into the stride scalars.
fn arith(t: &IntTuple) -> StrideScalar {
    match t {
        HTuple::Leaf(v) => StrideScalar::Int(*v),
        HTuple::Tuple(modes) => {
            StrideScalar::Arith(ArithTuple::from_data(modes.iter().map(arith).collect()))
        }
    }
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

/// A layout paired with the coordinate `A(i, j)` it evaluates to.
type CoordCase = (Layout, fn(Int, Int) -> StrideScalar);

#[test]
fn addition_is_elementwise() {
    assert_eq!(at!((1, 2, 3)).add(&at!((7, 8, 9))).unwrap(), at!((8, 10, 12)));
    assert_eq!(at!((1, 2, (3, 4))).add(&at!((7, 8, (9, 10)))).unwrap(), at!((8, 10, (12, 14))));
}

#[test]
fn addition_extends_the_shorter_side_with_zeros() {
    assert_eq!(at!((1, 2, 3)).add(&at!((7))).unwrap(), at!((8, 2, 3)));
}

#[test]
fn multiplication_scales_every_leaf() {
    assert_eq!(at!((1, 2, 3)).scale(4), at!((4, 8, 12)));
    assert_eq!(at!((1, 2, (3, 4))).scale(2), at!((2, 4, (6, 8))));
}

#[test]
fn adding_a_non_zero_integer_is_an_incompatibility() {
    assert!(at!((1, 2, 3)).add(&StrideScalar::Int(7)).is_err());
    assert!(StrideScalar::Int(7).add(&at!((1, 2, 3))).is_err());
}

#[test]
fn adding_zero_keeps_the_tuple() {
    assert_eq!(at!((1, 2, 3)).add(&StrideScalar::Int(0)).unwrap(), at!((1, 2, 3)));
    assert_eq!(StrideScalar::Int(0).add(&at!((1, 2, 3))).unwrap(), at!((1, 2, 3)));
}

#[test]
fn scaled_basis_places_one_value_at_a_path() {
    assert_eq!(scaled_basis(42, &[]), StrideScalar::Int(42));
    assert_eq!(scaled_basis(42, &[0]), at!((42, 0, 0, 0, 0)));
    assert_eq!(scaled_basis(42, &[1]), at!((0, 42, 0, 0, 0)));
    assert_eq!(scaled_basis(42, &[0, 0]), at!(((42, 0, 0, 0, 0), 0, 0, 0, 0)));
    assert_eq!(scaled_basis(42, &[0, 1]), at!(((0, 42, 0, 0, 0), 0, 0, 0, 0)));
    assert_eq!(scaled_basis(42, &[1, 0]), at!((0, (42, 0, 0, 0, 0), 0, 0, 0)));
    assert_eq!(scaled_basis(42, &[1, 1]), at!((0, (0, 42, 0, 0, 0), 0, 0, 0)));
}

#[test]
fn scaling_a_basis_scales_its_value() {
    assert_eq!(scaled_basis(42, &[]).scale(2), StrideScalar::Int(84));
    assert_eq!(scaled_basis(42, &[0]).scale(2), at!((84, 0, 0, 0, 0)));
    assert_eq!(scaled_basis(42, &[1, 0]).scale(2), at!((0, (84, 0, 0, 0, 0), 0, 0, 0)));
}

#[test]
fn the_order_on_coordinates_follows_the_colexicographic_index() {
    let shape = ht!((4, (5, 6), 2));
    let crd = |i: Int| arith(&shape.idx2crd(&ht!(i)).unwrap());
    let extent = shape.size();
    let precedes = |a: &StrideScalar, b: &StrideScalar| a.try_cmp(b).unwrap() == Ordering::Less;
    for i in 0..extent {
        assert!(precedes(&StrideScalar::Int(0), &crd(i + 1)));
        for j in i + 1..extent {
            assert!(precedes(&crd(i), &crd(j)));
            assert_eq!(crd(j).try_cmp(&crd(i)).unwrap(), Ordering::Greater);
        }
    }
}

#[test]
fn a_basis_strided_layout_evaluates_to_a_coordinate() {
    // Each case pairs a layout with the coordinate `A(i, j)` it lands
    // on, over the two mode extents PyCuTe sweeps.
    let cases: Vec<CoordCase> = vec![
        (strided(ht!((5, 4)), &t(vec![s(e(&[0])), s(e(&[1]))])), |i, j| at!((i, j))),
        (strided(ht!((5, 4)), &t(vec![s(e(&[0])), s(e(&[2]))])), |i, j| at!((i, 0, j))),
        (strided(ht!((5, 4)), &t(vec![s(e(&[2])), s(e(&[1]))])), |i, j| at!((0, j, i))),
        (strided(ht!((5, 4)), &t(vec![s(e(&[2])), s(StrideScalar::Int(0))])), |i, _| {
            at!((0, 0, i))
        }),
        (strided(ht!((5, 4)), &t(vec![s(e(&[2])), s(e(&[1, 3]))])), |i, j| {
            at!((0, (0, 0, 0, j), i))
        }),
        (
            strided(
                ht!((4, (4, 2))),
                &t(vec![s(e(&[1])), t(vec![s(e(&[0])), s(e(&[1]).scale(4))])]),
            ),
            |i, j| {
                // `at!` cannot spell an arithmetic leaf, so the two
                // coordinates are named first.
                let (row, col) = (j % 4, i + 4 * (j / 4));
                at!((row, col))
            },
        ),
    ];
    for (a, expected) in cases {
        let rows = a.shape.get(&[0]).unwrap().size();
        let cols = a.shape.get(&[1]).unwrap().size();
        for i in 0..rows {
            for j in 0..cols {
                assert_eq!(a.eval(&ht!((i, j))).unwrap(), expected(i, j), "{a} at ({i},{j})");
            }
        }
    }
}

/// The paths `test_sbasis` sweeps.
fn test_paths() -> Vec<Vec<usize>> {
    let mut paths = vec![vec![]];
    for i in 0..3 {
        paths.push(vec![i]);
    }
    for i in 0..3 {
        for j in 0..3 {
            paths.push(vec![i, j]);
        }
    }
    paths
}

#[test]
fn distinct_paths_give_distinct_basis_elements() {
    for a in test_paths() {
        for b in test_paths() {
            let (ea, eb) = (e(&a), e(&b));
            // Zero is the additive identity at every rank, so a scaled
            // basis compares equal to zero whatever its path.
            assert_eq!(ea.scale(0), eb.scale(0));
            match a == b {
                true => {
                    assert_eq!(ea, eb);
                    assert_eq!(ea.scale(3), eb.scale(3));
                    assert_ne!(ea.scale(3), eb.scale(4));
                }
                false => {
                    assert_ne!(ea, eb);
                    assert_ne!(ea.scale(0), eb.scale(3));
                }
            }
        }
    }
}

#[test]
fn zero_precedes_every_basis_element() {
    for path in test_paths() {
        assert_eq!(StrideScalar::Int(0).try_cmp(&e(&path)).unwrap(), Ordering::Less);
    }
}

#[test]
fn one_element_admits_several_representations() {
    assert_eq!(e(&[0]), at!((1, 0)));
    assert_eq!(at!((1, 0)), at!((1)));
}

#[test]
fn every_all_zero_tuple_equals_the_integer_zero() {
    assert_eq!(StrideScalar::Arith(ArithTuple::from_data(vec![])), StrideScalar::Int(0));
    assert_eq!(at!((0, 0, 0)), StrideScalar::Int(0));
    assert_eq!(at!(((0, 0))), StrideScalar::Int(0));
    assert_eq!(e(&[2]).scale(0), StrideScalar::Int(0));
    assert_eq!(scaled_basis(0, &[3]), StrideScalar::Int(0));
}

#[test]
fn an_integer_and_a_tuple_sit_at_different_ranks() {
    assert_ne!(at!((5)), StrideScalar::Int(5));
}

#[test]
fn equal_elements_hash_alike() {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash as _, Hasher as _};

    let hash_of = |x: &StrideScalar| {
        let mut hasher = DefaultHasher::new();
        x.hash(&mut hasher);
        hasher.finish()
    };
    assert_eq!(hash_of(&e(&[0])), hash_of(&at!((1, 0))));
    assert_eq!(hash_of(&at!((0, 0))), hash_of(&StrideScalar::Int(0)));
}

#[test]
fn basis_repr_of_an_integer_is_the_rank_zero_term() {
    assert_eq!(basis_repr(&StrideScalar::Int(0)), vec![(0, vec![])]);
    assert_eq!(basis_repr(&StrideScalar::Int(7)), vec![(7, vec![])]);
}

#[test]
fn basis_repr_of_an_all_zero_tuple_matches_the_integer_zero() {
    assert_eq!(basis_repr(&e(&[2]).scale(0)), vec![(0, vec![])]);
    assert_eq!(basis_repr(&at!((0))), vec![(0, vec![])]);
    assert_eq!(basis_repr(&at!((0, (0, 0)))), vec![(0, vec![])]);
}

#[test]
fn basis_repr_reads_the_path_of_a_unit() {
    assert_eq!(basis_repr(&e(&[0])), vec![(1, vec![0])]);
    assert_eq!(basis_repr(&e(&[1, 2])), vec![(1, vec![1, 2])]);
}

#[test]
fn basis_repr_splits_a_sum_into_its_terms() {
    let sum = e(&[0]).scale(3).add(&e(&[1]).scale(5)).unwrap();
    assert_eq!(basis_repr(&sum), vec![(3, vec![0]), (5, vec![1])]);
}

#[test]
fn basis_repr_round_trips_through_a_sum() {
    let cases = [
        StrideScalar::Int(0),
        StrideScalar::Int(7),
        e(&[0]),
        e(&[1, 2]).scale(5),
        e(&[0]).scale(3).add(&e(&[1]).scale(5)).unwrap(),
        at!((1, 2, (3, 4))),
    ];
    for x in cases {
        let rebuilt =
            basis_repr(&x).into_iter().try_fold(StrideScalar::Int(0), |acc, (value, path)| {
                acc.add(&scaled_basis(value, &path))
            });
        assert_eq!(rebuilt.unwrap(), x);
    }
}

#[test]
fn is_basis_accepts_one_term_and_rejects_a_sum() {
    assert!(is_basis(&e(&[0])));
    assert!(is_basis(&e(&[1, 2])));
    assert!(is_basis(&e(&[1, 2]).scale(5)));
    assert!(is_basis(&StrideScalar::Int(7)));
    assert!(is_basis(&StrideScalar::Int(0)));
    assert!(is_basis(&e(&[2]).scale(0)));
    assert!(!is_basis(&e(&[0]).scale(3).add(&e(&[1]).scale(5)).unwrap()));
}

#[test]
fn proj_reads_the_leaf_a_basis_names() {
    let x = at!((7, (8, 9)));
    assert_eq!(proj(&x, &e(&[])).unwrap(), &x);
    assert_eq!(proj(&x, &e(&[0])).unwrap(), &StrideScalar::Int(7));
    assert_eq!(proj(&x, &e(&[1])).unwrap(), &at!((8, 9)));
    assert_eq!(proj(&x, &e(&[1, 0])).unwrap(), &StrideScalar::Int(8));
    assert_eq!(proj(&x, &e(&[1, 1])).unwrap(), &StrideScalar::Int(9));
}

#[test]
fn unit_keeps_the_path_and_drops_the_coefficient() {
    assert_eq!(unit(&e(&[])).unwrap(), StrideScalar::Int(1));
    assert_eq!(unit(&e(&[0])).unwrap(), e(&[0]));
    assert_eq!(unit(&e(&[1, 2])).unwrap(), e(&[1, 2]));
    assert_eq!(unit(&e(&[1, 2]).scale(5)).unwrap(), e(&[1, 2]));
}

#[test]
fn proj_and_unit_reject_a_sum() {
    let sum = e(&[0]).scale(3).add(&e(&[1]).scale(5)).unwrap();
    assert!(proj(&at!((7, 8)), &sum).is_err());
    assert!(unit(&sum).is_err());
}

#[test]
fn make_basis_like_puts_a_unit_at_every_leaf() {
    let leaf = |path: &[usize]| HTuple::Leaf(e(path));
    assert_eq!(make_basis_like(&ht!((3, 4))), HTuple::Tuple(vec![leaf(&[0]), leaf(&[1])]));
    assert_eq!(
        make_basis_like(&ht!(((3, 4), 5))),
        HTuple::Tuple(vec![HTuple::Tuple(vec![leaf(&[0, 0]), leaf(&[0, 1])]), leaf(&[1]),])
    );
}

#[test]
fn proj_tuple_reads_the_sub_tuple_a_basis_names() {
    let x = ht!((7, (8, 9)));
    assert_eq!(proj_tuple(&x, &e(&[])).unwrap(), &x);
    assert_eq!(proj_tuple(&x, &e(&[0])).unwrap(), &ht!(7));
    assert_eq!(proj_tuple(&x, &e(&[1])).unwrap(), &ht!((8, 9)));
    assert_eq!(proj_tuple(&x, &e(&[1, 1])).unwrap(), &ht!(9));
    assert_eq!(proj_tuple(&x, &e(&[1, 1]).scale(5)).unwrap(), &ht!(9));
}

#[test]
fn proj_tuple_mut_writes_through() {
    let mut x = HTuple::Tuple(vec![HTuple::Leaf(vec![1]), HTuple::Leaf(vec![2])]);
    if let HTuple::Leaf(slot) = proj_tuple_mut(&mut x, &e(&[1])).unwrap() {
        slot.push(3);
    }
    assert_eq!(x, HTuple::Tuple(vec![HTuple::Leaf(vec![1]), HTuple::Leaf(vec![2, 3])]));
}

#[test]
fn proj_tuple_rejects_a_sum_and_a_path_off_the_end() {
    let sum = e(&[0]).scale(3).add(&e(&[1]).scale(5)).unwrap();
    assert!(proj_tuple(&ht!((7, 8)), &sum).is_err());
    assert!(proj_tuple(&ht!((7, 8)), &e(&[2])).is_err());
    assert!(proj_tuple_mut(&mut ht!((7, 8)), &e(&[2])).is_err());
}
