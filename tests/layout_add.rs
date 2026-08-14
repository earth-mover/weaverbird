//! Ported from `test/test_layout_add.py`.
//!
//! The `F2` cases are absent, because `F2` is not ported: PyCuTe's
//! stride scalars are open to any additive type, and this crate carries
//! only the integer and the arithmetic tuple. That drops
//! `F2_LAYOUTS` and every test parameterized on it —
//! `test_identity_f2`, `test_commutativity_f2`, `test_associativity_f2`,
//! `test_self_inverse_f2`, `test_f2_disjoint_bits`,
//! `test_f2_overlapping_bits_same_shape`, `test_f2_zero_stride_identity`,
//! `test_f2_universal_int_zero_identity`, `test_f2_multimode`, and
//! `test_f2_different_shapes_pow2`. `test_self_inverse_f2` is `F2`-only
//! by construction — no integer or arithmetic-tuple stride is its own
//! additive inverse.
//!
//! `test_non_layout_argument_raises` is gone for good: the argument
//! types are layouts, so the Python `TypeError` is a compile error here.
#![expect(clippy::unwrap_used, reason = "a test asserts the happy path")]

use pinstripe::{
    ArithTuple, HTuple, Int, IntTuple, Layout, Stride, StrideScalar, e, ht, layout_add, size,
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

/// PyCuTe's `ArithTuple(...)` over integer children.
fn at(children: &[Int]) -> StrideScalar {
    StrideScalar::Arith(ArithTuple::from_data(
        children.iter().map(|&v| StrideScalar::Int(v)).collect(),
    ))
}

/// The domain size of a layout.
fn extent(l: &Layout) -> Int {
    size(&l.shape, &[]).unwrap()
}

/// `L(i)` at an integral coordinate.
fn at_index(l: &Layout, i: Int) -> StrideScalar {
    l.call(&ht!(i)).unwrap()
}

/// `Layout(size(a), 0)`: the universal additive identity.
fn zero_like(a: &Layout) -> Layout {
    layout(ht!(extent(a)), ht!(0))
}

/// PyCuTe's `postcondition_layout_add`: pointwise correctness, size
/// preservation, symmetry.
fn postcondition(a: &Layout, b: &Layout) -> Layout {
    let result = layout_add(Some(a), b).unwrap();

    // Size matches both inputs.
    assert_eq!(extent(&result), extent(a), "{a} + {b} => {result}");
    assert_eq!(extent(&result), extent(b), "{a} + {b} => {result}");

    // R(i) == A(i) + B(i) for every i in [0, size(R)).
    for i in 0..extent(&result) {
        assert_eq!(
            at_index(&result, i),
            at_index(a, i).add(&at_index(b, i)).unwrap(),
            "i={i}: A={a}, B={b}, R={result}"
        );
    }

    // Symmetric in A and B.
    assert_eq!(
        layout_add(Some(b), a).unwrap(),
        result,
        "asymmetric: {a} + {b}"
    );

    result
}

// ---------------------------------------------------------------------------
// Algebraic-property helpers, parameterized by example list
// ---------------------------------------------------------------------------

/// `Layout(size(A), 0)` (int-zero stride) is the *universal* additive
/// identity: it evaluates to int `0` at every coordinate, and `0` is the
/// additive identity in every stride-scalar type.
fn assert_identity(examples: &[Layout]) {
    for a in examples {
        let result = postcondition(a, &zero_like(a));
        // Coalesce-equivalent to A: layout_add returns canonical form.
        assert_eq!(result, a.coalesce(&HTuple::Leaf(Some(1))).unwrap());
    }
}

/// `layout_add(A, B) == layout_add(B, A)` whenever both sides are
/// defined. The post-condition checks the pointwise version; this
/// asserts structural equality, which is stronger.
fn assert_commutativity(examples: &[Layout]) {
    for a in examples {
        for b in examples {
            if extent(a) != extent(b) {
                continue;
            }
            if let (Ok(ab), Ok(ba)) = (layout_add(Some(a), b), layout_add(Some(b), a)) {
                assert_eq!(ab, ba, "{a} + {b}");
            }
        }
    }
}

/// `(A + B) + C` and `A + (B + C)` agree pointwise with
/// `A(i) + B(i) + C(i)` whenever every pair and triple admits a layout
/// sum. Triples that fail are skipped.
fn assert_associativity(examples: &[Layout]) {
    for a in examples {
        for b in examples {
            for c in examples {
                if extent(a) != extent(b) || extent(a) != extent(c) {
                    continue;
                }
                let left = layout_add(Some(a), b).and_then(|ab| layout_add(Some(&ab), c));
                let right = layout_add(Some(b), c).and_then(|bc| layout_add(Some(a), &bc));
                let (Ok(left), Ok(right)) = (left, right) else {
                    continue;
                };
                for i in 0..extent(&left) {
                    let want = at_index(a, i)
                        .add(&at_index(b, i))
                        .unwrap()
                        .add(&at_index(c, i))
                        .unwrap();
                    assert_eq!(
                        at_index(&left, i),
                        at_index(&right, i),
                        "i={i}: {a},{b},{c}"
                    );
                    assert_eq!(at_index(&left, i), want, "i={i}: {a},{b},{c}");
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Example layouts grouped by stride-scalar type.
//
// Each list is a curated set of layouts of *uniform size* for which
// every pair and triple sum is a layout, so the property helpers cover
// every combination without skipping.
// ---------------------------------------------------------------------------

/// Integer strides, size 12.
fn int_layouts() -> Vec<Layout> {
    vec![
        layout(ht!(12), ht!(0)),                    // broadcast (additive identity)
        layout(ht!(12), ht!(1)),                    // row-major
        layout(ht!(12), ht!(2)),                    // scaled
        layout(ht!((4, 3)), ht!((1, 4))),           // (4,3) row-major
        layout(ht!((4, 3)), ht!((3, 1))),           // (4,3) col-major
        layout(ht!((4, 3)), ht!((1, 0))),           // broadcast in second mode
        layout(ht!((2, 6)), ht!((1, 2))),           // alternative factorization
        layout(ht!(((2, 2), 3)), ht!(((2, 1), 4))), // nested
    ]
}

/// Arithmetic-tuple strides, size 12.
fn at_layouts() -> Vec<Layout> {
    vec![
        // Universal int-zero identity.
        layout(ht!(12), ht!(0)),
        // Rank-1 unit basis.
        strided(ht!(12), s(e(&[0]))),
        // Scaled basis.
        strided(ht!(12), s(e(&[0]).scale(2))),
        // Disjoint paths.
        strided(ht!((4, 3)), t(vec![s(e(&[0])), s(e(&[1]))])),
        // Swapped paths.
        strided(ht!((4, 3)), t(vec![s(e(&[1])), s(e(&[0]))])),
        // Same path -- adds along one axis.
        strided(ht!((4, 3)), t(vec![s(e(&[0])), s(e(&[0]))])),
        // Scaled basis at disjoint paths.
        strided(
            ht!((4, 3)),
            t(vec![s(e(&[0]).scale(2)), s(e(&[1]).scale(3))]),
        ),
        // Different shape, disjoint paths.
        strided(ht!((2, 6)), t(vec![s(e(&[0])), s(e(&[1]))])),
    ]
}

// ---------------------------------------------------------------------------
// Algebraic properties per stride-scalar type
// ---------------------------------------------------------------------------

#[test]
fn identity_int() {
    assert_identity(&int_layouts());
}

#[test]
fn identity_atuple() {
    assert_identity(&at_layouts());
}

#[test]
fn commutativity_int() {
    assert_commutativity(&int_layouts());
}

#[test]
fn commutativity_atuple() {
    assert_commutativity(&at_layouts());
}

#[test]
fn associativity_int() {
    assert_associativity(&int_layouts());
}

#[test]
fn associativity_atuple() {
    assert_associativity(&at_layouts());
}

// ---------------------------------------------------------------------------
// Trivial / edge cases
// ---------------------------------------------------------------------------

#[test]
fn singleton() {
    // Size-1 layouts always evaluate to 0, regardless of stride.
    assert_eq!(
        extent(&postcondition(
            &layout(ht!(1), ht!(0)),
            &layout(ht!(1), ht!(0))
        )),
        1
    );
    assert_eq!(
        extent(&postcondition(
            &layout(ht!(1), ht!(5)),
            &layout(ht!(1), ht!(7))
        )),
        1
    );

    // A singleton with basis strides also evaluates to 0 at i=0.
    assert_eq!(
        extent(&postcondition(
            &strided(ht!(1), s(e(&[0]))),
            &strided(ht!(1), s(e(&[1]).scale(2)))
        )),
        1
    );
}

#[test]
fn zero_stride_broadcast() {
    // Adding a stride-0 (broadcast) layout reduces to the other one.
    let result = postcondition(&layout(ht!(5), ht!(0)), &layout(ht!(5), ht!(1)));
    assert_eq!(result, layout(ht!(5), ht!(1)));

    // Adding two zero-stride layouts gives the zero layout.
    let result = postcondition(&layout(ht!(5), ht!(0)), &layout(ht!(5), ht!(0)));
    assert_eq!(result, layout(ht!(5), ht!(0)));
}

// ---------------------------------------------------------------------------
// Pointwise correctness -- integer strides
// ---------------------------------------------------------------------------

#[test]
fn int_same_shape_same_stride() {
    // Adding a layout to itself doubles its strides.
    let result = postcondition(&layout(ht!(12), ht!(1)), &layout(ht!(12), ht!(1)));
    assert_eq!(result, layout(ht!(12), ht!(2)));

    // Coalescing folds the multi-mode form back to the canonical
    // doubling.
    let result = postcondition(
        &layout(ht!((4, 3)), ht!((1, 4))),
        &layout(ht!((4, 3)), ht!((1, 4))),
    );
    assert_eq!(result, layout(ht!(12), ht!(2)));

    let result = postcondition(
        &layout(ht!((4, 3)), ht!((3, 1))),
        &layout(ht!((4, 3)), ht!((3, 1))),
    );
    assert_eq!(result, layout(ht!((4, 3)), ht!((6, 2))));
}

#[test]
fn int_same_shape_diff_stride() {
    // Row-major plus column-major: leaf-wise stride sum.
    let result = postcondition(
        &layout(ht!((4, 3)), ht!((1, 4))),
        &layout(ht!((4, 3)), ht!((3, 1))),
    );
    assert_eq!(result, layout(ht!((4, 3)), ht!((4, 5))));

    // Strides scale linearly under addition.
    let result = postcondition(&layout(ht!((6)), ht!((5))), &layout(ht!((6)), ht!((3))));
    assert_eq!(result, layout(ht!(6), ht!(8)));
}

#[test]
fn int_both_row_major_same_size() {
    // Different shapes, A(i) = i and B(i) = i: the result is `2*i`.
    let result = postcondition(
        &layout(ht!((5, 3, 4)), ht!((1, 5, 15))),
        &layout(ht!((10, 6)), ht!((1, 10))),
    );
    assert_eq!(result, layout(ht!(60), ht!(2)));

    let result = postcondition(
        &layout(ht!((6)), ht!((1))),
        &layout(ht!((3, 2)), ht!((1, 3))),
    );
    assert_eq!(result, layout(ht!(6), ht!(2)));
}

#[test]
fn int_one_flat_one_decomposed() {
    let result = postcondition(
        &layout(ht!((6)), ht!((1))),
        &layout(ht!((3, 2)), ht!((2, 1))),
    );
    assert_eq!(result, layout(ht!((3, 2)), ht!((3, 4))));

    let result = postcondition(&layout(ht!(12), ht!(2)), &layout(ht!((4, 3)), ht!((3, 1))));
    assert_eq!(result, layout(ht!((4, 3)), ht!((5, 9))));
}

#[test]
fn int_nested_shapes() {
    // Nested shapes are flattened by greatest_common_domain.
    let result = postcondition(
        &layout(ht!((2, (3, 4))), ht!((12, (4, 1)))),
        &layout(ht!((6, 4)), ht!((4, 1))),
    );
    assert_eq!(result, layout(ht!((2, 3, 4)), ht!((16, 12, 2))));
}

#[test]
fn int_zero_stride_in_one_mode() {
    // Stride-0 in one mode of a multi-mode layout (broadcast row).
    let result = postcondition(
        &layout(ht!((4, 3)), ht!((1, 0))),
        &layout(ht!((4, 3)), ht!((3, 1))),
    );
    assert_eq!(result, layout(ht!((4, 3)), ht!((4, 1))));
}

// ---------------------------------------------------------------------------
// Pointwise correctness -- arithmetic-tuple strides
// ---------------------------------------------------------------------------

#[test]
fn atuple_disjoint_basis_paths() {
    // E(0) + E(1) on the same shape gives the rank-2 unit arithmetic
    // tuple at the same position: each output coordinate is (i, i).
    let result = postcondition(&strided(ht!(8), s(e(&[0]))), &strided(ht!(8), s(e(&[1]))));
    for i in 0..8 {
        assert_eq!(at_index(&result, i), at(&[i, i]));
    }
}

#[test]
fn atuple_same_basis_path() {
    // 2*E(0) + 3*E(0) = 5*E(0) at every coordinate.
    let result = postcondition(
        &strided(ht!(8), s(e(&[0]).scale(2))),
        &strided(ht!(8), s(e(&[0]).scale(3))),
    );
    assert_eq!(result, strided(ht!(8), s(e(&[0]).scale(5))));
}

#[test]
fn atuple_multimode() {
    // Multi-mode layouts with disjoint bases: leaf-wise stride sum.
    let result = postcondition(
        &strided(ht!((4, 3)), t(vec![s(e(&[0])), s(e(&[1]))])),
        &strided(ht!((4, 3)), t(vec![s(e(&[1])), s(e(&[0]))])),
    );
    // R(c0, c1) = (c0 + c1) * E(0) + (c0 + c1) * E(1).
    for c0 in 0..4 {
        for c1 in 0..3 {
            assert_eq!(
                result.call(&ht!((c0, c1))).unwrap(),
                at(&[c0 + c1, c0 + c1])
            );
        }
    }
}

#[test]
fn atuple_universal_int_zero_identity() {
    // Layout(N, 0) is the universal additive identity: it works even
    // for arithmetic-tuple-strided layouts.
    let a = strided(ht!((4, 3)), t(vec![s(e(&[0])), s(e(&[1]))]));
    assert_eq!(postcondition(&a, &zero_like(&a)), a);
}

#[test]
fn atuple_different_shapes() {
    // Layouts of different shapes that share a refinement -- the
    // arithmetic tuple's component-wise additivity is linear in any
    // integer decomposition, so this is unconditional.
    let a = strided(ht!(12), s(e(&[0])));
    let b = strided(ht!((4, 3)), t(vec![s(e(&[1])), s(e(&[2]))]));
    let result = postcondition(&a, &b);
    // R(c) = c*E(0) + (c%4)*E(1) + (c//4)*E(2).
    for i in 0..12 {
        assert_eq!(at_index(&result, i), at(&[i, i % 4, i / 4]));
    }
}

// ---------------------------------------------------------------------------
// Failure cases
// ---------------------------------------------------------------------------

#[test]
fn size_mismatch_errors() {
    assert!(layout_add(Some(&layout(ht!(6), ht!(1))), &layout(ht!(8), ht!(1))).is_err());
    assert!(
        layout_add(
            Some(&layout(ht!((4, 3)), ht!((3, 1)))),
            &layout(ht!((6, 5)), ht!((5, 1)))
        )
        .is_err()
    );
}

#[test]
fn no_common_refinement_errors() {
    // Pairwise coprime leaves: no aligned common refinement of size A.
    assert!(
        layout_add(
            Some(&layout(ht!((5, 3)), ht!((3, 1)))),
            &layout(ht!((3, 5)), ht!((5, 1)))
        )
        .is_err()
    );
    assert!(
        layout_add(
            Some(&layout(ht!((7, 11)), ht!((11, 1)))),
            &layout(ht!((11, 7)), ht!((7, 1)))
        )
        .is_err()
    );
}

#[test]
fn incompatible_stride_types_error() {
    // Basis-strided plus non-zero integer-strided: the per-leaf
    // `A(d) + B(d)` evaluation hits an arithmetic tuple plus a non-zero
    // integer, which has no value.
    assert!(layout_add(Some(&strided(ht!(8), s(e(&[0])))), &layout(ht!(8), ht!(1))).is_err());
}

// ---------------------------------------------------------------------------
// Seeding
// ---------------------------------------------------------------------------

#[test]
fn absent_left_operand_returns_the_right() {
    // PyCuTe's `if A is None: return B`, the seed of the `_composition`
    // fold.
    let b = layout(ht!((4, 3)), ht!((3, 1)));
    assert_eq!(layout_add(None, &b).unwrap(), b);
}
