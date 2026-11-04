//! The `serde` feature: a layout and its parts on a wire.
//!
//! The derives are mechanical, so these check the two things that are not:
//! a round trip keeps the hierarchy and the basis strides, and a decode
//! goes through [`Layout::new`], which refuses a stride that does not match
//! its shape.
#![cfg(feature = "serde")]
#![expect(clippy::unwrap_used, reason = "a test asserts the happy path")]

use weaverbird::{HTuple, Layout, StrideScalar, e, ht, layout, scaled_basis};

fn round_trip<T: serde::Serialize + serde::de::DeserializeOwned>(value: &T) -> T {
    serde_json::from_str(&serde_json::to_string(value).unwrap()).unwrap()
}

#[test]
fn a_flat_layout_round_trips() {
    let a = layout!((4, 8):(8, 1));
    assert_eq!(round_trip(&a), a);
}

#[test]
fn a_nested_layout_round_trips() {
    let a = layout!((4, (2, 3)):(2, (1, 8)));
    assert_eq!(round_trip(&a), a);
}

/// A view strides by basis vectors, and those are the scalars that must survive.
#[test]
fn basis_strides_round_trip() {
    let stride: HTuple<StrideScalar> = ht!((e(&[0]), scaled_basis(3, &[1])));
    assert_eq!(round_trip(&stride), stride);
}

#[test]
fn a_decode_refuses_an_incongruent_stride() {
    let bad = r#"{"shape":{"Tuple":[{"Leaf":4},{"Leaf":8}]},"stride":{"Leaf":{"Int":1}}}"#;
    let err = serde_json::from_str::<Layout>(bad).unwrap_err();
    assert!(err.to_string().contains("congruent"), "{err}");
}
