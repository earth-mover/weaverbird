//! A toy accessor stands in for a caller's storage: the tensor's own
//! methods are what these exercise.

use crate::{Accessor, Int, Layout, Result, StrideScalar, Tensor, e, ht, layout};

/// Reads a 2D grid of `(row, col)` through basis strides.
#[derive(Clone, Debug)]
struct Grid {
    cols: Int,
    origin: StrideScalar,
}

impl Accessor for Grid {
    type Value = (Int, Int);

    fn offset(&self, by: &StrideScalar) -> Result<Self> {
        Ok(Grid { cols: self.cols, origin: self.origin.add(by)? })
    }

    /// The stride is `r*E0 + c*E1`, so `to_tuple` reads the two coordinates back.
    fn get(&self, at: &StrideScalar) -> Result<Self::Value> {
        let at = self.origin.add(at)?;
        Ok((coordinate(&at, 0), coordinate(&at, 1)))
    }
}

/// The multiple of basis vector `k` in `at`. A missing position is implicitly zero.
fn coordinate(at: &StrideScalar, k: usize) -> Int {
    at.get(&[k]).ok().and_then(StrideScalar::as_int).unwrap_or(0)
}

fn grid(cols: Int) -> Grid {
    Grid { cols, origin: StrideScalar::Int(0) }
}

#[test]
fn identity_reads_its_own_coordinate() -> Result<()> {
    let t = Tensor::identity(grid(8), &[4, 8]);
    assert_eq!(t.shape(), vec![4, 8]);
    assert_eq!(t.get(&[2, 3])?, (2, 3));
    Ok(())
}

#[test]
fn permute_swaps_the_axes() -> Result<()> {
    let t = Tensor::identity(grid(8), &[4, 8]).permute(&[1, 0])?;
    assert_eq!(t.shape(), vec![8, 4]);
    assert_eq!(t.get(&[3, 2])?, (2, 3));
    Ok(())
}

#[test]
fn broadcast_adds_a_stride_0_axis() -> Result<()> {
    let t = Tensor::identity(grid(8), &[4, 8]).broadcast(0, 3)?;
    assert_eq!(t.shape(), vec![3, 4, 8]);
    assert_eq!(t.get(&[0, 2, 3])?, t.get(&[2, 2, 3])?);
    Ok(())
}

#[test]
fn new_keeps_the_layout_it_is_given() -> Result<()> {
    let t = Tensor::new(grid(8), Layout::from_parts(ht!((2, 2)), ht!((e(&[0]), e(&[1])))));
    assert_eq!(t.layout().shape, ht!((2, 2)));
    assert_eq!(t.get(&[1, 1])?, (1, 1));
    Ok(())
}

#[test]
fn squeeze_drops_the_named_mode() -> Result<()> {
    let t = Tensor::identity(grid(8), &[1, 8]).squeeze(&[0])?;
    assert_eq!(t.shape(), vec![8]);
    Ok(())
}

/// `compose` is the layout's own, lifted: the accessor rides along.
#[test]
fn compose_tiles_the_layout() -> Result<()> {
    let t = Tensor::identity(grid(8), &[4, 8]);
    assert_eq!(t.compose(layout!(4:1).into())?.shape(), vec![4]);
    Ok(())
}
