//! CuTe layouts. Mirrors `pycute/layout.py`.
//!
//! A [`Layout`] is a map from a coordinate domain to a codomain, given
//! by a shape and a congruent stride. Everything else in the crate
//! exists to build one, read one, or take one apart.
//!
//! PyCuTe carries an empty `LayoutBase` class so that `is_layout` can
//! ask `isinstance`. Rust has the type itself, so the marker class and
//! its predicate are both absent: `Layout` *is* the answer to
//! `is_layout`.
//!
//! The layout algebra — coalesce, composition, the inverses, complement,
//! divide, product — lands in a later module. This one holds the type,
//! its accessors, and the factories.

use std::{
    cmp::Ordering,
    fmt::{self, Debug, Display},
};

use crate::{
    atuple::{ArithTuple, StrideScalar, as_tuple, make_basis_like, proj, unit},
    error::{Error, Result},
    ht,
    htuple::{HTuple, congruent, slice_, transform_apply_leaf},
    shape::idx2crd,
    stride::{Coshape, coalesce_z, inner_product, prefix_product},
    typedefs::{Int, IntTuple, Stride},
};

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------

/// A map from a coordinate domain to a codomain, defined by a `shape`
/// and a `stride`.
///
/// The `shape` is an [`IntTuple`] of extents. The `stride` is congruent
/// with it and holds a [`StrideScalar`] at every leaf.
///
/// A layout evaluates as `L(c) == inner_product(idx2crd(c, shape),
/// stride)`, mapping any coordinate of `shape` — integral, flat, natural
/// — to a codomain value.
///
/// ```text
/// Layout::new((4, 8), 1)      == Layout((4, 8), (1, 4))   // compact column-major
/// Layout((4, 8), (8, 1))(2, 3) == 19                      // evaluate a coordinate
/// ```
#[derive(Clone, PartialEq, Eq)]
pub struct Layout {
    /// The domain.
    pub shape: IntTuple,
    /// The map. Congruent with `shape`.
    pub stride: Stride,
}

impl Layout {
    /// Constructs a layout from a `shape` and a `stride`. PyCuTe's
    /// `__init__`.
    ///
    /// The stride runs through [`prefix_product`], so a single scalar
    /// base fills in a compact, column-major stride. PyCuTe defaults
    /// that base to `1`; Rust has no default argument, so every caller
    /// spells it out.
    pub fn new(shape: IntTuple, stride: &Stride) -> Result<Self> {
        prefix_product(&shape, stride).map(|stride| Layout { shape, stride })
    }

    /// Constructs a layout from an already-computed shape and stride.
    /// PyCuTe's `_set`.
    ///
    /// Skips the [`prefix_product`] and the congruence check.
    pub fn set(shape: IntTuple, stride: Stride) -> Self {
        Layout { shape, stride }
    }

    /// Maps a coordinate to the layout's codomain:
    /// `L(c) == inner_product(idx2crd(c, shape), stride)`.
    ///
    /// PyCuTe's `__call__`. It accepts a coordinate in any form —
    /// integral, flat, or natural — passed either whole or as separate
    /// per-mode arguments. Rust has no varargs, so the coordinate always
    /// arrives whole: `L(2, 3)` is `l.call(&ht!((2, 3)))`.
    ///
    /// ```text
    /// L = Layout((4, 8), (8, 1))
    /// L(14) == 19
    /// L((2, 3)) == 19
    /// ```
    pub fn call(&self, crd: &IntTuple) -> Result<StrideScalar> {
        inner_product(&idx2crd(crd, &self.shape)?, &self.stride)
    }

    /// Evaluates `crd` to a codomain offset AND slices the layout,
    /// returning `(offset, sublayout)`. PyCuTe's `_offset_and_slice`.
    ///
    /// `None` entries in `crd` mark the modes the sublayout retains, so
    /// the coordinate is carried as an [`HTuple`] of [`Option`] — the
    /// shape [`slice_`] expects.
    pub fn offset_and_slice(&self, crd: &HTuple<Option<Int>>) -> Result<(StrideScalar, Self)> {
        Ok((
            inner_product(&idx2crd_opt(crd, &self.shape)?, &self.stride)?,
            Layout::set(slice_(crd, &self.shape)?, slice_(crd, &self.stride)?),
        ))
    }

    /// Gets mode `i` of the layout as a sublayout — tuple-like indexing
    /// over modes. PyCuTe's `__getitem__`.
    ///
    /// Returns [`Error::BadPath`] when `i` runs past the rank.
    pub fn index(&self, i: usize) -> Result<Self> {
        let bad_path = || Error::BadPath {
            path: vec![i],
            value: format!("{self}"),
        };
        if i >= self.shape.rank() {
            return Err(bad_path());
        }
        match self.shape.is_tuple() {
            true => Ok(Layout::set(
                self.shape.get(&[i]).ok_or_else(bad_path)?.clone(),
                self.stride.get(&[i]).ok_or_else(bad_path)?.clone(),
            )),
            false => Ok(self.clone()),
        }
    }

    /// Gets the sublayout at the given (possibly nested) `mode` path.
    /// PyCuTe's `get`.
    ///
    /// Returns [`Error::BadPath`] when `mode` does not address the
    /// layout.
    pub fn get(&self, mode: &[usize]) -> Result<Self> {
        let bad_path = || Error::BadPath {
            path: mode.to_vec(),
            value: format!("{self}"),
        };
        Ok(Layout::set(
            self.shape.get(mode).ok_or_else(bad_path)?.clone(),
            self.stride.get(mode).ok_or_else(bad_path)?.clone(),
        ))
    }

    /// Coalesces the layout per `profile`, keeping size-1 modes.
    /// PyCuTe's `Layout._coalesce_z`.
    ///
    /// Returns [`Error::RankMismatch`] when `profile` outranks the
    /// layout.
    pub fn coalesce_z(&self, profile: &Profile) -> Result<Self> {
        self.coalesce_with("coalesce_z", profile, &|shape, stride| Ok((shape, stride)))
    }

    /// Coalesces the layout per `profile`. PyCuTe's `Layout._coalesce`.
    ///
    /// This is [`Self::coalesce_z`] plus one step: a trailing size-1
    /// mode is dropped, so long as it is not the only one left.
    ///
    /// Returns [`Error::RankMismatch`] when `profile` outranks the
    /// layout.
    pub fn coalesce(&self, profile: &Profile) -> Result<Self> {
        self.coalesce_with(
            "coalesce",
            profile,
            &|shape, stride| match (&shape, &stride) {
                (HTuple::Tuple(s), HTuple::Tuple(d))
                    if s.len() > 1 && s.last() == Some(&ht!(1)) =>
                {
                    Ok((
                        HTuple::Tuple(s[..s.len() - 1].to_vec()),
                        HTuple::Tuple(d[..d.len() - 1].to_vec()),
                    ))
                }
                _ => Ok((shape, stride)),
            },
        )
    }

    /// The body the two coalesce variants share. PyCuTe repeats it; the
    /// two differ only in `trim`, which runs on the folded shape and
    /// stride before they are unwrapped.
    fn coalesce_with(
        &self,
        op: &'static str,
        profile: &Profile,
        trim: &impl Fn(IntTuple, Stride) -> Result<(IntTuple, Stride)>,
    ) -> Result<Self> {
        match profile {
            // A `None` profile is the no-op.
            HTuple::Leaf(None) => Ok(self.clone()),
            // A tuple profile dispatches by mode. PyCuTe zips the layout
            // against it with `zip_longest`, so the modes the profile
            // runs out on take the no-op.
            HTuple::Tuple(modes) => {
                if self.shape.rank() < modes.len() {
                    return Err(Error::RankMismatch {
                        op,
                        value: format!("{self}"),
                        profile: format!("{profile:?}"),
                    });
                }
                (0..self.shape.rank())
                    .map(|i| {
                        self.index(i)?.coalesce_with(
                            op,
                            modes.get(i).unwrap_or(&HTuple::Leaf(None)),
                            trim,
                        )
                    })
                    .collect::<Result<Vec<_>>>()
                    .map(make_layout)
            }
            HTuple::Leaf(Some(_)) => {
                let (shape, stride) = coalesce_z(&self.shape, &self.stride)?;
                // An empty fold means every mode dropped away. What is
                // left is the one-element layout.
                if shape.rank() == 0 {
                    return Ok(Layout::set(ht!(1), Stride::Leaf(0.into())));
                }
                let (shape, stride) = trim(shape, stride)?;
                Ok(Layout::set(shape.unwrap().clone(), stride.unwrap().clone()))
            }
        }
    }
}

/// The by-mode profile of a coalesce. PyCuTe writes `1` to coalesce a
/// mode, `None` to leave it alone, and a tuple to dispatch by mode.
///
/// Only the presence of the leaf is read, never its value — PyCuTe's `1`
/// is a placeholder. The [`Int`] is kept so the two spellings match.
pub type Profile = HTuple<Option<Int>>;

/// Shape of the codomain. PyCuTe's `_coshape`.
///
/// PyCuTe returns the coshape outright; here the trait carries a
/// [`Result`], because the inner product of a shape and a stride is
/// itself fallible — an incongruent pair, or a stride that mixes an
/// integer with a basis element, has no coshape. PyCuTe raises in both
/// cases.
impl Coshape for Layout {
    fn coshape(&self) -> Result<IntTuple> {
        let result = inner_product(&self.shape.transform_leaf(&|s| s - 1), &self.stride)?;
        result.add(&ones_like(&result)).map(|c| as_tuple(&c))
    }
}

/// Compact `shape:stride` form, e.g. `(4,8):(1,4)`. PyCuTe's `__str__`.
impl Display for Layout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}:{:?}", self.shape, self.stride)
    }
}

/// Constructor form, e.g. `Layout((4,8), (1,4))`. PyCuTe's `__repr__`.
impl Debug for Layout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Layout({:?}, {:?})", self.shape, self.stride)
    }
}

/// One at every leaf of `x`. PyCuTe writes this as
/// `repeat_like(1, shape(result))`, reading the profile off the element
/// itself.
fn ones_like(x: &StrideScalar) -> StrideScalar {
    match x {
        StrideScalar::Int(_) => StrideScalar::Int(1),
        StrideScalar::Arith(a) => StrideScalar::Arith(ArithTuple::from_data(
            a.data().iter().map(ones_like).collect(),
        )),
    }
}

/// [`idx2crd`] over a coordinate whose modes may be open.
///
/// PyCuTe's `idx2crd` answers `repeat_like(0, shape)` for a `None`
/// coordinate, so a sliced mode contributes nothing to the offset. The
/// ported [`idx2crd`] carries no `None`, so that one branch lives here,
/// beside its only caller.
fn idx2crd_opt(crd: &HTuple<Option<Int>>, shape: &IntTuple) -> Result<IntTuple> {
    match crd {
        HTuple::Leaf(None) => Ok(HTuple::repeat_like(&0, shape)),
        HTuple::Leaf(Some(i)) => idx2crd(&HTuple::Leaf(*i), shape),
        HTuple::Tuple(cs) => match shape {
            HTuple::Tuple(ss) if cs.len() == ss.len() => cs
                .iter()
                .zip(ss)
                .map(|(c, s)| idx2crd_opt(c, s))
                .collect::<Result<Vec<_>>>()
                .map(HTuple::Tuple),
            _ => Err(Error::BadCoord {
                idx: format!("{crd:?}"),
                shape: format!("{shape:?}"),
            }),
        },
    }
}

// ---------------------------------------------------------------------------
// Factories
// ---------------------------------------------------------------------------

/// Concatenates multiple layouts; each input becomes one mode of the
/// result. PyCuTe's `make_layout`.
///
/// Post-conditions:
///   `rank(result) == layouts.len()`,
///   `result[i] == layouts[i]` for every `i`.
///
/// ```text
/// make_layout([Layout(3, 1), Layout((5, 1), (7, 2)), Layout(2, 42)])
///     == Layout((3, (5, 1), 2), (1, (7, 2), 42))
/// ```
pub fn make_layout(layouts: Vec<Layout>) -> Layout {
    let (shape, stride) = layouts.into_iter().map(|a| (a.shape, a.stride)).unzip();
    Layout::set(HTuple::Tuple(shape), HTuple::Tuple(stride))
}

/// A compact layout with the same shape as `layout`, whose strides
/// follow the ordering induced by `layout`'s strides. PyCuTe's
/// `make_layout_like`.
///
/// The mode with the smallest non-zero source stride receives stride 1,
/// and the remaining non-zero modes receive compact (prefix-product)
/// strides in stable ascending order of the source strides. A mode that
/// carries no positional information — a stride of 0 — is pinned to
/// stride 0.
///
/// PyCuTe orders only the static strides and treats a symbolic one as
/// larger than every static one. [`Int`] is always static, so the sort
/// is plain — but still *stable*, which is what keeps ties in
/// left-to-right order.
///
/// Post-conditions:
///   `shape(result) == shape(layout)`,
///   the non-zero modes of `result` form a compact layout,
///   idempotent: `make_layout_like(make_layout_like(a)) == make_layout_like(a)`.
///
/// ```text
/// make_layout_like(Layout((4, 8), (100, 1)))            == Layout((4, 8), (8, 1))
/// make_layout_like(Layout((2, 3, 4, 5), (0, 42, 4, 0))) == Layout((2, 3, 4, 5), (0, 4, 1, 0))
/// ```
pub fn make_layout_like(layout: &Layout) -> Result<Layout> {
    let flat_s = layout.shape.leaves();
    let flat_d = layout.stride.leaves();

    // The modes in stride order. The sort is stable, so strides that do
    // not compare — a partial order, over the arithmetic tuples — keep
    // their left-to-right order.
    let mut order = (0..flat_s.len()).collect::<Vec<_>>();
    order.sort_by(|&a, &b| {
        flat_d
            .get(a)
            .zip(flat_d.get(b))
            .and_then(|(x, y)| x.partial_cmp(y))
            .unwrap_or(Ordering::Equal)
    });

    let (result_d, _) = order.into_iter().fold(
        (vec![StrideScalar::Int(0); flat_s.len()], 1),
        |(mut result_d, current), i| match flat_d.get(i).is_some_and(|d| d.is_zero()) {
            // A stride-0 mode keeps its result stride at 0.
            true => (result_d, current),
            false => {
                let extent = flat_s.get(i).copied().copied().unwrap_or(1);
                result_d[i] = StrideScalar::Int(current);
                (result_d, current * extent)
            }
        },
    );

    unflatten_stride(result_d, &layout.shape)
        .map(|stride| Layout::set(layout.shape.clone(), stride))
}

/// A compact layout with the same shape as `shape`, whose strides follow
/// the ordering induced by `order`. PyCuTe's `make_ordered_layout`.
///
/// The mode with the smallest `order` receives stride 1, and the rest
/// receive compact strides in ascending order of `order`. Only the
/// relative ordering matters, not the magnitudes, so `order` need not be
/// a contiguous `0..rank` permutation. Ties keep their left-to-right
/// position, because the sort is stable.
///
/// Pre-conditions:
///   `congruent(shape, order)`.
///
/// ```text
/// make_ordered_layout((4, 8), (1, 0))             == Layout((4, 8), (8, 1))
/// make_ordered_layout((2, 3, 4, 2), (0, 2, 3, 0)) == Layout((2, 3, 4, 2), (1, 4, 12, 2))
/// ```
///
/// Returns [`Error::NotCongruent`] when `order` does not match `shape`.
pub fn make_ordered_layout(shape: &IntTuple, order: &IntTuple) -> Result<Layout> {
    if !congruent(shape, order) {
        return Err(Error::NotCongruent {
            lhs: format!("{shape:?}"),
            rhs: format!("{order:?}"),
        });
    }
    let flat_s = shape.leaves();
    let flat_o = order.leaves();

    let mut modes = (0..flat_s.len()).collect::<Vec<_>>();
    modes.sort_by_key(|&i| flat_o.get(i).copied());

    let (result_d, _) = modes.into_iter().fold(
        (vec![StrideScalar::Int(0); flat_s.len()], 1),
        |(mut result_d, current), i| {
            let extent = flat_s.get(i).copied().copied().unwrap_or(1);
            result_d[i] = StrideScalar::Int(current);
            (result_d, current * extent)
        },
    );

    unflatten_stride(result_d, shape).map(|stride| Layout::set(shape.clone(), stride))
}

/// Rebuilds a flat run of stride leaves against a shape's profile.
///
/// The two factories above each produce exactly one leaf per leaf of the
/// shape, so the run never falls short; the [`Error::Incompatible`] is
/// there for the type, not for a reachable case.
fn unflatten_stride(leaves: Vec<StrideScalar>, shape: &IntTuple) -> Result<Stride> {
    HTuple::unflatten(&mut leaves.into_iter(), shape).ok_or_else(|| Error::Incompatible {
        lhs: format!("{shape:?}"),
        rhs: String::from("stride"),
    })
}

// ---------------------------------------------------------------------------
// Tiler type (Whitepaper, §3.3.5 By-mode Composition and Tilers).
// ---------------------------------------------------------------------------

/// The leaf of a [`Tiler`]: a mode extent, or a layout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TilerLeaf {
    Int(Int),
    Layout(Layout),
}

/// An [`HTuple`] of [`Int`] or [`Layout`]: the by-mode tiler argument to
/// the algebra.
///
/// It is the general right-hand side accepted by composition,
/// `logical_divide` and `logical_product`; [`tiler_to_layout`] turns it
/// into the equivalent single layout.
///
/// PyCuTe's `Tiler` also admits `None`, a per-mode no-op. That case is
/// not a leaf value here — the algebra methods that honour it take an
/// [`Option`] instead.
pub type Tiler = HTuple<TilerLeaf>;

/// Transforms a tiler into a layout that acts identically under
/// composition. PyCuTe's `tiler_to_layout`.
///
/// `e` scales the strides of a layout leaf. PyCuTe defaults it to `1`;
/// spell that as `StrideScalar::Int(1)`.
///
/// Post-conditions:
///   `shape(result) == shape(tiler)`,
///   `composition(a, result) == composition(a, tiler)`,
///   `logical_divide(a, result) == zipped_divide(a, tiler)`.
///
/// ```text
/// tiler_to_layout(3)                            == Layout(3, 1)
/// tiler_to_layout((4, 5))                       == Layout((4, 5), (1@0, 1@1))
/// tiler_to_layout((Layout(4, 2), Layout(5, 3))) == Layout((4, 5), (2@0, 3@1))
/// ```
pub fn tiler_to_layout(tiler: &Tiler, e: &StrideScalar) -> Result<Layout> {
    match tiler {
        HTuple::Leaf(TilerLeaf::Int(n)) => {
            Ok(Layout::set(HTuple::Leaf(*n), HTuple::Leaf(e.clone())))
        }
        // Each mode takes its own basis element, so the tiler's modes
        // land in separate positions of the codomain. The outer `e` is
        // dropped, as PyCuTe drops it.
        HTuple::Tuple(_) => transform_apply_leaf(
            &make_layout,
            &|t, basis| match (t, basis) {
                (Some(t), Some(HTuple::Leaf(basis))) => tiler_to_layout(t, basis),
                _ => Err(Error::BadPath {
                    path: vec![],
                    value: format!("{tiler:?}"),
                }),
            },
            Some(tiler),
            Some(&make_basis_like(tiler)),
        ),
        HTuple::Leaf(TilerLeaf::Layout(l)) => l
            .stride
            .leaves()
            .into_iter()
            .map(|d| scale_by(e, d))
            .collect::<Result<Vec<_>>>()
            .and_then(|leaves| unflatten_stride(leaves, &l.shape))
            .map(|stride| Layout::set(l.shape.clone(), stride)),
    }
}

/// The product `e * d`, as PyCuTe's `ArithTuple.__mul__` computes it:
/// one side must be an integer, which then scales the other.
///
/// Returns [`Error::NoProduct`] for two arithmetic tuples. Their product
/// leaves the module, so PyCuTe raises there.
fn scale_by(e: &StrideScalar, d: &StrideScalar) -> Result<StrideScalar> {
    match (e, d) {
        (StrideScalar::Int(a), _) => Ok(d.scale(*a)),
        (_, StrideScalar::Int(b)) => Ok(e.scale(*b)),
        _ => Err(Error::NoProduct {
            lhs: format!("{e:?}"),
            rhs: format!("{d:?}"),
        }),
    }
}

// ---------------------------------------------------------------------------
// recast
// ---------------------------------------------------------------------------

/// The element scale of [`recast`]: a positive rational.
///
/// PyCuTe passes an `int` or a `fractions.Fraction`, because the
/// argument must express both packing (`8` source elements per new one)
/// and unpacking (`Fraction(1, 2)`, two new elements per source one).
/// Rust has no rational in `std`, so this carries the numerator and
/// denominator itself — every use inside `recast` is then exact integer
/// arithmetic, and the crate takes on no dependency.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Scale {
    num: Int,
    den: Int,
}

impl Scale {
    /// `n` source elements per new element. PyCuTe's plain `int` scale.
    pub fn whole(n: Int) -> Result<Self> {
        Scale::fraction(n, 1)
    }

    /// `num / den` source elements per new element. PyCuTe's `Fraction`.
    ///
    /// Returns [`Error::Divisibility`] for a non-positive scale — the
    /// rescaling below divides by it.
    pub fn fraction(num: Int, den: Int) -> Result<Self> {
        match num > 0 && den > 0 {
            true => Ok(Scale { num, den }),
            false => Err(Error::Divisibility {
                detail: format!("scale {num}/{den} is not positive"),
            }),
        }
    }

    /// `divmod(d, scale)`: the floor quotient, and whether the division
    /// came out even. PyCuTe's `qdn, rdn`.
    fn divide_into(self, d: Int) -> (Int, bool) {
        let scaled = d * self.den;
        (scaled.div_euclid(self.num), scaled % self.num == 0)
    }

    /// `divmod(scale, d)`: the floor quotient, and whether the division
    /// came out even. PyCuTe's `qnd, rnd`.
    fn divide_by(self, d: Int) -> (Int, bool) {
        let divisor = self.den * d;
        (self.num.div_euclid(divisor), self.num % divisor == 0)
    }

    /// `-(-shape // scale)`: the extent that holds `shape` source
    /// elements at this scale.
    fn ceil_div(self, shape: Int) -> Int {
        ceil_div(shape * self.den, self.num)
    }
}

/// `-(-a // b)`, for a positive `b`.
fn ceil_div(a: Int, b: Int) -> Int {
    a.div_euclid(b) + Int::from(a.rem_euclid(b) != 0)
}

/// Recasts a layout to a new element scale. PyCuTe's `recast`.
///
/// Rewrites both shape and stride so the layout addresses a
/// differently-sized element. Each leaf `s:d` is rescaled by the ratio
/// between `d` and `scale` — shrinking the shape when packing, growing
/// it when unpacking.
///
/// PyCuTe projects `scale` through each stride's basis path, so a
/// tuple-valued scale could differ per codomain position. A [`Scale`]
/// carries no positions, so that projection is the identity here, and a
/// basis-strided layout recasts by its coefficient rather than raising.
///
/// Pre-conditions:
///   at each leaf the stride and the scale divide cleanly — one is a
///   multiple of the other.
///
/// ```text
/// recast(Layout(24, 1), 8)          == Layout(3, 1)
/// recast(Layout(24, 2), 4)          == Layout(12, 1)
/// recast(Layout((4, 4), (4, 1)), 4) == Layout((4, 1), (1, 1))
/// ```
///
/// Returns [`Error::Divisibility`] when a leaf divides unevenly.
pub fn recast(layout: &Layout, scale: Scale) -> Result<Layout> {
    let recast_elem = |shape: Option<&IntTuple>, stride: Option<&Stride>| match (shape, stride) {
        (Some(HTuple::Leaf(shape)), Some(HTuple::Leaf(stride))) => {
            recast_leaf(*shape, stride, scale)
        }
        _ => Err(Error::BadPath {
            path: vec![],
            value: format!("{layout}"),
        }),
    };
    transform_apply_leaf(
        &make_layout,
        &recast_elem,
        Some(&layout.shape),
        Some(&layout.stride),
    )
}

/// One leaf of [`recast`]. PyCuTe's `recast_elem`.
fn recast_leaf(shape: Int, stride: &StrideScalar, scale: Scale) -> Result<Layout> {
    let dd = match proj(stride, stride)? {
        StrideScalar::Int(v) => *v,
        other => {
            return Err(Error::NotBasis {
                value: format!("{other:?}"),
            });
        }
    };
    if dd == 0 {
        return Ok(Layout::set(
            HTuple::Leaf(shape),
            HTuple::Leaf(stride.clone()),
        ));
    }
    if dd == 1 {
        return Ok(Layout::set(
            HTuple::Leaf(scale.ceil_div(shape)),
            HTuple::Leaf(stride.clone()),
        ));
    }
    let (qdn, rdn) = scale.divide_into(dd);
    let (qnd, rnd) = scale.divide_by(dd);
    if !(rdn || rnd) {
        return Err(Error::Divisibility {
            detail: format!("recast {shape}, {stride:?}, {}/{}", scale.num, scale.den),
        });
    }
    Ok(Layout::set(
        HTuple::Leaf(ceil_div(shape, if rnd { qnd } else { 1 })),
        HTuple::Leaf(unit(stride)?.scale(if rdn { qdn } else { 1 })),
    ))
}
