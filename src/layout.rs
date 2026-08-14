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
//! The layout algebra lives here too, as it does in `layout.py`: the
//! `_`-prefixed methods of PyCuTe's `Layout` are the implementations,
//! and `algebra.rs` is the free-function facade over them. PyCuTe lets
//! the two modules import each other — `_composition` reaches for
//! `algebra.layout_add`, `_logical_divide` for `algebra.complement`.
//! Rust modules cannot circle that way, so the algorithms all stay on
//! this side and the dependency runs one way: `layout.rs → algebra.rs`,
//! for `layout_add` and `greatest_common_domain` alone.

use std::{
    cmp::Ordering,
    fmt::{self, Debug, Display},
};

use crate::{
    algebra::layout_add,
    atuple::{
        ArithTuple, StrideScalar, as_tuple, basis_repr, make_basis_like, proj, proj_tuple_mut, unit,
    },
    error::{Error, Result},
    ht,
    htuple::{
        HTuple, congruent, slice_, transform_apply_leaf, transform_apply_leaf4, zip3_transform_leaf,
    },
    shape::idx2crd,
    stride::{Coshape, coalesce_z, coprofile, inner_product, prefix_product},
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

// ---------------------------------------------------------------------------
// complement
// ---------------------------------------------------------------------------

impl Layout {
    /// The complement of this layout, optionally extended to cover
    /// `extend`. PyCuTe's `Layout._complement`.
    ///
    /// The complement walks this layout's modes in stride order and
    /// records the gaps they leave behind, one running position per
    /// codomain position. The result is the layout of those gaps, so
    /// that `make_layout([self, self.complement(None)])` covers the
    /// whole codomain without repeating a value.
    ///
    /// `extend` is PyCuTe's optional argument, so it arrives as an
    /// [`Option`]. When it is present, the trailing mode of each
    /// complement mode is grown until it spans `extend`.
    ///
    /// Post-conditions:
    ///   `weakly_congruent(coprofile(self), shape(result))`,
    ///   `result` is ordered — `result(i) < result(i+1)`,
    ///   the codomains of `self` and `result` are disjoint.
    ///
    /// ```text
    /// complement(Layout(4, 1))         == Layout(1, 4)
    /// complement(Layout((2, 4), (1, 6))) == Layout((3, 1), (2, 24))
    /// ```
    ///
    /// Returns [`Error::NonInjective`] when a mode falls behind the
    /// running position, and [`Error::NotBasis`] when a stride leaf is a
    /// sum of basis elements, which has no single codomain position.
    pub fn complement(&self, extend: Option<&IntTuple>) -> Result<Self> {
        let coprof = coprofile(self, &[])?;
        // PyCuTe accumulates into profile-shaped tuples whose leaves are
        // mutable lists. The lists are `Vec<Int>` here, for both halves:
        // every value pushed below is an integer — `d` is the
        // coefficient `proj` reads off a stride leaf, and `s` an extent
        // — so the strides become [`StrideScalar`]s only where the
        // per-position layouts are built.
        let mut result_s = HTuple::<Vec<Int>>::repeat_like(&Vec::new(), &coprof);
        let mut result_d = HTuple::<Vec<Int>>::repeat_like(&Vec::from([1]), &coprof);

        // Modes are ordered by stride. PyCuTe orders only the static
        // strides, since a symbolic one has no known place; [`Int`] is
        // always static, so the sort is plain — but still *stable*,
        // which keeps equal strides in left-to-right order.
        let mut modes = self
            .stride
            .leaves()
            .into_iter()
            .zip(self.shape.leaves())
            .collect::<Vec<_>>();
        modes.sort_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap_or(Ordering::Equal));

        // A mutating fold, as PyCuTe writes it: each mode reads the
        // running position its own codomain position left behind, and
        // leaves a new one.
        for (de, &s) in modes {
            let d = proj(de, de)?;
            if d.is_zero() || s == 1 {
                continue;
            }
            let d = coefficient(d)?;
            let result_s = accumulator(&mut result_s, de)?;
            let result_d = accumulator(&mut result_d, de)?;
            let position = *result_d.last().ok_or_else(|| Error::BadPath {
                path: vec![],
                value: format!("{self}"),
            })?;
            if d < position {
                return Err(Error::NonInjective {
                    detail: format!("complement({self})"),
                });
            }
            result_s.push(d.checked_div(position).ok_or_else(|| Error::Divisibility {
                detail: format!("complement({self}): a mode of extent 0"),
            })?);
            result_d.push(d * s);
        }

        // One layout per codomain position, each closed with the extent
        // `1` that `extend` below grows. PyCuTe's `transform_leaf` pads
        // with `zip_longest`; the three tuples are congruent by
        // construction, so the strict zip does the same work.
        let tiler = zip3_transform_leaf(
            &|_: &Int, result_s: &Vec<Int>, result_d: &Vec<Int>| {
                Layout::set(
                    HTuple::Tuple(
                        result_s
                            .iter()
                            .copied()
                            .chain([1])
                            .map(HTuple::Leaf)
                            .collect(),
                    ),
                    HTuple::Tuple(
                        result_d
                            .iter()
                            .map(|&d| HTuple::Leaf(StrideScalar::Int(d)))
                            .collect(),
                    ),
                )
                // PyCuTe's default coalesce profile, `1`.
                .coalesce_z(&HTuple::Leaf(Some(1)))
                .map(TilerLeaf::Layout)
            },
            &coprof,
            &result_s,
            &result_d,
        )?;
        let result = tiler_to_layout(&tiler, &StrideScalar::Int(1))?;

        let Some(extend) = extend else {
            return Ok(result);
        };
        // PyCuTe drives this with `zip_longest` padding and *reads* the
        // padding: an absent complement mode means `extend` outranks the
        // codomain there, and the extend mode stands on its own.
        let extend_complement = |_: Option<&IntTuple>,
                                 shape_c: Option<&IntTuple>,
                                 stride_c: Option<&Stride>,
                                 shape_a: Option<&IntTuple>,
                                 stride_a: Option<&Stride>| {
            let rank_mismatch = || Error::RankMismatch {
                op: "complement",
                value: format!("{self}"),
                profile: format!("{extend:?}"),
            };
            let (Some(shape_c), Some(stride_c)) = (shape_c, stride_c) else {
                return match (shape_a, stride_a) {
                    (Some(shape_a), Some(stride_a)) => {
                        Ok(Layout::set(shape_a.clone(), stride_a.clone()))
                    }
                    _ => Err(rank_mismatch()),
                };
            };
            let shape_a = shape_a.ok_or_else(rank_mismatch)?;
            // The last extent of the complement is always 1, so its
            // stride is the size the extension starts from.
            let size_c = match stride_c.back() {
                HTuple::Leaf(d) => coefficient(proj(d, d)?)?,
                other => {
                    return Err(Error::NotBasis {
                        value: format!("{other:?}"),
                    });
                }
            };
            let (shape_r, _) = shape_a.leaves().into_iter().try_fold(
                (Vec::new(), size_c),
                |(mut shape_r, size_c), &s| match s > 0 && size_c > 0 {
                    // PyCuTe writes `(s + sizeC - 1) // sizeC` and
                    // `(s + sizeC - 1) // s`; over positive operands
                    // those are the two ceiling divisions below.
                    true => {
                        shape_r.push(HTuple::Leaf(ceil_div(s, size_c)));
                        Ok((shape_r, ceil_div(size_c, s)))
                    }
                    false => Err(Error::Divisibility {
                        detail: format!("complement({self}) extended by {extend:?}"),
                    }),
                },
            )?;
            Layout::new(
                shape_c.clone().replace_back(HTuple::Tuple(shape_r)),
                stride_c,
            )?
            // PyCuTe's default coalesce profile, `1`.
            .coalesce(&HTuple::Leaf(Some(1)))
        };
        transform_apply_leaf4(
            &make_layout,
            &extend_complement,
            Some(&coprof),
            Some(&result.shape),
            Some(&result.stride),
            Some(extend),
            Some(&make_basis_like(extend)),
        )
    }
}

/// The integer coefficient of a stride leaf.
///
/// Every non-zero leaf [`proj`] reaches is one, so the error stands for
/// a stride that is a sum of basis elements — an element `complement`
/// cannot place in the codomain.
fn coefficient(d: &StrideScalar) -> Result<Int> {
    match d {
        StrideScalar::Int(v) => Ok(*v),
        other => Err(Error::NotBasis {
            value: format!("{other:?}"),
        }),
    }
}

/// The accumulator list at the codomain position `de` names. PyCuTe
/// writes `proj(result_S, de)`, and appends to what comes back.
fn accumulator<'a>(x: &'a mut HTuple<Vec<Int>>, de: &StrideScalar) -> Result<&'a mut Vec<Int>> {
    match proj_tuple_mut(x, de)? {
        HTuple::Leaf(list) => Ok(list),
        other => Err(Error::BadPath {
            path: vec![],
            value: format!("{other:?}"),
        }),
    }
}

// ---------------------------------------------------------------------------
// the inverses and the nullspace
// ---------------------------------------------------------------------------

impl Layout {
    /// Largest right inverse of this layout. PyCuTe's `_right_inverse`.
    ///
    /// Walks the modes in stride order and keeps the ones that continue
    /// the chain `d_k == s_{k-1} * d_{k-1}`, recording each survivor's
    /// extent against its position in the domain. A mode that carries no
    /// information — stride 0, or extent 1 — is skipped, and so is any
    /// mode that breaks the chain, which is what makes the result the
    /// *largest* right inverse rather than a failure.
    ///
    /// Post-conditions:
    ///   `weakly_congruent(coprofile(self), shape(result))`,
    ///   `result(self(result(i))) == result(i)` for every `i` in the
    ///   domain of `result`,
    ///   `self(result(i)) == i` as well, when the codomain is `Z`.
    ///
    /// ```text
    /// right_inverse(Layout((8, 4), (4, 1))) == Layout((4, 8), (8, 1))
    /// right_inverse(Layout(4, 2))           == Layout(1, 0)
    /// ```
    pub fn right_inverse(&self) -> Result<Self> {
        let coprof = coprofile(self, &[])?;
        // One accumulator per position of the codomain. PyCuTe warns
        // "Avoid aliasing [] from repeat_like"; `repeat_like` clones, so
        // each leaf holds a vector of its own.
        //
        // PyCuTe's `result_S` / `result_D` / `curr_D` are these trees;
        // its `result_s` / `result_d` / `curr_d` are the leaves projected
        // out of them inside the loop.
        let mut result_shape = HTuple::repeat_like(&Vec::<Int>::new(), &coprof);
        let mut result_stride = HTuple::repeat_like(&Vec::<StrideScalar>::new(), &coprof);
        let mut curr_stride = HTuple::repeat_like(&StrideScalar::Int(1), &coprof);

        let (flat_s, flat_d) = coalesce_z(&self.shape, &self.stride)?;
        let pps = prefix_product(&flat_s, &HTuple::Leaf(StrideScalar::Int(1)))?;

        // The chain is followed in stride order. PyCuTe orders only the
        // static strides; [`Int`] is always static, so `_stride_key`
        // collapses to a plain sort. It stays *stable*, which is what
        // keeps equal strides in left-to-right order — and what carries
        // the strides that do not compare at all.
        let mut modes = collect_modes(&flat_d, &flat_s, &pps);
        modes.sort_by(|a, b| a.0.partial_cmp(b.0).unwrap_or(Ordering::Equal));

        // A mutating fold, as PyCuTe writes it: each mode reads and
        // rewrites the accumulator at its own codomain position.
        for (de, s, pps) in modes {
            let d = proj(de, de)?;
            // Stride-0 / size-1 modes carry no information.
            //
            // PyCuTe projects first and guards second. A stride of
            // integer 0 is the rank-0 basis, whose path is empty, so its
            // `proj` answers the whole accumulator tree and the mode is
            // then skipped. `proj_leaf_mut` needs a leaf, so the guard
            // comes first here. The projections have no side effect, so
            // the order does not otherwise matter.
            if d.is_zero() || s == 1 {
                continue;
            }
            let result_s = proj_leaf_mut(&mut result_shape, de)?;
            let result_d = proj_leaf_mut(&mut result_stride, de)?;
            let curr_d = proj_leaf_mut(&mut curr_stride, de)?;

            // A mode that does not continue the chain is dropped.
            if d != curr_d {
                continue;
            }
            result_s.push(s);
            result_d.push(pps.clone());
            *curr_d = d.scale(s);
        }

        Layout::set(modes_of(result_shape), modes_of(result_stride)).coalesce(&as_profile(&coprof))
    }

    /// Left inverse of this layout. PyCuTe's `_left_inverse`.
    ///
    /// Walks the modes in stride order and pads each recorded extent out
    /// to the next stride, so the holes between the modes become
    /// stride-0 filler. Unlike [`Self::right_inverse`], a mode that
    /// breaks the chain is an error rather than a stopping point: a left
    /// inverse must account for the whole domain.
    ///
    /// Post-conditions:
    ///   `weakly_congruent(coprofile(self), shape(result))`,
    ///   `self(result(self(i))) == self(i)` for every `i` in the domain.
    ///
    /// ```text
    /// left_inverse(Layout((8, 4), (4, 1)))  == Layout((4, 8), (8, 1))
    /// left_inverse(Layout((4, 2), (1, 16))) == Layout((16, 2), (1, 4))
    /// ```
    ///
    /// Returns [`Error::Divisibility`] when the strides do not form an
    /// ordered chain, and [`Error::NonInjective`] when a mode overlaps
    /// the one before it.
    pub fn left_inverse(&self) -> Result<Self> {
        let coprof = coprofile(self, &[])?;
        // As in [`Self::right_inverse`], one independent accumulator per
        // codomain position. These start non-empty: the seed `1:0` mode
        // is what the first stride pads out to.
        let mut result_shape = HTuple::repeat_like(&vec![1], &coprof);
        let mut result_stride = HTuple::repeat_like(&vec![StrideScalar::Int(0)], &coprof);
        let mut curr_shape = HTuple::repeat_like(&1, &coprof);

        let (flat_s, flat_d) = coalesce_z(&self.shape, &self.stride)?;
        let pps = prefix_product(&flat_s, &HTuple::Leaf(StrideScalar::Int(1)))?;

        // PyCuTe sorts the raw triples with no key, so the comparison
        // runs `(stride, extent, position)` lexicographically. The
        // leading [`StrideScalar`] orders only partially — an integer
        // does not compare with an arithmetic tuple — so an incomparable
        // pair falls back on the stable order, as it does above.
        let mut modes = collect_modes(&flat_d, &flat_s, &pps);
        modes.sort_by(|a, b| {
            a.0.partial_cmp(b.0)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.1.cmp(&b.1))
                .then_with(|| a.2.partial_cmp(b.2).unwrap_or(Ordering::Equal))
        });

        for (de, s, pps) in modes {
            let d = proj(de, de)?;
            // Stride-0 / size-1 modes carry no information. The guard
            // precedes the projections for the reason given in
            // [`Self::right_inverse`].
            if d.is_zero() || s == 1 {
                continue;
            }
            let result_s = proj_leaf_mut(&mut result_shape, de)?;
            let result_d = proj_leaf_mut(&mut result_stride, de)?;
            let curr_s = proj_leaf_mut(&mut curr_shape, de)?;

            // The chain is walked with integer arithmetic, so a stride
            // that projects to anything else has no place on it.
            let d = match d {
                StrideScalar::Int(v) => *v,
                other => {
                    return Err(Error::NotBasis {
                        value: format!("{other:?}"),
                    });
                }
            };
            // gap = d_k / d_{k-1}, the span to the next stride.
            let (gap, rem) = (d.div_euclid(*curr_s), d.rem_euclid(*curr_s));
            if rem != 0 {
                return Err(Error::Divisibility {
                    detail: format!("left_inverse({self}): strides do not form an ordered chain"),
                });
            }
            // d_k must clear the previous mode: d_k >= d_{k-1} * s_{k-1}.
            if result_s.last().is_some_and(|&last| gap < last) {
                return Err(Error::NonInjective {
                    detail: format!("left_inverse({self})"),
                });
            }
            // Pad the previous mode out to d_k; the extra entries are
            // holes.
            if let Some(last) = result_s.last_mut() {
                *last = gap;
            }
            // Advance the consumed stride to d_k.
            *curr_s *= gap;
            // Record this mode. A later mode overwrites `s` with its own
            // gap.
            result_s.push(s);
            result_d.push(pps.clone());
        }

        Layout::set(modes_of(result_shape), modes_of(result_stride))
            .coalesce_z(&as_profile(&coprof))
    }

    /// Nullspace of this layout. PyCuTe's `_nullspace`.
    ///
    /// The stride-0 modes, gathered into a layout over the domain: every
    /// coordinate it produces maps to zero. A layout with no stride-0
    /// mode has the trivial nullspace `1:0`.
    ///
    /// Post-conditions:
    ///   `self(result(i)) == 0` for every `i` in the domain of `result`.
    ///
    /// ```text
    /// nullspace(Layout((2, 4, 6), (1, 2, 0))) == Layout(6, 8)
    /// nullspace(Layout((8, 4), (4, 1)))       == Layout(1, 0)
    /// ```
    pub fn nullspace(&self) -> Result<Self> {
        let fstride = self.stride.flatten();
        let iseq = fstride
            .leaves()
            .into_iter()
            .enumerate()
            .filter(|(_, d)| d.is_zero())
            .map(|(i, _)| i)
            .collect::<Vec<_>>();
        if iseq.is_empty() {
            return Ok(Layout::set(ht!(1), HTuple::Leaf(StrideScalar::Int(0))));
        }
        let fshape = self.shape.flatten();
        let pshape = prefix_product(&fshape, &HTuple::Leaf(StrideScalar::Int(1)))?;
        Ok(Layout::set(
            fshape.select(&iseq)?.unwrap().clone(),
            pshape.select(&iseq)?.unwrap().clone(),
        ))
    }
}

/// The `(stride, extent, position)` triples the two inverses sort.
///
/// PyCuTe writes `zip(flat_d, flat_s, prefix_product(flat_s))`; the three
/// arrive already flat, so this is that zip.
fn collect_modes<'a>(
    flat_d: &'a Stride,
    flat_s: &IntTuple,
    pps: &'a Stride,
) -> Vec<(&'a StrideScalar, Int, &'a StrideScalar)> {
    flat_d
        .leaves()
        .into_iter()
        .zip(flat_s.leaves())
        .zip(pps.leaves())
        .map(|((de, s), pps)| (de, *s, pps))
        .collect()
}

/// The accumulator at the codomain position `de` names, for writing
/// through. PyCuTe's `proj(result_S, de)`.
///
/// Returns [`Error::BadPath`] when the position addresses an interior
/// node rather than one accumulator — PyCuTe reaches for `.append` on a
/// tuple there.
fn proj_leaf_mut<'a, T: Debug>(x: &'a mut HTuple<T>, de: &StrideScalar) -> Result<&'a mut T> {
    match proj_tuple_mut(x, de)? {
        HTuple::Leaf(v) => Ok(v),
        other => Err(Error::BadPath {
            path: vec![],
            value: format!("{other:?}"),
        }),
    }
}

/// Turns each accumulated vector into a mode of its own.
///
/// PyCuTe's `tuple(result_S)`: its leaves are Python lists, and
/// `is_tuple` counts a list as a tuple, so the accumulated entries are
/// already modes by the time the layout is built.
fn modes_of<T>(acc: HTuple<Vec<T>>) -> HTuple<T> {
    match acc {
        HTuple::Leaf(v) => HTuple::Tuple(v.into_iter().map(HTuple::Leaf).collect()),
        HTuple::Tuple(modes) => HTuple::Tuple(modes.into_iter().map(modes_of).collect()),
    }
}

/// A coprofile as a coalesce [`Profile`]. Only the presence of each leaf
/// is read, so the value carries over untouched.
fn as_profile(coprof: &IntTuple) -> Profile {
    coprof.transform_leaf(&|v| Some(*v))
}

// ---------------------------------------------------------------------------
// composition
// ---------------------------------------------------------------------------

/// The right-hand side of a composition: a [`Tiler`] whose modes may be
/// absent.
///
/// PyCuTe annotates `_composition`'s argument `Tiler`, then opens the
/// function by reading a `None` its own alias does not admit — the
/// per-mode no-op, as in `composition(A, (X, None))`. The [`Option`]
/// here is what carries that case, and it sits at the leaf rather than
/// around the whole tuple, so a mode can be absent on its own.
///
/// That is where [`Profile`] puts it too, and for the same reason: the
/// whole no-op and the per-mode no-op are then one spelling at two
/// depths, `HTuple::Leaf(None)`.
pub type OptTiler = HTuple<Option<TilerLeaf>>;

impl Layout {
    /// Composes this layout with `b`: the group composition `A o B`
    /// (Whitepaper, §3.3). PyCuTe's `Layout._composition`.
    ///
    /// The result has `b`'s domain and this layout's values — walk `b`,
    /// then map what it produces through `self`. A tuple `b` composes
    /// by-mode, and an absent `b` is the no-op.
    ///
    /// Pre-conditions:
    ///   `self` and `b` satisfy the shape- and stride-divisibility
    ///   conditions (Whitepaper, Eqs. (20)-(21)).
    ///
    /// Post-conditions:
    ///   `compatible(shape(b), shape(result))`,
    ///   `result(i) == self(b(i))` for every `i` in the domain of `b`.
    ///
    /// ```text
    /// composition(Layout((6, 2), (8, 2)), Layout((4, 3), (3, 1))) == Layout(((2, 2), 3), ((24, 2), 8))
    /// composition(Layout(20, 2),          Layout((5, 4), (4, 1))) == Layout((5, 4), (8, 2))
    /// composition(Layout(12),             Layout((4, 3)))         == Layout((4, 3), (1, 4))
    /// ```
    ///
    /// Returns [`Error::RankMismatch`] when a tiler tuple outranks this
    /// layout, and [`Error::Divisibility`] when either divisibility
    /// condition fails.
    pub fn composition(&self, b: &OptTiler) -> Result<Self> {
        let b = match b {
            // RHS None, noop.
            HTuple::Leaf(None) => return Ok(self.clone()),
            // RHS tuple, (A0,A1,...) o <X,Y,...> => (A0 o X, A1 o Y, ...).
            HTuple::Tuple(modes) => {
                if self.shape.rank() < modes.len() {
                    return Err(Error::RankMismatch {
                        op: "composition",
                        value: format!("{self}"),
                        profile: format!("{b:?}"),
                    });
                }
                // PyCuTe zips the layout against the tiler with
                // `zip_longest`, so the modes the tiler runs out on take
                // the no-op.
                return (0..self.shape.rank())
                    .map(|i| {
                        self.index(i)?
                            .composition(modes.get(i).unwrap_or(&HTuple::Leaf(None)))
                    })
                    .collect::<Result<Vec<_>>>()
                    .map(make_layout);
            }
            // RHS int, A o N -> A o N:1.
            HTuple::Leaf(Some(TilerLeaf::Int(n))) => {
                Layout::set(HTuple::Leaf(*n), HTuple::Leaf(StrideScalar::Int(1)))
            }
            HTuple::Leaf(Some(TilerLeaf::Layout(l))) => l.clone(),
        };

        //
        // Special cases with A: Layout and B: Layout
        //

        let a = self.coalesce_z(&as_profile(&coprofile(&b, &[])?))?;

        // RHS distributive, A o (X,Y,...) => (A o X, A o Y, ...).
        if b.shape.is_tuple() {
            return (0..b.shape.rank())
                .map(|i| a.composition(&HTuple::Leaf(Some(TilerLeaf::Layout(b.index(i)?)))))
                .collect::<Result<Vec<_>>>()
                .map(make_layout);
        }
        // Special case stride-0, A o N:0 => N:0.
        if b.stride == HTuple::Leaf(StrideScalar::Int(0)) {
            return Ok(Layout::set(
                b.shape.clone(),
                HTuple::Leaf(StrideScalar::Int(0)),
            ));
        }
        // `b.shape` is no longer a tuple, so both halves are leaves. A
        // stride that is not is an incongruent layout, which PyCuTe
        // walks into rather than checks for.
        let (HTuple::Leaf(b_shape), HTuple::Leaf(b_stride)) = (&b.shape, &b.stride) else {
            return Err(Error::BadPath {
                path: vec![],
                value: format!("{b}"),
            });
        };
        // Special case shape-1, A o 1:M => 1:A(M).
        if *b_shape == 1 {
            return Ok(Layout::set(
                b.shape.clone(),
                HTuple::Leaf(a.call(&as_tuple(b_stride))?),
            ));
        }

        //
        // General case   (A0,A1,...) o N:M
        //

        let shape_divisibility = || Error::Divisibility {
            detail: format!("shape condition: composition({self}, {b})"),
        };
        let stride_divisibility = || Error::Divisibility {
            detail: format!("stride condition: composition({self}, {b})"),
        };

        // A mutating index-walk over `result_s` / `result_d`, as PyCuTe
        // writes it: each basis term of `B`'s stride truncates one
        // sublayout of `A` and shifts it into place.
        let mut acc: Option<Layout> = None;
        for (mut stride_b, basis_b) in basis_repr(b_stride) {
            let ab = a.get(&basis_b)?;
            let (mut result_s, mut result_d) = flat_modes(&ab)?;

            // Truncate/extend result_s based on strideB * B.shape.
            let last = result_s.len() - 1;
            result_s[last] = stride_b * *b_shape;
            for i in 0..last {
                let (quotient, r_es) =
                    divmod(result_s[last], result_s[i]).ok_or_else(shape_divisibility)?;
                result_s[last] = quotient;
                if result_s[last] == 0 {
                    result_s[i] = r_es;
                    result_s.truncate(i + 1);
                    result_d.truncate(i + 1);
                    break;
                }
                if r_es != 0 {
                    return Err(shape_divisibility());
                }
            }

            // Remove result_s prefix strideB.
            //
            // PyCuTe closes this loop with a `for ... else`: the `else`
            // body runs only when the loop completes *without* `break`.
            // Rust has no such form, so `broke` stands in for it.
            let mut broke = false;
            for i in 0..result_s.len() - 1 {
                let (q_sd, r_sd) = divmod(result_s[i], stride_b).ok_or_else(stride_divisibility)?;
                if r_sd == 0 {
                    result_s[i] = q_sd;
                    result_d[i] = result_d[i].scale(stride_b);
                    broke = true;
                    break;
                }
                let (q_ds, r_ds) = divmod(stride_b, result_s[i]).ok_or_else(stride_divisibility)?;
                stride_b = q_ds;
                result_s[i] = 1;
                if r_ds != 0 {
                    return Err(stride_divisibility());
                }
            }
            // The `else` clause.
            if !broke {
                let last = result_s.len() - 1;
                result_s[last] = divmod(result_s[last], stride_b)
                    .ok_or_else(stride_divisibility)?
                    .0;
                result_d[last] = result_d[last].scale(stride_b);
            }

            // Accumulate into resultL. PyCuTe seeds `resultL` with
            // `None`, which is why only the left operand of `layout_add`
            // is optional.
            acc = Some(layout_add(
                acc.as_ref(),
                &Layout::set(
                    HTuple::Tuple(result_s.into_iter().map(HTuple::Leaf).collect()),
                    HTuple::Tuple(result_d.into_iter().map(HTuple::Leaf).collect()),
                ),
            )?);
        }

        match acc {
            // PyCuTe's default coalesce profile, `1`.
            Some(result) => result.coalesce(&HTuple::Leaf(Some(1))),
            // [`basis_repr`] yields at least one term — an all-zero
            // element decomposes to `(0, [])` — so the fold above always
            // runs and this arm stands for the type alone.
            None => Err(Error::NotBasis {
                value: format!("{b_stride:?}"),
            }),
        }
    }
}

/// The top-level modes of a layout, as PyCuTe's `list(wrap(Ab.shape))`
/// and `list(wrap(Ab.stride))`.
///
/// `Ab` is a sublayout of the coalesced `A`, so it is flat and every mode
/// is a leaf. A nested mode is where PyCuTe's `divmod` meets a tuple.
///
/// The result is non-empty and the two halves have equal length, which is
/// what lets the walk above index `result_s.len() - 1`.
fn flat_modes(ab: &Layout) -> Result<(Vec<Int>, Vec<StrideScalar>)> {
    let bad_path = || Error::BadPath {
        path: vec![],
        value: format!("{ab}"),
    };
    let result_s = top_level_leaves(&ab.shape).ok_or_else(bad_path)?;
    let result_d = top_level_leaves(&ab.stride).ok_or_else(bad_path)?;
    match !result_s.is_empty() && result_s.len() == result_d.len() {
        true => Ok((result_s, result_d)),
        false => Err(bad_path()),
    }
}

/// The leaves one level down. PyCuTe's `list(wrap(x))`, which is a list
/// of leaves exactly when `x` is flat.
///
/// Returns `None` for a mode that is itself a tuple.
fn top_level_leaves<T: Clone>(x: &HTuple<T>) -> Option<Vec<T>> {
    match x {
        HTuple::Leaf(v) => Some(vec![v.clone()]),
        HTuple::Tuple(modes) => modes
            .iter()
            .map(|m| match m {
                HTuple::Leaf(v) => Some(v.clone()),
                HTuple::Tuple(_) => None,
            })
            .collect(),
    }
}

/// Python's `divmod`, over the positive extents and strides this walk
/// carries. A zero divisor raises there; here the caller turns the
/// `None` into the divisibility error of its own loop.
fn divmod(a: Int, b: Int) -> Option<(Int, Int)> {
    (b != 0).then(|| (a.div_euclid(b), a.rem_euclid(b)))
}

// ---------------------------------------------------------------------------
// logical_divide and logical_product
// ---------------------------------------------------------------------------

impl Layout {
    /// Splits this layout into the elements of `b` — the Tile — and a
    /// grid over those tiles. PyCuTe's `Layout._logical_divide`.
    ///
    /// The Tile is `b` itself and the grid is [`Self::complement`] of `b`
    /// extended over this layout's shape, so composing with the two of
    /// them together re-reads the whole domain, tile first.
    ///
    /// The dispatch head is [`Self::composition`]'s: a tuple `b` divides
    /// by-mode, an integer promotes to `N:1`, and an absent `b` is the
    /// no-op. A tuple `b` therefore interleaves Tile and grid per mode —
    /// the `(Tile, Grid)` regrouping is `zipped_divide`, which promotes
    /// the tiler through [`tiler_to_layout`] first.
    ///
    /// Post-conditions, for a `b` that is a single layout:
    ///   `rank(result) == 2`,
    ///   `compatible(shape(b), shape(result[0]))`,
    ///   `result(i, 0) == self(b(i))` for every `i` in the domain of `b`,
    ///   every element of `self` appears in `result`.
    ///
    /// ```text
    /// logical_divide(Layout(24), Layout(4, 2))          == Layout((4, (2, 3)), (2, (1, 8)))
    /// logical_divide(Layout((6, 4), (4, 1)), Layout(2, 1)) == Layout((2, 3), (4, 8))
    /// ```
    ///
    /// Returns [`Error::RankMismatch`] when a tiler tuple outranks this
    /// layout, and whatever [`Self::complement`] or [`Self::composition`]
    /// raise otherwise.
    pub fn logical_divide(&self, b: &OptTiler) -> Result<Self> {
        let b = match b {
            // RHS None, noop.
            HTuple::Leaf(None) => return Ok(self.clone()),
            // RHS tuple, (A0,A1,...) / <X,Y,...> => (A0 / X, A1 / Y, ...).
            HTuple::Tuple(modes) => {
                if self.shape.rank() < modes.len() {
                    return Err(Error::RankMismatch {
                        op: "logical_divide",
                        value: format!("{self}"),
                        profile: format!("{b:?}"),
                    });
                }
                // PyCuTe zips the layout against the tiler with
                // `zip_longest`, so the modes the tiler runs out on take
                // the no-op.
                return (0..self.shape.rank())
                    .map(|i| {
                        self.index(i)?
                            .logical_divide(modes.get(i).unwrap_or(&HTuple::Leaf(None)))
                    })
                    .collect::<Result<Vec<_>>>()
                    .map(make_layout);
            }
            // RHS int, A / N -> A / N:1.
            HTuple::Leaf(Some(TilerLeaf::Int(n))) => {
                Layout::set(HTuple::Leaf(*n), HTuple::Leaf(StrideScalar::Int(1)))
            }
            HTuple::Leaf(Some(TilerLeaf::Layout(l))) => l.clone(),
        };

        // PyCuTe imports `complement` from `algebra` inside the function,
        // to dodge a circular import. Here it is the method above.
        let grid = b.complement(Some(&self.shape))?;
        self.composition(&HTuple::Leaf(Some(TilerLeaf::Layout(make_layout(vec![
            b, grid,
        ])))))
    }

    /// Reproduces this layout over `b`. PyCuTe's
    /// `Layout._logical_product`.
    ///
    /// Mode 0 of the result is this layout untouched, and mode 1 walks
    /// `b` through the codomain this layout leaves free — its
    /// [`Self::complement`] — so each coordinate of `b` places one
    /// disjoint copy.
    ///
    /// The dispatch head is [`Self::composition`]'s: a tuple `b`
    /// multiplies by-mode, an integer promotes to `N:1`, and an absent
    /// `b` is the no-op.
    ///
    /// Post-conditions:
    ///   `rank(result) == 2`,
    ///   `result[0] == self`,
    ///   `compatible(shape(b), shape(result[1]))`.
    ///
    /// ```text
    /// logical_product(Layout(3, 1), Layout(4, 1))           == Layout((3, 4), (1, 3))
    /// logical_product(Layout((2, 2), (4, 1)), Layout(6, 1)) == Layout(((2, 2), (2, 3)), ((4, 1), (2, 8)))
    /// ```
    ///
    /// Returns [`Error::RankMismatch`] when a tiler tuple outranks this
    /// layout, and whatever [`Self::complement`] or [`Self::composition`]
    /// raise otherwise.
    pub fn logical_product(&self, b: &OptTiler) -> Result<Self> {
        let b = match b {
            // RHS None, noop.
            HTuple::Leaf(None) => return Ok(self.clone()),
            // RHS tuple, (A0,A1,...) x <X,Y,...> => (A0 x X, A1 x Y, ...).
            HTuple::Tuple(modes) => {
                if self.shape.rank() < modes.len() {
                    return Err(Error::RankMismatch {
                        op: "logical_product",
                        value: format!("{self}"),
                        profile: format!("{b:?}"),
                    });
                }
                // PyCuTe zips the layout against the tiler with
                // `zip_longest`, so the modes the tiler runs out on take
                // the no-op.
                return (0..self.shape.rank())
                    .map(|i| {
                        self.index(i)?
                            .logical_product(modes.get(i).unwrap_or(&HTuple::Leaf(None)))
                    })
                    .collect::<Result<Vec<_>>>()
                    .map(make_layout);
            }
            // RHS int, A x N -> A x N:1.
            HTuple::Leaf(Some(TilerLeaf::Int(n))) => {
                Layout::set(HTuple::Leaf(*n), HTuple::Leaf(StrideScalar::Int(1)))
            }
            HTuple::Leaf(Some(TilerLeaf::Layout(l))) => l.clone(),
        };

        Ok(make_layout(vec![
            self.clone(),
            self.complement(None)?
                .composition(&HTuple::Leaf(Some(TilerLeaf::Layout(b))))?,
        ]))
    }
}
