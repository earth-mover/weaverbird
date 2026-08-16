//! Layouts: a map from a coordinate domain to a codomain.
//!
//! A [`Layout`] pairs a shape with a congruent stride. Everything else in
//! the crate exists to build one, read one, or take one apart. The layout
//! algebra lives here as methods; [`algebra`](crate::algebra) holds the
//! two operations that are not methods of a single layout.

use std::{
    cmp::Ordering,
    fmt::{self, Debug, Display},
};

use crate::{
    algebra::layout_add,
    atuple::{
        ArithTuple, StrideScalar, basis_repr, e, make_basis_like, proj, proj_tuple_mut, unit,
    },
    error::{Condition, Error, Result},
    ht,
    htuple::HTuple,
    stride::coalesce_modes,
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
/// ```
/// # use weaverbird::{layout, ht, Layout, StrideScalar};
/// assert_eq!(layout!((4, 8)), layout!((4, 8):(1, 4)));
/// assert_eq!(layout!((4, 8):(8, 1)).eval(&ht!((2, 3))).unwrap(), StrideScalar::Int(19));
/// ```
#[derive(Clone, PartialEq, Eq)]
pub struct Layout {
    /// The domain.
    pub shape: IntTuple,
    /// The map. Congruent with `shape`.
    pub stride: Stride,
}

impl Layout {
    /// A layout from a shape and a congruent stride.
    ///
    /// Returns [`Error::NotCongruent`] when the two profiles differ.
    pub fn new(shape: IntTuple, stride: Stride) -> Result<Self> {
        match shape.congruent(&stride) {
            true => Ok(Layout { shape, stride }),
            false => Err(Error::NotCongruent { shape: shape.clone(), stride: stride.clone() }),
        }
    }

    /// A layout from a shape and a stride, unchecked.
    ///
    /// The algebra builds both halves together, so the congruence of
    /// [`Self::new`] would re-walk a tree it just made.
    pub fn from_parts(shape: IntTuple, stride: Stride) -> Self {
        Layout { shape, stride }
    }

    /// The compact, column-major layout of `shape`.
    pub fn compact(shape: IntTuple) -> Self {
        let stride = shape.compact_stride();
        Layout { shape, stride }
    }

    /// The compact layout of `shape`, from a stride base.
    ///
    /// `base` seeds the prefix product. It is one scalar, or one per
    /// mode.
    ///
    /// Returns [`Error::Incompatible`] when `base` does not weakly
    /// coarsen `shape`.
    pub fn from_base(shape: IntTuple, base: &Stride) -> Result<Self> {
        shape.prefix_product(base).map(|stride| Layout { shape, stride })
    }

    /// One mode of the result per input layout.
    ///
    /// ```
    /// # use weaverbird::{layout, Layout};
    /// let joined = Layout::from_modes(vec![layout!(3:1), layout!(2:42)]);
    /// assert_eq!(joined, layout!((3, 2):(1, 42)));
    /// ```
    pub fn from_modes(layouts: Vec<Layout>) -> Self {
        layouts.into_iter().collect()
    }

    /// A compact layout of `shape`, ordered by `order`.
    ///
    /// The mode with the smallest `order` takes stride 1, and the rest
    /// take compact strides in ascending order. Only the relative
    /// ordering matters, so `order` need not be a `0..rank` permutation.
    /// Ties keep their left-to-right position.
    ///
    /// ```
    /// # use weaverbird::{ht, layout, Layout};
    /// let l = Layout::ordered(&ht!((4, 8)), &ht!((1, 0))).unwrap();
    /// assert_eq!(l, layout!((4, 8):(8, 1)));
    /// ```
    ///
    /// Returns [`Error::NotCongruent`] when `order` does not match
    /// `shape`.
    pub fn ordered(shape: &IntTuple, order: &IntTuple) -> Result<Self> {
        if !shape.congruent(order) {
            return Err(Error::BadOrder { shape: shape.clone(), order: order.clone() });
        }
        let extents = shape.leaves().copied().collect::<Vec<_>>();
        let keys = order.leaves().copied().collect::<Vec<_>>();

        let mut modes = (0..extents.len()).collect::<Vec<_>>();
        modes.sort_by_key(|&i| keys.get(i).copied());
        Ok(Layout::from_parts(shape.clone(), compact_in_order(&extents, modes.into_iter(), shape)))
    }

    /// CuTe `rank(layout)`: the top-level mode count, read off the shape.
    /// A layout over a leaf shape has rank 1.
    ///
    /// ```
    /// # use weaverbird::layout;
    /// assert_eq!(layout!((4, 8):(8, 1)).rank(), 2);
    /// assert_eq!(layout!(((3, 2), 4):((4, 12), 1)).rank(), 2);
    /// assert_eq!(layout!(4:1).rank(), 1);
    /// ```
    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    /// A compact layout with this shape, ordered by this layout's
    /// strides.
    ///
    /// The mode with the smallest non-zero stride takes stride 1, and the
    /// remaining non-zero modes take compact strides in ascending order.
    /// A mode that carries no positional information — a stride of 0 —
    /// stays at 0.
    ///
    /// ```
    /// # use weaverbird::layout;
    /// assert_eq!(layout!((4, 8):(100, 1)).compacted(), layout!((4, 8):(8, 1)));
    /// let sparse = layout!((2, 3, 4, 5):(0, 42, 4, 0));
    /// assert_eq!(sparse.compacted(), layout!((2, 3, 4, 5):(0, 4, 1, 0)));
    /// ```
    pub fn compacted(&self) -> Self {
        let extents = self.shape.leaves().copied().collect::<Vec<_>>();
        let strides = self.stride.leaves().collect::<Vec<_>>();

        // The modes in stride order. The sort is stable, so strides that
        // do not compare keep their left-to-right order.
        let mut order = (0..extents.len()).collect::<Vec<_>>();
        order.sort_by(|&a, &b| match (strides.get(a), strides.get(b)) {
            (Some(x), Some(y)) => x.sort_cmp(y),
            _ => Ordering::Equal,
        });
        let placed = order.into_iter().filter(|&i| strides.get(i).is_some_and(|d| !d.is_zero()));
        Layout::from_parts(self.shape.clone(), compact_in_order(&extents, placed, &self.shape))
    }

    /// Maps a coordinate to the codomain:
    /// `L(c) == inner_product(idx2crd(c, shape), stride)`.
    ///
    /// The coordinate takes any form — integral, flat, or natural.
    ///
    /// ```
    /// # use weaverbird::{ht, layout, StrideScalar};
    /// let l = layout!((4, 8):(8, 1));
    /// assert_eq!(l.eval(&ht!(14)).unwrap(), StrideScalar::Int(19));
    /// assert_eq!(l.eval(&ht!((2, 3))).unwrap(), StrideScalar::Int(19));
    /// ```
    pub fn eval(&self, crd: &IntTuple) -> Result<StrideScalar> {
        self.shape.idx2crd(crd)?.inner_product(&self.stride)
    }

    /// Evaluates `crd` to a codomain offset and slices the layout,
    /// returning `(offset, sublayout)`.
    ///
    /// An absent entry in `crd` marks a mode the sublayout retains.
    pub fn offset_and_slice(&self, crd: &HTuple<Option<Int>>) -> Result<(StrideScalar, Self)> {
        Ok((
            idx2crd_open(crd, &self.shape)?.inner_product(&self.stride)?,
            Layout::from_parts(crd.slice(&self.shape)?, crd.slice(&self.stride)?),
        ))
    }

    /// Mode `i` of the layout, as a sublayout.
    ///
    /// Returns [`Error::BadPath`] when `i` runs past the rank.
    pub fn mode(&self, i: usize) -> Result<Self> {
        match self.shape.is_tuple() {
            true => self.get(&[i]),
            false if i < self.shape.rank() => Ok(self.clone()),
            false => Err(self.bad_path(&[i])),
        }
    }

    /// The sublayout at the given (possibly nested) `path`.
    ///
    /// Returns [`Error::BadPath`] when `path` does not address the
    /// layout.
    pub fn get(&self, path: &[usize]) -> Result<Self> {
        match (self.shape.get(path), self.stride.get(path)) {
            (Some(shape), Some(stride)) => Ok(Layout::from_parts(shape.clone(), stride.clone())),
            _ => Err(self.bad_path(path)),
        }
    }

    /// Shape of the codomain.
    ///
    /// Returns [`Error::Incompatible`] when the shape and the stride do
    /// not pair, or when a stride mixes an integer with a basis element.
    pub fn coshape(&self) -> Result<IntTuple> {
        let extremum = self.shape.transform_leaf(&|s| s - 1).inner_product(&self.stride)?;
        extremum.add(&ones_like(&extremum)).map(|c| c.to_tuple())
    }

    /// Coalesces every mode, keeping size-1 modes.
    ///
    /// ```
    /// # use weaverbird::layout;
    /// let l = layout!((2, 1, 6, 1):(1, 7, 8, 0));
    /// assert_eq!(l.coalesced_z().unwrap(), layout!((2, 6, 1):(1, 8, 0)));
    /// ```
    pub fn coalesced_z(&self) -> Result<Self> {
        self.coalesce_z(&Profile::Merge)
    }

    /// Coalesces every mode.
    ///
    /// ```
    /// # use weaverbird::layout;
    /// assert_eq!(layout!((2, (1, 6)):(1, (6, 2))).coalesced().unwrap(), layout!(12:1));
    /// let l = layout!((2, 1, 6, 1):(1, 7, 8, 0));
    /// assert_eq!(l.coalesced().unwrap(), layout!((2, 6):(1, 8)));
    /// ```
    pub fn coalesced(&self) -> Result<Self> {
        self.coalesce(&Profile::Merge)
    }

    /// Coalesces per `profile`, keeping size-1 modes.
    ///
    /// Returns [`Error::RankMismatch`] when `profile` outranks the
    /// layout.
    pub fn coalesce_z(&self, profile: &Profile) -> Result<Self> {
        self.coalesce_with("coalesce_z", profile, &|shape, stride| Ok((shape, stride)))
    }

    /// Coalesces per `profile`.
    ///
    /// This is [`Self::coalesce_z`] plus one step: a trailing size-1 mode
    /// drops, so long as it is not the only one left.
    ///
    /// ```
    /// # use weaverbird::{layout, Profile};
    /// let by_mode = Profile::ByMode(vec![Profile::Merge, Profile::Merge]);
    /// let l = layout!((2, (1, 6)):(1, (6, 2)));
    /// assert_eq!(l.coalesce(&by_mode).unwrap(), layout!((2, 6):(1, 2)));
    /// ```
    ///
    /// Returns [`Error::RankMismatch`] when `profile` outranks the
    /// layout.
    pub fn coalesce(&self, profile: &Profile) -> Result<Self> {
        self.coalesce_with("coalesce", profile, &|shape, stride| match (&shape, &stride) {
            (HTuple::Tuple(s), HTuple::Tuple(d)) if s.len() > 1 && s.last() == Some(&ht!(1)) => {
                Ok((
                    HTuple::Tuple(s[..s.len() - 1].to_vec()),
                    HTuple::Tuple(d[..d.len() - 1].to_vec()),
                ))
            }
            _ => Ok((shape, stride)),
        })
    }

    /// The body the two coalesce variants share. They differ only in
    /// `trim`, which runs on the folded shape and stride.
    fn coalesce_with(
        &self,
        op: &'static str,
        profile: &Profile,
        trim: &impl Fn(IntTuple, Stride) -> Result<(IntTuple, Stride)>,
    ) -> Result<Self> {
        match profile {
            Profile::Keep => Ok(self.clone()),
            // A by-mode profile dispatches. The modes it runs out on keep
            // their layout.
            Profile::ByMode(modes) => {
                if self.shape.rank() < modes.len() {
                    return Err(Error::ProfileRank {
                        op,
                        layout: Box::new(self.clone()),
                        profile: profile.clone(),
                    });
                }
                (0..self.shape.rank())
                    .map(|i| {
                        self.mode(i)?.coalesce_with(
                            op,
                            modes.get(i).unwrap_or(&Profile::Keep),
                            trim,
                        )
                    })
                    .collect()
            }
            Profile::Merge => {
                let (shape, stride) = coalesce_modes(&self.shape, &self.stride)?;
                // An empty fold means every mode dropped away. What is
                // left is the one-element layout.
                if shape.rank() == 0 {
                    return Ok(Layout::from_parts(ht!(1), Stride::Leaf(0.into())));
                }
                let (shape, stride) = trim(shape, stride)?;
                Ok(Layout::from_parts(
                    shape.strip_singletons().clone(),
                    stride.strip_singletons().clone(),
                ))
            }
        }
    }

    /// The error a bad mode path raises.
    fn bad_path(&self, path: &[usize]) -> Error {
        Error::BadPath { path: path.to_vec(), value: format!("{self}") }
    }
}

/// Collects modes into one layout.
impl FromIterator<Layout> for Layout {
    fn from_iter<I: IntoIterator<Item = Layout>>(layouts: I) -> Self {
        let (shape, stride) = layouts.into_iter().map(|a| (a.shape, a.stride)).unzip();
        Layout::from_parts(HTuple::Tuple(shape), HTuple::Tuple(stride))
    }
}

/// CuTe notation, e.g. `(4,8):(1,4)`.
impl Display for Layout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.shape, self.stride)
    }
}

/// Constructor form, e.g. `Layout((4,8), (1,4))`.
impl Debug for Layout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Layout({:?}, {:?})", self.shape, self.stride)
    }
}

/// A compact stride that visits `order` first, and pins every other mode
/// to stride 0.
fn compact_in_order(
    extents: &[Int],
    order: impl Iterator<Item = usize>,
    profile: &IntTuple,
) -> Stride {
    let mut strides = vec![StrideScalar::Int(0); extents.len()];
    order.fold(1, |current, i| match extents.get(i) {
        Some(&extent) => {
            strides[i] = StrideScalar::Int(current);
            current * extent
        }
        None => current,
    });
    // One stride per leaf of the profile, so the rebuild cannot run dry.
    HTuple::unflatten(&mut strides.into_iter(), profile)
        .unwrap_or_else(|| HTuple::repeat_like(&StrideScalar::Int(0), profile))
}

/// One at every leaf of `x`.
fn ones_like(x: &StrideScalar) -> StrideScalar {
    match x {
        StrideScalar::Int(_) => StrideScalar::Int(1),
        StrideScalar::Arith(a) => {
            StrideScalar::Arith(ArithTuple::from_data(a.data().iter().map(ones_like).collect()))
        }
    }
}

/// [`IntTuple::idx2crd`] over a coordinate whose modes may be open. An
/// absent mode contributes nothing to the offset.
fn idx2crd_open(crd: &HTuple<Option<Int>>, shape: &IntTuple) -> Result<IntTuple> {
    match crd {
        HTuple::Leaf(None) => Ok(HTuple::repeat_like(&0, shape)),
        HTuple::Leaf(Some(i)) => shape.idx2crd(&HTuple::Leaf(*i)),
        HTuple::Tuple(cs) => match shape {
            HTuple::Tuple(ss) if cs.len() == ss.len() => {
                cs.iter().zip(ss).map(|(c, s)| idx2crd_open(c, s)).collect()
            }
            _ => Err(Error::BadCoord { crd: HTuple::repeat_like(&0, shape), shape: shape.clone() }),
        },
    }
}

// ---------------------------------------------------------------------------
// Profile
// ---------------------------------------------------------------------------

/// How a by-mode coalesce treats one mode.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Profile {
    /// Merge every mode of this sub-tree.
    Merge,
    /// Leave this sub-tree alone.
    Keep,
    /// One profile per mode.
    ByMode(Vec<Profile>),
}

/// The profile that merges every mode of `shape`.
fn merge_profile(shape: &IntTuple) -> Profile {
    match shape {
        HTuple::Leaf(_) => Profile::Merge,
        HTuple::Tuple(modes) => Profile::ByMode(modes.iter().map(merge_profile).collect()),
    }
}

// ---------------------------------------------------------------------------
// Tiler (Whitepaper, §3.3.5 By-mode Composition and Tilers)
// ---------------------------------------------------------------------------

/// The right-hand side of a by-mode operation: composition,
/// [`Layout::logical_divide`], and [`Layout::logical_product`].
///
/// An extent stands for the compact layout `n:1`, and
/// [`Self::to_layout`] turns a whole tiler into the layout that acts as
/// it does.
///
/// ```
/// # use weaverbird::{layout, Tiler};
/// let by_mode: Tiler = vec![Tiler::from(4), Tiler::Skip].into();
/// assert_eq!(layout!((4, 8):(8, 1)).compose(by_mode).unwrap(), layout!((4, 8):(8, 1)));
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Tiler {
    /// The no-op. One mode of the operand passes through untouched.
    Skip,
    /// An extent, which stands for `n:1`.
    Extent(Int),
    /// A layout.
    Layout(Layout),
    /// One tiler per mode.
    ByMode(Vec<Tiler>),
}

impl From<Int> for Tiler {
    fn from(extent: Int) -> Self {
        Tiler::Extent(extent)
    }
}

impl From<Layout> for Tiler {
    fn from(layout: Layout) -> Self {
        Tiler::Layout(layout)
    }
}

impl From<&Layout> for Tiler {
    fn from(layout: &Layout) -> Self {
        Tiler::Layout(layout.clone())
    }
}

impl From<Vec<Tiler>> for Tiler {
    fn from(modes: Vec<Tiler>) -> Self {
        Tiler::ByMode(modes)
    }
}

/// So a caller can hand the same tiler to several operations.
impl From<&Tiler> for Tiler {
    fn from(tiler: &Tiler) -> Self {
        tiler.clone()
    }
}

impl Tiler {
    /// The layout that acts as this tiler does under composition.
    ///
    /// Each mode of a [`Self::ByMode`] takes its own basis element, so
    /// the modes land in separate positions of the codomain.
    ///
    /// ```
    /// # use weaverbird::{layout, Tiler};
    /// assert_eq!(Tiler::Extent(3).to_layout().unwrap(), layout!(3:1));
    /// ```
    ///
    /// Returns [`Error::BadPath`] for a [`Self::Skip`], which has no
    /// layout.
    pub fn to_layout(&self) -> Result<Layout> {
        self.to_layout_at(&mut Vec::new())
    }

    /// [`Self::to_layout`] at one position of the codomain. The scale of
    /// a leaf is the basis element of its own path.
    fn to_layout_at(&self, path: &mut Vec<usize>) -> Result<Layout> {
        match self {
            Tiler::Skip => Err(Error::BadPath { path: path.clone(), value: format!("{self:?}") }),
            Tiler::Extent(n) => Ok(Layout::from_parts(HTuple::Leaf(*n), HTuple::Leaf(e(path)))),
            Tiler::Layout(l) => {
                let scale = e(path);
                l.stride
                    .try_transform_leaf(&|d| scale_by(&scale, d))
                    .map(|stride| Layout::from_parts(l.shape.clone(), stride))
            }
            Tiler::ByMode(modes) => modes
                .iter()
                .enumerate()
                .map(|(i, mode)| {
                    path.push(i);
                    let layout = mode.to_layout_at(path);
                    path.pop();
                    layout
                })
                .collect(),
        }
    }

    /// This tiler as a layout, for the operand of a by-mode dispatch.
    /// A [`Self::ByMode`] has no single layout, so it returns `None` and
    /// the caller dispatches instead.
    fn as_layout(&self) -> Option<Layout> {
        match self {
            Tiler::Extent(n) => {
                Some(Layout::from_parts(HTuple::Leaf(*n), HTuple::Leaf(StrideScalar::Int(1))))
            }
            Tiler::Layout(l) => Some(l.clone()),
            Tiler::Skip | Tiler::ByMode(_) => None,
        }
    }
}

/// The product `e * d`: one side must be an integer, which then scales
/// the other.
///
/// Returns [`Error::NoProduct`] for two arithmetic tuples. Their product
/// leaves `Z^S`.
fn scale_by(e: &StrideScalar, d: &StrideScalar) -> Result<StrideScalar> {
    match (e, d) {
        (StrideScalar::Int(a), _) => Ok(d.scale(*a)),
        (_, StrideScalar::Int(b)) => Ok(e.scale(*b)),
        _ => Err(Error::NoProduct { lhs: e.clone(), rhs: d.clone() }),
    }
}

// ---------------------------------------------------------------------------
// recast
// ---------------------------------------------------------------------------

/// The element scale of [`Layout::recast`]: a positive rational.
///
/// The argument expresses both packing (`8` source elements per new one)
/// and unpacking (`1/2`, two new elements per source one). `std` has no
/// rational, so this carries the numerator and the denominator itself,
/// and every use inside `recast` is exact integer arithmetic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Scale {
    num: Int,
    den: Int,
}

impl Scale {
    /// `n` source elements per new element.
    pub fn whole(n: Int) -> Result<Self> {
        Scale::fraction(n, 1)
    }

    /// `num / den` source elements per new element.
    ///
    /// Returns [`Error::Divisibility`] for a non-positive scale — the
    /// rescaling divides by it.
    pub fn fraction(num: Int, den: Int) -> Result<Self> {
        match num > 0 && den > 0 {
            true => Ok(Scale { num, den }),
            false => Err(Error::BadScale { num, den }),
        }
    }

    /// `divmod(d, scale)`: the floor quotient, and whether the division
    /// came out even.
    fn divide_into(self, d: Int) -> (Int, bool) {
        let scaled = d * self.den;
        (scaled.div_euclid(self.num), scaled % self.num == 0)
    }

    /// `divmod(scale, d)`: the floor quotient, and whether the division
    /// came out even.
    fn divide_by(self, d: Int) -> (Int, bool) {
        let divisor = self.den * d;
        (self.num.div_euclid(divisor), self.num % divisor == 0)
    }

    /// The extent that holds `shape` source elements at this scale.
    fn ceil_div(self, shape: Int) -> Int {
        ceil_div(shape * self.den, self.num)
    }
}

/// `-(-a // b)`, for a positive `b`.
fn ceil_div(a: Int, b: Int) -> Int {
    a.div_euclid(b) + Int::from(a.rem_euclid(b) != 0)
}

impl Layout {
    /// Recasts the layout to a new element scale.
    ///
    /// Rewrites both shape and stride so the layout addresses a
    /// differently-sized element. Each leaf `s:d` rescales by the ratio
    /// between `d` and `scale` — the shape shrinks when packing, and
    /// grows when unpacking.
    ///
    /// Pre-conditions:
    ///   at each leaf the stride and the scale divide cleanly — one is a
    ///   multiple of the other.
    ///
    /// ```
    /// # use weaverbird::{layout, Scale};
    /// assert_eq!(layout!(24:1).recast(Scale::whole(8).unwrap()).unwrap(), layout!(3:1));
    /// assert_eq!(layout!(24:2).recast(Scale::whole(4).unwrap()).unwrap(), layout!(12:1));
    /// ```
    ///
    /// Returns [`Error::Divisibility`] when a leaf divides unevenly.
    pub fn recast(&self, scale: Scale) -> Result<Self> {
        match (&self.shape, &self.stride) {
            (HTuple::Leaf(shape), HTuple::Leaf(stride)) => recast_leaf(*shape, stride, scale),
            (HTuple::Tuple(shapes), HTuple::Tuple(strides)) if shapes.len() == strides.len() => {
                shapes
                    .iter()
                    .zip(strides)
                    .map(|(s, d)| Layout::from_parts(s.clone(), d.clone()).recast(scale))
                    .collect()
            }
            _ => Err(self.bad_path(&[])),
        }
    }
}

/// One leaf of [`Layout::recast`].
fn recast_leaf(shape: Int, stride: &StrideScalar, scale: Scale) -> Result<Layout> {
    let dd = match proj(stride, stride)? {
        StrideScalar::Int(v) => *v,
        other => {
            return Err(Error::NotBasis { value: other.clone() });
        }
    };
    if dd == 0 {
        return Ok(Layout::from_parts(HTuple::Leaf(shape), HTuple::Leaf(stride.clone())));
    }
    if dd == 1 {
        return Ok(Layout::from_parts(
            HTuple::Leaf(scale.ceil_div(shape)),
            HTuple::Leaf(stride.clone()),
        ));
    }
    let (qdn, rdn) = scale.divide_into(dd);
    let (qnd, rnd) = scale.divide_by(dd);
    if !(rdn || rnd) {
        return Err(Error::Recast {
            shape,
            stride: stride.clone(),
            num: scale.num,
            den: scale.den,
        });
    }
    Ok(Layout::from_parts(
        HTuple::Leaf(ceil_div(shape, if rnd { qnd } else { 1 })),
        HTuple::Leaf(unit(stride)?.scale(if rdn { qdn } else { 1 })),
    ))
}

// ---------------------------------------------------------------------------
// complement
// ---------------------------------------------------------------------------

impl Layout {
    /// The complement of this layout.
    ///
    /// The complement walks this layout's modes in stride order and
    /// records the gaps they leave behind, one running position per
    /// codomain position. The result is the layout of those gaps, so
    /// `Layout::from_modes(vec![self, self.complement()?])` covers the
    /// whole codomain without repeating a value.
    ///
    /// Post-conditions:
    ///   `weakly_congruent(coshape(self), shape(result))`,
    ///   `result` is ordered — `result(i) < result(i+1)`,
    ///   the codomains of `self` and `result` are disjoint.
    ///
    /// ```
    /// # use weaverbird::layout;
    /// assert_eq!(layout!(4:1).complement().unwrap(), layout!(1:4));
    /// assert_eq!(layout!((2, 4):(1, 6)).complement().unwrap(), layout!((3, 1):(2, 24)));
    /// ```
    ///
    /// Returns [`Error::NonInjective`] when a mode falls behind the
    /// running position, and [`Error::NotBasis`] when a stride leaf is a
    /// sum of basis elements, which has no single codomain position.
    pub fn complement(&self) -> Result<Self> {
        let coprofile = self.coshape()?;
        // One accumulator pair per position of the codomain. Every value
        // pushed below is an integer: `d` is the coefficient `proj` reads
        // off a stride leaf, and `s` an extent.
        let mut result_s = HTuple::<Vec<Int>>::repeat_like(&Vec::new(), &coprofile);
        let mut result_d = HTuple::<Vec<Int>>::repeat_like(&Vec::from([1]), &coprofile);

        // Modes are ordered by stride. The sort is stable, so equal and
        // incomparable strides keep their left-to-right order.
        let mut modes = self.stride.leaves().zip(self.shape.leaves()).collect::<Vec<_>>();
        modes.sort_by(|(a, _), (b, _)| a.sort_cmp(b));

        // Each mode reads the running position its own codomain position
        // left behind, and leaves a new one.
        for (de, &s) in modes {
            let d = proj(de, de)?;
            if d.is_zero() || s == 1 {
                continue;
            }
            let d = coefficient(d)?;
            let result_s = accumulator(&mut result_s, de)?;
            let result_d = accumulator(&mut result_d, de)?;
            let position = *result_d.last().ok_or_else(|| self.bad_path(&[]))?;
            if d < position {
                return Err(Error::NonInjective {
                    op: "complement",
                    layout: Box::new(self.clone()),
                });
            }
            result_s.push(
                d.checked_div(position)
                    .ok_or_else(|| Error::ZeroExtent { layout: Box::new(self.clone()) })?,
            );
            result_d.push(d * s);
        }

        // One layout per codomain position, each closed with the extent
        // `1` that `complement_to` grows.
        complement_tiler(&coprofile, &result_s, &result_d)?.to_layout()
    }

    /// The complement, extended to cover `extend`.
    ///
    /// The trailing mode of each complement mode grows until it spans
    /// `extend`. An extend mode that no complement mode faces stands on
    /// its own.
    ///
    /// ```
    /// # use weaverbird::{ht, layout};
    /// assert_eq!(layout!(4:2).complement_to(&ht!(20)).unwrap(), layout!((2, 3):(1, 8)));
    /// ```
    ///
    /// Returns [`Error::RankMismatch`] when `extend` and the codomain do
    /// not pair up.
    pub fn complement_to(&self, extend: &IntTuple) -> Result<Self> {
        let coprofile = self.coshape()?;
        let complement = self.complement()?;
        let basis = make_basis_like(extend);
        extend_walk(
            Some(&coprofile),
            Some(&complement.shape),
            Some(&complement.stride),
            Some(extend),
            Some(&basis),
            self,
        )
    }
}

/// One layout per codomain position, from the accumulated extents and
/// positions. Each closes with the extent `1` that an extension grows.
fn complement_tiler(
    coprofile: &IntTuple,
    result_s: &HTuple<Vec<Int>>,
    result_d: &HTuple<Vec<Int>>,
) -> Result<Tiler> {
    match (coprofile, result_s, result_d) {
        (HTuple::Leaf(_), HTuple::Leaf(shape), HTuple::Leaf(stride)) => Layout::from_parts(
            shape.iter().copied().chain([1]).map(HTuple::Leaf).collect(),
            stride.iter().map(|&d| HTuple::Leaf(StrideScalar::Int(d))).collect(),
        )
        .coalesced_z()
        .map(Tiler::Layout),
        (HTuple::Tuple(profile), HTuple::Tuple(shapes), HTuple::Tuple(strides))
            if profile.len() == shapes.len() && shapes.len() == strides.len() =>
        {
            profile
                .iter()
                .zip(shapes)
                .zip(strides)
                .map(|((p, s), d)| complement_tiler(p, s, d))
                .collect::<Result<Vec<_>>>()
                .map(Tiler::ByMode)
        }
        _ => Err(Error::BadPath { path: vec![], value: format!("{coprofile}") }),
    }
}

/// Grows the trailing mode of each complement mode until it spans the
/// extension.
///
/// The codomain profile drives the walk. The other four tuples step in
/// parallel, and an absent one means the extension outranks the codomain
/// there.
fn extend_walk(
    coprofile: Option<&IntTuple>,
    shape_c: Option<&IntTuple>,
    stride_c: Option<&Stride>,
    shape_e: Option<&IntTuple>,
    stride_e: Option<&Stride>,
    origin: &Layout,
) -> Result<Layout> {
    if let Some(HTuple::Tuple(modes)) = coprofile {
        let width = modes
            .len()
            .max(rank_of(shape_c))
            .max(rank_of(stride_c))
            .max(rank_of(shape_e))
            .max(rank_of(stride_e));
        return (0..width)
            .map(|i| {
                extend_walk(
                    child(coprofile, i),
                    child(shape_c, i),
                    child(stride_c, i),
                    child(shape_e, i),
                    child(stride_e, i),
                    origin,
                )
            })
            .collect();
    }
    let mismatch = || Error::ExtensionMismatch {
        layout: Box::new(origin.clone()),
        extend: shape_e.cloned().unwrap_or(HTuple::Leaf(0)),
    };
    // No complement mode here: the extension mode stands on its own.
    let (Some(shape_c), Some(stride_c)) = (shape_c, stride_c) else {
        return match (shape_e, stride_e) {
            (Some(shape), Some(stride)) => Ok(Layout::from_parts(shape.clone(), stride.clone())),
            _ => Err(mismatch()),
        };
    };
    let shape_e = shape_e.ok_or_else(mismatch)?;
    // The last extent of the complement is always 1, so its stride is the
    // size the extension starts from.
    let size_c = match stride_c.back() {
        HTuple::Leaf(d) => coefficient(proj(d, d)?)?,
        _ => return Err(mismatch()),
    };
    let (shape_r, _) =
        shape_e.leaves().try_fold((Vec::new(), size_c), |(mut shape_r, size_c), &s| {
            match s > 0 && size_c > 0 {
                true => {
                    shape_r.push(HTuple::Leaf(ceil_div(s, size_c)));
                    Ok((shape_r, ceil_div(size_c, s)))
                }
                false => Err(Error::ExtensionMismatch {
                    layout: Box::new(origin.clone()),
                    extend: shape_e.clone(),
                }),
            }
        })?;
    // The extension splits the trailing mode, so the complement's stride
    // seeds a prefix product rather than standing as the whole stride.
    Layout::from_base(shape_c.clone().replace_back(HTuple::Tuple(shape_r)), stride_c)?.coalesced()
}

/// The `i`th child of an optional tuple, or `None` past the end.
fn child<T>(x: Option<&HTuple<T>>, i: usize) -> Option<&HTuple<T>> {
    match x {
        Some(HTuple::Tuple(modes)) => modes.get(i),
        _ => None,
    }
}

/// The rank an optional tuple contributes to a parallel walk. An absent
/// tuple, or a leaf, contributes nothing.
fn rank_of<T>(x: Option<&HTuple<T>>) -> usize {
    match x {
        Some(HTuple::Tuple(modes)) => modes.len(),
        _ => 0,
    }
}

/// The integer coefficient of a stride leaf.
///
/// Every non-zero leaf [`proj`] reaches is one, so the error stands for a
/// stride that is a sum of basis elements — an element `complement`
/// cannot place in the codomain.
fn coefficient(d: &StrideScalar) -> Result<Int> {
    match d {
        StrideScalar::Int(v) => Ok(*v),
        other => Err(Error::NotBasis { value: other.clone() }),
    }
}

/// The accumulator list at the codomain position `de` names.
fn accumulator<'a>(x: &'a mut HTuple<Vec<Int>>, de: &StrideScalar) -> Result<&'a mut Vec<Int>> {
    match proj_tuple_mut(x, de)? {
        HTuple::Leaf(list) => Ok(list),
        other => Err(Error::BadPath { path: vec![], value: format!("{other:?}") }),
    }
}

// ---------------------------------------------------------------------------
// the inverses and the nullspace
// ---------------------------------------------------------------------------

impl Layout {
    /// The largest right inverse of this layout.
    ///
    /// Walks the modes in stride order and keeps the ones that continue
    /// the chain `d_k == s_{k-1} * d_{k-1}`, recording each survivor's
    /// extent against its position in the domain. A mode that carries no
    /// information — stride 0, or extent 1 — is skipped, and so is any
    /// mode that breaks the chain. That is what makes the result the
    /// *largest* right inverse rather than a failure.
    ///
    /// Post-conditions:
    ///   `weakly_congruent(coshape(self), shape(result))`,
    ///   `result(self(result(i))) == result(i)` for every `i` in the
    ///   domain of `result`,
    ///   `self(result(i)) == i` as well, when the codomain is `Z`.
    ///
    /// ```
    /// # use weaverbird::layout;
    /// assert_eq!(layout!((8, 4):(4, 1)).right_inverse().unwrap(), layout!((4, 8):(8, 1)));
    /// assert_eq!(layout!(4:2).right_inverse().unwrap(), layout!(1:0));
    /// ```
    pub fn right_inverse(&self) -> Result<Self> {
        let coprofile = self.coshape()?;
        // One accumulator per position of the codomain.
        let mut result_shape = HTuple::repeat_like(&Vec::<Int>::new(), &coprofile);
        let mut result_stride = HTuple::repeat_like(&Vec::<StrideScalar>::new(), &coprofile);
        let mut curr_stride = HTuple::repeat_like(&StrideScalar::Int(1), &coprofile);

        let (flat_s, flat_d) = coalesce_modes(&self.shape, &self.stride)?;
        let positions = flat_s.compact_stride();

        // The chain is followed in stride order. The sort is stable, so
        // equal and incomparable strides keep their input order.
        let mut modes = zip_modes(&flat_d, &flat_s, &positions);
        modes.sort_by(|a, b| a.0.sort_cmp(b.0));

        // Each mode reads and rewrites the accumulator at its own
        // codomain position.
        for (de, s, position) in modes {
            let d = proj(de, de)?;
            // Stride-0 and size-1 modes carry no information. The guard
            // precedes the projections, because a stride of integer 0 is
            // the rank-0 basis and projects to the whole tree.
            if d.is_zero() || s == 1 {
                continue;
            }
            let result_s = leaf_mut(&mut result_shape, de)?;
            let result_d = leaf_mut(&mut result_stride, de)?;
            let curr_d = leaf_mut(&mut curr_stride, de)?;

            // A mode that does not continue the chain is dropped.
            if d != curr_d {
                continue;
            }
            result_s.push(s);
            result_d.push(position.clone());
            *curr_d = d.scale(s);
        }

        Layout::from_parts(modes_of(result_shape), modes_of(result_stride))
            .coalesce(&merge_profile(&coprofile))
    }

    /// The left inverse of this layout.
    ///
    /// Walks the modes in stride order and pads each recorded extent out
    /// to the next stride, so the holes between the modes become stride-0
    /// filler. Unlike [`Self::right_inverse`], a mode that breaks the
    /// chain is an error rather than a stopping point: a left inverse
    /// must account for the whole domain.
    ///
    /// Post-conditions:
    ///   `weakly_congruent(coshape(self), shape(result))`,
    ///   `self(result(self(i))) == self(i)` for every `i` in the domain.
    ///
    /// ```
    /// # use weaverbird::layout;
    /// assert_eq!(layout!((8, 4):(4, 1)).left_inverse().unwrap(), layout!((4, 8):(8, 1)));
    /// assert_eq!(layout!((4, 2):(1, 16)).left_inverse().unwrap(), layout!((16, 2):(1, 4)));
    /// ```
    ///
    /// Returns [`Error::Divisibility`] when the strides do not form an
    /// ordered chain, and [`Error::NonInjective`] when a mode overlaps
    /// the one before it.
    pub fn left_inverse(&self) -> Result<Self> {
        let coprofile = self.coshape()?;
        // One accumulator per codomain position, as in
        // [`Self::right_inverse`]. These start non-empty: the seed `1:0`
        // mode is what the first stride pads out to.
        let mut result_shape = HTuple::repeat_like(&vec![1], &coprofile);
        let mut result_stride = HTuple::repeat_like(&vec![StrideScalar::Int(0)], &coprofile);
        let mut curr_shape = HTuple::repeat_like(&1, &coprofile);

        let (flat_s, flat_d) = coalesce_modes(&self.shape, &self.stride)?;
        let positions = flat_s.compact_stride();

        // The comparison runs `(stride, extent, position)`
        // lexicographically. The leading stride orders only partially, so
        // an incomparable pair falls back on the stable order.
        let mut modes = zip_modes(&flat_d, &flat_s, &positions);
        modes.sort_by(|a, b| {
            a.0.sort_cmp(b.0).then_with(|| a.1.cmp(&b.1)).then_with(|| a.2.sort_cmp(b.2))
        });

        for (de, s, position) in modes {
            let d = proj(de, de)?;
            if d.is_zero() || s == 1 {
                continue;
            }
            let result_s = leaf_mut(&mut result_shape, de)?;
            let result_d = leaf_mut(&mut result_stride, de)?;
            let curr_s = leaf_mut(&mut curr_shape, de)?;

            // The chain is walked with integer arithmetic, so a stride
            // that projects to anything else has no place on it.
            let d = match d {
                StrideScalar::Int(v) => *v,
                other => {
                    return Err(Error::NotBasis { value: other.clone() });
                }
            };
            // gap = d_k / d_{k-1}, the span to the next stride.
            let (gap, rem) = (d.div_euclid(*curr_s), d.rem_euclid(*curr_s));
            if rem != 0 {
                return Err(Error::UnorderedStrides { layout: Box::new(self.clone()) });
            }
            // d_k must clear the previous mode: d_k >= d_{k-1} * s_{k-1}.
            if result_s.last().is_some_and(|&last| gap < last) {
                return Err(Error::NonInjective {
                    op: "left_inverse",
                    layout: Box::new(self.clone()),
                });
            }
            // Pad the previous mode out to d_k. The extra entries are
            // holes.
            if let Some(last) = result_s.last_mut() {
                *last = gap;
            }
            // Advance the consumed stride to d_k.
            *curr_s *= gap;
            // Record this mode. A later mode overwrites `s` with its own
            // gap.
            result_s.push(s);
            result_d.push(position.clone());
        }

        Layout::from_parts(modes_of(result_shape), modes_of(result_stride))
            .coalesce_z(&merge_profile(&coprofile))
    }

    /// The nullspace of this layout.
    ///
    /// The stride-0 modes, gathered into a layout over the domain: every
    /// coordinate it produces maps to zero. A layout with no stride-0
    /// mode has the trivial nullspace `1:0`.
    ///
    /// Post-conditions:
    ///   `self(result(i)) == 0` for every `i` in the domain of `result`.
    ///
    /// ```
    /// # use weaverbird::layout;
    /// assert_eq!(layout!((2, 4, 6):(1, 2, 0)).nullspace().unwrap(), layout!(6:8));
    /// assert_eq!(layout!((8, 4):(4, 1)).nullspace().unwrap(), layout!(1:0));
    /// ```
    pub fn nullspace(&self) -> Result<Self> {
        let zeros = self
            .stride
            .leaves()
            .enumerate()
            .filter(|(_, d)| d.is_zero())
            .map(|(i, _)| i)
            .collect::<Vec<_>>();
        if zeros.is_empty() {
            return Ok(Layout::from_parts(ht!(1), HTuple::Leaf(StrideScalar::Int(0))));
        }
        let flat_shape = self.shape.flatten();
        let positions = flat_shape.compact_stride();
        Ok(Layout::from_parts(
            flat_shape.select(&zeros)?.strip_singletons().clone(),
            positions.select(&zeros)?.strip_singletons().clone(),
        ))
    }
}

/// The `(stride, extent, position)` triples the two inverses sort.
fn zip_modes<'a>(
    flat_d: &'a Stride,
    flat_s: &IntTuple,
    positions: &'a Stride,
) -> Vec<(&'a StrideScalar, Int, &'a StrideScalar)> {
    flat_d
        .leaves()
        .zip(flat_s.leaves())
        .zip(positions.leaves())
        .map(|((de, s), position)| (de, *s, position))
        .collect()
}

/// The accumulator at the codomain position `de` names, for writing
/// through.
///
/// Returns [`Error::BadPath`] when the position addresses an interior
/// node rather than one accumulator.
fn leaf_mut<'a, T: Debug>(x: &'a mut HTuple<T>, de: &StrideScalar) -> Result<&'a mut T> {
    match proj_tuple_mut(x, de)? {
        HTuple::Leaf(v) => Ok(v),
        other => Err(Error::BadPath { path: vec![], value: format!("{other:?}") }),
    }
}

/// Turns each accumulated vector into a mode of its own.
fn modes_of<T>(acc: HTuple<Vec<T>>) -> HTuple<T> {
    match acc {
        HTuple::Leaf(v) => v.into_iter().map(HTuple::Leaf).collect(),
        HTuple::Tuple(modes) => modes.into_iter().map(modes_of).collect(),
    }
}

// ---------------------------------------------------------------------------
// composition, logical_divide and logical_product
// ---------------------------------------------------------------------------

impl Layout {
    /// Composes this layout with `b`: the group composition `A o B`
    /// (Whitepaper, §3.3).
    ///
    /// The result has `b`'s domain and this layout's values — walk `b`,
    /// then map what it produces through `self`. A [`Tiler::ByMode`]
    /// composes by mode, and a [`Tiler::Skip`] is the no-op.
    ///
    /// Pre-conditions:
    ///   `self` and `b` satisfy the shape- and stride-divisibility
    ///   conditions (Whitepaper, Eqs. (20)-(21)).
    ///
    /// Post-conditions:
    ///   `compatible(shape(b), shape(result))`,
    ///   `result(i) == self(b(i))` for every `i` in the domain of `b`.
    ///
    /// ```
    /// # use weaverbird::layout;
    /// let a = layout!((6, 2):(8, 2));
    /// assert_eq!(a.compose(layout!((4, 3):(3, 1))).unwrap(), layout!(((2, 2), 3):((24, 2), 8)));
    /// assert_eq!(layout!(12:1).compose(layout!((4, 3))).unwrap(), layout!((4, 3):(1, 4)));
    /// ```
    ///
    /// Returns [`Error::RankMismatch`] when a by-mode tiler outranks this
    /// layout, and [`Error::Divisibility`] when either divisibility
    /// condition fails.
    pub fn compose(&self, b: impl Into<Tiler>) -> Result<Self> {
        self.compose_tiler(&b.into())
    }

    /// [`Self::compose`] over a borrowed tiler, so the by-mode recursion
    /// does not rebuild one per mode.
    fn compose_tiler(&self, b: &Tiler) -> Result<Self> {
        let b = match self.dispatch_by_mode("composition", b, Layout::compose_tiler)? {
            Dispatch::NoOp => return Ok(self.clone()),
            Dispatch::Done(result) => return Ok(result),
            Dispatch::Single(b) => b,
        };

        //
        // Special cases with A: Layout and B: Layout
        //

        let a = self.coalesce_z(&merge_profile(&b.coshape()?))?;

        // RHS distributive, A o (X,Y,...) => (A o X, A o Y, ...).
        if b.shape.is_tuple() {
            return (0..b.shape.rank())
                .map(|i| a.compose_tiler(&Tiler::Layout(b.mode(i)?)))
                .collect();
        }
        // Special case stride-0, A o N:0 => N:0.
        if b.stride == HTuple::Leaf(StrideScalar::Int(0)) {
            return Ok(Layout::from_parts(b.shape.clone(), HTuple::Leaf(StrideScalar::Int(0))));
        }
        // `b.shape` is no longer a tuple, so both halves are leaves. A
        // stride that is not belongs to an incongruent layout.
        let (HTuple::Leaf(b_shape), HTuple::Leaf(b_stride)) = (&b.shape, &b.stride) else {
            return Err(b.bad_path(&[]));
        };
        // Special case shape-1, A o 1:M => 1:A(M).
        if *b_shape == 1 {
            return Ok(Layout::from_parts(
                b.shape.clone(),
                HTuple::Leaf(a.eval(&b_stride.to_tuple())?),
            ));
        }

        //
        // General case   (A0,A1,...) o N:M
        //

        let shape_divisibility = || Error::Divisibility {
            condition: Condition::Shape,
            a: Box::new(self.clone()),
            b: Box::new(b.clone()),
        };
        let stride_divisibility = || Error::Divisibility {
            condition: Condition::Stride,
            a: Box::new(self.clone()),
            b: Box::new(b.clone()),
        };

        // Each basis term of `B`'s stride truncates one sublayout of `A`
        // and shifts it into place.
        let mut acc: Option<Layout> = None;
        for (mut stride_b, basis_b) in basis_repr(b_stride) {
            let ab = a.get(&basis_b)?;
            let (mut result_s, mut result_d) = flat_modes(&ab)?;

            // Truncate or extend result_s by strideB * B.shape.
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

            // Remove the result_s prefix strideB. The trailing mode takes
            // the division when no earlier mode does.
            let mut divided = false;
            for i in 0..result_s.len() - 1 {
                let (q_sd, r_sd) = divmod(result_s[i], stride_b).ok_or_else(stride_divisibility)?;
                if r_sd == 0 {
                    result_s[i] = q_sd;
                    result_d[i] = result_d[i].scale(stride_b);
                    divided = true;
                    break;
                }
                let (q_ds, r_ds) = divmod(stride_b, result_s[i]).ok_or_else(stride_divisibility)?;
                stride_b = q_ds;
                result_s[i] = 1;
                if r_ds != 0 {
                    return Err(stride_divisibility());
                }
            }
            if !divided {
                let last = result_s.len() - 1;
                result_s[last] =
                    divmod(result_s[last], stride_b).ok_or_else(stride_divisibility)?.0;
                result_d[last] = result_d[last].scale(stride_b);
            }

            let term = Layout::from_parts(
                result_s.into_iter().map(HTuple::Leaf).collect(),
                result_d.into_iter().map(HTuple::Leaf).collect(),
            );
            acc = Some(match acc {
                Some(sum) => layout_add(&sum, &term)?,
                None => term,
            });
        }

        match acc {
            Some(result) => result.coalesced(),
            // [`basis_repr`] yields at least one term, so the fold above
            // always runs. This arm stands for the type alone.
            None => Err(Error::NotBasis { value: b_stride.clone() }),
        }
    }

    /// Splits this layout into the elements of `b` — the Tile — and a
    /// grid over those tiles.
    ///
    /// The Tile is `b` itself and the grid is [`Self::complement_to`] of
    /// `b` over this layout's shape, so composing with the two of them
    /// together re-reads the whole domain, tile first.
    ///
    /// The dispatch is [`Self::compose`]'s: a [`Tiler::ByMode`] divides
    /// by mode, an extent promotes to `n:1`, and a [`Tiler::Skip`] is the
    /// no-op. A by-mode tiler therefore interleaves Tile and grid per
    /// mode.
    ///
    /// Post-conditions, for a `b` that is one layout:
    ///   `rank(result) == 2`,
    ///   `compatible(shape(b), shape(result[0]))`,
    ///   `result(i, 0) == self(b(i))` for every `i` in the domain of `b`,
    ///   every element of `self` appears in `result`.
    ///
    /// ```
    /// # use weaverbird::layout;
    /// let split = layout!(24:1).logical_divide(layout!(4:2)).unwrap();
    /// assert_eq!(split, layout!((4, (2, 3)):(2, (1, 8))));
    /// ```
    ///
    /// Returns [`Error::RankMismatch`] when a by-mode tiler outranks this
    /// layout.
    pub fn logical_divide(&self, b: impl Into<Tiler>) -> Result<Self> {
        self.logical_divide_tiler(&b.into())
    }

    /// [`Self::logical_divide`] over a borrowed tiler.
    fn logical_divide_tiler(&self, b: &Tiler) -> Result<Self> {
        let b = match self.dispatch_by_mode("logical_divide", b, Layout::logical_divide_tiler)? {
            Dispatch::NoOp => return Ok(self.clone()),
            Dispatch::Done(result) => return Ok(result),
            Dispatch::Single(b) => b,
        };
        let grid = b.complement_to(&self.shape)?;
        self.compose(Layout::from_modes(vec![b, grid]))
    }

    /// Reproduces this layout over `b`.
    ///
    /// Mode 0 of the result is this layout untouched, and mode 1 walks
    /// `b` through the codomain this layout leaves free — its
    /// [`Self::complement`] — so each coordinate of `b` places one
    /// disjoint copy.
    ///
    /// Post-conditions:
    ///   `rank(result) == 2`,
    ///   `result[0] == self`,
    ///   `compatible(shape(b), shape(result[1]))`.
    ///
    /// ```
    /// # use weaverbird::layout;
    /// assert_eq!(layout!(3:1).logical_product(layout!(4:1)).unwrap(), layout!((3, 4):(1, 3)));
    /// ```
    ///
    /// Returns [`Error::RankMismatch`] when a by-mode tiler outranks this
    /// layout.
    pub fn logical_product(&self, b: impl Into<Tiler>) -> Result<Self> {
        self.logical_product_tiler(&b.into())
    }

    /// [`Self::logical_product`] over a borrowed tiler.
    fn logical_product_tiler(&self, b: &Tiler) -> Result<Self> {
        let b = match self.dispatch_by_mode("logical_product", b, Layout::logical_product_tiler)? {
            Dispatch::NoOp => return Ok(self.clone()),
            Dispatch::Done(result) => return Ok(result),
            Dispatch::Single(b) => b,
        };
        Ok(Layout::from_modes(vec![self.clone(), self.complement()?.compose(b)?]))
    }

    /// The dispatch head the three by-mode operations share.
    ///
    /// Returns [`Error::RankMismatch`] when a by-mode tiler outranks this
    /// layout.
    fn dispatch_by_mode(
        &self,
        op: &'static str,
        b: &Tiler,
        each: impl Fn(&Layout, &Tiler) -> Result<Layout>,
    ) -> Result<Dispatch> {
        match b {
            Tiler::Skip => Ok(Dispatch::NoOp),
            Tiler::ByMode(modes) => {
                if self.shape.rank() < modes.len() {
                    return Err(Error::TilerRank {
                        op,
                        layout: Box::new(self.clone()),
                        tiler: b.clone(),
                    });
                }
                // The modes the tiler runs out on take the no-op.
                (0..self.shape.rank())
                    .map(|i| each(&self.mode(i)?, modes.get(i).unwrap_or(&Tiler::Skip)))
                    .collect::<Result<Layout>>()
                    .map(Dispatch::Done)
            }
            Tiler::Extent(_) | Tiler::Layout(_) => match b.as_layout() {
                Some(layout) => Ok(Dispatch::Single(layout)),
                None => Ok(Dispatch::NoOp),
            },
        }
    }
}

/// What a by-mode dispatch leaves for the general case.
enum Dispatch {
    /// The tiler is the no-op. The operand passes through.
    NoOp,
    /// The dispatch ran per mode and holds the whole result.
    Done(Layout),
    /// One layout for the general case to work on.
    Single(Layout),
}

/// The top-level modes of a layout.
///
/// The layout is a sublayout of a coalesced one, so it is flat and every
/// mode is a leaf.
///
/// The result is non-empty and the two halves have equal length, which is
/// what lets the composition walk index `result_s.len() - 1`.
fn flat_modes(ab: &Layout) -> Result<(Vec<Int>, Vec<StrideScalar>)> {
    let result_s = top_level_leaves(&ab.shape).ok_or_else(|| ab.bad_path(&[]))?;
    let result_d = top_level_leaves(&ab.stride).ok_or_else(|| ab.bad_path(&[]))?;
    match !result_s.is_empty() && result_s.len() == result_d.len() {
        true => Ok((result_s, result_d)),
        false => Err(ab.bad_path(&[])),
    }
}

/// The leaves one level down. Returns `None` for a mode that is itself a
/// tuple.
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

/// Floor quotient and remainder, over the positive extents and strides
/// these walks carry. A zero divisor answers `None`, which the caller
/// turns into the divisibility error of its own loop.
fn divmod(a: Int, b: Int) -> Option<(Int, Int)> {
    (b != 0).then(|| (a.div_euclid(b), a.rem_euclid(b)))
}

/// Builds a [`Layout`] from `shape:stride`, or a compact one from a shape
/// alone.
///
/// ```
/// use weaverbird::{layout, Layout};
/// let strided = layout!((4, 8):(1, 4));
/// let compact = layout!((4, 8));
/// assert_eq!(strided, compact);
/// ```
#[macro_export]
macro_rules! layout {
    ($shape:tt : $stride:tt) => {
        $crate::layout::Layout::from_parts($crate::ht!($shape), $crate::stride!($stride))
    };
    ($shape:tt) => {
        $crate::layout::Layout::compact($crate::ht!($shape))
    };
}
