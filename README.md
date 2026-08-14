# weaverbird: Weave access patterns for your arrays

A Rust transliteration of the CuTe layout algebra.

The source is [PyCuTe].

## The name

A weaverbird builds its nest by interlacing strips on a regular pitch.
That is what this algebra describes. `logical_product` lays a tile over a
grid, and the two variants are the two weaves:

```text
blocked_product              raked_product
  A A A C C C                  A C A C A C
  A A A C C C                  B D B D B D
  B B B D D D                  A C A C A C
  B B B D D D                  B D B D B D
```

Each cell carries the tile it came from. `blocked` keeps a tile
contiguous; `raked` spreads it on a sublattice, so all four tiles
interpenetrate and every one runs through the whole cloth.

## Divergences from PyCuTe

Python tells a tuple from a leaf at run time. Rust needs a type, so
`HTuple<T>` makes the two cases variants of one enum.

Python raises. This crate returns `Result`.

`ArithTuple` stores its children as PyCuTe stores them, and its equality
extends trailing positions by zero. `Hash` therefore hashes a trimmed
form, so that equal values hash alike.

PyCuTe tells a concrete integer from a symbolic one with `is_static`, and
guards its stride orderings with it. `Int` is the only integer here, so
`is_static` would always be true. The crate omits it, and those orderings
collapse to a plain — but stable — sort.

PyCuTe's `layout.py` and `algebra.py` import each other: `_composition`
reaches for `algebra.layout_add`, and `_logical_divide` for
`algebra.complement`. Rust modules cannot circle that way, so the
algorithms all live in `layout.rs`, where `layout.py` puts them, and
`algebra.rs` is the thin free-function facade over them. `layout_add` and
`greatest_common_domain` are the exception: they are `algebra.py`
functions that `layout.rs` calls, so the dependency runs
`layout.rs → algebra.rs` and stays one-way.

## State

| module                          | PyCuTe source | state                                                              |
|---------------------------------|---------------|--------------------------------------------------------------------|
| `htuple`                        | `htuple.py`   | ported                                                             |
| `atuple`                        | `atuple.py`   | ported                                                             |
| `shape`                         | `shape.py`    | ported                                                             |
| `stride`                        | `stride.py`   | ported                                                             |
| `layout`                        | `layout.py`   | ported                                                             |
| `algebra`                       | `algebra.py`  | ported, less `zipped_divide` / `blocked_product` / `raked_product` |
| `swizzle`, `accessor`, `tensor` | —             | not planned                                                        |

`tensor.py` and `accessor.py` are out of scope on purpose. This crate is
the layout algebra — shapes, strides, and the operations over them. It
carries no data and addresses no memory.

## Licence

Apache-2.0. See `LICENSE` and `NOTICE`.

CuTe and CUTLASS are names of NVIDIA Corporation. This project is not
affiliated with NVIDIA. It uses those names only to state what it is a
port of.

[PyCuTe]: https://github.com/NVIDIA/cutlass
