# pinstripe

A Rust transliteration of the CuTe layout algebra.

The source is [PyCuTe]. Each module here mirrors one PyCuTe module, with
the same names, the same decomposition, and the same algorithms, so you
can check a file against its Python counterpart line by line. The test
suite is a port of the PyCuTe test suite.

A layout is `Shape : Stride`. Both halves are hierarchical integer
tuples. A layout maps a coordinate to an offset. Give it basis strides
instead, and it maps a coordinate to a coordinate — that is an identity
tensor, and it is what makes this algebra useful for describing how one
iteration space reads another.

## What this crate does not do

- **No row-major variant.** The algebra is colexicographic, as CuTe
  defines it. A caller that wants row-major order converts at its own
  boundary.
- **No axis names.** A layout is positional. Names are a labelling, and a
  labelling belongs to the caller.

## Divergences from PyCuTe

Python tells a tuple from a leaf at run time. Rust needs a type, so
`HTuple<T>` makes the two cases variants of one enum.

Python raises. This crate returns `Result`.

`ArithTuple` stores its children as PyCuTe stores them, and its equality
extends trailing positions by zero. `Hash` therefore hashes a trimmed
form, so that equal values hash alike.

## State

| module | PyCuTe source | state |
|---|---|---|
| `htuple` | `htuple.py` | ported |
| `atuple` | `atuple.py` | ported |
| `shape` | `shape.py` | to do |
| `stride` | `stride.py` | to do |
| `layout` | `layout.py` | to do |
| `algebra` | `algebra.py` | to do |
| `swizzle`, `accessor`, `tensor` | — | not planned |

## Licence

Apache-2.0. See `LICENSE` and `NOTICE`.

CuTe and CUTLASS are names of NVIDIA Corporation. This project is not
affiliated with NVIDIA. It uses those names only to state what it is a
port of.

[PyCuTe]: https://github.com/NVIDIA/cutlass
