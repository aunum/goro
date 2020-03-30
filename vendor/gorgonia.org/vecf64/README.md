# vecf64  [![GoDoc](https://godoc.org/gorgonia.org/vecf64?status.svg)](https://godoc.org/gorgonia.org/vecf64) [![Build Status](https://travis-ci.org/gorgonia/vecf64.svg?branch=master)](https://travis-ci.org/gorgonia/vecf64) [![Coverage Status](https://coveralls.io/repos/github/gorgonia/vecf64/badge.svg?branch=master)](https://coveralls.io/github/gorgonia/vecf64?branch=master)

Package vecf64 provides common functions and methods for slices of float64

# Installation

`go get -u gorgonia.org/vecf64`

This package uses the standard library only. For testing this package uses [testify/assert](https://github.com/stretchr/testify), which is licenced with a [MIT/BSD-like licence](https://github.com/stretchr/testify/blob/master/LICENSE)

# Build Tags

The point of this package is to provide operations that are accelerated by SIMD. However, this pakcage by default does not use SIMD. To use SIMD, build tags must be used. The supported build tags are `sse` and `avx`. Here's an example on how to use them:

* [SSE](https://en.wikipedia.org/wiki/Streaming_SIMD_Extensions) - `go build -tags='sse' ...
* [AVX](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions) - `go build -tags='avx' ...

### Why are there so many `a = a[:len(a)]` lines?
This is mainly done to eliminate bounds checking in a loop. The idea is the bounds of the slice is checked early on, and if need be, panics early. Then if everything is normal, there won't be bounds checking while in the loop.

To check for boundschecking and bounds check elimination (an amazing feature that landed in Go 1.7), compile your programs with `-gcflags='-d=ssa/check_bce/debug=1'`. 

# Contributing

Contributions are welcome. The typical process works like this:

1. File an issue  on the topic you want to contribute
2. Fork this repo
3. Add your contribution
4. Make a pull request
5. The pull request will be merged once tests pass, and code reviewed.
6. Add your name (if it hasn't already been added to CONTRIBUTORS.md)

## Pull Requests

This package is very well tested. Please ensure tests are written if any new features are added. If bugs are fixed, please add the bugs to the tests as well.

# Licence

Package vecf64 is licenced under the MIT licence.