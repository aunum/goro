# hm [![GoDoc](https://godoc.org/github.com/chewxy/hm?status.svg)](https://godoc.org/github.com/chewxy/hm) [![Build Status](https://travis-ci.org/chewxy/hm.svg?branch=master)](https://travis-ci.org/chewxy/hm) [![Coverage Status](https://coveralls.io/repos/github/chewxy/hm/badge.png)](https://coveralls.io/github/chewxy/hm)

Package hm is a simple Hindley-Milner type inference system in Go. It provides the necessary data structures and functions for creating such a system. 

# Installation #

This package is go-gettable: `go get -u github.com/chewxy/hm`

There are very few dependencies that this package uses. Therefore there isn't a need for vendoring tools. However, package hm DOES provide a `Gopkg.toml` and `Gopkg.lock` for any potential users of the [dep](https://github.com/golang/dep) tool.

Here is a listing of the dependencies of `hm`:

|Package|Used For|Vitality|Notes|Licence|
|-------|--------|--------|-----|-------|
|[errors](https://github.com/pkg/errors)|Error wrapping|Can do without, but this is by far the superior error solution out there|Stable API for the past 6 months|[errors licence](https://github.com/pkg/errors/blob/master/LICENSE) (MIT/BSD-like)|
|[testify/assert](https://github.com/stretchr/testify)|Testing|Can do without but will be a massive pain in the ass to test||[testify licence](https://github.com/stretchr/testify/blob/master/LICENSE) (MIT/BSD-like)|

# Usage

TODO: Write this

# Notes

This package is used by [Gorgonia](https://github.com/chewxy/gorgonia) as part of the graph building process. It is also used by several other internal projects of this author, all sharing a similar theme of requiring a type system, which is why this was abstracted out.


# Contributing

This library is developed using Github. Therefore the workflow is very github-centric. 

# Licence

Package `hm` is licenced under the MIT licence.
