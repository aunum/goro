// Package hm provides a Hindley-Milner type and type inference system.
//
// If you are creating a new programming language and you'd like it to be
// strongly typed with parametric polymorphism (or just have Haskell-envy),
// this library provides the necessary types and functions for creating such a system.
//
// The key to the HM type inference system is in the Unify() function.

package hm

const letters = `abcdefghijklmnopqrstuvwxyz`
