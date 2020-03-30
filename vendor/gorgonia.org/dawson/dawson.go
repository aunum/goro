package dawson

import (
	"fmt"
	"math"
	"math/cmplx"

	"github.com/chewxy/math32"
)

// ToleranceF64 is a test to see if two float64s, a and b are equal,
// within the specified tolerance e.
// 		a: actual value
//		b: expected value
//		e: allowed errors (i.e. the values are within this range)
//
// This function was taken from tthe test files in the Go stdlib package math,
// which has the Go licence.
func ToleranceF64(a, b, e float64) bool {
	d := a - b
	if d < 0 {
		d = -d
	}

	// note: b is correct (expected) value, a is actual value.
	// make error tolerance a fraction of b, not a.
	if b != 0 {
		e = e * b
		if e < 0 {
			e = -e
		}
	}
	return d <= e
}

// TolereranceF32 is a test to see if two float64s, a and b are equal,
// within the specified tolerance e.
// 		a: actual value
//		b: expected value
//		e: allowed errors (i.e. the values are within this range)
//
// This function was adapted from the test files of the package github.com/chewxy/math32,
// which in turn was adapted from the test files of the Go stdlib package math,
// which has the Go licence.
func ToleranceF32(a, b, e float32) bool {
	d := a - b
	if d < 0 {
		d = -d
	}

	// note: b is correct (expected) value, a is actual value.
	// make error tolerance a fraction of b, not a.
	if b != 0 {
		e = e * b
		if e < 0 {
			e = -e
		}
	}
	return d <= e
}

// ToleranceC128 is a test to see if two float64s, a and b are equal,
// within the specified tolerance e.
// 		a: actual value
//		b: expected value
//		e: allowed errors (i.e. the values are within this range)
//
// NOTE: e is a float64, which will be used in the individual comparison
// of both real and imaginary components
//
// This function was adapted from the test files in the Go stdlib package math/cmplx,
// which has the Go licence.
func ToleranceC128(a, b complex128, e float64) bool {
	d := cmplx.Abs(a - b)
	if b != 0 {
		e = e * cmplx.Abs(b)
		if e < 0 {
			e = -e
		}
	}
	return d < e
}

// CloseEnoughF64 checks that a and b are within 1e-8 tolerance.
func CloseEnoughF64(a, b float64) bool { return ToleranceF64(a, b, 1e-8) }

// CloseF64 checks that a and b are within 1e-14 tolerance.
func CloseF64(a, b float64) bool { return ToleranceF64(a, b, 1e-14) }

// VeryCloseF64 checks that a and b are within 4e-16 tolerance.
func VeryCloseF64(a, b float64) bool { return ToleranceF64(a, b, 4e-16) }

// AlikeF64 checks that a and b are alike:
//		- NaNs are considered to be equal
//		- Both have the same sign bits
func AlikeF64(a, b float64) bool {
	switch {
	case math.IsNaN(a) && math.IsNaN(b):
		return true
	case a == b:
		return math.Signbit(a) == math.Signbit(b)
	}
	return false
}

// CloseF32 checks that a and b are within 1e-5 tolerance.
// The tolerance number gotten from the cfloat standard.
// By contrast, Haskell's Linear package uses 1e-6 for floats
func CloseF32(a, b float32) bool { return ToleranceF32(a, b, 1e-5) }

// VeryCloseF32 checks that a and b are within 1e-6 tolerance.
// This number was acquired from Haskell's linear package, as well as wikipedia
func VeryCloseF32(a, b float32) bool { return ToleranceF32(a, b, 1e-6) }

// AlikeF32 checks that a and b are alike:
//		- NaNs are considered to be equal
//		- Both have the same sign bits
func AlikeF32(a, b float32) bool {
	switch {
	case math32.IsNaN(a) && math32.IsNaN(b):
		return true
	case a == b:
		return math32.Signbit(a) == math32.Signbit(b)
	}
	return false
}

// CloseC128 checks that a and b are within 1e-14 tolerance
func CloseC128(a, b complex128) bool { return ToleranceC128(a, b, 1e-14) }

// VeryCloseC128 checks that a and b are within 1e-16 tolerance
func VeryCloseC128(a, b complex128) bool { return ToleranceC128(a, b, 4e-16) }

// AlikeC128 checks that a and b are alike:
//		- NaNs are considered to be equal
//		- Both have the same sign bits for both the real and imaginary components
func AlikeC128(a, b complex128) bool {
	switch {
	case cmplx.IsNaN(a) && cmplx.IsNaN(b):
		return true
	case a == b:
		return math.Signbit(real(a)) == math.Signbit(real(b)) &&
			math.Signbit(imag(a)) == math.Signbit(imag(b))
	}
	return false
}

// AllClose checks slices a and b are close together. An optional approximation function is accepted.
// If nothing is passed in, the CloseF64, CloseF32, CloseC128 functions will be used
//
// This is not an exhasutive function. It only recognizes these types:
//		[]float64
//		[]float32
//		[]complex64
//		[]complex128
//
// This function will panic if other types are passed in, or if a and b do not have matching types.
func AllClose(a, b interface{}, approxFn ...interface{}) bool {
	switch at := a.(type) {
	case []float64:
		closeness := CloseF64
		var ok bool
		if len(approxFn) > 0 {
			if closeness, ok = approxFn[0].(func(a, b float64) bool); !ok {
				closeness = CloseF64
			}
		}
		bt := b.([]float64)
		for i, v := range at {
			if math.IsNaN(v) {
				if !math.IsNaN(bt[i]) {
					return false
				}
				continue
			}
			if math.IsInf(v, 0) {
				if !math.IsInf(bt[i], 0) {
					return false
				}
				continue
			}
			if !closeness(v, bt[i]) {
				return false
			}
		}
		return true
	case []float32:
		closeness := CloseF32
		var ok bool
		if len(approxFn) > 0 {
			if closeness, ok = approxFn[0].(func(a, b float32) bool); !ok {
				closeness = CloseF32
			}
		}
		bt := b.([]float32)
		for i, v := range at {
			if math32.IsNaN(v) {
				if !math32.IsNaN(bt[i]) {
					return false
				}
				continue
			}
			if math32.IsInf(v, 0) {
				if !math32.IsInf(bt[i], 0) {
					return false
				}
				continue
			}
			if !closeness(v, bt[i]) {
				return false
			}
		}
		return true
	case []complex64:
		bt := b.([]complex64)
		for i, v := range at {
			if cmplx.IsNaN(complex128(v)) {
				if !cmplx.IsNaN(complex128(bt[i])) {
					return false
				}
				continue
			}
			if cmplx.IsInf(complex128(v)) {
				if !cmplx.IsInf(complex128(bt[i])) {
					return false
				}
				continue
			}
			if !ToleranceC128(complex128(v), complex128(bt[i]), 1e-5) {
				return false
			}
		}
		return true
	case []complex128:
		bt := b.([]complex128)
		for i, v := range at {
			if cmplx.IsNaN(v) {
				if !cmplx.IsNaN(bt[i]) {
					return false
				}
				continue
			}
			if cmplx.IsInf(v) {
				if !cmplx.IsInf(bt[i]) {
					return false
				}
				continue
			}
			if !CloseC128(v, bt[i]) {
				return false
			}
		}
		return true
	default:
		panic(fmt.Sprintf("Unable to perform AllClose on %T and %T", a, b))
		// return reflect.DeepEqual(a, b)
	}
}
