// +build !avx,!sse

package vecf64

import "math"

// Add performs a̅ + b̅. a̅ will be clobbered
func Add(a, b []float64) {
	b = b[:len(a)]
	for i, v := range a {
		a[i] = v + b[i]
	}
}

// Sub performs a̅ - b̅. a̅ will be clobbered
func Sub(a, b []float64) {
	b = b[:len(a)]
	for i, v := range a {
		a[i] = v - b[i]
	}
}

// Mul performs a̅ × b̅. a̅ will be clobbered
func Mul(a, b []float64) {
	b = b[:len(a)]
	for i, v := range a {
		a[i] = v * b[i]
	}
}

// Div performs a̅ ÷ b̅. a̅ will be clobbered
func Div(a, b []float64) {
	b = b[:len(a)]
	for i, v := range a {
		if b[i] == 0 {
			a[i] = math.Inf(0)
			continue
		}

		a[i] = v / b[i]
	}
}

// Sqrt performs √a̅ elementwise. a̅ will be clobbered
func Sqrt(a []float64) {
	for i, v := range a {
		a[i] = math.Sqrt(v)
	}
}

// InvSqrt performs 1/√a̅ elementwise. a̅ will be clobbered
func InvSqrt(a []float64) {
	for i, v := range a {
		a[i] = float64(1) / math.Sqrt(v)
	}
}
