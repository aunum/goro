// +build !avx,!sse

package vecf32

import "github.com/chewxy/math32"

// Add performs a̅ + b̅. a̅ will be clobbered
func Add(a, b []float32) {
	b = b[:len(a)]
	for i, v := range a {
		a[i] = v + b[i]
	}
}

// Sub performs a̅ - b̅. a̅ will be clobbered
func Sub(a, b []float32) {
	b = b[:len(a)]
	for i, v := range a {
		a[i] = v - b[i]
	}
}

// Mul performs a̅ × b̅. a̅ will be clobbered
func Mul(a, b []float32) {
	b = b[:len(a)]
	for i, v := range a {
		a[i] = v * b[i]
	}
}

// Div performs a̅ ÷ b̅. a̅ will be clobbered
func Div(a, b []float32) {
	b = b[:len(a)]
	for i, v := range a {
		if b[i] == 0 {
			a[i] = math32.Inf(0)
			continue
		}

		a[i] = v / b[i]
	}
}

// Sqrt performs √a̅ elementwise. a̅ will be clobbered
func Sqrt(a []float32) {
	for i, v := range a {
		a[i] = math32.Sqrt(v)
	}
}

// InvSqrt performs 1/√a̅ elementwise. a̅ will be clobbered
func InvSqrt(a []float32) {
	for i, v := range a {
		a[i] = float32(1) / math32.Sqrt(v)
	}
}
