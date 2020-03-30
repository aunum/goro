// +build sse avx

package vecf32

// Add performs a̅ + b̅. a̅ will be clobbered
func Add(a, b []float32) {
	if len(a) != len(b) {
		panic("vectors must be the same length")
	}
	addAsm(a, b)
}
func addAsm(a, b []float32)

// Sub performs a̅ - b̅. a̅ will be clobbered
func Sub(a, b []float32) {
	if len(a) != len(b) {
		panic("vectors must be the same length")
	}
	subAsm(a, b)
}
func subAsm(a, b []float32)

// Mul performs a̅ × b̅. a̅ will be clobbered
func Mul(a, b []float32) {
	if len(a) != len(b) {
		panic("vectors must be the same length")
	}
	mulAsm(a, b)
}
func mulAsm(a, b []float32)

// Div performs a̅ ÷ b̅. a̅ will be clobbered
func Div(a, b []float32) {
	if len(a) != len(b) {
		panic("vectors must be the same length")
	}
	divAsm(a, b)
}
func divAsm(a, b []float32)

// Sqrt performs √a̅ elementwise. a̅ will be clobbered
func Sqrt(a []float32)

// InvSqrt performs 1/√a̅ elementwise. a̅ will be clobbered
func InvSqrt(a []float32)

/*

func Pow(a, b []float32)
*/

/*
func Scale(s float32, a []float32)
func ScaleFrom(s float32, a []float32)
func Trans(s float32, a []float32)
func TransFrom(s float32, a []float32)
func Power(s float32, a []float32)
func PowerFrom(s float32, a []float32)
*/
