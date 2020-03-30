// +build sse avx

package vecf64

// Add performs a̅ + b̅. a̅ will be clobbered
func Add(a, b []float64) {
	if len(a) != len(b) {
		panic("vectors must be the same length")
	}
	addAsm(a, b)
}
func addAsm(a, b []float64)

// Sub performs a̅ - b̅. a̅ will be clobbered
func Sub(a, b []float64) {
	if len(a) != len(b) {
		panic("vectors must be the same length")
	}
	subAsm(a, b)
}
func subAsm(a, b []float64)

// Mul performs a̅ × b̅. a̅ will be clobbered
func Mul(a, b []float64) {
	if len(a) != len(b) {
		panic("vectors must be the same length")
	}
	mulAsm(a, b)
}
func mulAsm(a, b []float64)

// Div performs a̅ ÷ b̅. a̅ will be clobbered
func Div(a, b []float64) {
	if len(a) != len(b) {
		panic("vectors must be the same length")
	}
	divAsm(a, b)
}
func divAsm(a, b []float64)

// Sqrt performs √a̅ elementwise. a̅ will be clobbered
func Sqrt(a []float64)

// InvSqrt performs 1/√a̅ elementwise. a̅ will be clobbered
func InvSqrt(a []float64)

/*
func Pow(a, b []float64)
*/

/*
func Scale(s float64, a []float64)
func ScaleFrom(s float64, a []float64)
func Trans(s float64, a []float64)
func TransFrom(s float64, a []float64)
func Power(s float64, a []float64)
func PowerFrom(s float64, a []float64)
*/
