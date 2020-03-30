package vecf64

import "math"

// IncrAdd performs a̅ + b̅ and then adds it elementwise to the incr slice
func IncrAdd(a, b, incr []float64) {
	b = b[:len(a)]
	incr = incr[:len(a)]
	for i, v := range a {
		incr[i] += v + b[i]
	}
}

// IncrSub performs a̅ = b̅ and then adds it elementwise to the incr slice
func IncrSub(a, b, incr []float64) {
	b = b[:len(a)]
	incr = incr[:len(a)]
	for i, v := range a {
		incr[i] += v - b[i]
	}
}

// IncrMul performs a̅ × b̅ and then adds it elementwise to the incr slice
func IncrMul(a, b, incr []float64) {
	b = b[:len(a)]
	incr = incr[:len(a)]
	for i, v := range a {
		incr[i] += v * b[i]
	}
}

func IncrDiv(a, b, incr []float64) {
	b = b[:len(a)]
	incr = incr[:len(a)]
	for i, v := range a {
		if b[i] == 0 {
			incr[i] = math.Inf(0)
			continue
		}
		incr[i] += v / b[i]
	}
}

// IncrDiv performs a̅ ÷ b̅. a̅ will be clobbered
func IncrPow(a, b, incr []float64) {
	b = b[:len(a)]
	incr = incr[:len(a)]
	for i, v := range a {
		switch b[i] {
		case 0:
			incr[i]++
		case 1:
			incr[i] += v
		case 2:
			incr[i] += v * v
		case 3:
			incr[i] += v * v * v
		default:
			incr[i] += math.Pow(v, b[i])
		}
	}
}

// IncrMod performs a̅ % b̅ then adds it to incr
func IncrMod(a, b, incr []float64) {
	b = b[:len(a)]
	incr = incr[:len(a)]

	for i, v := range a {
		incr[i] += math.Mod(v, b[i])
	}
}

// Scale multiplies all values in the slice by the scalar and then increments the incr slice
// 		incr += a̅ * s
func IncrScale(a []float64, s float64, incr []float64) {
	incr = incr[:len(a)]
	for i, v := range a {
		incr[i] += v * s
	}
}

// IncrScaleInv divides all values in the slice by the scalar and then increments the incr slice
// 		incr += a̅ / s
func IncrScaleInv(a []float64, s float64, incr []float64) {
	IncrScale(a, 1/s, incr)
}

/// IncrScaleInvR divides all numbers in the slice by a scalar and then increments the incr slice
// 		incr += s / a̅
func IncrScaleInvR(a []float64, s float64, incr []float64) {
	incr = incr[:len(a)]
	for i, v := range a {
		incr[i] += s / v
	}
}

// IncrTrans adds all the values in the slice by a scalar and then increments the incr slice
// 		incr += a̅ + s
func IncrTrans(a []float64, s float64, incr []float64) {
	incr = incr[:len(a)]
	for i, v := range a {
		incr[i] += v + s
	}
}

// IncrTransInv subtracts all the values in the slice by a scalar and then increments the incr slice
//		incr += a̅ - s
func IncrTransInv(a []float64, s float64, incr []float64) {
	IncrTrans(a, -s, incr)
}

// IncrTransInvR subtracts all the numbers in a slice from a scalar and then increments the incr slice
//	 incr += s - a̅
func IncrTransInvR(a []float64, s float64, incr []float64) {
	incr = incr[:len(a)]
	for i, v := range a {
		incr[i] += s - v
	}
}

// IncrPowOf performs elementwise power function and then increments the incr slice
//		incr += a̅ ^ s
func IncrPowOf(a []float64, s float64, incr []float64) {
	incr = incr[:len(a)]
	for i, v := range a {
		incr[i] += math.Pow(v, s)
	}
}

// PowOfR performs elementwise power function below and then increments the incr slice.
//		incr += s ^ a̅
func IncrPowOfR(a []float64, s float64, incr []float64) {
	incr = incr[:len(a)]
	for i, v := range a {
		incr[i] += math.Pow(s, v)
	}
}
