package vecf64

// Range is a function to create arithmetic progressions of float32
func Range(start, end int) []float64 {
	size := end - start
	incr := true
	if start > end {
		incr = false
		size = start - end
	}

	if size < 0 {
		panic("Cannot create a float range that is negative in size")
	}

	r := make([]float64, size)
	for i, v := 0, float64(start); i < size; i++ {
		r[i] = v

		if incr {
			v++
		} else {
			v--
		}
	}
	return r
}

// Reduce takes a function to reduce by, a defalut, and a splatted list of float64s
func Reduce(f func(a, b float64) float64, def float64, l ...float64) (retVal float64) {
	retVal = def
	if len(l) == 0 {
		return
	}

	for _, v := range l {
		retVal = f(retVal, v)
	}
	return
}
