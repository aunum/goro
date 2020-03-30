package vecf32

// Range is a function to create arithmetic progressions of float32
func Range(start, end int) []float32 {
	size := end - start
	incr := true
	if start > end {
		incr = false
		size = start - end
	}

	if size < 0 {
		panic("Cannot create a float range that is negative in size")
	}

	r := make([]float32, size)
	for i, v := 0, float32(start); i < size; i++ {
		r[i] = v

		if incr {
			v++
		} else {
			v--
		}
	}
	return r
}

// Reduce takes a function to reduce by, a defalut, and a splatted list of float32s
func Reduce(f func(a, b float32) float32, def float32, l ...float32) (retVal float32) {
	retVal = def
	if len(l) == 0 {
		return
	}

	for _, v := range l {
		retVal = f(retVal, v)
	}
	return
}
