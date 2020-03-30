package dense

// Note: these are taken from gorgonias core, because they aren't currently exported.

// RangedSlice is a ranged slice that implements the Slice interface.
type RangedSlice struct {
	start, end, step int
}

// Start of the slice.
func (s RangedSlice) Start() int { return s.start }

// End of the slice.
func (s RangedSlice) End() int { return s.end }

// Step of slice.
func (s RangedSlice) Step() int { return s.step }

// MakeRangedSlice creates a ranged slice. It takes an optional step param.
func MakeRangedSlice(start, end int, opts ...int) RangedSlice {
	step := 1
	if len(opts) > 0 {
		step = opts[0]
	}
	return RangedSlice{
		start: start,
		end:   end,
		step:  step,
	}
}

// SingleSlice is a single slice, representing this: [start:start+1:0]
type SingleSlice int

// Start of slice.
func (s SingleSlice) Start() int { return int(s) }

// End of the slice.
func (s SingleSlice) End() int { return int(s) + 1 }

// Step of slice.
func (s SingleSlice) Step() int { return 0 }
