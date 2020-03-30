package num

import (
	"fmt"
)

// EqWidthBinner implements the equal width binning algorithm. This is used to
// discretize continuous spaces.
type EqWidthBinner struct {
	// NumIntervals is the number of bins.
	NumIntervals int

	// High is the upper bound of the continuous space.
	High float32

	// Low is the lower bound of the continuous space.
	Low float32

	width      float32
	boundaries []float32
}

// NewEqWidthBinner returns an EqWidthBin.
func NewEqWidthBinner(numIntervals int, high, low float32) *EqWidthBinner {
	width := (high - low) / float32(numIntervals)
	return &EqWidthBinner{
		NumIntervals: numIntervals,
		High:         high,
		Low:          low,
		width:        width,
	}
}

// Bin the given value.
func (e *EqWidthBinner) Bin(v float32) (int, error) {
	for i := 0; i <= e.NumIntervals; i++ {
		if v < e.Low {
			return 0, fmt.Errorf("v: %f not in range %f to %f", v, e.Low, e.High)
		}
		if v < (e.Low + (float32((i + 1)) * e.width)) {
			return i, nil
		}
	}
	return 0, fmt.Errorf("v: %f not in range %f to %f", v, e.Low, e.High)
}
