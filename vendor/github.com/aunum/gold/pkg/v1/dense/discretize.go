package dense

import (
	"fmt"

	"gorgonia.org/tensor"
)

// EqWidthBinner bins values within diminsions to the given intervals.
type EqWidthBinner struct {
	// Intervals to bin.
	Intervals *tensor.Dense

	// Low values are the lower bounds.
	Low *tensor.Dense

	// High values are the upper bounds.
	High *tensor.Dense

	widths *tensor.Dense
	bounds []*tensor.Dense
}

// NewEqWidthBinner creates a new Equal Width Binner.
func NewEqWidthBinner(intervals, low, high *tensor.Dense) (*EqWidthBinner, error) {
	var err error

	// make types homogenous
	low, err = ToF32(low)
	if err != nil {
		return nil, err
	}
	high, err = ToF32(high)
	if err != nil {
		return nil, err
	}
	intervals, err = ToF32(intervals)
	if err != nil {
		return nil, err
	}

	// width = (max - min)/n
	spread, err := high.Sub(low)
	if err != nil {
		return nil, err
	}
	widths, err := spread.Div(intervals)
	if err != nil {
		return nil, err
	}
	var bounds []*tensor.Dense
	iterator := widths.Iterator()
	for i, err := iterator.Next(); err == nil; i, err = iterator.Next() {
		interval := intervals.GetF32(i)
		l := low.GetF32(i)
		width := widths.GetF32(i)
		backing := []float32{l}
		for j := 0; j <= int(interval); j++ {
			backing = append(backing, backing[j]+width)
		}
		bound := tensor.New(tensor.WithShape(1, len(backing)), tensor.WithBacking(backing))
		bounds = append(bounds, bound)
	}
	return &EqWidthBinner{
		Intervals: intervals,
		Low:       low,
		High:      high,
		widths:    widths,
		bounds:    bounds,
	}, nil
}

// Bin the values.
func (d *EqWidthBinner) Bin(values *tensor.Dense) (*tensor.Dense, error) {
	iterator := values.Iterator()
	backing := []int{}
	for i, err := iterator.Next(); err == nil; i, err = iterator.Next() {
		v := values.GetF32(i)
		bounds := d.bounds[i]
		bIter := bounds.Iterator()
		for j, err := bIter.Next(); err == nil; j, err = bIter.Next() {
			if v < bounds.GetF32(0) {
				return nil, fmt.Errorf("could not bin %v, out of range %v", v, bounds)
			}
			if v < bounds.GetF32(j) {
				backing = append(backing, j)
				break
			}
		}
		if len(backing) <= i {
			return nil, fmt.Errorf("could not bin %v, out of range %v", v, bounds)
		}
	}
	binned := tensor.New(tensor.WithShape(values.Shape()...), tensor.WithBacking(backing))
	return binned, nil
}

// Widths used in binning.
func (d *EqWidthBinner) Widths() *tensor.Dense {
	return d.widths
}

// Bounds used in binning.
func (d *EqWidthBinner) Bounds() []*tensor.Dense {
	return d.bounds
}
