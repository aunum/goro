package dense

import (
	"fmt"

	t "gorgonia.org/tensor"
)

// MinMaxNorm normalizes the input x using pointwise min-max normalization along the axis.
// This is a pointwise operation and requires the shape of the min/max tensors be equal to x.
//
// y=(x-min)/(max-min)
func MinMaxNorm(x, min, max *t.Dense) (*t.Dense, error) {
	if !min.Shape().Eq(x.Shape()) {
		return nil, fmt.Errorf("min shape %v must match x shape %v", min.Shape(), x.Shape())
	}
	if !max.Shape().Eq(x.Shape()) {
		return nil, fmt.Errorf("max shape %v must match x shape %v", max.Shape(), x.Shape())
	}

	ret, err := x.Sub(min)
	if err != nil {
		return nil, err
	}
	space, err := max.Sub(min)
	if err != nil {
		return nil, err
	}
	ret, err = ret.Div(space)
	if err != nil {
		return nil, err
	}
	return ret, nil
}

// ZNorm normalizes x using z-score normalization along the axis.
//
// y=x-μ/σ
func ZNorm(x *t.Dense, along ...int) (*t.Dense, error) {
	if len(along) == 0 {
		along = []int{0}
	}
	axis := along[0]
	mu, err := Mean(x, axis)
	if err != nil {
		return nil, err
	}
	mus, err := mu.Repeat(0, x.Shape()[axis])
	if err != nil {
		return nil, err
	}
	ret, err := x.Sub(mus.(*t.Dense))
	if err != nil {
		return nil, err
	}
	sigma, err := StdDev(x, axis)
	if err != nil {
		return nil, err
	}
	sigmas, err := sigma.Repeat(0, x.Shape()[axis])
	if err != nil {
		return nil, err
	}
	ret, err = Div(ret, sigmas.(*t.Dense))
	if err != nil {
		return nil, err
	}
	return ret, nil
}

// NormalizeZeros normalizes the zero values.
func NormalizeZeros(d *t.Dense) error {
	contains, indicies := Contains(d, ZeroValue(d.Dtype()))
	if !contains {
		return nil
	}
	for _, i := range indicies {
		d.Set(i, FauxZeroValue(d.Dtype()))
	}
	return nil
}
