package dense

import (
	"gorgonia.org/tensor"
)

// ExpandDims expands the dimensions of a tensor along the given axis.
func ExpandDims(t *tensor.Dense, axis int) error {
	dims := []int{}
	if axis == 0 {
		dims = append(dims, 1)
		dims = append(dims, t.Shape()...)

	} else {
		dims = append(dims, t.Shape()...)
		dims = append(dims, 1)
	}
	err := t.Reshape(dims...)
	return err
}

// Squeeze the tensor removing any dimensions of size 1.
func Squeeze(t *tensor.Dense) error {
	return t.Reshape(SqueezeShape(t.Shape())...)
}

// SqueezeShape removes any dimensions of size 1.
func SqueezeShape(shape tensor.Shape) tensor.Shape {
	newShape := []int{}
	for _, size := range shape {
		if size != 1 {
			newShape = append(newShape, size)
		}
	}
	return newShape
}

// OneOfMany ensures the given tensor starts with a shape of 1.
func OneOfMany(t *tensor.Dense) error {
	if t.Shape()[0] != 1 {
		return ExpandDims(t, 0)
	}
	return nil
}

// ManyOfOne ensures the given tensor ends with a shape of 1.
func ManyOfOne(t *tensor.Dense) error {
	if t.Shape()[len(t.Shape())-1] != 1 {
		return ExpandDims(t, 1)
	}
	return nil
}

// Repeat the value along the axis for the given number of repeats.
func Repeat(t *tensor.Dense, axis int, repeats ...int) (*tensor.Dense, error) {
	v, err := t.Repeat(axis, repeats...)
	if err != nil {
		return nil, err
	}
	return v.(*tensor.Dense), nil
}
