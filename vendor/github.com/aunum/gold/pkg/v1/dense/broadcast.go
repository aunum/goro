// Package dense provides methods for Gorgonia's Dense Tensors.
package dense

import (
	"fmt"

	"gorgonia.org/tensor"
)

// BroadcastAdd adds 'a' to 'b' element-wise using broadcasting rules.
//
// Shapes are compared element-wise starting with trailing dimensions and working its
// way forward.
// Dimensions are compatible if:
// - they are equal
// - one of them is 1
func BroadcastAdd(a, b *tensor.Dense) (retVal *tensor.Dense, err error) {
	err = matchShape(a, b)
	if err != nil {
		return
	}
	return a.Add(b)
}

// BroadcastSub subtracts 'a' from 'b' element-wise using broadcasting rules.
//
// Shapes are compared element-wise starting with trailing dimensions and working its
// way forward.
// Dimensions are compatible if:
// - they are equal
// - one of them is 1
func BroadcastSub(a, b *tensor.Dense) (retVal *tensor.Dense, err error) {
	err = matchShape(a, b)
	if err != nil {
		return
	}
	return a.Sub(b)
}

// BroadcastMul multiplies 'a' to 'b' element-wise using broadcasting rules.
//
// Shapes are compared element-wise starting with trailing dimensions and working its
// way forward.
// Dimensions are compatible if:
// - they are equal
// - one of them is 1
func BroadcastMul(a, b *tensor.Dense) (retVal *tensor.Dense, err error) {
	err = matchShape(a, b)
	if err != nil {
		return
	}
	return a.Mul(b)
}

// BroadcastDiv safely divides 'a' to 'b' element-wise using broadcasting rules.
// Any zero values in 'b' will be slightly augmented.
//
// Shapes are compared element-wise starting with trailing dimensions and working its
// way forward.
// Dimensions are compatible if:
// - they are equal
// - one of them is 1
func BroadcastDiv(a, b *tensor.Dense) (retVal *tensor.Dense, err error) {
	err = matchShape(a, b)
	if err != nil {
		return
	}
	return Div(a, b)
}

// matches the shape using broadcast rules.
func matchShape(a, b *tensor.Dense) error {
	err := normalizeShape(a, b)
	if err != nil {
		return err
	}
	for i := a.Dims() - 1; i >= 0; i-- {
		ai := a.Shape()[i]
		bi := b.Shape()[i]
		if ai == bi {
			continue
		}
		if ai == 1 {
			ar, err := a.Repeat(i, bi)
			if err != nil {
				return err
			}
			*a = *ar.(*tensor.Dense)
		} else if bi == 1 {
			br, err := b.Repeat(i, ai)
			if err != nil {
				return err
			}
			*b = *br.(*tensor.Dense)
		} else {
			return fmt.Errorf("a shape %v b shape %v are incompatible", a.Shape(), b.Shape())
		}
	}
	return nil
}

// normalizes the shapes to be of equal dimensions.
func normalizeShape(a, b *tensor.Dense) (err error) {
	if a.Dims() == b.Dims() {
		return
	}
	if a.Dims() > b.Dims() {
		err = expandDimsTo(b, a)
	} else {
		err = expandDimsTo(a, b)
	}
	return
}

// expand dims of 'a' to 'b'.
func expandDimsTo(a, b *tensor.Dense) error {
	diff := b.Dims() - a.Dims()
	shape := []int{}
	for i := 0; i < diff; i++ {
		shape = append(shape, 1)
	}
	shape = append(shape, a.Shape()...)
	return a.Reshape(shape...)
}
