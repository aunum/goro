package dense

import (
	"fmt"

	"gorgonia.org/tensor"
	t "gorgonia.org/tensor"
)

// Div safely divides 'a' by 'b' by slightly augmenting any zero values in 'b'.
func Div(a, b *t.Dense) (*t.Dense, error) {
	err := NormalizeZeros(b)
	if err != nil {
		return nil, err
	}
	return a.Div(b)
}

// Neg negates the tensor elementwise.
func Neg(v *tensor.Dense) (*t.Dense, error) {
	ret, err := BroadcastMul(v, tensor.New(tensor.FromScalar(NegValue(v.Dtype()))))
	if err != nil {
		return nil, err
	}
	return ret, nil
}

// Contains checks if a tensor contains a value.
func Contains(d *t.Dense, val interface{}) (contains bool, indicies []int) {
	iterator := d.Iterator()
	for i, err := iterator.Next(); err == nil; i, err = iterator.Next() {
		if val == d.Get(i) {
			contains = true
			indicies = append(indicies, i)
		}
	}
	return
}

// AMax returns the maximum value in a tensor along an axis.
// TODO: support returning slice.
func AMax(d *tensor.Dense, axis int) (interface{}, error) {
	maxIndex, err := d.Argmax(axis)
	if err != nil {
		return nil, err
	}
	max := d.Get(maxIndex.GetI(0))
	return max, nil
}

// AMaxF32 returns the maximum value in a tensor along an axis as a float32.
func AMaxF32(d *tensor.Dense, axis int) (float32, error) {
	max, err := AMax(d, axis)
	if err != nil {
		return 0, err
	}
	f, ok := max.(float32)
	if !ok {
		return f, fmt.Errorf("could not cast %v to float32", max)
	}
	return f, nil
}

// Concat a list of tensors along a given axis.
func Concat(axis int, tensors ...*t.Dense) (retVal *t.Dense, err error) {
	if len(tensors) == 0 {
		return nil, fmt.Errorf("no tensors provided to concat")
	}
	retVal = tensors[0]
	retVal, err = retVal.Concat(axis, tensors[1:]...)
	return
}

// ConcatOr return b if a is nil.
func ConcatOr(axis int, a, b *t.Dense) (retVal *t.Dense, err error) {
	if a == nil {
		return b, nil
	}
	return a.Concat(axis, b)
}
