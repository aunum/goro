package dense

import (
	"fmt"

	"gorgonia.org/tensor"
)

// ToF32 will attempt to cast the given tensor int values to float32 vals.
func ToF32(t *tensor.Dense) (*tensor.Dense, error) {
	new := tensor.New(tensor.WithShape(t.Shape()...), tensor.Of(tensor.Float32))
	iterator := t.Iterator()
	for i, err := iterator.Next(); err == nil; i, err = iterator.Next() {
		v := t.Get(i)

		switch a := v.(type) {
		case int:
			new.Set(i, float32(a))
		case int8:
			new.Set(i, float32(a))
		case int32:
			new.Set(i, float32(a))
		case int64:
			new.Set(i, float32(a))
		case uint:
			new.Set(i, float32(a))
		case uint8:
			new.Set(i, float32(a))
		case uint16:
			new.Set(i, float32(a))
		case uint32:
			new.Set(i, float32(a))
		case uint64:
			new.Set(i, float32(a))
		case float32:
			return t, nil
		case float64:
			new.Set(i, float32(a))
		default:
			return nil, fmt.Errorf("could not cast type: %v", t.Dtype())
		}
	}
	return new, nil
}

// SizeAsDType returns the size of a tensor as a tensor along the axis with the same dtype.
func SizeAsDType(x *tensor.Dense, along ...int) (size *tensor.Dense, err error) {
	if len(along) == 0 {
		along = []int{0}
	}
	axis := along[0]

	switch x.Dtype() {
	case tensor.Float32:
		size = tensor.New(tensor.WithBacking([]float32{float32(x.Shape()[axis])}))
	case tensor.Float64:
		size = tensor.New(tensor.WithBacking([]float64{float64(x.Shape()[axis])}))
	case tensor.Int:
		size = tensor.New(tensor.WithBacking([]int{int(x.Shape()[axis])}))
	case tensor.Int8:
		size = tensor.New(tensor.WithBacking([]int8{int8(x.Shape()[axis])}))
	case tensor.Int32:
		size = tensor.New(tensor.WithBacking([]int32{int32(x.Shape()[axis])}))
	case tensor.Int64:
		size = tensor.New(tensor.WithBacking([]int64{int64(x.Shape()[axis])}))
	case tensor.Uint:
		size = tensor.New(tensor.WithBacking([]uint{uint(x.Shape()[axis])}))
	case tensor.Uint8:
		size = tensor.New(tensor.WithBacking([]uint8{uint8(x.Shape()[axis])}))
	case tensor.Uint16:
		size = tensor.New(tensor.WithBacking([]uint16{uint16(x.Shape()[axis])}))
	case tensor.Uint32:
		size = tensor.New(tensor.WithBacking([]uint32{uint32(x.Shape()[axis])}))
	case tensor.Uint64:
		size = tensor.New(tensor.WithBacking([]uint64{uint64(x.Shape()[axis])}))
	default:
		return nil, fmt.Errorf("cannot return size as tensor dtype %v; not implemented", x.Dtype())
	}
	return
}
