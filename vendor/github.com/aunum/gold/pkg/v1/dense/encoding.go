package dense

import (
	"fmt"

	"gorgonia.org/tensor"
	t "gorgonia.org/tensor"
)

// OneHotVector creates a one hot vector for the given id within the number of classses.
//
// Note: this is mostly taken from gorgonia core, but was needed as a basic tensor function, should be
// contributed back upstream.
func OneHotVector(id, classes int, dt t.Dtype) (retVal *t.Dense, err error) {
	retVal = t.New(t.Of(dt), t.WithShape(classes))

	switch dt {
	case tensor.Float32:
		err = retVal.SetAt(float32(1), id)
	case tensor.Float64:
		err = retVal.SetAt(float64(1), id)
	case tensor.Int:
		err = retVal.SetAt(int(1), id)
	case tensor.Int32:
		err = retVal.SetAt(int32(1), id)
	case tensor.Int64:
		err = retVal.SetAt(int64(1), id)
	default:
		return nil, fmt.Errorf("tensor dtype %v is not supported for one hot vector", dt)
	}
	if err != nil {
		return nil, err
	}
	return
}
