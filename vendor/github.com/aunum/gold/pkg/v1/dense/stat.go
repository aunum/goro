package dense

import (
	"fmt"
	"math/rand"
	"reflect"
	"time"

	t "gorgonia.org/tensor"
)

// Mean of the tensor along the axis.
//
// y=Σx/n
func Mean(x *t.Dense, along ...int) (*t.Dense, error) {
	if len(along) == 0 {
		along = []int{0}
	}
	axis := along[0]
	sum, err := x.Sum(axis)
	if err != nil {
		return nil, err
	}
	if len(x.Shape()) < axis {
		return nil, fmt.Errorf("tensor shape %v does not contain the axis %v", x.Shape(), along)
	}

	size, err := SizeAsDType(x, along...)
	if err != nil {
		return nil, err
	}
	mean, err := sum.Div(size)
	if err != nil {
		return nil, err
	}
	return mean, nil
}

// StdDev is the standard deviation of the tensor along the axis.
//
// y=√(Σ((x-μ)^2)/n)
func StdDev(x *t.Dense, along ...int) (*t.Dense, error) {
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
	distance, err := x.Sub(mus.(*t.Dense))
	if err != nil {
		return nil, err
	}

	abs, err := t.Square(distance)
	if err != nil {
		return nil, err
	}
	sum, err := abs.(*t.Dense).Sum(0)
	if err != nil {
		return nil, err
	}

	size, err := SizeAsDType(x, along...)
	if err != nil {
		return nil, err
	}
	inner, err := sum.Div(size)
	if err != nil {
		return nil, err
	}
	ret, err := t.Sqrt(inner)
	if err != nil {
		return nil, err
	}

	return ret.(*t.Dense), nil
}

// RandN generates a new dense tensor of the given shape with
// values populated from the standard normal distribution.
func RandN(dt t.Dtype, shape ...int) *t.Dense {
	size := t.Shape(shape).TotalSize()
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	switch dt.Kind() {
	case reflect.Int:
		backing := make([]int, size)
		for i := range backing {
			backing[i] = int(r.NormFloat64())
		}
		return t.New(t.WithShape(shape...), t.WithBacking(backing))
	case reflect.Int8:
		backing := make([]int8, size)
		for i := range backing {
			backing[i] = int8(r.NormFloat64())
		}
		return t.New(t.WithShape(shape...), t.WithBacking(backing))
	case reflect.Int16:
		backing := make([]int16, size)
		for i := range backing {
			backing[i] = int16(r.NormFloat64())
		}
		return t.New(t.WithShape(shape...), t.WithBacking(backing))
	case reflect.Int32:
		backing := make([]int32, size)
		for i := range backing {
			backing[i] = int32(r.NormFloat64())
		}
		return t.New(t.WithShape(shape...), t.WithBacking(backing))
	case reflect.Int64:
		backing := make([]int64, size)
		for i := range backing {
			backing[i] = int64(r.NormFloat64())
		}
		return t.New(t.WithShape(shape...), t.WithBacking(backing))
	case reflect.Uint:
		backing := make([]uint, size)
		for i := range backing {
			backing[i] = uint(r.Uint32())
		}
		return t.New(t.WithShape(shape...), t.WithBacking(backing))
	case reflect.Uint8:
		backing := make([]uint8, size)
		for i := range backing {
			backing[i] = uint8(r.Uint32())
		}
		return t.New(t.WithShape(shape...), t.WithBacking(backing))
	case reflect.Uint16:
		backing := make([]uint16, size)
		for i := range backing {
			backing[i] = uint16(r.Uint32())
		}
		return t.New(t.WithShape(shape...), t.WithBacking(backing))
	case reflect.Uint32:
		backing := make([]uint32, size)
		for i := range backing {
			backing[i] = uint32(r.Uint32())
		}
		return t.New(t.WithShape(shape...), t.WithBacking(backing))
	case reflect.Uint64:
		backing := make([]uint64, size)
		for i := range backing {
			backing[i] = uint64(r.Uint32())
		}
		return t.New(t.WithShape(shape...), t.WithBacking(backing))
	case reflect.Float32:
		backing := make([]float32, size)
		for i := range backing {
			backing[i] = float32(r.NormFloat64())
		}
		return t.New(t.WithShape(shape...), t.WithBacking(backing))
	case reflect.Float64:
		backing := make([]float64, size)
		for i := range backing {
			backing[i] = rand.NormFloat64()
		}
		return t.New(t.WithShape(shape...), t.WithBacking(backing))
	case reflect.Complex64:
		backing := make([]complex64, size)
		for i := range backing {
			backing[i] = complex(r.Float32(), float32(0))
		}
		return t.New(t.WithShape(shape...), t.WithBacking(backing))
	case reflect.Complex128:
		backing := make([]complex128, size)
		for i := range backing {
			backing[i] = complex(r.Float64(), float64(0))
		}
		return t.New(t.WithShape(shape...), t.WithBacking(backing))
	default:
		panic("unknown type")
	}
}
