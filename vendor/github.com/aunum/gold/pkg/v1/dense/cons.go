package dense

import (
	"reflect"

	t "gorgonia.org/tensor"
)

// Fill creates a tensor and fills it with the given value.
func Fill(val interface{}, shape ...int) *t.Dense {
	size := t.Shape(shape).TotalSize()
	switch v := val.(type) {
	case int:
		backing := make([]int, size)
		for i := range backing {
			backing[i] = v
		}
		return t.New(t.WithShape(shape...), t.WithBacking(backing))
	case int8:
		backing := make([]int8, size)
		for i := range backing {
			backing[i] = v
		}
		return t.New(t.WithShape(shape...), t.WithBacking(backing))
	case int16:
		backing := make([]int16, size)
		for i := range backing {
			backing[i] = v
		}
		return t.New(t.WithShape(shape...), t.WithBacking(backing))
	case int32:
		backing := make([]int32, size)
		for i := range backing {
			backing[i] = v
		}
		return t.New(t.WithShape(shape...), t.WithBacking(backing))
	case int64:
		backing := make([]int64, size)
		for i := range backing {
			backing[i] = v
		}
		return t.New(t.WithShape(shape...), t.WithBacking(backing))
	case uint:
		backing := make([]uint, size)
		for i := range backing {
			backing[i] = v
		}
		return t.New(t.WithShape(shape...), t.WithBacking(backing))
	case uint8:
		backing := make([]uint8, size)
		for i := range backing {
			backing[i] = v
		}
		return t.New(t.WithShape(shape...), t.WithBacking(backing))
	case uint16:
		backing := make([]uint16, size)
		for i := range backing {
			backing[i] = v
		}
		return t.New(t.WithShape(shape...), t.WithBacking(backing))
	case uint32:
		backing := make([]uint32, size)
		for i := range backing {
			backing[i] = v
		}
		return t.New(t.WithShape(shape...), t.WithBacking(backing))
	case uint64:
		backing := make([]uint64, size)
		for i := range backing {
			backing[i] = v
		}
		return t.New(t.WithShape(shape...), t.WithBacking(backing))
	case float32:
		backing := make([]float32, size)
		for i := range backing {
			backing[i] = v
		}
		return t.New(t.WithShape(shape...), t.WithBacking(backing))
	case float64:
		backing := make([]float64, size)
		for i := range backing {
			backing[i] = v
		}
		return t.New(t.WithShape(shape...), t.WithBacking(backing))
	case complex64:
		backing := make([]complex64, size)
		for i := range backing {
			backing[i] = v
		}
		return t.New(t.WithShape(shape...), t.WithBacking(backing))
	case complex128:
		backing := make([]complex128, size)
		for i := range backing {
			backing[i] = v
		}
		return t.New(t.WithShape(shape...), t.WithBacking(backing))
	default:
		panic("unknown type")
	}
}

// Zeros inits a tensor with the given shape and type with zeros.
func Zeros(dt t.Dtype, shape ...int) *t.Dense {
	size := t.Shape(shape).TotalSize()
	switch dt.Kind() {
	case reflect.Int:
		backing := make([]int, size)
		for i := range backing {
			backing[i] = int(0)
		}
		return t.New(t.WithShape(shape...), t.WithBacking(backing))
	case reflect.Int8:
		backing := make([]int8, size)
		for i := range backing {
			backing[i] = int8(0)
		}
		return t.New(t.WithShape(shape...), t.WithBacking(backing))
	case reflect.Int16:
		backing := make([]int16, size)
		for i := range backing {
			backing[i] = int16(0)
		}
		return t.New(t.WithShape(shape...), t.WithBacking(backing))
	case reflect.Int32:
		backing := make([]int32, size)
		for i := range backing {
			backing[i] = int32(0)
		}
		return t.New(t.WithShape(shape...), t.WithBacking(backing))
	case reflect.Int64:
		backing := make([]int64, size)
		for i := range backing {
			backing[i] = int64(0)
		}
		return t.New(t.WithShape(shape...), t.WithBacking(backing))
	case reflect.Uint:
		backing := make([]uint, size)
		for i := range backing {
			backing[i] = uint(0)
		}
		return t.New(t.WithShape(shape...), t.WithBacking(backing))
	case reflect.Uint8:
		backing := make([]uint8, size)
		for i := range backing {
			backing[i] = uint8(0)
		}
		return t.New(t.WithShape(shape...), t.WithBacking(backing))
	case reflect.Uint16:
		backing := make([]uint16, size)
		for i := range backing {
			backing[i] = uint16(0)
		}
		return t.New(t.WithShape(shape...), t.WithBacking(backing))
	case reflect.Uint32:
		backing := make([]uint32, size)
		for i := range backing {
			backing[i] = uint32(0)
		}
		return t.New(t.WithShape(shape...), t.WithBacking(backing))
	case reflect.Uint64:
		backing := make([]uint64, size)
		for i := range backing {
			backing[i] = uint64(0)
		}
		return t.New(t.WithShape(shape...), t.WithBacking(backing))
	case reflect.Float32:
		backing := make([]float32, size)
		for i := range backing {
			backing[i] = float32(0)
		}
		return t.New(t.WithShape(shape...), t.WithBacking(backing))
	case reflect.Float64:
		backing := make([]float64, size)
		for i := range backing {
			backing[i] = float64(0)
		}
		return t.New(t.WithShape(shape...), t.WithBacking(backing))
	case reflect.Complex64:
		backing := make([]complex64, size)
		for i := range backing {
			backing[i] = complex64(0)
		}
		return t.New(t.WithShape(shape...), t.WithBacking(backing))
	case reflect.Complex128:
		backing := make([]complex128, size)
		for i := range backing {
			backing[i] = complex128(0)
		}
		return t.New(t.WithShape(shape...), t.WithBacking(backing))
	default:
		panic("unknown type")
	}
}
