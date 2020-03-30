package dense

import (
	"fmt"
	"reflect"

	t "gorgonia.org/tensor"
)

// ZeroValue for the given datatype.
func ZeroValue(dt t.Dtype) interface{} {
	switch dt.Kind() {
	case reflect.Int:
		return int(0)
	case reflect.Int8:
		return int8(0)
	case reflect.Int16:
		return int16(0)
	case reflect.Int32:
		return int32(0)
	case reflect.Int64:
		return int64(0)
	case reflect.Uint:
		return uint(0)
	case reflect.Uint8:
		return uint8(0)
	case reflect.Uint16:
		return uint16(0)
	case reflect.Uint32:
		return uint32(0)
	case reflect.Uint64:
		return uint64(0)
	case reflect.Float32:
		return float32(0)
	case reflect.Float64:
		return float64(0)
	case reflect.Complex64:
		return complex64(0)
	case reflect.Complex128:
		return complex128(0)
	default:
		panic(fmt.Sprintf("type not supported: %#v", dt))
	}
}

// FauxZero is the faux zero value used to prevent divde by zero errors.
const FauxZero = 1e-6

// FauxZeroValue is a faux zero value for the given datatype.
func FauxZeroValue(dt t.Dtype) interface{} {
	switch dt.Kind() {
	case reflect.Float32:
		return float32(FauxZero)
	case reflect.Float64:
		return float64(FauxZero)
	case reflect.Complex64:
		return complex64(FauxZero)
	case reflect.Complex128:
		return complex128(FauxZero)
	default:
		panic(fmt.Sprintf("type not supported: %#v", dt))
	}
}

// NegValue for the given datatype.
func NegValue(dt t.Dtype) interface{} {
	switch dt.Kind() {
	case reflect.Int:
		return int(-1)
	case reflect.Int8:
		return int8(-1)
	case reflect.Int16:
		return int16(-1)
	case reflect.Int32:
		return int32(-1)
	case reflect.Int64:
		return int64(-1)
	case reflect.Float32:
		return float32(-1)
	case reflect.Float64:
		return float64(-1)
	case reflect.Complex64:
		return complex64(-1)
	case reflect.Complex128:
		return complex128(-1)
	default:
		panic(fmt.Sprintf("type not supported: %#v", dt))
	}
}
