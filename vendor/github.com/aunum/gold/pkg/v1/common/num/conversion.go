// Package num provides various numeric functions.
package num

import "fmt"

// ToF32 attempts to convert the given interface to float32.
func ToF32(i interface{}) (float32, error) {
	switch a := i.(type) {
	case float32:
		return a, nil
	case int:
		return float32(a), nil
	case int32:
		return float32(a), nil
	case int64:
		return float32(a), nil
	case float64:
		return float32(a), nil
	default:
		return 0, fmt.Errorf("type not supported: %v", a)
	}
}

// ToF64 attempts to convert the given interface to float64.
func ToF64(i interface{}) (float64, error) {
	switch a := i.(type) {
	case float64:
		return a, nil
	case int:
		return float64(a), nil
	case int32:
		return float64(a), nil
	case int64:
		return float64(a), nil
	case float32:
		return float64(a), nil
	default:
		return 0, fmt.Errorf("type not supported: %v", a)
	}
}

// BoolToInt converts a boolean to an int, with 1=true 0=false.
func BoolToInt(b bool) int {
	if b {
		return 1
	}
	return 0
}
