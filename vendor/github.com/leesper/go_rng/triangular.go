package rng

import (
	"fmt"
	"math"
)

// TriangularGenerator is a random number generator for Triangular distribution.
type TriangularGenerator struct {
	uniform *UniformGenerator
}

// NewTriangularGenerator returns a Triangular-distribution generator
func NewTriangularGenerator(seed int64) *TriangularGenerator {
	urng := NewUniformGenerator(seed)
	return &TriangularGenerator{urng}
}

//TriangularObj returns value
//when x in[a,c] then cdf =  (x-a)^2/((b-a)(c-a)) ,when x in(c,b] then 1 - (b-x)^2/((b-a)(b-c))
//when x in[0,(c-a)/(b-a)] then invcdf = a+ sqrt(x*(b-1)(c-a)) ,when x in((c-a)/(b-a),1] then invcdf = b - sqrt((1-x)*(b-a)*(b-c))
func (Trng TriangularGenerator) TriangularObj(a, b, c float64) float64 {
	if !(a < c && c < b) {
		fmt.Println("Invalid parameters a,b,c:", a, b, c)
		return 0.0
	}
	urng := Trng.uniform.Float64Range(0.0, 1.0)
	mode := (c - a) / (b - a)
	if urng < mode {
		return a + math.Sqrt(urng*(b-1.0)*(c-a))
	} else {
		return b - math.Sqrt((1.0-urng)*(b-a)*(b-c))
	}
}
