package rng

import (
	"fmt"
	"math"
)

// LogisticGenerator is a random number generator for logistic distribution.
// The zero value is invalid, use NewLogisticGenerator to create a generator
type LogisticGenerator struct {
	uniform *UniformGenerator
}

// NewLogisticGenerator returns a logistic-distribution generator
// it is recommended using time.Now().UnixNano() as the seed, for example:
// lrng := rng.NewLogisticGenerator(time.Now().UnixNano())
func NewLogisticGenerator(seed int64) *LogisticGenerator {
	urng := NewUniformGenerator(seed)
	return &LogisticGenerator{urng}
}

// Logistic returns a random number of logistic distribution
func (lrng LogisticGenerator) Logistic(mu, s float64) float64 {
	if !(s > 0.0) {
		panic(fmt.Sprintf("Invalid parameter s: %.2f", s))
	}
	return lrng.logistic(mu, s)
}

func (lrng LogisticGenerator) logistic(mu, s float64) float64 {
	return mu + s*math.Log(lrng.uniform.Float64()/(1-lrng.uniform.Float64()))
}
