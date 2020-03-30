package num

import (
	"github.com/aunum/log"
	"github.com/chewxy/math32"
)

// MinMaxNorm is min-max normalization.
func MinMaxNorm(x, min, max float32) float32 {
	if x < min || x > max || min >= max {
		log.Fatalf("parameters for min-max norm not valid")
	}
	return (x - min) / (max - min)
}

// MeanNorm is mean normalization.
func MeanNorm(x, min, max, average float32) float32 {
	if x < min || x > max || min >= max {
		log.Fatalf("parameters for mean norm not valid")
	}
	return (x - average) / (max - min)
}

// ZNorm uses z-score normalization.
func ZNorm(x, mean, stdDev float32) float32 {
	return (x - mean) / stdDev
}

// Mean of the values.
func Mean(vals []float32) float32 {
	n := float32(len(vals))
	var sum float32
	for _, val := range vals {
		sum += val
	}
	return sum / n
}

// Variance is the average distance from the mean.
func Variance(vals []float32) float32 {
	mu := Mean(vals)
	var distance float32
	for _, val := range vals {
		diff := val - mu
		distance += math32.Pow(diff, 2)
	}
	return distance / float32((len(vals) - 1))
}

// StdDev returns the standard deviation of x.
func StdDev(x []float32) float32 {
	mu := Mean(x)
	var distance float32
	for _, z := range x {
		diff := z - mu
		p := math32.Pow(diff, 2)
		distance += p
	}
	return math32.Sqrt(distance / float32(len(x)))
}
