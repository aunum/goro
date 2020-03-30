package track

import (
	"fmt"

	"gonum.org/v1/plot/plotter"
)

// AggregatorName is the name of an aggregator.
type AggregatorName string

const (
	// MeanAggregatorName is the name of the mean aggregator.
	MeanAggregatorName AggregatorName = "mean"

	// ModeAggregatorName is the name of the mode aggregator.
	ModeAggregatorName AggregatorName = "mode"

	// MaxAggregatorName is the name of the max aggregator.
	MaxAggregatorName AggregatorName = "max"
)

// AggregatorNames holds all of the current aggregator names.
var AggregatorNames = []AggregatorName{MeanAggregatorName, ModeAggregatorName, MaxAggregatorName}

// AggregatorFromName returns an aggregator from its name.
func AggregatorFromName(name string) (Aggregator, error) {
	switch AggregatorName(name) {
	case MeanAggregatorName:
		return Mean, nil
	case ModeAggregatorName:
		return Mode, nil
	case MaxAggregatorName:
		return Max, nil
	default:
		return nil, fmt.Errorf("aggregator %q unknown", name)
	}
}

// Aggregable is a value that can be aggregated.
type Aggregable interface {
	// Scalar value.
	Scalar() float64

	// Ep is the episode for this value.
	// Note: this is shortened to deal with name conflicts.
	Ep() int
}

// Aggregables are a slice of aggregable values.
type Aggregables []Aggregable

// AggregatedValue is an aggregated value.
type AggregatedValue struct {
	value   float64
	episode int
}

// Scalar value
func (a *AggregatedValue) Scalar() float64 {
	return a.value
}

// Ep is the episode this value is linked to.
func (a *AggregatedValue) Ep() int {
	return a.episode
}

// AggregatedValues is a slice of aggregated value.
type AggregatedValues []AggregatedValue

// Scalar values.
func (a Aggregables) Scalar() []float64 {
	ret := []float64{}
	for _, aggregable := range a {
		ret = append(ret, aggregable.Scalar())
	}
	return ret
}

// Aggregator aggregates historical values into a single value.
type Aggregator interface {
	// Aggregate the values.
	Aggregate(vals Aggregables) *Aggregates
}

// MeanAggregator returns the mean of the historical values.
type MeanAggregator struct{ Slicer }

// NewMeanAggregator returns a new mode aggregator.
func NewMeanAggregator(slicer Slicer) *MeanAggregator {
	return &MeanAggregator{slicer}
}

// Aggregate the values.
func (m *MeanAggregator) Aggregate(vals Aggregables) *Aggregates {
	sliced := m.Slice(vals)
	aggs := NewAggregates(m.Label(), "value")
	for i, slice := range sliced {
		aggs.values = append(aggs.values, &AggregatedValue{value: mean(slice.Scalar()), episode: i})
	}
	return aggs
}

func mean(vals []float64) float64 {
	l := float64(len(vals))
	var sum float64
	for _, val := range vals {
		sum += val
	}
	return sum / l
}

// Mean aggregator.
var Mean = &MeanAggregator{SingleEpisodeSlicer}

// MaxAggregator returns the max of the historical values.
type MaxAggregator struct{ Slicer }

// NewMaxAggregator returns a new max aggregator.
func NewMaxAggregator(slicer Slicer) *MaxAggregator {
	return &MaxAggregator{slicer}
}

// Aggregate the values.
func (m *MaxAggregator) Aggregate(vals Aggregables) *Aggregates {
	sliced := m.Slice(vals)
	aggs := NewAggregates(m.Label(), "value")
	for i, slice := range sliced {
		aggs.values = append(aggs.values, &AggregatedValue{value: max(slice.Scalar()), episode: i})
	}
	return aggs
}

func max(vals []float64) float64 {
	var max float64
	for _, val := range vals {
		if val > max {
			max = val
		}
	}
	return max
}

// Max aggregator.
var Max = &MaxAggregator{SingleEpisodeSlicer}

// ModeAggregator returns the most common of the historical values.
type ModeAggregator struct{ Slicer }

// NewModeAggregator returns a new mode aggregator.
func NewModeAggregator(slicer Slicer) *ModeAggregator {
	return &ModeAggregator{slicer}
}

// Aggregate the values.
func (m *ModeAggregator) Aggregate(vals Aggregables) *Aggregates {
	sliced := m.Slice(vals)
	aggs := NewAggregates(m.Label(), "value")
	for i, slice := range sliced {
		aggs.values = append(aggs.values, &AggregatedValue{value: mode(slice.Scalar()), episode: i})
	}
	return aggs
}

// Mode aggregator.
var Mode = &ModeAggregator{SingleEpisodeSlicer}

func mode(vals []float64) float64 {
	modes := map[float64]int{}
	for _, val := range vals {
		v, ok := modes[val]
		if !ok {
			v = 0
		}
		modes[val] = v + 1
	}
	var maxI int
	var maxV float64
	for val, i := range modes {
		if i > maxI {
			maxV = val
		}
	}
	return maxV
}

// ChainAggregator is a chain of aggregators.
type ChainAggregator struct {
	aggregators []Aggregator
}

// NewChainAggregator returns a new chain aggregator.
func NewChainAggregator(aggregators ...Aggregator) *ChainAggregator {
	return &ChainAggregator{
		aggregators: aggregators,
	}
}

// Aggregate the values.
func (c *ChainAggregator) Aggregate(vals Aggregables) (retVal *Aggregates) {
	for _, aggregator := range c.aggregators {
		retVal := aggregator.Aggregate(vals)
		vals = retVal.values
	}
	return
}

// DefaultRateAggregator is a default aggregator to take the rate of a value, uses the
// mean of the last 100 episodes.
var DefaultRateAggregator = NewMeanAggregator(DefaultCummulativeSlicer)

// Aggregates are generic aggregated values.
type Aggregates struct {
	values Aggregables
	xLabel string
	yLabel string
}

// NewAggregates returns a new aggregates.
func NewAggregates(xLabel, yLabel string) *Aggregates {
	return &Aggregates{
		values: []Aggregable{},
		xLabel: xLabel,
		yLabel: yLabel,
	}
}

// Sort the episodes into a slice.
func (a Aggregates) Sort() []float64 {
	sorted := make([]float64, len(a.values))
	for i, v := range a.values {
		sorted[i] = v.Scalar()
	}
	return sorted
}

// GonumXYs returns the episode aggregates as gonum xy pairs.
func (a Aggregates) GonumXYs() plotter.XYs {
	xys := plotter.XYs{}
	for _, value := range a.values {
		xy := plotter.XY{
			X: float64(value.Ep()),
			Y: value.Scalar(),
		}
		xys = append(xys, xy)
	}
	return xys
}

// Chartjs is a ChartJS chart.
type Chartjs struct {
	// XLabel is the label for the x values.
	XLabel string `json:"xLabel"`
	// YLabel is the label for the y values.
	YLabel string `json:"yLabel"`
	// XYS are the xy values.
	XYs ChartjsXYs `json:"xys"`
}

// ChartjsXY conforms to the expected point data structure for chartjs charts.
type ChartjsXY struct {
	// X value.
	X float64 `json:"x"`
	// Y value.
	Y float64 `json:"y"`
}

// ChartjsXYs conforms to the expected set of point data structure for chartjs charts.
type ChartjsXYs []ChartjsXY

// Chartjs returns the data for a ChartJS chart.
func (a Aggregates) Chartjs() *Chartjs {
	return &Chartjs{
		XLabel: a.xLabel,
		YLabel: a.yLabel,
		XYs:    a.ChartjsXYs(),
	}
}

// ChartjsXYs returns the episode aggregates as gonum xy pairs.
func (a Aggregates) ChartjsXYs() ChartjsXYs {
	xys := ChartjsXYs{}
	for _, value := range a.values {
		xy := ChartjsXY{
			X: float64(value.Ep()),
			Y: value.Scalar(),
		}
		xys = append(xys, xy)
	}
	return xys.Order()
}

// Order the xys by x.
func (c ChartjsXYs) Order() ChartjsXYs {
	xys := make([]ChartjsXY, len(c))
	for _, xy := range c {
		xys[int(xy.X)] = xy
	}
	return xys
}
