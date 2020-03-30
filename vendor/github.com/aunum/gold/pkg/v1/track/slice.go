package track

import "fmt"

// Slicer slices historical data into chunks that are then aggregated.
type Slicer interface {
	// Slice the data.
	Slice(vals Aggregables) AggregableSlices

	// Label for these slices of data.
	Label() string
}

// AggregableSlices are slices of historical values.
type AggregableSlices map[int]Aggregables

// Sort the historical value slices returning an ordered list.
func (h AggregableSlices) Sort() []Aggregables {
	sorted := make([]Aggregables, len(h))
	for i, v := range h {
		sorted[i] = v
	}
	return sorted
}

// EpisodicSlicer slices episodes.
type EpisodicSlicer struct{}

// NewEpisodicSlicer returns a new episodic slicer.
func NewEpisodicSlicer() *EpisodicSlicer {
	return &EpisodicSlicer{}
}

// Slice the values.
func (e *EpisodicSlicer) Slice(vals Aggregables) AggregableSlices {
	hvs := AggregableSlices{}
	for _, val := range vals {
		if _, ok := hvs[val.Ep()]; !ok {
			hvs[val.Ep()] = Aggregables{val}
			continue
		}
		hvs[val.Ep()] = append(hvs[val.Ep()], val)
	}
	return hvs
}

// Label applied to slice.
func (e *EpisodicSlicer) Label() string {
	return "episode"
}

// SingleEpisodeSlicer slices all historical data by single epidodes.
var SingleEpisodeSlicer = &EpisodicSlicer{}

// CummulativeRangeSlicer slices cummulative episodes i.e. each slice will contain
// the current episode plus however many episodes back are specified.
type CummulativeRangeSlicer struct {
	// back is how far bach you would like to cummulate.
	back int
	// wtart is the episode to start with.
	start int
	// end is the episode to end with. If end <=0 will go to last episode
	end int
}

// NewCummulativeRangeSlicer returns a new cummulative slicer.
func NewCummulativeRangeSlicer(back, start, end int) *CummulativeRangeSlicer {
	return &CummulativeRangeSlicer{
		back:  back,
		start: start,
		end:   end,
	}
}

// Slice the values.
func (e *CummulativeRangeSlicer) Slice(vals Aggregables) AggregableSlices {
	episodic := AggregableSlices{}
	for _, val := range vals {
		if _, ok := episodic[val.Ep()]; !ok {
			episodic[val.Ep()] = Aggregables{val}
			continue
		}
		episodic[val.Ep()] = append(episodic[val.Ep()], val)
	}
	cummulative := AggregableSlices{}
	for episode := range episodic {
		hv := Aggregables{}
		back := episode - e.back
		if back < 0 {
			// if we don't have enough back data return empty
			cummulative[episode] = Aggregables{&AggregatedValue{}}
			continue
		}
		for i := back; i <= episode; i++ {
			hv = append(hv, episodic[i]...)
		}
		cummulative[episode] = hv
	}
	return cummulative
}

// Label applied to slice.
func (e *CummulativeRangeSlicer) Label() string {
	if e.back == 0 {
		return "episode"
	}
	return fmt.Sprintf("per last %d episodes", e.back)
}

// DefaultCummulativeSlicer is the default slicer for cummulative slices.
var DefaultCummulativeSlicer = &CummulativeRangeSlicer{back: 100, start: 0, end: -1}
