package track

import (
	"fmt"

	g "gorgonia.org/gorgonia"
)

// Episode represents a training episode.
type Episode struct {
	tracker *Tracker
	I       int
	Values  map[string]TrackedValue
}

// Episodes is a slice of episode.
type Episodes []*Episode

// MakeEpisodes creates a number of episodes.
func (t *Tracker) MakeEpisodes(num int) Episodes {
	episodes := Episodes{}
	for i := 0; i < num; i++ {
		episodes = append(episodes, &Episode{tracker: t, I: i, Values: map[string]TrackedValue{}})
	}
	t.episodes = episodes
	return episodes
}

// Steps to take in an episode.
func (e *Episode) Steps(num int) Timesteps {
	timesteps := Timesteps{}
	for i := 0; i < num; i++ {
		timesteps = append(timesteps, &Timestep{I: i, episode: e})
	}
	return timesteps
}

// TrackValue tracks a value for only an episode.
func (e *Episode) TrackValue(name string, value interface{}, opts ...TrackedValueOpt) TrackedValue {
	var tv TrackedValue
	if n, ok := value.(*g.Node); ok {
		nv := NewTrackedNodeValue(name, opts...)
		e.Values[nv.name] = nv
		g.Read(n, &nv.value)
		tv = nv
	} else {
		sv := NewTrackedScalarValue(name, value, opts...)
		e.Values[sv.name] = sv
		tv = sv
	}
	return tv
}

// TrackScalar tracks a scarlar value for only an episode.
func (e *Episode) TrackScalar(name string, value interface{}, opts ...TrackedValueOpt) *TrackedScalarValue {
	sv := NewTrackedScalarValue(name, value, opts...)
	e.Values[sv.name] = sv
	return sv
}

// GetValue a tracked value by name.
func (e *Episode) GetValue(name string) (TrackedValue, error) {
	for _, value := range e.Values {
		if value.Name() == name {
			return value, nil
		}
	}
	return nil, fmt.Errorf("%q value does not exist", name)
}

// Data returns episodic data.
func (e *Episode) Data() *History {
	vals := []*HistoricalValue{}
	for _, v := range e.tracker.values {
		vals = append(vals, v.Data(e.I, 0))
	}
	for _, v := range e.Values {
		vals = append(vals, v.Data(e.I, 0))
	}
	return &History{
		Values:   vals,
		Timestep: 0,
		Episode:  e.I,
	}
}

// Log values for an episode.
func (e *Episode) Log() {
	e.tracker.encoder.Encode(e.Data())
}

// Timestep represents a training timestep.
type Timestep struct {
	episode *Episode
	I       int
}

// Timesteps is a slice of timestep.
type Timesteps []*Timestep

// Log values for a timestep.
func (t *Timestep) Log() {
	t.episode.tracker.encoder.Encode(t.Data())
}

// Data returns timestep data.
func (t *Timestep) Data() *History {
	vals := []*HistoricalValue{}
	for _, v := range t.episode.tracker.values {
		vals = append(vals, v.Data(t.episode.I, t.I))
	}
	for _, v := range t.episode.Values {
		vals = append(vals, v.Data(t.episode.I, t.I))
	}
	return &History{
		Values:   vals,
		Timestep: t.I,
		Episode:  t.episode.I,
	}
}
