# Tracker

The tracker is used to track values as an agent is performing. It can track 
arbitrary scalar values or values from a gorgonia graph. It comes with aggregation 
capabilities and an HTTP interface.

## Usage

Create a new default tracker
```go
tracker, _ := NewTracker()
```

Create a new tracker with a specific directory
```go
tracker, _ := NewTracker(WithDir("./logs"))
```

Track a scalar value
```go
score := tracker.TrackValue("score", 0)
score.Inc()
```

Track a scalar with a custom aggregator and increment it
```go
score := tracker.TrackValue("score", 0, track.WithAggregator(track.Max))
score.Inc()
```

Track a scalar value with a custom slicer with gives the rate over an episode size
```go
score := tracker.TrackValue("score", 0,track.WithAggregator(NewMeanAggregator(DefaultCummulativeSlicer)))
score.Inc()
```

Track a Gorgonia graph node value
```go
var loss *gorgonia.Node

lossVal := tracker.TrackValue("loss", loss)
```

Create episodes and timesteps to track
```go
for _, episode := range tracker.MakeEpisodes(100) {

    // create an episode bound scalar that will only hold its value per episode
    score := episode.TrackScalar("score", 0, track.WithAggregator(track.Max))

    for _, timestep := range episode.Steps(200) {
        // log all values per timestamp
        timstep.Log()
    }
    // log all values per episode
    episode.Log()
}
```

Access values via API
```go
resp, _ := http.Get("my.host.com/api/values/myValue")
```
Returns the values as chartjs xy's aggregated according to the values aggregator. A custom aggregator 
can be provided using the `?aggregator=` query param.