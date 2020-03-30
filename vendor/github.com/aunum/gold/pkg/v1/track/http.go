package track

import (
	"encoding/json"
	"net/http"
	"strings"
)

// ApplyHandlers applies tracker handlers to a mux.
func (t *Tracker) ApplyHandlers(mux *http.ServeMux) {
	mux.HandleFunc("/api/aggregators", t.AggregatorsHandler)
	mux.HandleFunc("/api/values/", t.AggregateValuesHandler)
	mux.HandleFunc("/api/values", t.ValuesHandler)
}

// AggregatorsHandler returns all possible aggregators.
func (t *Tracker) AggregatorsHandler(w http.ResponseWriter, req *http.Request) {
	b, err := json.Marshal(AggregatorNames)
	if err != nil {
		t.logger.Error(err)
		w.WriteHeader(500)
		w.Write([]byte(err.Error()))
		return
	}
	w.WriteHeader(200)
	w.Write(b)
}

// AggregateValuesHandler is an HTTP handler for the tracker serving aggregates.
func (t *Tracker) AggregateValuesHandler(w http.ResponseWriter, req *http.Request) {
	valueName := strings.TrimPrefix(req.URL.Path, "/api/values/")
	if valueName == "" {
		t.logger.Error("value name blank")
		w.WriteHeader(500)
		w.Write([]byte(`request must include /values/:name, 
		to find value names use the /values endpoint`))
		return
	}
	v, err := t.GetValue(valueName)
	if err != nil {
		t.logger.Error(err)
		w.WriteHeader(500)
		w.Write([]byte(err.Error()))
		return
	}
	query := req.URL.Query()
	aggName := query.Get("aggregator")
	agg := v.Aggregator()
	if aggName != "" {
		agg, err = AggregatorFromName(aggName)
		if err != nil {
			t.logger.Error(err)
			w.WriteHeader(500)
			w.Write([]byte(err.Error()))
			return
		}
	}
	h, err := t.GetHistory(valueName)
	if err != nil {
		t.logger.Error(err)
		w.WriteHeader(500)
		w.Write([]byte(err.Error()))
		return
	}
	aggs := h.Aggregate(agg)
	chartJS := aggs.Chartjs()
	b, err := json.Marshal(chartJS)
	if err != nil {
		t.logger.Error(err)
		w.WriteHeader(500)
		w.Write([]byte(err.Error()))
		return
	}
	w.WriteHeader(200)
	w.Write(b)
}

// ValuesHandler is an HTTP handler for revealing what values are tracked over the network.
func (t *Tracker) ValuesHandler(w http.ResponseWriter, req *http.Request) {
	valueNames := t.ValueNames()
	b, err := json.Marshal(valueNames)
	if err != nil {
		t.logger.Error(err)
		w.WriteHeader(500)
		w.Write([]byte(err.Error()))
		return
	}
	w.WriteHeader(200)
	w.Write(b)
}
