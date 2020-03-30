package layer

import (
	"fmt"

	g "gorgonia.org/gorgonia"
)

// Dropout implements layer dropout.
type Dropout struct {
	// Probability of dropping out.
	// Defaults to 0.6
	Probability float64
}

// Validate the config.
func (d Dropout) Validate() error {
	if d.Probability > 1.0 || d.Probability < 0 {
		return fmt.Errorf("dropout probability must be between 0 and 1")
	}
	return nil
}

// ApplyDefaults applys defaults to the layers.
func (d Dropout) ApplyDefaults() Config {
	if d.Probability == 0 {
		d.Probability = 0.6
	}
	return d
}

// Compile the config as a layer.
func (d Dropout) Compile(graph *g.ExprGraph, opts ...CompileOpt) Layer {
	drop := newDropout(&d)
	drop.graph = graph
	return drop
}

// Clone the config.
func (d Dropout) Clone() Config {
	return &Dropout{
		Probability: d.Probability,
	}
}

type dropout struct {
	*Dropout
	graph *g.ExprGraph
}

func newDropout(config *Dropout) *dropout {
	config.ApplyDefaults()
	return &dropout{
		Dropout: config,
	}
}

// Fwd is a forward pass through the layer.
func (d *dropout) Fwd(x *g.Node) (*g.Node, error) {
	return g.Dropout(x, d.Probability)
}

// Learnables returns all learnable nodes within this layer.
func (d *dropout) Learnables() g.Nodes {
	return g.Nodes{}
}

// Clone the layer.
func (d *dropout) Clone() Layer {
	return &dropout{
		Dropout: d.Dropout.Clone().(*Dropout),
	}
}

// Graph returns the graph for this layer.
func (d *dropout) Graph() *g.ExprGraph {
	return d.graph
}
