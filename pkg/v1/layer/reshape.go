package layer

import (
	"fmt"

	g "gorgonia.org/gorgonia"
	t "gorgonia.org/tensor"
)

// Reshape the incoming tensor.
type Reshape struct {
	// To shape
	// required
	To t.Shape
}

// ApplyDefaults to the flatten layer.
func (r Reshape) ApplyDefaults() Config { return r }

// Compile the layer.
func (r Reshape) Compile(graph *g.ExprGraph, opts ...CompileOpt) Layer {
	rshp := newReshape(&r)
	rshp.graph = graph
	return rshp
}

// Clone the config.
func (r Reshape) Clone() Config {
	return Reshape{To: r.To}
}

// Validate the config.
func (r Reshape) Validate() error {
	if len(r.To) == 0 {
		return fmt.Errorf("Shape must be set on reshape")
	}
	return nil
}

type reshape struct {
	*Reshape
	graph *g.ExprGraph
}

func newReshape(config *Reshape) *reshape {
	return &reshape{Reshape: config}
}

// Fwd is a forward pass through the layer.
func (r *reshape) Fwd(x *g.Node) (*g.Node, error) {
	batch := []int{x.Shape()[0]}
	newShape := append(batch, r.To...)
	return g.Reshape(x, newShape)
}

// Learnables returns all learnable nodes within this layer.
func (r *reshape) Learnables() g.Nodes {
	return g.Nodes{}
}

// Clone the layer.
func (r *reshape) Clone() Layer {
	return &reshape{Reshape: r.Reshape.Clone().(*Reshape)}
}

// Graph returns the graph for this layer.
func (r *reshape) Graph() *g.ExprGraph {
	return r.graph
}
