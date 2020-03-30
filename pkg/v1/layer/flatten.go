package layer

import (
	"fmt"

	"github.com/aunum/log"

	g "gorgonia.org/gorgonia"
)

// Flatten reshapes the incoming tensor to be flat, preserving the batch.
type Flatten struct{}

// ApplyDefaults to the flatten layer.
func (f Flatten) ApplyDefaults() Config { return f }

// Compile the layer.
func (f Flatten) Compile(graph *g.ExprGraph, opts ...CompileOpt) Layer {
	flat := newFlatten(&f)
	flat.graph = graph
	return flat
}

// Clone the config.
func (f Flatten) Clone() Config {
	return Flatten{}
}

// Validate the config.
func (f Flatten) Validate() error {
	return nil
}

type flatten struct {
	*Flatten
	graph *g.ExprGraph
}

func newFlatten(config *Flatten) *flatten {
	return &flatten{Flatten: config}
}

// Fwd is a forward pass through the layer.
func (f *flatten) Fwd(x *g.Node) (*g.Node, error) {
	if len(x.Shape()) < 2 {
		return nil, fmt.Errorf("flatten expects input in the shape (batch, x...), to few dimensions in %v", x.Shape())
	}
	batch := x.Shape()[0]
	s := x.Shape()[1:]
	product := 1
	for _, d := range s {
		product *= d
	}
	newShape := []int{batch, product}
	n, err := g.Reshape(x, newShape)
	if err != nil {
		return nil, err
	}
	log.Debugf("flatten output shape: %v", n.Shape())
	return n, nil
}

// Learnables returns all learnable nodes within this layer.
func (f *flatten) Learnables() g.Nodes {
	return g.Nodes{}
}

// Clone the layer.
func (f *flatten) Clone() Layer {
	return &flatten{Flatten: f.Flatten.Clone().(*Flatten)}
}

// Graph returns the graph for this layer.
func (f *flatten) Graph() *g.ExprGraph {
	return f.graph
}
