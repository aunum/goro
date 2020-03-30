package layers

import (
	g "gorgonia.org/gorgonia"
	t "gorgonia.org/tensor"
)

// Layer in a network.
type Layer interface {
	// Compile the layer.
	Compile(graph *g.ExprGraph, opts ...LayerOpt)

	// Fwd is a forward pass through the layer.
	Fwd(x *g.Node) (*g.Node, error)

	// Learnables returns all learnable nodes within this layer.
	Learnables() g.Nodes

	// Clone the layer.
	Clone() Layer

	// Graph returns the graph for this layer.
	Graph() *g.ExprGraph
}

// LayerOpt is a layer option.
type LayerOpt func(Layer)

// WithSharedLearnables shares the learnables from another layer.
func WithSharedLearnables(shared Layer) func(Layer) {
	return func(l Layer) {
		switch lay := l.(type) {
		case *FC:
			lay.shared = shared.(*FC)
		}
	}
}

// AsBatch informs the layer compilation that it is a batch.
func AsBatch() func(Layer) {
	return func(l Layer) {
		fc := l.(*FC)
		fc.isBatched = true
	}
}

// AsType sets the datatype for the layer.
func AsType(dtype t.Dtype) func(Layer) {
	return func(l Layer) {
		switch lay := l.(type) {
		case *FC:
			lay.dtype = dtype
		}
	}
}
