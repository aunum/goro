// Package layer provides the layers for sequential models.
package layer

import (
	g "gorgonia.org/gorgonia"
	t "gorgonia.org/tensor"
)

// Config is the config for a layer.
type Config interface {
	// Compile the layer.
	Compile(graph *g.ExprGraph, opts ...CompileOpt) Layer

	// ApplyDefaults to the config.
	ApplyDefaults() Config

	// Validate the config.
	Validate() error

	// Clone the layer config.
	Clone() Config
}

// Layer in a network.
type Layer interface {
	// Fwd is a forward pass through the layer.
	Fwd(x *g.Node) (*g.Node, error)

	// Learnables returns all learnable nodes within this layer.
	Learnables() g.Nodes

	// Clone the layer.
	Clone() Layer

	// Graph returns the graph for this layer.
	Graph() *g.ExprGraph
}

// CompileOpt is a layer compile option.
type CompileOpt func(Layer)

// WithSharedLearnables shares the learnables from another layer.
func WithSharedLearnables(shared Layer) func(Layer) {
	return func(l Layer) {
		switch lay := l.(type) {
		case *fc:
			lay.shared = shared.(*fc)
		case *conv2D:
			lay.shared = shared.(*conv2D)
		}
	}
}

// AsBatch informs the layer compilation that it is a batch.
func AsBatch() func(Layer) {
	return func(l Layer) {
		switch lay := l.(type) {
		case *fc:
			lay.isBatched = true
		case *conv2D:
			lay.isBatched = true
		}
	}
}

// AsType sets the datatype for the layer.
func AsType(dtype t.Dtype) func(Layer) {
	return func(l Layer) {
		switch lay := l.(type) {
		case *fc:
			lay.dtype = dtype
		case *conv2D:
			lay.dtype = dtype
		}
	}
}
