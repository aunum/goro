package layer

import (
	"github.com/aunum/log"

	g "gorgonia.org/gorgonia"
	t "gorgonia.org/tensor"
)

// MaxPooling2D implements the max pooling 2d function.
type MaxPooling2D struct {
	// Shape of the kernel.
	// Defaults to (2, 2)
	Kernel t.Shape

	// Pad
	// Defaults to (0, 0)
	Pad []int

	// Stride
	// Defaults to (2, 2)
	Stride []int

	// Name
	Name string
}

// Validate the config.
func (m MaxPooling2D) Validate() error {
	return nil
}

// ApplyDefaults applys defaults to the layers.
func (m MaxPooling2D) ApplyDefaults() Config {
	if len(m.Kernel) == 0 {
		m.Kernel = []int{2, 2}
	}
	if len(m.Pad) == 0 {
		m.Pad = []int{0, 0}
	}
	if len(m.Stride) == 0 {
		m.Stride = []int{2, 2}
	}
	return m
}

// Compile the config as a layer.
func (m MaxPooling2D) Compile(graph *g.ExprGraph, opts ...CompileOpt) Layer {
	mp := newMaxPooling2d(&m)
	mp.graph = graph
	return mp
}

// Clone the config.
func (m MaxPooling2D) Clone() Config {
	return &MaxPooling2D{
		Kernel: m.Kernel,
		Pad:    m.Pad,
		Stride: m.Stride,
		Name:   m.Name,
	}
}

type maxPooling2D struct {
	*MaxPooling2D
	graph *g.ExprGraph
}

func newMaxPooling2d(config *MaxPooling2D) *maxPooling2D {
	config.ApplyDefaults()
	return &maxPooling2D{
		MaxPooling2D: config,
	}
}

// Fwd is a forward pass through the layer.
func (m *maxPooling2D) Fwd(x *g.Node) (*g.Node, error) {
	n, err := g.MaxPool2D(x, m.Kernel, m.Pad, m.Stride)
	if err != nil {
		return nil, err
	}
	log.Debugf("pooling2d %q output shape: %v", m.Name, n.Shape())
	return n, nil
}

// Learnables returns all learnable nodes within this layer.
func (m *maxPooling2D) Learnables() g.Nodes {
	return g.Nodes{}
}

// Clone the layer.
func (m *maxPooling2D) Clone() Layer {
	return &maxPooling2D{
		MaxPooling2D: m.MaxPooling2D.Clone().(*MaxPooling2D),
	}
}

// Graph returns the graph for this layer.
func (m *maxPooling2D) Graph() *g.ExprGraph {
	return m.graph
}
