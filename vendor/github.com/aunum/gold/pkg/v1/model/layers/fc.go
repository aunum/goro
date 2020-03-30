package layers

import (
	"fmt"

	"github.com/aunum/log"
	g "gorgonia.org/gorgonia"
	t "gorgonia.org/tensor"
)

// FC is a fully connected layer of neurons.
type FC struct {
	// Input is the number of units in input.
	Input int

	// Output is the number of units in the output.
	Output int

	// Name of the layer.
	Name string

	activation Activation
	weights    *g.Node
	init       g.InitWFn
	dtype      t.Dtype
	bias       *g.Node
	useBias    bool
	biasInit   g.InitWFn
	isBatched  bool
	shared     *FC
}

// FCOpts are options for a fully connected layer.
type FCOpts func(*FC)

// NewFC returns a new fully connected layer.
func NewFC(input, output int, opts ...FCOpts) *FC {
	fc := &FC{
		Input:    input,
		Output:   output,
		dtype:    g.Float32,
		init:     g.GlorotU(1.0),
		useBias:  true,
		biasInit: g.Zeroes(),
	}
	for _, opt := range opts {
		opt(fc)
	}
	return fc
}

// WithActivation adds an activation function to the layer.
func WithActivation(fn Activation) func(*FC) {
	return func(f *FC) {
		f.activation = fn
	}
}

// WithName gives a name to this layer.
func WithName(name string) func(*FC) {
	return func(f *FC) {
		f.Name = name
	}
}

// WithInit adds an init function to the weights.
// Defaults to Glorot.
func WithInit(fn g.InitWFn) func(*FC) {
	return func(f *FC) {
		f.init = fn
	}
}

// WithType sets the type for the layer
// Defaults to Float32.
func WithType(dtype t.Dtype) func(*FC) {
	return func(f *FC) {
		f.dtype = dtype
	}
}

// WithBiasInit adds an init function to the bias.
// Defaults to zeros.
func WithBiasInit(fn g.InitWFn) func(*FC) {
	return func(f *FC) {
		f.init = fn
	}
}

// WithNoBias removes the bias.
func WithNoBias() func(*FC) {
	return func(f *FC) {
		f.useBias = false
	}
}

// Compile the layer into the graph.
func (f *FC) Compile(graph *g.ExprGraph, opts ...LayerOpt) {
	for _, opt := range opts {
		opt(f)
	}
	if f.shared != nil {
		f.weights = g.NewMatrix(graph, f.dtype, g.WithShape(f.Input, f.Output), g.WithName(f.Name), g.WithValue(f.shared.weights.Value()))
		if f.useBias {
			f.bias = g.NewMatrix(graph, f.dtype, g.WithShape(1, f.Output), g.WithName(fmt.Sprintf("%s-bias", f.Name)), g.WithValue(f.shared.bias.Value()))
		}
		return
	}
	f.weights = g.NewMatrix(graph, f.dtype, g.WithShape(f.Input, f.Output), g.WithInit(f.init), g.WithName(f.Name))
	if f.useBias {
		f.bias = g.NewMatrix(graph, f.dtype, g.WithShape(1, f.Output), g.WithInit(f.biasInit), g.WithName(fmt.Sprintf("%s-bias", f.Name)))
	}
}

// Fwd is a forward pass on a single fully connected layer.
func (f *FC) Fwd(x *g.Node) (*g.Node, error) {
	var xw, xwb *g.Node
	var err error
	if x.IsVector() {
		s := t.Shape{1}
		s = append(s, x.Shape()...)
		x, err = g.Reshape(x, s)
		if err != nil {
			return nil, err
		}
		log.Debugf("normalizing dimensions of x to %v", s)
	}

	// Note: parts of this are borrowed from https://github.com/gorgonia/golgi
	if xw, err = g.Mul(x, f.weights); err != nil {
		return nil, err
	}

	if f.bias == nil {
		xwb = xw
		goto act
	}
	if f.isBatched {
		if xwb, err = g.BroadcastAdd(xw, f.bias, nil, []byte{0}); err != nil {
			return nil, err
		}
	} else {
		if xwb, err = g.Add(xw, f.bias); err != nil {
			return nil, err
		}
	}

act:
	if f.activation == nil {
		return xwb, nil
	}
	a, err := f.activation.Fwd(xwb)
	if err != nil {
		return nil, err
	}
	return a, err
}

// Learnables are the learnable parameters of the fully connected layer.
func (f *FC) Learnables() g.Nodes {
	if f.bias != nil {
		return g.Nodes{f.weights, f.bias}
	}
	return g.Nodes{f.weights}
}

// Clone the layer without any nodes. (nodes cannot be shared)
func (f *FC) Clone() Layer {
	return &FC{
		Input:      f.Input,
		Output:     f.Output,
		Name:       f.Name,
		activation: f.activation.Clone(),
		init:       f.init,
		dtype:      f.dtype,
		useBias:    f.useBias,
		biasInit:   f.biasInit,
		isBatched:  f.isBatched,
	}
}

// Graph returns the graph this layer was compiled with.
func (f *FC) Graph() *g.ExprGraph {
	if f.weights == nil {
		return nil
	}
	return f.weights.Graph()
}
