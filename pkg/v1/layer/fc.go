package layer

import (
	"fmt"

	"github.com/aunum/log"

	g "gorgonia.org/gorgonia"
	t "gorgonia.org/tensor"
)

// FC is a fully connected layer of neurons.
type FC struct {
	// Input is the number of units in input.
	// required
	Input int

	// Output is the number of units in the output.
	// required
	Output int

	// Name of the layer.
	Name string

	// Activation is the activation function.
	// Defaults to ReLU
	Activation ActivationFn

	// Init is the init function.
	// Defaults to GlorotN(1)
	Init g.InitWFn

	// NoBias indicates to not use a bias with the layer
	// Defaults to true.
	NoBias bool

	// BiasInit is the init function for the bias.
	// Defaults to GlorotN(1)
	BiasInit g.InitWFn
}

type fc struct {
	*FC

	weights   *g.Node
	dtype     t.Dtype
	bias      *g.Node
	isBatched bool
	shared    *fc
}

func newFC(config *FC) *fc {
	config.ApplyDefaults()
	return &fc{
		FC:    config,
		dtype: t.Float32,
	}
}

// Validate the config.
func (f FC) Validate() error {
	if f.Input == 0 {
		return fmt.Errorf("input must be set")
	}
	if f.Output == 0 {
		return fmt.Errorf("output must be set")
	}
	return nil
}

// ApplyDefaults to the config.
func (f FC) ApplyDefaults() Config {
	if f.Activation == nil {
		f.Activation = ReLU
	}
	if f.Init == nil {
		f.Init = g.GlorotN(1)
	}
	if f.BiasInit == nil {
		f.BiasInit = g.GlorotN(1)
	}
	return f
}

// Compile the layer into the graph.
func (f FC) Compile(graph *g.ExprGraph, opts ...CompileOpt) Layer {
	fcn := newFC(&f)
	for _, opt := range opts {
		opt(fcn)
	}
	if fcn.shared != nil {
		fcn.weights = g.NewMatrix(graph, fcn.dtype, g.WithShape(f.Input, f.Output), g.WithName(f.Name), g.WithValue(fcn.shared.weights.Value()))
		if !fcn.NoBias {
			fcn.bias = g.NewMatrix(graph, fcn.dtype, g.WithShape(1, f.Output), g.WithName(fmt.Sprintf("%s-bias", f.Name)), g.WithValue(fcn.shared.bias.Value()))
		}
		return fcn
	}
	fcn.weights = g.NewMatrix(graph, fcn.dtype, g.WithShape(f.Input, f.Output), g.WithInit(f.Init), g.WithName(f.Name))
	if !f.NoBias {
		fcn.bias = g.NewMatrix(graph, fcn.dtype, g.WithShape(1, f.Output), g.WithInit(f.BiasInit), g.WithName(fmt.Sprintf("%s-bias", f.Name)))
	}
	return fcn
}

// Clone the config.
func (f FC) Clone() Config {
	return &FC{
		Input:      f.Input,
		Output:     f.Output,
		Name:       f.Name,
		Activation: f.Activation.Clone(),
		Init:       f.Init,
		NoBias:     f.NoBias,
		BiasInit:   f.BiasInit,
	}
}

// Fwd is a forward pass on a single fully connected layer.
func (f *fc) Fwd(x *g.Node) (*g.Node, error) {
	var xw, xwb *g.Node
	var err error
	if x.IsVector() {
		s := append(t.Shape{1}, x.Shape()...)
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
	if f.Activation == nil {
		log.Debugf("fc %q output shape: %v", f.Name, xwb.Shape())
		return xwb, nil
	}
	a, err := f.Activation.Fwd(xwb)
	if err != nil {
		return nil, err
	}
	log.Debugf("fc name %q output shape: %v", f.Name, a.Shape())
	return a, err
}

// Learnables are the learnable parameters of the fully connected layer.
func (f *fc) Learnables() g.Nodes {
	if f.bias != nil {
		return g.Nodes{f.weights, f.bias}
	}
	return g.Nodes{f.weights}
}

// Clone the layer without any nodes. (nodes cannot be shared)
func (f *fc) Clone() Layer {
	configCloned := f.FC.Clone().(FC)
	return &fc{
		FC:        &configCloned,
		dtype:     f.dtype,
		isBatched: f.isBatched,
		shared:    f.shared,
	}
}

// Graph returns the graph this layer was compiled with.
func (f *fc) Graph() *g.ExprGraph {
	if f.weights == nil {
		return nil
	}
	return f.weights.Graph()
}
