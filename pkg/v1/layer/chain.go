package layer

import (
	"github.com/aunum/log"

	g "gorgonia.org/gorgonia"
)

// Chain of layers.
type Chain struct {
	// Layers are the layers to chain together.
	Layers []Config

	sharedLearnables *Chain
	compileOpts      []CompileOpt
	layers           []Layer
}

// NewChain returns a new chain of layers.
func NewChain(layers ...Config) *Chain {
	c := &Chain{}
	c.Add(layers...)
	return c
}

// Fwd is a forward pass thorugh all layers of the chain.
func (c *Chain) Fwd(x *g.Node) (prediction *g.Node, err error) {
	prediction = x
	for _, layer := range c.layers {
		if prediction, err = layer.Fwd(prediction); err != nil {
			return nil, err
		}
	}
	return prediction, nil
}

// Learnables are all of the learnable parameters in the chain.
func (c *Chain) Learnables() g.Nodes {
	retVal := []*g.Node{}
	for _, layer := range c.layers {
		retVal = append(retVal, layer.Learnables()...)
	}
	return retVal
}

// Add to the chain.
func (c *Chain) Add(l ...Config) {
	for _, layer := range l {
		err := layer.Validate()
		if err != nil {
			log.Fatalf("layer %#v \nfailed validation: %v", layer, err)
		}
		layer = layer.ApplyDefaults()
		c.Layers = append(c.Layers, layer)
	}
}

// Clone the chain without any nodes.
func (c *Chain) Clone() *Chain {
	ch := &Chain{}
	for _, layer := range c.Layers {
		ch.Add(layer.Clone())
	}
	return ch
}

// ChainOpt is a chain option.
type ChainOpt func(*Chain)

// WithSharedChainLearnables shares the learnables from another chain.
func WithSharedChainLearnables(shared *Chain) func(*Chain) {
	return func(c *Chain) {
		c.sharedLearnables = shared
	}
}

// WithLayerOpts adds the given layer opts to all layers.
func WithLayerOpts(opts ...CompileOpt) func(*Chain) {
	return func(c *Chain) {
		c.compileOpts = opts
	}
}

// Compile the chain of layers into the model.
func (c *Chain) Compile(graph *g.ExprGraph, opts ...ChainOpt) {
	for _, opt := range opts {
		opt(c)
	}
	if c.sharedLearnables != nil {
		for i, layer := range c.Layers {
			compileOpts := append(c.compileOpts, WithSharedLearnables(c.sharedLearnables.layers[i]))
			l := layer.Compile(graph, compileOpts...)
			c.layers = append(c.layers, l)
		}
		return
	}
	for _, layer := range c.Layers {
		l := layer.Compile(graph, c.compileOpts...)
		c.layers = append(c.layers, l)
	}
}
