package layers

import (
	g "gorgonia.org/gorgonia"
)

// Chain of layers.
type Chain struct {
	// Layers are the layers to chain together.
	Layers []Layer

	sharedLearnables *Chain
	layerOpts        []LayerOpt
}

// NewChain returns a new chain of layers.
func NewChain(layers ...Layer) *Chain {
	return &Chain{
		Layers: layers,
	}
}

// Fwd is a forward pass thorugh all layers of the chain.
func (c *Chain) Fwd(x *g.Node) (prediction *g.Node, err error) {
	prediction = x
	for _, layer := range c.Layers {
		if prediction, err = layer.Fwd(prediction); err != nil {
			return nil, err
		}
	}
	return prediction, nil
}

// Learnables are all of the learnable parameters in the chain.
func (c *Chain) Learnables() g.Nodes {
	retVal := []*g.Node{}
	for _, layer := range c.Layers {
		retVal = append(retVal, layer.Learnables()...)
	}
	return retVal
}

// Add to the chain.
func (c *Chain) Add(l ...Layer) {
	for _, layer := range l {
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
func WithLayerOpts(opts ...LayerOpt) func(*Chain) {
	return func(c *Chain) {
		c.layerOpts = opts
	}
}

// Compile the chain of layers into the model.
func (c *Chain) Compile(graph *g.ExprGraph, opts ...ChainOpt) {
	for _, opt := range opts {
		opt(c)
	}
	if c.sharedLearnables != nil {
		for i, layer := range c.Layers {
			c.layerOpts = append(c.layerOpts, WithSharedLearnables(c.sharedLearnables.Layers[i]))
			layer.Compile(graph, c.layerOpts...)
		}
	}
	for _, layer := range c.Layers {
		layer.Compile(graph, c.layerOpts...)
	}
}
