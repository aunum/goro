package model

import (
	"github.com/aunum/log"

	g "gorgonia.org/gorgonia"
)

// Loss is the loss of a model.
type Loss interface {
	// Comput the loss.
	Compute(yHat, y *g.Node) (loss *g.Node, err error)

	// Clone the loss to another graph.
	CloneTo(graph *g.ExprGraph, opts ...CloneOpt) Loss

	// Inputs return any inputs the loss function utilizes.
	Inputs() Inputs
}

// MSE is standard mean squared error loss.
var MSE = &MSELoss{}

// MSELoss is mean squared error loss.
type MSELoss struct{}

// Compute the loss
func (m *MSELoss) Compute(yHat, y *g.Node) (loss *g.Node, err error) {
	loss, err = g.Sub(yHat, y)
	if err != nil {
		return nil, err
	}
	loss, err = g.Square(loss)
	if err != nil {
		return nil, err
	}
	loss, err = g.Mean(loss)
	if err != nil {
		return nil, err
	}
	return
}

// CloneTo another graph.
func (m *MSELoss) CloneTo(graph *g.ExprGraph, opts ...CloneOpt) Loss {
	return &MSELoss{}
}

// Inputs returns any inputs the loss function utilizes.
func (m *MSELoss) Inputs() Inputs {
	return Inputs{}
}

// CrossEntropy loss.
var CrossEntropy = &CrossEntropyLoss{}

// CrossEntropyLoss is standard cross entropy loss.
type CrossEntropyLoss struct{}

// Compute the loss.
func (c *CrossEntropyLoss) Compute(yHat, y *g.Node) (loss *g.Node, err error) {
	loss, err = g.Log(yHat)
	if err != nil {
		return nil, err
	}
	loss, err = g.HadamardProd(y, loss)
	if err != nil {
		return nil, err
	}
	loss, err = g.Neg(loss)
	if err != nil {
		return nil, err
	}
	loss, err = g.Mean(loss)
	if err != nil {
		return nil, err
	}
	return
}

// CloneTo another graph.
func (c *CrossEntropyLoss) CloneTo(graph *g.ExprGraph, opts ...CloneOpt) Loss {
	return &CrossEntropyLoss{}
}

// Inputs returns any inputs the loss function utilizes.
func (c *CrossEntropyLoss) Inputs() Inputs {
	return Inputs{}
}

// PseudoHuberLoss is a loss that is less sensetive to outliers.
// Can be thought of as absolute error when large, and quadratic when small.
// The larger the Delta param the steeper the loss.
//
// !blocked on https://github.com/gorgonia/gorgonia/issues/373
type PseudoHuberLoss struct {
	// Delta determines where the function switches behavior.
	Delta float32
}

// PseudoHuber is the Huber loss function.
var PseudoHuber = &PseudoHuberLoss{
	Delta: 1.0,
}

// NewPseudoHuberLoss return a new huber loss.
func NewPseudoHuberLoss(delta float32) *PseudoHuberLoss {
	return &PseudoHuberLoss{
		Delta: delta,
	}
}

// Compute the loss.
func (h *PseudoHuberLoss) Compute(yHat, y *g.Node) (loss *g.Node, err error) {
	loss, err = g.Sub(yHat, y)
	if err != nil {
		return nil, err
	}
	loss, err = g.Div(loss, g.NewScalar(yHat.Graph(), g.Float32, g.WithValue(float32(h.Delta))))
	if err != nil {
		return nil, err
	}
	loss, err = g.Square(loss)
	if err != nil {
		return nil, err
	}
	loss, err = g.Add(g.NewScalar(yHat.Graph(), g.Float32, g.WithValue(float32(1.0))), loss)
	if err != nil {
		return nil, err
	}
	loss, err = g.Sqrt(loss)
	if err != nil {
		return nil, err
	}
	loss, err = g.Sub(loss, g.NewScalar(yHat.Graph(), g.Float32, g.WithValue(float32(1.0))))
	if err != nil {
		return nil, err
	}
	deltaSquare, err := g.Square(g.NewScalar(yHat.Graph(), g.Float32, g.WithValue(float32(h.Delta))))
	if err != nil {
		return nil, err
	}
	loss, err = g.Mul(deltaSquare, loss)
	if err != nil {
		return nil, err
	}
	/// arg! https://github.com/gorgonia/gorgonia/issues/373
	loss, err = g.Sum(loss, 1)
	if err != nil {
		return nil, err
	}
	return
}

// CloneTo another graph.
func (h *PseudoHuberLoss) CloneTo(graph *g.ExprGraph, opts ...CloneOpt) Loss {
	return &PseudoHuberLoss{Delta: h.Delta}
}

// Inputs returns any inputs the loss function utilizes.
func (h *PseudoHuberLoss) Inputs() Inputs {
	return Inputs{}
}

// PseudoCrossEntropy loss.
var PseudoCrossEntropy = &PseudoCrossEntropyLoss{}

// PseudoCrossEntropyLoss is standard cross entropy loss.
type PseudoCrossEntropyLoss struct{}

// Compute the loss.
func (c *PseudoCrossEntropyLoss) Compute(yHat, y *g.Node) (loss *g.Node, err error) {
	loss, err = g.HadamardProd(yHat, y)
	if err != nil {
		log.Fatal(err)
	}
	loss, err = g.Mean(loss)
	if err != nil {
		log.Fatal(err)
	}
	loss, err = g.Neg(loss)
	if err != nil {
		log.Fatal(err)
	}
	return
}

// CloneTo another graph.
func (c *PseudoCrossEntropyLoss) CloneTo(graph *g.ExprGraph, opts ...CloneOpt) Loss {
	return &PseudoCrossEntropyLoss{}
}

// Inputs returns any inputs the loss function utilizes.
func (c *PseudoCrossEntropyLoss) Inputs() Inputs {
	return Inputs{}
}
