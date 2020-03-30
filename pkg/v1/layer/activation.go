package layer

import (
	"fmt"

	"github.com/aunum/log"
	"github.com/pkg/errors"

	g "gorgonia.org/gorgonia"
	t "gorgonia.org/tensor"
)

// ActivationFn is an activation function.
type ActivationFn interface {
	// Fwd is a forward pass through x.
	Fwd(x *g.Node) (*g.Node, error)

	// Clone the activation.
	Clone() ActivationFn
}

// SigmoidActivation is a sigmoid activation layer.
type SigmoidActivation struct{}

// Sigmoid activation function.
var Sigmoid = &SigmoidActivation{}

// NewSigmoid returns a new sigmoid activation layer.
func NewSigmoid() *SigmoidActivation {
	return &SigmoidActivation{}
}

// Fwd is a forward pass through the layer.
func (s *SigmoidActivation) Fwd(x *g.Node) (*g.Node, error) {
	return g.Sigmoid(x)
}

// Learnables returns all learnable nodes within this layer.
func (s *SigmoidActivation) Learnables() (n g.Nodes) {
	return n
}

// Compile the layer.
func (s *SigmoidActivation) Compile(x *g.Node, opts ...CompileOpt) {}

// Clone the activation.
func (s *SigmoidActivation) Clone() ActivationFn {
	return NewSigmoid()
}

// TanhActivation is a tanh activation layer.
type TanhActivation struct{}

// Tanh activation.
var Tanh = &TanhActivation{}

// NewTanh returns a new tanh activation layer.
func NewTanh() *TanhActivation {
	return &TanhActivation{}
}

// Fwd is a forward pass through the layer.
func (t *TanhActivation) Fwd(x *g.Node) (*g.Node, error) {
	return g.Tanh(x)
}

// Learnables returns all learnable nodes within this layer.
func (t *TanhActivation) Learnables() (n g.Nodes) {
	return n
}

// Compile the layer.
func (t *TanhActivation) Compile(x *g.Node, opts ...CompileOpt) {}

// Clone the activation.
func (t *TanhActivation) Clone() ActivationFn {
	return NewTanh()
}

// ReLUActivation is a relu activation layer.
type ReLUActivation struct{}

// ReLU activation.
var ReLU = &ReLUActivation{}

// NewReLU returns a new relu activation layer.
func NewReLU() *ReLUActivation {
	return &ReLUActivation{}
}

// Fwd is a forward pass through the layer.
func (r *ReLUActivation) Fwd(x *g.Node) (*g.Node, error) {
	return g.Rectify(x)
}

// Learnables returns all learnable nodes within this layer.
func (r *ReLUActivation) Learnables() (n g.Nodes) {
	return n
}

// Compile the layer.
func (r *ReLUActivation) Compile(x *g.Node, opts ...CompileOpt) {}

// Clone the activation.
func (r *ReLUActivation) Clone() ActivationFn {
	return NewReLU()
}

// LeakyReLUActivation is a leaky relu activation layer.
type LeakyReLUActivation struct {
	alpha float64
}

// LeakyReLU is default leaky relu activation.
var LeakyReLU = &LeakyReLUActivation{0.01}

// NewLeakyReLU returns a new leaky relu activation layer.
func NewLeakyReLU(alpha float64) *LeakyReLUActivation {
	return &LeakyReLUActivation{alpha: alpha}
}

// Fwd is a forward pass through the layer.
func (r *LeakyReLUActivation) Fwd(x *g.Node) (*g.Node, error) {
	return g.LeakyRelu(x, r.alpha)
}

// Learnables returns all learnable nodes within this layer.
func (r *LeakyReLUActivation) Learnables() (n g.Nodes) {
	return n
}

// Compile the layer.
func (r *LeakyReLUActivation) Compile(x *g.Node, opts ...CompileOpt) {}

// Clone the activation.
func (r *LeakyReLUActivation) Clone() ActivationFn {
	return NewLeakyReLU(r.alpha)
}

// SoftmaxActivation is a softmax activation layer.
type SoftmaxActivation struct {
	axis []int
}

// Softmax is the default softmax activation.
var Softmax = &SoftmaxActivation{}

// NewSoftmax returns a new leaky softmax activation layer.
func NewSoftmax(axis ...int) *SoftmaxActivation {
	// if len(axis) == 0 {
	// 	axis = append(axis, 0)
	// }
	return &SoftmaxActivation{axis: axis}
}

// Fwd is a forward pass through the layer.
func (s *SoftmaxActivation) Fwd(x *g.Node) (*g.Node, error) {
	// fmt.Printf("running softmax with x shape: %v dims: %v \n", x.Shape(), x.Dims())
	return softMax(x, s.axis...)
}

// Learnables returns all learnable nodes within this layer.
func (s *SoftmaxActivation) Learnables() (n g.Nodes) {
	return n
}

// Compile the layer.
func (s *SoftmaxActivation) Compile(x *g.Node, opts ...CompileOpt) {}

// Clone the activation.
func (s *SoftmaxActivation) Clone() ActivationFn {
	return NewSoftmax(s.axis...)
}

// LinearActivation is a linear (identity) activation layer.
type LinearActivation struct{}

// Linear activation.
var Linear = &LinearActivation{}

// NewLinear is a linear activation layer.
func NewLinear() *LinearActivation {
	return &LinearActivation{}
}

// Fwd is a forward pass through the layer.
func (l *LinearActivation) Fwd(x *g.Node) (*g.Node, error) {
	return x, nil
}

// Learnables returns all learnable nodes within this layer.
func (l *LinearActivation) Learnables() (n g.Nodes) {
	return n
}

// Compile the layer.
func (l *LinearActivation) Compile(x *g.Node, opts ...CompileOpt) {}

// Clone the activation.
func (l *LinearActivation) Clone() ActivationFn {
	return NewLinear()
}

// softMax performs softmax on the input. Specifically this is used:
//		e^(a[i]) / sum((e^(a[i])))
// For a more numerically stable SoftMax, use StableSoftMax.
//
// This is ripped from Gorgonia core and tweaked as there was a bug in it https://github.com/gorgonia/gorgonia/issues/373
// which is currently being worked on.
func softMax(a *g.Node, axes ...int) (retVal *g.Node, err error) {
	aShape := a.Shape()

	if aShape[0] == 1 {
		aShape = aShape[1:]
		a, err = g.Reshape(a, aShape)
		log.Debugf("a reshaped to %v", a.Shape())
	}
	axis := aShape.Dims() - 1 // default: last dim
	if a.IsColVec() || (a.IsVector() && !a.IsRowVec()) {
		axis = 0
	}

	if len(axes) > 0 {
		if axes[0] >= axis+1 || axes[0] < 0 {
			return nil, fmt.Errorf("Cannot perform SoftMax on axis %d. Input has shape %v", axes[0], a.Shape())
		}
		axis = axes[0]
	}
	var exp, sum *g.Node
	if exp, err = g.Exp(a); err != nil {
		return nil, err
	}
	if sum, err = g.Sum(exp, axis); err != nil {
		return nil, err
	}

	if sum.IsScalar() {
		return g.HadamardDiv(exp, sum)
	}

	// reshape if necessary
	ss := sum.Shape()
	diff := exp.Shape().Dims() - ss.Dims()

	// TODO: multirank softmax
	if diff > 0 {
		newShape := t.Shape(t.BorrowInts(ss.Dims() + diff))
		copy(newShape, ss)
		copy(newShape[axis+1:], newShape[axis:])
		newShape[axis] = 1

		if sum, err = g.Reshape(sum, newShape); err != nil {
			return nil, errors.Wrap(err, "Failed to reshape")
		}
	}
	retVal, err = g.BroadcastHadamardDiv(exp, sum, nil, []byte{byte(axis)})
	if err != nil {
		return
	}
	return

}
