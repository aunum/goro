package model

import (
	"fmt"

	l "github.com/aunum/goro/pkg/v1/layer"
	"github.com/aunum/log"

	g "gorgonia.org/gorgonia"
	t "gorgonia.org/tensor"
)

// InputOr is a sum type of input or inputs.
type InputOr interface {
	// Input present.
	Input() *Input

	// Inputs present.
	Inputs() Inputs
}

// Input into the model.
type Input struct {
	name  string
	shape t.Shape
	dtype t.Dtype
	node  *g.Node
}

// InputOpt is an input option.
type InputOpt func(*Input)

// AsType explicitly sets the type of the input.
// Defaults to Float32.
func AsType(dtype t.Dtype) func(*Input) {
	return func(i *Input) {
		i.dtype = dtype
	}
}

// NewInput returns a new input.
func NewInput(name string, shape t.Shape, opts ...InputOpt) *Input {
	i := &Input{
		name:  name,
		shape: shape,
		dtype: t.Float32,
	}
	for _, opt := range opts {
		opt(i)
	}
	return i
}

// Input implements an In.
func (i *Input) Input() *Input {
	return i
}

// Inputs implements an In.
func (i *Input) Inputs() Inputs {
	return Inputs{i}
}

// Compile an input into a graph.
func (i *Input) Compile(graph *g.ExprGraph, opts ...InputOpt) *g.Node {
	if i.node != nil {
		panic(fmt.Sprintf("trying ot compile input %q that is already compiled", i.name))
	}
	for _, opt := range opts {
		opt(i)
	}
	var n *g.Node
	if i.shape.IsScalar() {
		n = g.NewScalar(graph, i.dtype, g.WithName(i.name))
	} else {
		n = g.NewTensor(graph, i.dtype, len(i.shape), g.WithShape(i.shape...), g.WithName(i.name))
	}
	i.node = n
	return n
}

// Shape is the shape of the input.
func (i *Input) Shape() t.Shape {
	return i.shape
}

// Name of the input.
func (i *Input) Name() string {
	return i.name
}

// DType data type of the input.
func (i *Input) DType() t.Dtype {
	return i.dtype
}

// Node returns the graph node.
func (i *Input) Node() *g.Node {
	return i.node
}

// CloneOpt are clone options.
type CloneOpt func(*Input)

// AsBatch adds a batch size to a clone opt
func AsBatch(size int) func(*Input) {
	return func(i *Input) {
		*i = *i.AsBatch(size)
	}
}

// Clone an input.
func (i *Input) Clone(opts ...CloneOpt) *Input {
	ret := &Input{
		name:  i.name,
		shape: i.shape.Clone(),
		dtype: i.dtype,
	}
	for _, opt := range opts {
		opt(ret)
	}
	return ret
}

// CloneTo clones an input with the node value (if present) to another graph.
func (i *Input) CloneTo(graph *g.ExprGraph, opts ...CloneOpt) *Input {
	ret := i.Clone(opts...)
	if ret.node != nil {
		ret.node = i.node.CloneTo(graph)
	} else {
		ret.Compile(graph)
	}
	return ret
}

// Check that the dimensions and type of the given value are congruent with the
// expected input.
func (i *Input) Check(value g.Value) error {
	vShape := value.Shape()
	if len(vShape) != len(i.Shape()) {
		return fmt.Errorf("shape mismatch: input %v expects %v got %v", i.name, i.shape, vShape)
	}

	for index, s := range i.Shape() {
		if vShape[index] != s {
			return fmt.Errorf("shape mismatch: input %v expects %v got %v", i.name, i.shape, vShape)
		}
	}
	if i.dtype != value.Dtype() {
		return fmt.Errorf("data type mismatch: input %v expects %v got %v", i.name, i.dtype, value.Dtype())
	}
	return nil
}

// Set the value of the input.
func (i *Input) Set(value g.Value) error {
	err := i.Check(value)
	if err != nil {
		return err
	}
	return g.Let(i.node, value)
}

// AsBatch converts an input to a batched representation.
func (i *Input) AsBatch(size int) *Input {
	ret := i.Clone()
	if len(ret.Shape()) == 1 {
		err := ret.OneOfMany()
		if err != nil {
			panic(err)
		}
	}
	ret.shape[0] = size
	ret.name = NameAsBatch(ret.Name())
	return ret
}

// NameAsBatch takes an input name and converts it to its batch name.
func NameAsBatch(name string) string {
	if name == "" {
		return ""
	}
	return fmt.Sprintf("%v_batch", name)
}

// OneOfMany normalizes the input shape to be one of many.
// Any incoming singular input will also be normalized to this shape.
func (i *Input) OneOfMany() (err error) {
	i.shape = append([]int{1}, i.shape...)
	if i.node != nil {
		i.node, err = g.Reshape(i.node, i.shape)
		if err != nil {
			return err
		}
	}
	return nil
}

// EnsureBatch checks that the first dimension is 1 or reshapes it to be so.
func (i *Input) EnsureBatch() *Input {
	if i.Shape()[0] != 1 || len(i.Shape()) == 1 {
		i.shape = append([]int{1}, i.shape...)
		log.Debugf("reshaping %v to %v to have a batch of 1", i.Name(), i.Shape())
	}
	return i
}

// Validate the input.
func (i *Input) Validate() error {
	if len(i.shape) == 0 {
		return fmt.Errorf("no input shape provided")
	}
	if i.shape[0] != 1 {
		return fmt.Errorf("input shape %q %v must be a scalar or have the first dimension 1 e.g. [1, 4]", i.name, i.shape)
	}
	return nil
}

// Squeeze returns the shape of the input with any leading dimensions of size 1 removed.
func (i *Input) Squeeze() t.Shape {
	if i.Shape()[0] == 1 {
		return i.Shape().Clone()[1:]
	}
	return i.Shape().Clone()
}

// Inputs is a slice of input.
type Inputs []*Input

// Input implements an In, returns the first element of the inputs.
func (i Inputs) Input() *Input {
	return nil
}

// Inputs implements an In.
func (i Inputs) Inputs() Inputs {
	return i
}

// Get an input by name.
func (i Inputs) Get(name string) (*Input, error) {
	for _, input := range i {
		if input.name == name {
			return input, nil
		}
	}
	return nil, fmt.Errorf("could not find input %s", name)
}

// Compile all inputs into the given graph.
func (i Inputs) Compile(graph *g.ExprGraph, opts ...InputOpt) g.Nodes {
	nodes := g.Nodes{}
	for _, input := range i {
		n := input.Compile(graph, opts...)
		nodes = append(nodes, n)
	}
	return nodes
}

// Clone the inputs.
func (i Inputs) Clone() Inputs {
	inputs := Inputs{}
	for _, input := range i {
		inputs = append(inputs, input.Clone())
	}
	return inputs
}

// Set the values to the inputs.
func (i Inputs) Set(values Values) error {
	for index, value := range values {
		err := i[index].Set(value)
		if err != nil {
			return err
		}
	}
	return nil
}

// Contains tests whether the given input set contains an input.
func (i Inputs) Contains(name string) bool {
	for _, input := range i {
		if input.Name() == name {
			return true
		}
	}
	return false
}

// InputLayer is an input layer to be used in a chain.
type InputLayer struct {
	input *Input
}

// AsLayer converts the input to a layer.
func (i *Input) AsLayer() l.Layer {
	return &InputLayer{
		input: i,
	}
}

// Compile the layer.
func (i *InputLayer) Compile(graph *g.ExprGraph, opts ...l.CompileOpt) {
	if i.Graph() != nil {
		return
	}
	i.input.Compile(graph)
}

// Fwd is a forward pass through the layer.
func (i *InputLayer) Fwd(x *g.Node) (*g.Node, error) {
	return x, nil
}

// Learnables returns all learnable nodes within this layer.
func (i *InputLayer) Learnables() g.Nodes {
	return g.Nodes{}
}

// Clone the layer.
func (i *InputLayer) Clone() l.Layer {
	return &InputLayer{
		input: i.input.Clone(),
	}
}

// Graph returns the graph for this layer.
func (i *InputLayer) Graph() *g.ExprGraph {
	if i.input.Node() == nil {
		return nil
	}
	return i.input.Node().Graph()
}

// Node of the input.
func (i *InputLayer) Node() *g.Node {
	if i.input.Node() == nil {
		return nil
	}
	return i.input.Node()
}

// Values is a slice of value.
type Values []g.Value

// ValueOr is a sum type that represents a gorgonia.Value or []gorgonia.Value.
type ValueOr interface{}

// ValuesFrom returns the value as an array of gorgonia values.
func ValuesFrom(v ValueOr) Values {
	switch val := v.(type) {
	case g.Value:
		return []g.Value{val}
	case []g.Value:
		return val
	default:
		log.Fatalf("value type %v is not supported; gorgonia.Value and []gorgonia.Values are currently supported.", val)
	}
	return nil
}
