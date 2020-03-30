// Package model provides an interface for machine learning models.
package model

import (
	"fmt"
	golog "log"

	"github.com/aunum/gold/pkg/v1/track"
	cgraph "github.com/aunum/goro/pkg/v1/common/graph"
	"github.com/aunum/goro/pkg/v1/layer"
	"github.com/aunum/log"

	g "gorgonia.org/gorgonia"
)

// Model is a prediction model.
type Model interface {
	// Compile the model.
	Compile(x InputOr, y *Input, opts ...Opt) error

	// Predict x.
	Predict(x g.Value) (prediction g.Value, err error)

	// Fit x to y.
	Fit(x ValueOr, y g.Value) error

	// FitBatch fits x to y as batches.
	FitBatch(x ValueOr, y g.Value) error

	// PredictBatch predicts x as a batch
	PredictBatch(x g.Value) (prediction g.Value, err error)

	// ResizeBatch resizes the batch graphs.
	ResizeBatch(n int) error

	// Visualize the model by graph name.
	Visualize(name string)

	// Graph returns the expression graph for the model.
	Graphs() map[string]*g.ExprGraph

	// X is the inputs to the model.
	X() InputOr

	// Y is the expected output of the model.
	Y() *Input

	// Learnables for the model.
	Learnables() g.Nodes
}

// Sequential model.
type Sequential struct {
	// Chain of layers in the model.
	Chain *layer.Chain

	// Tracker of values.
	Tracker   *track.Tracker
	noTracker bool
	logger    *log.Logger

	name string

	x   Inputs
	y   *Input
	fwd *Input

	trainChain, trainBatchChain   *layer.Chain
	onlineChain, onlineBatchChain *layer.Chain
	backwardChain                 *layer.Chain

	trainGraph, trainBatchGraph   *g.ExprGraph
	onlineGraph, onlineBatchGraph *g.ExprGraph
	backwardGraph                 *g.ExprGraph

	xTrain, xTrainBatch         Inputs
	xTrainFwd, xTrainBatchFwd   *Input
	xOnline, xOnlineBatch       Inputs
	xOnlineFwd, xOnlineBatchFwd *Input

	yTrain, yTrainBatch *Input

	trainPredVal, trainBatchPredVal   g.Value
	onlinePredVal, onlineBatchPredVal g.Value

	loss                      Loss
	trainLoss, trainBatchLoss Loss

	metrics Metrics

	batchSize int
	optimizer g.Solver

	trainVM, trainBatchVM   g.VM
	onlineVM, onlineBatchVM g.VM
	backwardVM              g.VM
	vmOpts                  []g.VMOpt
}

// NewSequential returns a new sequential model.
func NewSequential(name string) (*Sequential, error) {
	return &Sequential{
		Chain:     layer.NewChain(),
		name:      name,
		batchSize: 32,
		metrics:   AllMetrics,
	}, nil
}

// Opt is a model option.
type Opt func(Model)

// Metric tracked by the model.
type Metric string

const (
	// TrainLossMetric is the metric for training loss.
	TrainLossMetric Metric = "train_loss"

	// TrainBatchLossMetric is the metric for batch training loss.
	TrainBatchLossMetric Metric = "train_batch_loss"
)

// Metrics is a set of metric.
type Metrics []Metric

// Contains tells whether the set contains the given metric.
func (m Metrics) Contains(metric Metric) bool {
	for _, mt := range m {
		if mt == metric {
			return true
		}
	}
	return false
}

// AllMetrics are all metrics.
var AllMetrics = Metrics{TrainLossMetric, TrainBatchLossMetric}

// WithMetrics sets the metrics that the model should track.
// Defaults to AllMetrics.
func WithMetrics(metrics ...Metric) func(Model) {
	return func(m Model) {
		switch t := m.(type) {
		case *Sequential:
			t.metrics = metrics
		default:
			log.Fatal("unknown model type")
		}
	}
}

// WithLoss uses a specific loss function with the model.
// Defaults to MSE.
func WithLoss(loss Loss) func(Model) {
	return func(m Model) {
		switch t := m.(type) {
		case *Sequential:
			t.loss = loss
		default:
			log.Fatal("unknown model type")
		}
	}
}

// WithOptimizer uses a specific optimizer function.
// Defaults to Adam.
func WithOptimizer(optimizer g.Solver) func(Model) {
	return func(m Model) {
		switch t := m.(type) {
		case *Sequential:
			t.optimizer = optimizer
		default:
			log.Fatal("unknown model type")
		}
	}
}

// WithTracker adds a tracker to the model, if not provided one will be created.
func WithTracker(tracker *track.Tracker) func(Model) {
	return func(m Model) {
		switch t := m.(type) {
		case *Sequential:
			t.Tracker = tracker
		default:
			log.Fatal("unknown model type")
		}
	}
}

// WithoutTracker uses no tracking with the model.
func WithoutTracker() func(Model) {
	return func(m Model) {
		switch t := m.(type) {
		case *Sequential:
			t.noTracker = true
		default:
			log.Fatal("unknown model type")
		}
	}
}

// WithBatchSize sets the batch size for the model.
// Defaults to 32.
func WithBatchSize(size int) func(Model) {
	return func(m Model) {
		switch t := m.(type) {
		case *Sequential:
			t.batchSize = size
		default:
			log.Fatal("unknown model type")
		}
	}
}

// WithGraphLogger adds a logger to the model which will print out the graph operations
// as they occur.
func WithGraphLogger(log *golog.Logger) func(Model) {
	return func(m Model) {
		switch t := m.(type) {
		case *Sequential:
			t.vmOpts = append(t.vmOpts, g.WithLogger(log))
		default:
			log.Fatal("unknown model type")
		}
	}
}

// WithLogger adds a logger to the model.
func WithLogger(logger *log.Logger) func(Model) {
	return func(m Model) {
		switch t := m.(type) {
		case *Sequential:
			t.logger = logger
		default:
			log.Fatal("unknown model type")
		}
	}
}

// AddLayer adds a layer.
func (s *Sequential) AddLayer(layer layer.Config) {
	s.Chain.Add(layer)
}

// AddLayers adds a number of layer.
func (s *Sequential) AddLayers(layers ...layer.Config) {
	for _, layer := range layers {
		s.Chain.Add(layer)
	}
}

// Fwd tells the model which input should be sent through the layer.
// If not provided, the first input will be used.
func (s *Sequential) Fwd(x *Input) {
	s.fwd = x
}

// Compile the model.
func (s *Sequential) Compile(x InputOr, y *Input, opts ...Opt) error {
	s.x = x.Inputs()
	err := y.Validate()
	if err != nil {
		return err
	}
	s.y = y

	for _, opt := range opts {
		opt(s)
	}
	if s.logger == nil {
		s.logger = log.DefaultLogger
	}
	if s.loss == nil {
		s.loss = MSE
	}
	if s.optimizer == nil {
		s.optimizer = g.NewAdamSolver()
	}
	if s.Tracker == nil && !s.noTracker {
		tracker, err := track.NewTracker(track.WithLogger(s.logger))
		if err != nil {
			return err
		}
		s.Tracker = tracker
	}
	if s.fwd == nil {
		s.fwd = x.Inputs()[0]
		err = s.fwd.Validate()
		if err != nil {
			return err
		}
		s.logger.Infof("setting forward for layers to input %q", s.fwd.Name())
	}
	err = s.buildTrainGraph(s.x, s.y)
	if err != nil {
		return err
	}
	err = s.buildTrainBatchGraph(s.x, s.y)
	if err != nil {
		return err
	}
	err = s.buildOnlineGraph(s.x)
	if err != nil {
		return err
	}
	err = s.buildOnlineBatchGraph(s.x)
	if err != nil {
		return err
	}
	return nil
}

func (s *Sequential) buildTrainGraph(x Inputs, y *Input) (err error) {
	s.trainGraph = g.NewGraph()

	s.trainLoss = s.loss.CloneTo(s.trainGraph)
	for _, input := range x {
		if i, err := s.trainLoss.Inputs().Get(input.Name()); err == nil {
			s.xTrain = append(s.xTrain, i)
			continue
		}
		i := input.CloneTo(s.trainGraph)
		s.xTrain = append(s.xTrain, i)
	}

	s.xTrainFwd, err = s.xTrain.Get(s.fwd.Name())
	if err != nil {
		return err
	}

	s.trainLoss = s.loss.CloneTo(s.trainGraph)

	s.yTrain = y.Clone()
	s.yTrain.Compile(s.trainGraph)

	s.trainChain = s.Chain.Clone()
	s.trainChain.Compile(s.trainGraph)

	prediction, err := s.trainChain.Fwd(s.xTrainFwd.Node())
	if err != nil {
		return err
	}
	g.Read(prediction, &s.trainPredVal)

	loss, err := s.trainLoss.Compute(prediction, s.yTrain.Node())
	if err != nil {
		return err
	}

	if s.metrics.Contains(TrainLossMetric) {
		if s.Tracker != nil {
			s.Tracker.TrackValue("train_loss", loss, track.WithNamespace(s.name))
		}
	}

	_, err = g.Grad(loss, s.trainChain.Learnables()...)
	if err != nil {
		return err
	}

	vmOpts := []g.VMOpt{}
	copy(vmOpts, s.vmOpts)
	vmOpts = append(vmOpts, g.BindDualValues(s.trainChain.Learnables()...))
	s.trainVM = g.NewTapeMachine(s.trainGraph, vmOpts...)
	return nil
}

func (s *Sequential) buildTrainBatchGraph(x Inputs, y *Input) (err error) {
	s.trainBatchGraph = g.NewGraph()

	s.trainBatchLoss = s.loss.CloneTo(s.trainBatchGraph, AsBatch(s.batchSize))
	for _, input := range x {
		// TODO: need to validate input names for duplicates.
		if i, err := s.trainBatchLoss.Inputs().Get(input.Name()); err == nil {
			s.xTrainBatch = append(s.xTrainBatch, i)
			continue
		}
		i := input.CloneTo(s.trainBatchGraph, AsBatch(s.batchSize))
		s.xTrainBatch = append(s.xTrainBatch, i)
	}

	s.xTrainBatchFwd, err = s.xTrainBatch.Get(NameAsBatch(s.fwd.Name()))
	if err != nil {
		return err
	}

	s.yTrainBatch = s.y.AsBatch(s.batchSize)
	s.yTrainBatch.Compile(s.trainBatchGraph)

	s.trainBatchChain = s.Chain.Clone()
	s.trainBatchChain.Compile(s.trainBatchGraph, layer.WithSharedChainLearnables(s.trainChain), layer.WithLayerOpts(layer.AsBatch()))

	prediction, err := s.trainBatchChain.Fwd(s.xTrainBatchFwd.Node())
	if err != nil {
		return err
	}
	g.Read(prediction, &s.trainBatchPredVal)
	loss, err := s.trainBatchLoss.Compute(prediction, s.yTrainBatch.Node())
	if err != nil {
		return err
	}

	if s.metrics.Contains(TrainBatchLossMetric) {
		if s.Tracker != nil {
			s.Tracker.TrackValue("train_batch_loss", loss, track.WithNamespace(s.name))
		}
	}

	_, err = g.Grad(loss, s.trainBatchChain.Learnables()...)
	if err != nil {
		return err
	}

	vmOpts := []g.VMOpt{}
	copy(vmOpts, s.vmOpts)
	vmOpts = append(vmOpts, g.BindDualValues(s.trainBatchChain.Learnables()...))
	s.trainBatchVM = g.NewTapeMachine(s.trainBatchGraph, vmOpts...)

	return nil
}

func (s *Sequential) buildOnlineGraph(x Inputs) (err error) {
	s.onlineGraph = g.NewGraph()

	s.xOnline = s.x.Clone()
	s.xOnline.Compile(s.onlineGraph)

	s.xOnlineFwd, err = s.xOnline.Get(s.fwd.Name())
	if err != nil {
		return err
	}

	s.onlineChain = s.Chain.Clone()
	s.onlineChain.Compile(s.onlineGraph, layer.WithSharedChainLearnables(s.trainChain))

	prediction, err := s.onlineChain.Fwd(s.xOnlineFwd.Node())
	if err != nil {
		return err
	}
	g.Read(prediction, &s.onlinePredVal)

	vmOpts := []g.VMOpt{}
	copy(vmOpts, s.vmOpts)
	s.onlineVM = g.NewTapeMachine(s.onlineGraph, vmOpts...)
	return nil
}

func (s *Sequential) buildOnlineBatchGraph(x Inputs) (err error) {
	s.onlineBatchGraph = g.NewGraph()

	for _, input := range x {
		if input.Name() == s.fwd.Name() {
			s.xOnlineBatchFwd = input.AsBatch(s.batchSize)
			s.xOnlineBatchFwd.Compile(s.onlineBatchGraph)
			s.xOnlineBatch = append(s.xOnlineBatch, s.xOnlineBatchFwd)
			continue
		}
		i := input.CloneTo(s.onlineBatchGraph)
		s.xOnlineBatch = append(s.xOnlineBatch, i)
	}

	s.xOnlineBatchFwd, err = s.xOnlineBatch.Get(NameAsBatch(s.fwd.Name()))
	if err != nil {
		return err
	}

	s.onlineBatchChain = s.Chain.Clone()
	s.onlineBatchChain.Compile(s.onlineBatchGraph, layer.WithSharedChainLearnables(s.trainChain), layer.WithLayerOpts(layer.AsBatch()))

	prediction, err := s.onlineBatchChain.Fwd(s.xOnlineBatchFwd.Node())
	if err != nil {
		return err
	}
	g.Read(prediction, &s.onlineBatchPredVal)

	vmOpts := []g.VMOpt{}
	copy(vmOpts, s.vmOpts)
	s.onlineBatchVM = g.NewTapeMachine(s.onlineBatchGraph, vmOpts...)
	return nil
}

// ResizeBatch will resize the batch graph.
// Note: this is expensive as it recompiles the graph.
func (s *Sequential) ResizeBatch(n int) (err error) {
	log.Debugf("resizing batch graphs to %d", n)
	s.batchSize = n
	s.xTrainBatch = Inputs{}
	s.xTrainBatchFwd = nil
	err = s.buildTrainBatchGraph(s.x, s.y)
	if err != nil {
		return
	}
	s.xOnlineBatch = Inputs{}
	s.xOnlineBatchFwd = nil
	return s.buildOnlineBatchGraph(s.x)
}

// Predict x.
func (s *Sequential) Predict(x g.Value) (prediction g.Value, err error) {
	err = s.xOnlineFwd.Set(x)
	if err != nil {
		return prediction, err
	}
	err = s.onlineVM.RunAll()
	if err != nil {
		return prediction, err
	}
	prediction = s.onlinePredVal
	s.onlineVM.Reset()
	return
}

// PredictBatch predicts x as a batch.
func (s *Sequential) PredictBatch(x g.Value) (prediction g.Value, err error) {
	err = s.xOnlineBatchFwd.Set(x)
	if err != nil {
		return prediction, err
	}
	err = s.onlineBatchVM.RunAll()
	if err != nil {
		return prediction, err
	}
	prediction = s.onlineBatchPredVal
	s.onlineBatchVM.Reset()
	return
}

// Fit x to y.
func (s *Sequential) Fit(x ValueOr, y g.Value) error {
	err := s.yTrain.Set(y)
	if err != nil {
		return err
	}
	xVals := ValuesFrom(x)
	err = s.xTrain.Set(xVals)
	if err != nil {
		return err
	}

	err = s.trainVM.RunAll()
	if err != nil {
		return err
	}
	grads := g.NodesToValueGrads(s.trainChain.Learnables())
	s.optimizer.Step(grads)
	s.trainVM.Reset()
	return nil
}

// FitBatch fits x to y as a batch.
func (s *Sequential) FitBatch(x ValueOr, y g.Value) error {
	err := s.yTrainBatch.Set(y)
	if err != nil {
		return err
	}

	xVals := ValuesFrom(x)
	err = s.xTrainBatch.Set(xVals)
	if err != nil {
		return err
	}

	err = s.trainBatchVM.RunAll()
	if err != nil {
		return err
	}
	// log.Infovb("pred val", s.trainBatchPredVal)
	grads := g.NodesToValueGrads(s.trainBatchChain.Learnables())
	s.optimizer.Step(grads)
	s.trainBatchVM.Reset()
	return nil
}

// Visualize the model by graph name.
func (s *Sequential) Visualize(name string) {
	cgraph.Visualize(s.Graphs()[name])
}

// Graphs returns the expression graphs for the model.
func (s *Sequential) Graphs() map[string]*g.ExprGraph {
	return map[string]*g.ExprGraph{
		"train":       s.trainGraph,
		"trainBatch":  s.trainBatchGraph,
		"online":      s.onlineGraph,
		"onlineBatch": s.onlineBatchGraph,
	}
}

// X is is the input to the model.
func (s *Sequential) X() InputOr {
	return s.x
}

// Y is is the output of the model.
func (s *Sequential) Y() *Input {
	return s.y
}

// Learnables are the model learnables.
func (s *Sequential) Learnables() g.Nodes {
	return s.trainChain.Learnables()
}

// CloneLearnablesTo another model.
func (s *Sequential) CloneLearnablesTo(to *Sequential) error {
	desired := s.trainChain.Learnables()
	destination := to.trainChain.Learnables()
	if len(desired) != len(destination) {
		return fmt.Errorf("models must be identical to clone learnables")
	}
	for i, learnable := range destination {
		c := desired[i].Clone()
		err := g.Let(learnable, c.(*g.Node).Value())
		if err != nil {
			return err
		}
	}
	new := to.trainChain.Learnables()
	shared := map[string]*layer.Chain{
		"trainBatch":  to.trainBatchChain,
		"online":      to.onlineChain,
		"onlineBatch": to.onlineBatchChain,
	}
	for name, chain := range shared {
		s.logger.Debugv("chain", name)
		for i, learnable := range chain.Learnables() {
			err := g.Let(learnable, new[i].Value())
			if err != nil {
				return err
			}
			s.logger.Debugvb(learnable.Name(), learnable.Value())
		}
	}
	return nil
}

// Opts are optsion for a model
type Opts struct {
	opts []Opt
}

// NewOpts returns a new set of options for a model.
func NewOpts() *Opts {
	return &Opts{opts: []Opt{}}
}

// Add an option to the options.
func (o *Opts) Add(opts ...Opt) {
	o.opts = append(o.opts, opts...)
}

// Values are the options.
func (o *Opts) Values() []Opt {
	return o.opts
}
