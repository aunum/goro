package model_test

import (
	"fmt"
	golog "log"
	"os"
	"testing"

	"github.com/aunum/gold/pkg/v1/dense"
	. "github.com/aunum/gold/pkg/v1/model"
	"github.com/aunum/goro/pkg/v1/layer"
	"github.com/aunum/log"

	"github.com/stretchr/testify/require"
	g "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func TestSequential(t *testing.T) {
	batchSize := 10

	x := tensor.New(tensor.WithShape(batchSize, 5), tensor.WithBacking(tensor.Range(tensor.Float32, 0, 50)))
	xnot, err := x.Slice(dense.MakeRangedSlice(0, 1))
	require.NoError(t, err)

	x0 := xnot.Materialize().(*tensor.Dense)
	err = dense.ExpandDims(x0, 0)
	require.NoError(t, err)
	xi := NewInput("x", x0.Shape())
	fmt.Println("xi shape: ", xi.Shape())

	y := tensor.New(tensor.WithShape(batchSize, 2), tensor.WithBacking(tensor.Range(tensor.Float32, 15, 35)))
	ynot, err := y.Slice(dense.MakeRangedSlice(0, 1))
	require.NoError(t, err)

	y0 := ynot.Materialize().(*tensor.Dense)
	err = dense.ExpandDims(y0, 0)
	require.NoError(t, err)
	yi := NewInput("y", y0.Shape())
	fmt.Println("yi shape: ", yi.Shape())

	log.Infovb("x", x)
	log.Infovb("y", y)
	log.Break()

	log.Infov("x0", x0)
	log.Infov("y0", y0)
	log.Break()

	model, err := NewSequential("test")
	require.NoError(t, err)
	model.AddLayers(
		layer.FC{Input: 5, Output: 24, Activation: layer.Sigmoid, Name: "w0"},
		layer.FC{Input: 24, Output: 24, Activation: layer.Sigmoid, Name: "w1"},
		layer.FC{Input: 24, Output: 2, Activation: layer.Linear, Name: "w2"},
	)

	optimizer := g.NewAdamSolver()
	model.Fwd(xi)

	logger := golog.New(os.Stdout, "", 0)
	err = model.Compile(xi, yi,
		WithOptimizer(optimizer),
		WithLoss(MSE),
		WithBatchSize(batchSize),
		WithGraphLogger(logger),
	)
	require.NoError(t, err)
	log.Break()

	prediction, err := model.Predict(x0)
	require.NoError(t, err)
	log.Infov("y0", y0)
	log.Infov("initial single prediction", prediction)
	log.Break()

	numSteps := 10000
	log.Infof("fitting for %v steps", numSteps)
	for i := 0; i < numSteps; i++ {
		err = model.FitBatch(x, y)
		require.NoError(t, err)
	}
	log.Break()
	prediction, err = model.PredictBatch(x)
	require.NoError(t, err)
	log.Infovb("y", y)
	log.Infovb("final batch prediction", prediction)
	log.Break()

	prediction, err = model.Predict(x0)
	require.NoError(t, err)
	log.Infov("y0", y0)
	log.Infov("final single prediction", prediction)
}
