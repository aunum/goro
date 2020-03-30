package main

import (
	"github.com/aunum/gold/pkg/v1/common/num"
	"github.com/aunum/gold/pkg/v1/common/require"
	"github.com/aunum/gold/pkg/v1/dense"
	"github.com/aunum/goro/pkg/v1/layer"
	m "github.com/aunum/goro/pkg/v1/model"
	"github.com/aunum/log"

	g "gorgonia.org/gorgonia"
	"gorgonia.org/gorgonia/examples/mnist"
	"gorgonia.org/tensor"
)

func main() {
	// load the train set
	x, y, err := mnist.Load("train", "./testdata", g.Float32)
	require.NoError(err)

	exampleSize := x.Shape()[0]
	log.Infov("exampleSize", exampleSize)

	// load our test set
	testX, testY, err := mnist.Load("test", "./testdata", g.Float32)
	require.NoError(err)

	log.Infov("x example shape", x.Shape())
	log.Infov("y example shape", y.Shape())

	batchSize := 100
	log.Infov("batchsize", batchSize)

	batches := exampleSize / batchSize
	log.Infov("num batches", batches)

	xi := m.NewInput("x", []int{1, 1, 28, 28})
	log.Infov("x input shape", xi.Shape())

	yi := m.NewInput("y", []int{1, 10})
	log.Infov("y input shape", yi.Shape())

	model, err := m.NewSequential("mnist")
	require.NoError(err)

	model.AddLayers(
		layer.Conv2D{Input: 1, Output: 32, Width: 3, Height: 3},
		layer.MaxPooling2D{},
		layer.Conv2D{Input: 32, Output: 64, Width: 3, Height: 3},
		layer.MaxPooling2D{},
		layer.Conv2D{Input: 64, Output: 128, Width: 3, Height: 3},
		layer.MaxPooling2D{},
		layer.Flatten{},
		layer.FC{Input: 128 * 3 * 3, Output: 100},
		layer.FC{Input: 100, Output: 10, Activation: layer.Softmax},
	)

	optimizer := g.NewRMSPropSolver(g.WithBatchSize(float64(batchSize)))
	err = model.Compile(xi, yi,
		m.WithOptimizer(optimizer),
		m.WithLoss(m.CrossEntropy),
		m.WithBatchSize(batchSize),
	)
	require.NoError(err)

	epochs := 20

	log.Infov("epochs", epochs)
	for epoch := 0; epoch < epochs; epoch++ {
		for batch := 0; batch < batches; batch++ {
			start := batch * batchSize
			end := start + batchSize
			if start >= exampleSize {
				break
			}
			if end > exampleSize {
				end = exampleSize
			}

			xi, err := x.Slice(dense.MakeRangedSlice(start, end))
			require.NoError(err)
			xi.Reshape(batchSize, 1, 28, 28)

			yi, err := y.Slice(dense.MakeRangedSlice(start, end))
			require.NoError(err)
			yi.Reshape(batchSize, 10)

			err = model.FitBatch(xi, yi)
			require.NoError(err)
			model.Tracker.LogStep(epoch, batch)
		}
		accuracy, loss, err := evaluate(testX.(*tensor.Dense), testY.(*tensor.Dense), model, batchSize)
		require.NoError(err)
		log.Infof("completed train epoch %v with accuracy %v and loss %v", epoch, accuracy, loss)
	}
	err = model.Tracker.Clear()
	require.NoError(err)
}

func evaluate(x, y *tensor.Dense, model *m.Sequential, batchSize int) (acc, loss float32, err error) {
	exampleSize := x.Shape()[0]
	batches := exampleSize / batchSize

	accuracies := []float32{}
	for batch := 0; batch < batches; batch++ {
		start := batch * batchSize
		end := start + batchSize
		if start >= exampleSize {
			break
		}
		if end > exampleSize {
			end = exampleSize
		}

		xi, err := x.Slice(dense.MakeRangedSlice(start, end))
		require.NoError(err)
		xi.Reshape(batchSize, 1, 28, 28)

		yi, err := y.Slice(dense.MakeRangedSlice(start, end))
		require.NoError(err)
		yi.Reshape(batchSize, 10)

		yHat, err := model.PredictBatch(xi)
		require.NoError(err)

		acc, err := accuracy(yHat.(*tensor.Dense), yi.(*tensor.Dense), model)
		require.NoError(err)
		accuracies = append(accuracies, acc)
	}
	lossVal, err := model.Tracker.GetValue("mnist_train_batch_loss")
	require.NoError(err)
	loss = float32(lossVal.Scalar())
	acc = num.Mean(accuracies)
	return
}

func accuracy(yHat, y *tensor.Dense, model m.Model) (float32, error) {
	yMax, err := y.Argmax(1)
	require.NoError(err)

	yHatMax, err := yHat.Argmax(1)
	require.NoError(err)

	eq, err := tensor.ElEq(yMax, yHatMax, tensor.AsSameType())
	require.NoError(err)
	eqd := eq.(*tensor.Dense)
	len := eqd.Len()

	numTrue, err := eqd.Sum()
	if err != nil {
		return 0, err
	}

	return float32(numTrue.Data().(int)) / float32(len), nil
}
