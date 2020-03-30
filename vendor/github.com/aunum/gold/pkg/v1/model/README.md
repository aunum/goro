# Model

Model is a high-level helper library for Gorgonia, it aims to have a similar feel to Keras.

## Usage

```go
import (
    . "github.com/aunum/gold/pkg/v1/model"
    l "github.com/aunum/gold/pkg/v1/model/layers"
)

// create the 'x' input
x := NewInput("x", []int{784})

// create the 'y' or expect output
y := NewInput("y", []int{10})

// create a new sequential model with the name 'mnist'
model, _ := NewSequential("mnist")

// add layers to the model
model.AddLayers(
    l.NewFC(784, 300, l.WithActivation(l.ReLU), l.WithInit(g.GlorotN(1)), l.WithName("w0")),
    l.NewFC(300, 100, l.WithActivation(l.ReLU), l.WithInit(g.GlorotN(1)), l.WithName("w1")),
    l.NewFC(100, 10, l.WithActivation(l.Softmax), l.WithInit(g.GlorotN(1)), l.WithName("w2")),
)

// pick an optimizer
optimizer := g.NewRMSPropSolver()

// compile the model with options
model.Compile(xi, yi,
    WithOptimizer(optimizer),
    WithLoss(CrossEntropy),
    WithBatchSize(100),
)

// Use the model to predict an 'x'
prediction, _ := model.Predict(xTest)

// fit the model
model.Fit(xTrain, yTrain)

// Use the model to predict a batch of 'x'
prediction, _ = model.PredictBatch(xTestBatch)

// fit the model with a batch
model.FitBatch(xTrainBatch, yTrainBatch)
```

## Examples
See the [experiments](./experiments) folder for example implementations.