# Model

Model is a high-level helper library for Gorgonia, it aims to have a similar feel to Keras.

## Usage

```go
import (
    . "github.com/aunum/gold/pkg/v1/model"
    "github.com/aunum/gold/pkg/v1/model/layer"
)

// create the 'x' input
x := NewInput("x", []int{784})

// create the 'y' or expect output
y := NewInput("y", []int{10})

// create a new sequential model with the name 'mnist'
model, _ := NewSequential("mnist")

// add layers to the model
model.AddLayers(
    layer.FC{Input: 784, Output: 300},
    layer.FC{Input: 300, Output: 100},
    layer.FC{Input: 100, Output: 10, Activation: layer.Softmax},
)

// pick an optimizer
optimizer := g.NewRMSPropSolver()

// compile the model with options
model.Compile(x, y,
    WithOptimizer(optimizer),
    WithLoss(CrossEntropy),
    WithBatchSize(100),
)

// fit the model
model.Fit(xTrain, yTrain)

// Use the model to predict an 'x'
prediction, _ := model.Predict(xTest)

// fit the model with a batch
model.FitBatch(xTrainBatch, yTrainBatch)

// Use the model to predict a batch of 'x'
prediction, _ = model.PredictBatch(xTestBatch)
```

## Examples
See the [experiments](./experiments) folder for example implementations.