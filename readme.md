# Number Recognition

This project implements a basic number recognition AI using backpropagation

## Training a Model

To train a new model, run:

```shell
./bin/training --train --out model.out
```

You can customize training parameters, for example:

```shell
./bin/training --train --out model.out --max-iters 1000 --tolerance 0.05 --lr 0.01
```

To iterate better on the training, you can write the hyperparameters in a config file (`model_config` is a example of it) and press space in the UI to run the training again

Once training finishes, you can press the key `T` to run the test data from the UI.

You can also run the tests manually at any time with:

```shell
./bin/training --test --model model.out
```

## Running the Guess GUI

You can launch the graphical interface to test and play with the model:
```shell
./bin/guess --model model.out
```

Custom Training Data

Currently, the API does not support changing the training dataset directly — but feel free to hack the code and modify it to suit your needs!

### Results

With the testing data, we achieve 96~% of accuracy depending on the hyperparameters. Wich is good to me :)

Checkout this cool looking result:

```shell
./bin/training --test --model models/BEST_lr_0.2000-tl_0.0100-itrs_50.model

+----------------------------------+
| Model Statistics                 |
+------------------------+---------+
| Correct Predicitions   |  9636   |
| Incorrect Predicitions |  0364   |
| Accuracy               |  96.36% |
| Error                  |  03.64% |
+------------------------+---------+
```