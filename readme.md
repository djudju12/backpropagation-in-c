# Number Recognition

This project implements a basic number recognition AI using 10 perceptrons — each one trained to recognize a digit from 0 to 9.
Training a Model

To train a new model, run:

```shell
./bin/perceptron --train --out model.out
```

You can customize training parameters, for example:

```shell
./bin/perceptron --train --out model.out --max-iters 1000 --tolerance 0.05 --lr 0.01 --threads 4
```

Once training finishes, the model is tested automatically and its performance stats are displayed.

You can also run the tests manually at any time with:

```shell
./bin/perceptron --test --model model.out
```

## Running the GUI

You can launch the graphical interface to test and play with the model:
```shell
./bin/perceptron --gui --model model.out
```

Custom Training Data

Currently, the API does not support changing the training dataset directly — but feel free to hack the code and modify it to suit your needs!

### Results

With the testing data, we achieve 92~% of accuracy. Wich is good to me :)
Testing with the GUI can be clunky, sometimes it just dont get quite right. BUT, i think mostly is the mouse fault (copium).

Checkout this cool looking result:

```shell
./bin/perceptron --test --model models/lr_0.2000-tl_0.0300-itrs_50.model
+----------------------------------+
| Model Statistics                 |
+------------------------+---------+
| Correct Predicitions   |  9193   |
| Incorrect Predicitions |  0807   |
| Accuracy               |  91.93% |
| Error                  |  08.07% |
+------------------------+---------+
```