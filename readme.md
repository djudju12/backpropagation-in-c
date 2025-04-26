# Number Recognition

This project implements a basic number recognition AI using 10 perceptrons — each one trained to recognize a digit from 0 to 9.
Training a Model

To train a new model, run:

```shell
./perceptron --train --out model.out
```

You can customize training parameters, for example:

```shell
./perceptron --train --out model.out --max-iters 1000 --tolerance 0.05 --lr 0.01 --threads 4
```

Once training finishes, the model is tested automatically and its performance stats are displayed.

You can also run the tests manually at any time with:

```shell
./perceptron --test --model model.out
```

## Running the GUI

You can launch the graphical interface to test and play with the model:
```shell
./perceptron --gui --model model.out
```

Custom Training Data

Currently, the API does not support changing the training dataset directly — but feel free to hack the code and modify it to suit your needs!


## TODOs

- [X] add train mode
- [X] add gui mode
- [X] print errors to stderr
- [ ] print state of training in training mode
- [ ] create a better ui to show the result
- [X] remove all '28' values from code
- [X] remove all '28*28' values from code
- [ ] optimize training by applying some techniques
  - [ ] calculate all norm data values in advance
  - [ ] pre calculate the sigmoid values
  - [ ] and....?