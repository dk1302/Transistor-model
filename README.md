# Surrogate Model for MMT Transistors

## CNN based model for predicting drain IV characteristics of MMT transistors

### How to use

Use train.py to train the model and test.py to run. Pre-trained models are provided in the models directory.

A parameter set can be chosen with the index variable

```
model = cnn.use_model('datasets/val.csv', index=2)

```

The index variable corresponds to a list of val parameters and output coordinates found in the dataset directory
(index starts at 0)
