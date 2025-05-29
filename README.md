# Surrogate Model for MMT Transistors

## CNN based model for predicting drain IV characteristics of MMT transistors

### How to Use

Install requirements.txt and run test.py. Pre-trained models are provided in the models directory.
A parameter set can be chosen for testing with the index variable

```
index = 66
plot.plot(index)
model = cnn.use_model('datasets/val.csv', index=index)

```
The index variable corresponds to a list of val parameters and output coordinates found in the dataset directory
(index starts at 0, ends at 72)

To train new models, edit the train.py file with desired changes and run to create new models.
