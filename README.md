# inf573vegetalisation
The file "project_paper.pdf" describes the approach and the main results.

## Organisation
In this project, there are mostly three things :
- a python module that trains a Unet model
- a python module that implements the two algorithms mentionned in the paper.
- some data

## What you can do
the story.ipynb notebook is directly runnable. To run the train_cnn notebook, you need to downlaod the big dataset that can be found here: https://ignf.github.io/FLAIR/ .
However all the parameters are the one we used for the training, and the results are still shown in the train_cnn.ipynb file.
We exported a few predictions, a few labels and a few images in the data_sample folder for you to try if you want.

## Installation
We used poetry for creating the virtual environment associated with the project. Here is how to install it :
```pip install poetry```
then 
```poetry install```
which should create a virtual environment to run the notebook.
