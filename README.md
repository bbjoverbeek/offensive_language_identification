# offensive_language_identification
 
## Data
- offensive word list dataset: https://www.cs.cmu.edu/~biglou/resources/

## How to run
For this experiment a bash script is provided. To run the experiment, run the following command:

```
bash ./run_experiment.sh <model_type>
```

Here `model_type` can be `features`, `lstm`, `plm`.

This will train the models, predict labels with the trained models and evaluate their performance on a test set. For each run a directory in the `experiments` folder is created. In here you will find the python files for your model type, for some utility functions, for predicting the labels, and for evaluating the scores. You will also find the trained model and the predictions on the test set. The evaluation scores shown in a subdirectory called `evaluation`. 

The evaluation directory contains files for individual models which contain precision, recall, f1-score, and accuracy. The files also contains a confusion matrix. The `evaluation` directory also contains a file called `all_scores.csv` which contains the scores for all models in one file. Only the precision, recall, f1-score, and accuracy are shown in this file.

## How to configure

With the command above, you can run the experiment with the default settings. In the configs directory, you can find the configuration files for each model type. You can change the settings in these files to change the experiment settings.
For example, you can change the directory in the script can find the train, dev, and test set. To see all the options, you can take a look at the config files.

To create a default config file for a model type, run the following command:

```
python3 <model_type>.py -x
```

## Requirements
To install the Python requirements, run the following command:

```
pip install -r requirements.txt
```

