#!/bin/bash
# This program is a bash script that runs the experiments for the paper. 
# It takes one command line argument to run: features, lstm, plm.
# make sure that there is a python script wih the same name in the src folder, 
# and a config file with the same name in the configs folder.

# Variables below can be adjusted

# --- set variables ---
EXPERIMENTS_FOLDER="./experiments/"
CONFIGS_FOLDER="./configs/"
RESULTS_FOLDER="./results/"
CODE_FOLDER="./src/"
ENV_FOLDER="./env/"
OFFENSIVE_WORD_REPLACE_OPTION="none"

# provide parameter 'test' if you want to predict on test set instead of dev
if [ "$2" == "test" ]; then
    TEST_SET=true
    else
    TEST_SET=false
fi

# --- create folders if they do not exist yet ---
mkdir -p $EXPERIMENTS_FOLDER $RESULTS_FOLDER

# --- Create a new folder for the experiment. ---
# Get a list of all the existing folders with the desired filename.
# Do not show any error messages if no folders exist yet.
folders=$(ls -d $EXPERIMENTS_FOLDER$1"-"* 2> /dev/null)

# Find the highest folder number.
highest_folder_number=-1
for folder in $folders; do
    folder_number=$(basename $folder | cut -d "-" -f 2)
    if [ $folder_number -gt $highest_folder_number ]; then
        highest_folder_number=$folder_number
  fi
done

# Increment the highest folder number by 1.
new_folder_number=$((highest_folder_number + 1))


# Create the new folder.
EXPERIMENT_FOLDER="${EXPERIMENTS_FOLDER}$1-$new_folder_number/"
mkdir $EXPERIMENT_FOLDER

# --- Use command line arguments to copy the correct train file ---
case $1 in

  "features")
    cp $CODE_FOLDER"features.py" $EXPERIMENT_FOLDER"train.py"
    cp $CONFIGS_FOLDER"features.json" $EXPERIMENT_FOLDER"config.json"
    ;;

  "lstm")
    cp $CODE_FOLDER"lstm.py" $EXPERIMENT_FOLDER"train.py"
    cp $CONFIGS_FOLDER"lstm.json" $EXPERIMENT_FOLDER"config.json"
    ;;

  "plm")
    cp $CODE_FOLDER"plm.py" $EXPERIMENT_FOLDER"train.py"
    cp $CONFIGS_FOLDER"plm.json" $EXPERIMENT_FOLDER"config.json"
    ;;

  *)
    echo "Provide a valid argument: baseline, features, lstm, plm"
    exit 1
    ;;
esac

# --- Copy the util, evaluation and prediction scripts ---
touch $EXPERIMENT_FOLDER"__init__.py"
cp $CODE_FOLDER"util.py" $EXPERIMENT_FOLDER"util.py"
cp $CODE_FOLDER"evaluate.py" $EXPERIMENT_FOLDER"evaluate.py"
cp $CODE_FOLDER"predict.py" $EXPERIMENT_FOLDER"predict.py"

# --- Activate virtual environment ---
source $ENV_FOLDER"bin/activate"

# --- Train the model ---
python3 $EXPERIMENT_FOLDER"train.py" -c $EXPERIMENT_FOLDER"config.json" -o $EXPERIMENT_FOLDER"model/">> $EXPERIMENT_FOLDER"training_log.txt"

# --- Predict and evaluate on dev or test set based on TEST_SET variable ---
if [ $TEST_SET = false ]; then
    python3 $EXPERIMENT_FOLDER"predict.py" \
            --model $EXPERIMENT_FOLDER"model.bin" \
            --directory $DATA_FOLDER \
            --test-data "dev" \
            --predictions-directory $EXPERIMENT_FOLDER"dev_predictions" \
            --model-type $1

    python3 $EXPERIMENT_FOLDER"evaluate.py" \
            --directory $DATA_FOLDER \
            --test-data "test" \
            --predictions-directory $EXPERIMENT_FOLDER"dev_predictions" \
            --evaluation-directory $EXPERIMENT_FOLDER"evaluation.txt" \
            --evaluation-overview $RESULTS_FOLDER"$1".md

else
    python3 $EXPERIMENT_FOLDER"predict.py" \
        --model $EXPERIMENT_FOLDER"model.bin" \
        --directory $DATA_FOLDER \
        --test-data "test" \
        --predictions-directory $EXPERIMENT_FOLDER"test_predictions" \
        --model-type $1

    python3 $EXPERIMENT_FOLDER"evaluate.py" \
        --directory $DATA_FOLDER \
        --test-data "test" \
        --predictions-directory $EXPERIMENT_FOLDER"test_predictions" \
        --evaluation-directory $EXPERIMENT_FOLDER"evaluation.txt" \
        --evaluation-overview $RESULTS_FOLDER"$1".md
fi

# --- Deactivate virtual environment---
# deactivate