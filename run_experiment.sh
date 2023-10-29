#!/bin/bash
# This program is a bash script that runs the experiments for the paper. 
# It takes one command line argument to run: baseline, features, lstm, plm.
# Variables below can be adjusted

# --- set variables ---
EXPERIMENTS_FOLDER="./experiments/"
DATA_FOLDER="./data/"
RESULTS_FOLDER="./results/"
CODE_FOLDER="./src/"
ENV_FOLDER="./env/"

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

  "baseline")
    cp $CODE_FOLDER"baseline.py" $EXPERIMENT_FOLDER"train.py"
    ;;

  "features")
    cp $CODE_FOLDER"features.py" $EXPERIMENT_FOLDER"train.py"
    ;;

  "lstm")
    cp $CODE_FOLDER"lstm.py" $EXPERIMENT_FOLDER"train.py"
    ;;

  "plm")
    cp $CODE_FOLDER"plm.py" $EXPERIMENT_FOLDER"train.py"
    ;;

  *)
    echo "Provide a valid argument: baseline, features, lstm, plm"
    exit 1
    ;;
esac

# --- Activate virtual environment ---
source $ENV_FOLDER"bin/activate"

# --- Train the model ---
python3 $EXPERIMENT_FOLDER"train.py" \
        --train-data $DATA_FOLDER"train.tsv" \
        --model-outp $EXPERIMENT_FOLDER"model.bin"

# predict and evaluate on dev or test set based on TEST_SET variable
if [ $TEST_SET = false ]; then
    python3 $CODE_FOLDER"predict.py" \
            --model $EXPERIMENT_FOLDER"model.bin" \
            --test-data $DATA_FOLDER"dev.tsv" \
            --predictions-outp $EXPERIMENT_FOLDER"dev_predictions.txt"

    python3 $CODE_FOLDER"evaluate.py" \
            --gold-labels $DATA_FOLDER"dev.tsv" \
            --predictions $EXPERIMENT_FOLDER"dev_predictions.txt" \
            --evaluation-outp $EXPERIMENT_FOLDER"evaluation.txt" \
            --evaluation-overview $RESULTS_FOLDER"$1".md

else
    python3 $CODE_FOLDER"predict.py" \
        --model $EXPERIMENT_FOLDER"model.bin" \
        --test-data $DATA_FOLDER"test.tsv" \
        --predictions-outp $EXPERIMENT_FOLDER"test_predictions.txt"

    python3 $CODE_FOLDER"evaluate.py" \
        --gold-labels $DATA_FOLDER"test.tsv" \
        --predictions $EXPERIMENT_FOLDER"test_predictions.txt" \
        --evaluation-outp $EXPERIMENT_FOLDER"evaluation.txt" \
        --evaluation-overview $RESULTS_FOLDER"$1".md
fi

# --- Deactivate virtual environment ---
deactivate