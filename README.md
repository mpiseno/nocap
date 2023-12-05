# Replicating Results

## Data Preprocessing

First, we process the raw AMASS data by running the following command.

```bash
python scripts/process_amass_data.py --datasets SFU
```

This will parse all the data files and create sequences of motion data.

## Training

To train the model, run the following command. Replace the `data_dir` with the appropriate directory.

```bash
python scripts/train.py --data_dir data/amass_processed/SFU
```

# TODO

* Autrogressive models with a free-form variance do not generalize because the variance becomes very small around the training data. Use a fixed variance instead.