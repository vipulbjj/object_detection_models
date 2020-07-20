# End-to-end object detection with Transformers

This is the repository for the "End-to-end object detection with Transformers" for object detection for Uplara.

## Instructions

**NOTE:** Python 3.6+ is needed to run the code.

All scripts use argparse to parse command-line arguments.
For viewing the list of all positional and optional arguments for any script, run:
```sh
./script.py --help
```

### Setup
Install all required Python libraries:
```sh
pip install -r requirements.txt
```
    

### Training
Run `train.py`:
```sh
./train.py
```
The trained model is saved in Keras default `.h5` format (to the directory given by the `--save-dir` argument).
The training logs are by default stored inside an ISO 8601 timestamp named subdirectory, which is stored in a parent directory (as given by the `--log-dir` argument).

### Evaluation  --- yet to finalise this
Run `evaluate.py`:
```sh
./evaluate.py
```


## Training Details

### Specs
* CPU: Intel(R) Core(TM) i7-6850K
* OS: Arch Linux
* CUDA Version: 10.1

**NOTE**: All the following results are reported using the default command-line arguments and hyper-parameters.

### Training model
```sh
python train.py
```
