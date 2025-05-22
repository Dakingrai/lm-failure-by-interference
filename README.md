## Environment Setup
This project is tested in Python 3.9.9.

To get started, set up the environment:
```
python -m venv env 
source env/bin/activate
pip install -r requirements.txt
```
## Dataset synthesis
We provide the data used in the experiment in the data folder. Alternatively, you can generate your own dataset by running the following Python script.
```
synthesize_data.py
```
## Accuracy
You can compute the accuracy of all models on the synthesized train, dev, and test datasets by running the following Python script:
```
get_accuracy.py
```
