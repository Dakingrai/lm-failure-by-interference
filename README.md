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
python synthesize_data.py
```
## Accuracy
You can compute the accuracy of all models on the synthesized train, dev, and test datasets by running the following Python script:
```
python get_accuracy.py
```
## LM Components Projection on Train Set
To obtain the generalizability of attention heads and MLP neurons, please run the following script.
```
python proj_attn.py
python proj_neuron.py
```
Running these scripts saves all the necessary metadata for subsequent experiments, including the LM steering(RASteer) experiment.

## Generalizability of LM Components
To obtain the results of Section 3.2, please run the following script. The results can be found in results/meta_results/generalization_heads.json and results/mlp_results/model_name/neuron_generalization_0.json
```
python analyze_attn_proj.py
python analyze_neuron_proj.py
```

Now, figure 1 results can be obtained by running the following script.
```
python attention_acc_distribution.py
```

## Steering Experiment
### Running steering to obtain the optimal multiplier. The results are saved in results/steer_results
```
python steer_attn.py
python steer_neuron.py
python steer_both.py
```
### Running script to extract the best multiplier for each number of heads
```
python analyze_dev_steer.py
```
### Run steer on the test set
```
python steer_attn_test.py
python steer_neuron_test.py
python steer_both_test.py
```
The test set results are saved in the results/steer_results/model





