# ENAS-PdM
![check](cnnlstm.png)
ENAS-PdM is a custom Evolutionary Algorithm specifically designed to optimize a Deep Network architecture used to predict the Remaining Useful Life (RUL) for predictive maintenance (PdM). Based on our previous work, [Multi-Head CNN-LSTM with Prediction Error Analysis for Remaining Useful Life Prediction](https://ieeexplore.ieee.org/abstract/document/9211058/), the goal of this study is to find the best multi-head Convolutional Neural Network with Long Short Term Memory (CNN-LSTM) architecture for the RUL prediction. For that, we use evolutionary search to explore the combinatorial parameter space of a multi-head CNN-LSTM as shown below figure. 
<p align="center">
  <img height="600" src="/cnn_c.png">
</p>

## Prerequisites
You can download the benchmark dataset used in our experiments, C-MAPSS from [here](https://drive.google.com/drive/folders/1xHLtx9laqSTO_8LOFCdOBEkouMpbkAFM?usp=sharing).
The files should be placed in /tmp folder.
The ENAS-PdM library has the following dependencies:
```bash
pip install -r py_pkg_requirements.txt
```
- pandas
- numpy
- scikit-learn
- tqdm
- tensorflow-gpu
- deap
- matplotlib

## Descriptions
- launcher.py: Launcher for the experiments
  - evolutionary_algorithm.py: implementations of evolutionary algorithms to evolve neural networks in the context of predictive mainteinance
  - task.py: implementation of a Task, used to load the data and compute the fitness of an individual
  - utils.py: generating the multi-head CNN-LSTM network & training the network
    - network_training.py: class for network generation and training
    - ts_preprocessing.py: class for preprocessing and data preparation
    - ts_window.py: class for time series window application



## Run
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
