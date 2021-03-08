# ENAS-PdM
![check](cnnlstm.png =100x20)
ENAS-PdM is a custom Evolutionary Algorithm specifically designed to optimize a Deep Network architecture used to predict the Remaining Useful Life (RUL) for predictive maintenance (PdM). Based on our [previous work](https://ieeexplore.ieee.org/abstract/document/9211058/), the goal of this study is to find the best multi-head Convolutional Neural Network with Long Short Term Memory (CNN-LSTM) architecture for the RUL prediction. For that, we use evolutionary search to explore the combinatorial parameter space of a multi-head CNN-LSTM as shown below figure. 
<p align="center">
  <img height="600" src="/cnn_c.png">
</p>

## Prerequisites
You can download the benchmark dataset used in our experiments, C-MAPSS from [here](https://drive.google.com/drive/folders/1xHLtx9laqSTO_8LOFCdOBEkouMpbkAFM?usp=sharing).
The files should be placed in /tmp folder.
The ENAS-PdM library has the following dependencies:
- pandas
- numpy
- scikit-learn
- tqdm
- tensorflow-gpu
- deap
- matplotlib

## Descriptions


## Run
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
