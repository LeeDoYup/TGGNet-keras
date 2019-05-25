# TGNet-keras
Author's implementation of TGNet: Demand Forecasting from Spatiotemporal Data with Graph Networks and Temporal-Guided Embedding

## TGNet: Model Description
TGNet consists 3 components: (a) baseline model (GN), (b) temporal-guided embedding, and (c) late fusion with external data sources.  

<img src="./assets/model_arch.png" width='800'>

Temporal guided embedding learns temporal contexts and helps FCN extract hidden feature maps conditioned on temporal contexts.  
It is similar with the generator in coniditional GAN as we see the forecasting model as generative model.  

You can use external data sources (if you want), encoding in various ways and concatenated into the baseline.  
In TGNet, we encode taxi drop-off volumes and improve forecasting performances on atypical samples with extremely large volumes.

# Main results
## Forecasting Accuracy
TGNet shows state-of-the-art performances on real-world datsets and has **20 times smaller** number of parameters.

<img src="./assets/forecasting_table.jpeg" width='800'>


## Visualization of Temporal Guided Embedding
Temporal-guided embedding explicitly learns temporal contexts in training data.  
If your data is large-scale, Temporal guided embedding shows meaningful and interpretable visualization.  
The results of SEO-taxi dataset (private) are below.

### Visualization Result 1
The basic concept of time-of-day is high correlations between adjacent time.  
Temporal guided embedding can learn the fact, from 0/1 categorical vectors of time-of-day.

<img src="./assets/tge_visualization_00.gif" width="400">

### Visualization Result 2
Based on taxi demand patterns, temporal guided embedding classify time-of-day into 4 groups.  

- Night: 0:00 ~ 06:00
- Commute Time: 06:00 ~ 9:00, 18:00 ~ 21:00
- Daytime: 09:00 ~ 18:00
- Evening: 21:00 ~ 24:00


<img src="./assets/tge_visualization.png" width="400">

### Visualization Result 3
Temporal guided embedding can learn day-of-week and holiday concept.
Regardless time-of-day, all weekend & hoiday (but not weekend) vectors are adjacent to each other.  
Also, weekend and weekday are totally divided.  

<img src="./assets/tge_visualization_2.png" width="400">

### Visualization Result 4 : NYC-taxi Dataset  
In the case of NYC-taxi, temporal guided embedding does not many meaningful insights, because the dataset is small scale.  
However, the adjacent time-of-day vectors tend to be located in adjacent to.  
In addition, working day and weekend&holiday are divided in the embedding space.

<img src="./assets/tge_nyc_visualization.gif" width="400">

<img src="./assets/nyc_segmentation.jpg" width="400">


## Prerequisites (my environments)
- Python 3.5.2 .
- Tensorflow-gpu 1.7.0 .
- Keras 2.2.2 .
- Pandas 0.22.0 .
- Numpy 1.14.2 .


## Usage
We attach \*Sampler.ipynb files for reference to make dataset.  
We do not open SEO-taxi dataset in original paper.  

## Run Saved Model and Teest
```
python main.py --model_name NYC --dataset_name NYC --test --alpha (0.01 or 0.05)
python main.py --model_name NYCB --dataset_name NYC_bike --test --alpha (0.01 or 0.05)
```

## Training a Model from Scratch
```
python3 main.py --model_name (model_name) --num_gpu (gpu counts) --dataset_name (NYC or NYC_bike) --alpha (0.01 or 0.05)
```
In here, alpha means the threshold level of atypical sample selection.    
There are many arguments to change hyper-parameters, see main.py


## Contact
Doyup Lee (doyup.lee@postech.ac.kr)
