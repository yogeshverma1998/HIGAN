# HIGAN
Heavy Ion Generative Adversarial Network \
GANs(Generative Adversarial Networks)-based generative modelling method to generate heavy ion collision events. 

# Data
Training Data for HIGAN of PbPb events with all centralities at √s<sub>NN</sub> = 5.02 TeV is generated using [JETSCAPE](https://github.com/JETSCAPE) Monte Carlo (MC) Simulator.

# Training
To train the HIGAN model, change the location of dataset and model save path in all files which will be used later for evaluation. After, run the following command.
```js
python Train.py
```

# Evaluation
One can test the saved GAN model by evaluating [generate_event.py](https://github.com/yogeshverma1998/HIGAN/blob/main/generate_event.py) to generate events by GAN and modifying [Plot.py](https://github.com/yogeshverma1998/HIGAN/blob/main/Plot.py) to plot the various variables. 
