''' Author: Yogesh Verma '''
'''HIGAN'''

#Generate events from GAN and store in ROOT files
from array import array
from keras.models import load_model
from numpy.random import randn,randint
from matplotlib import pyplot
import numpy as np
from Data_load import *
from Analysis import *
import ROOT
 
# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input

def generate_real_samples(dataset, n_samples):
	# choose random instances
	ix = randint(0, dataset.shape[0], n_samples)
	# retrieve selected images
	X = dataset[ix]
	# generate 'real' class labels (1)
	y = np.ones((n_samples, 1))
	return X, y


latent_dim = 300
n_samples = 10000



for i in range(1,251):
    model_name = '/home/datab/Trained_GAN/C70080/generator_model_%03d.h5'%(i) #Saved model file
    print("model number currently generating events",i)
    model = load_model(model_name)
   
    Size=400


    latent_points  = generate_latent_points(latent_dim,n_samples)
    X = model.predict(latent_points)
    value = array('d',Size*[0])
    rootfilename = '/home/datab/Trained_GAN/GAN_data/C70080/model_%03d.root'%(i) #ROOT filename
    file = ROOT.TFile(rootfilename,'recreate')
    tree = ROOT.TTree("Hist","data")
    tree.Branch('value',value,'value[400]/D')
    
    
    for index in range(len(X)):
        data_gen = np.reshape(X[index],(400))
        for i_row in range(400):
                value[i_row] = data_gen[i_row]
        tree.Fill()
    file.Write()
    file.Close()
