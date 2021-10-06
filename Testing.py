''' Author: Yogesh Verma '''
''' HIGAN '''

# example of loading the generator model and generating images
from keras.models import load_model
from numpy.random import randn,randint
from matplotlib import pyplot
import numpy as np
from Data_load import *
from Analysis import *
 
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
n_samples = 5

data,xedges,yedges = get_2d_hist_matrix()
X_real, y_real = generate_real_samples(data, n_samples)


for i in range(1,150):
    model_name = '/home/yogesh/JETSCAPE_data/gan_3_h5/generator_model_%03d.h5'%(i) #saved model name which you want to test
    model = load_model(model_name)
    latent_points  = generate_latent_points(latent_dim,n_samples)
    X = model.predict(latent_points)
    avg_get_pT_hist(X_real,X,xedges,i)
