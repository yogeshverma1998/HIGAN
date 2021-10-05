''' Author: Yogesh Verma '''
'''HIGAN'''

from Network import *
from Data_load import *
import keras.backend as K
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.utils.generic_utils import Progbar
import numpy as np
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
from numpy.random import randn


nb_epochs = 40
batch_size = 50
latent_size = 200
verbose = 1

adam_lr = 0.0002
adam_beta_1 = 0.5


print('[INFO] Building Discriminator') 
Discriminator = discriminator()
Discriminator.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),loss='binary_crossentropy',metrics=['accuracy'])


data,xedges,yedges,T_arr,Cup_arr,Cdown_arr,EtaS_arr = get_2d_hist_matrix()

def train_discriminator(model,dataset,n_iter=20,n_batch=data.shape[0]):
    for i in range(n_iter):
        X_real,y_real = generate_real_samples(dataset,n_batch)
        _,real_acc = model.train_on_batch(X_real,y_real)
        X_fake,y_fake = generate_fake_samples(n_batch)
        _,fake_acc = model.train_on_batch(X_fake,y_fake)
        print('>%d real=%.0f%% fake=%.0f%%' % (i+1, real_acc*100, fake_acc*100))



latent_dim = 400
n_samples = 25



def generate_latent_points(latent_dim,n_samples,Temp_arr,centup_arr,centdown_arr,shear_arr):
    Input_arr = []
    for num in range(n_samples):
        temp = Temp_arr[num]
        Cup = centup_arr[num]
        Cdown = centdown_arr[num]
        eta_s = shear_arr[num]
        param = np.array([temp,Cup,Cdown,eta_s])
        param = np.reshape(param,(4,1))
        latent = randn(100)
        Latent_arr = np.array(latent)
        latent = np.reshape(latent,(1,100))
        final = np.matmul(param,latent)
        flat = final.flatten()
        Input_arr.append(flat)
    x_input = np.array(Input_arr)
    x_input = np.reshape(Input_arr,(n_samples,400))
    return x_input


def generate_real_samples(dataset, n_samples,Temp_arr,centup_arr,centdown_arr,shear_arr):
	# choose random instances
	ix = randint(0, dataset.shape[0], n_samples)
	# retrieve selected images
	X = dataset[ix]
        temp = Temp_arr[ix]
        cup = centup_arr[ix]
        cdown = centdown_arr[ix]
        eta_by_s = shear_arr[ix]
	# generate 'real' class labels (1)
	y = np.ones((n_samples, 1))
	return X, y,temp,cup,cdown,eta_by_s



def generate_fake_samples(g_model,latent_dim,n_samples,Temp_arr,centup_arr,centdown_arr,shear_arr):
    x_input = generate_latent_points(latent_dim,n_samples,Temp_arr,centup_arr,centdown_arr,shear_arr)
    X = g_model.predict(x_input)
    y = np.zeros((n_samples,1))
    return X,y


def QGAN(g_model,d_model):
    d_model.trainable = False
    model = Sequential()
    model.add(g_model)
    model.add(d_model)
    opt = Adam(lr=0.0002,beta_1=0.5)
    model.compile(loss='binary_crossentropy',optimizer=opt)
    return model





from matplotlib import pyplot
def save_plot(examples, epoch, n=5):
	# plot images
	plt.figure(figsize=(10,10))
	for i in range(n * n):
		# define subplot
		pyplot.subplot(n, n, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(examples[i, :, :, 0], cmap=plt.cm.jet)
	# save plot to file
	filename = '/home/datab/Trained_GAN/GAN_param/generated_plot_e%03d.png' % (epoch+1)
	pyplot.savefig(filename)
	pyplot.close()



def save_plot_real(examples, epoch, n=5):
	# plot images
	plt.figure(figsize=(10,10))
	for i in range(n * n):
		# define subplot
		pyplot.subplot(n, n, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(examples[i, :, :, 0], cmap=plt.cm.jet)
	# save plot to file
	filename = '/home/datab/Trained_GAN/GAN_param/real_plot_e%03d.png' % (epoch+1)
	pyplot.savefig(filename)
	pyplot.close()


def summarize_performance(epoch, g_model, d_model, dataset, latent_dim,Temp_arr,centup_arr,centdown_arr,shear_arr,n_samples=10000):
	# prepare real samples
	X_real, y_real,temp,cup,cdown,etas = generate_real_samples(dataset, n_samples,Temp_arr,centup_arr,centdown_arr,shear_arr)
	# evaluate discriminator on real examples
	_, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
	# prepare fake examples
	x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples,temp,cup,cdown,etas)
	# evaluate discriminator on fake examples
	_, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
	# summarize discriminator performance
	print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
	# save plot
	#save_plot(x_fake, epoch)
	#save_plot_real(X_real, epoch)
        #print("Real",X_real[1])
        #print("Fake",X_fake[1])
	# save the generator model tile file
        X_real, y_real,x_fake,y_fake=0,0,0,0
	filename = '/home/datab/Trained_GAN/GAN_param/generator_model_%03d.h5' % (epoch + 1)
	g_model.save(filename)



	
# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim,Temp_arr,centup_arr,centdown_arr,shear_arr,n_epochs=250,n_batch=2500):
    bat_per_epo = int(dataset.shape[0]/n_batch)
    half_batch = int(n_batch/2)
    for i in range(n_epochs):
        # enumerate batches over the training set
        for j in range(bat_per_epo):
        # get randomly selected 'real' samples
           X_real, y_real,temp,cup,cdown,etas = generate_real_samples(dataset, half_batch,Temp_arr,centup_arr,centdown_arr,shear_arr)
        # generate 'fake' examples
           X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch,temp,cup,cdown,etas)
			# create training set for the discriminator
           X, y = np.vstack((X_real, X_fake)), np.vstack((y_real, y_fake))
	# update discriminator model weights
           d_loss, _ = d_model.train_on_batch(X, y)
        # prepare points in latent space as input for the generator
           X_gan = generate_latent_points(latent_dim, n_batch,Temp_arr,centup_arr,centdown_arr,shear_arr)
			# create inverted labels for the fake samples
           y_gan = np.ones((n_batch, 1))
			# update the generator via the discriminator's error
           g_loss = gan_model.train_on_batch(X_gan, y_gan)
			# summarize loss on this batch
           print('>%d, %d/%d, d=%.3f, g=%.3f' % (i+1, j+1, bat_per_epo, d_loss, g_loss))
           X_real,y_real,X_fake,y_fake,X,y = 0,0,0,0,0,0
           X_gan,y_gan = 0,0
		# evaluate the model performance, sometimes
        if (i+1) % 1 == 0:
           summarize_performance(i, g_model, d_model, dataset, latent_dim,Temp_arr,centup_arr,centdown_arr,shear_arr)

Generator = generator(latent_dim)
QGP_GAN = QGAN(Generator,Discriminator)
plot_model(QGP_GAN, to_file='/home/datab/Trained_GAN/GAN_param/gan_plot.png', show_shapes=True, show_layer_names=True)

train(Generator,Discriminator, QGP_GAN, data, latent_dim,T_arr,Cup_arr,Cdown_arr,EtaS_arr)

