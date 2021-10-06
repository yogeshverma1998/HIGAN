import numpy as np
from os import path
home = path.expanduser("~")

import matplotlib.pyplot as plt

# define plot style
width = 0.05
plotMarkerSize = 8
labelfontsize = 15
import matplotlib as mpl
import ROOT


EPS = 5e-16  # a small number

#Loads event from the final hadron file and returns a 2-D weighted plot of eta vs phi weighted with E/cosh(eta)

def get_2d_hist_matrix():
    data_filename = "/home/datab/Jetscape_param_root/Hist_param.root"
    inFile = ROOT.TFile.Open(data_filename,"READ")
    hist_list = []
    T_list = []
    Cup_list = []
    Cdown_list = []
    Eta_S_list = []
    tree = inFile.Get("Hist")
    for entryNum in range(0,tree.GetEntries()):
        tree.GetEntry(entryNum)
        t1= []
        arr_value = getattr(tree,"value")
        temp = getattr(tree,"Temp")
        Cup = getattr(tree,"Cup")
        Cdown = getattr(tree,"Cdown")
        Etabys = getattr(tree,"eta_s")
        for i in arr_value:
            t1.append(i)
        if entryNum%100==0: print("Entry loaded ",entryNum)
        if True in np.isnan(t1): continue
        hist_list.append(t1)
        T_list.append(temp)
        Cup_list.append(Cup)
        Cdown_list.append(Cdown)
        Eta_S_list.append(Etabys)
    Hist = np.array(hist_list)
    Temp = np.array(T_list)
    Centrality_up = np.array(Cup_list)
    Centrality_down = np.array(Cdown_list)
    Shear_vis = np.array(Eta_S_list)
   
    Hist = np.reshape(Hist,(len(hist_list),20,20,1))
    hist_list= 0
   
    phi_bins = [-2.68520483, -2.39596058, -2.10671634, -1.8174721 , -1.52822786, -1.23898362, -0.94973938, -0.66049514, -0.3712509 , -0.08200666, 0.20723758,  0.49648183,  0.78572607,  1.07497031,  1.36421455, 1.65345879, 1.94270303,  2.23194727,  2.52119151,  2.81043575, 3.09967999]
    eta_bins = [-0.857548 , -0.7443981, -0.6312482, -0.5180983, -0.4049484, -0.2917985, -0.1786486, -0.0654987,  0.0476512,  0.1608011, 0.273951 ,  0.3871009,  0.5002508,  0.6134007,  0.7265506, 0.8397005,  0.9528504,  1.0660003,  1.1791502,  1.2923001, 1.40545  ]


    return Hist, phi_bins,eta_bins,Temp,Centrality_up,Centrality_down,Shear_vis




