import numpy as np
import math
import ROOT
kBlue = ROOT.kBlue
kRed = ROOT.kRed
from ROOT import gStyle

phi_bins = [-2.68520483, -2.39596058, -2.10671634, -1.8174721 , -1.52822786, -1.23898362, -0.94973938, -0.66049514, -0.3712509 , -0.08200666, 0.20723758,  0.49648183,  0.78572607,  1.07497031,  1.36421455, 1.65345879, 1.94270303,  2.23194727,  2.52119151,  2.81043575, 3.09967999]
eta_bins = [-0.857548 , -0.7443981, -0.6312482, -0.5180983, -0.4049484, -0.2917985, -0.1786486, -0.0654987,  0.0476512,  0.1608011, 0.273951 ,  0.3871009,  0.5002508,  0.6134007,  0.7265506, 0.8397005,  0.9528504,  1.0660003,  1.1791502,  1.2923001, 1.40545  ]

def plot(gan_filename,real_filename,epoch):
	
	legend = ROOT.TLegend(0.6,0.7,0.8,0.88)
	h1 = ROOT.TH1D("h1","",20,0,0.5)
	h2 = ROOT.TH1D("h2","",20,0,0.5)
	gStyle.SetFrameBorderMode(0)
	gStyle.SetFrameFillColor(0)
	gStyle.SetCanvasBorderMode(0)
	gStyle.SetPadBorderMode(0)
	gStyle.SetPadColor(10)
	gStyle.SetCanvasColor(10)
	gStyle.SetCanvasDefH(625)
	gStyle.SetCanvasDefW(950)
	gStyle.SetTitleFillColor(10)
	gStyle.SetTitleBorderSize(1)
	gStyle.SetStatColor(10)
	gStyle.SetStatBorderSize(1)
	gStyle.SetLegendBorderSize(1)
	gStyle.SetDrawBorder(0)
	gStyle.SetTextFont(42)
	gStyle.SetStatFont(42)
	gStyle.SetStatFontSize(0.05)
	gStyle.SetStatX(0.97)
	gStyle.SetStatY(0.98)
	gStyle.SetStatH(0.03)
	gStyle.SetStatW(0.3)
	gStyle.SetTickLength(0.02,"y")
	gStyle.SetEndErrorSize(3)
	gStyle.SetLabelSize(0.05,"xyz")
	gStyle.SetLabelFont(42,"xyz")
	gStyle.SetLabelOffset(0.01,"xyz")
	gStyle.SetTitleFont(42,"xyz")
	gStyle.SetTitleOffset(1.0,"xyz")
	gStyle.SetTitleSize(0.06,"xyz")
	gStyle.SetMarkerSize(1)
	gStyle.SetPalette(1,0)
	ROOT.gROOT.ForceStyle()
	
	c1 = ROOT.TCanvas('c1','',900,800)
	pad1 = ROOT.TPad("pad1","",0,0.3,1,1.0)
	pad2 = ROOT.TPad("pad2","",0,0.0,1,0.3)
	pad1.SetLeftMargin(0.15)
	pad1.SetTopMargin(0.04)
	pad1.SetRightMargin(0.04)
	pad1.SetBottomMargin(0.01)
	pad1.Draw()
	c1.cd()
	pad2.SetLeftMargin(0.15)
	pad2.SetTopMargin(0.00)
	pad2.SetRightMargin(0.04)
	pad2.SetBottomMargin(0.35)
	pad2.Draw()

	inFile_gan = ROOT.TFile.Open(gan_filename,"READ")
	gan_tree = inFile_gan.Get("Hist")
	N = 10000
	for entrynum in range(0,N):
	    int_val = []
	    gan_tree.GetEntry(entrynum)
	    int_val = getattr(gan_tree,"value")
	    val = [] 
	    for i in range(400):
	        val.append(int_val[i])
	    val_arr = np.reshape(val,(20,20))
	    pt = []
	    for i_row in range(20):
	        for i_column in range(20):
	            temp = (val_arr[i_row][i_column]*math.cos(phi_bins[i_column]))**2 +  (val_arr[i_row][i_column]*math.sin(phi_bins[i_column]))**2
	            pt.append(np.sqrt(temp))

	    for j in range(len(pt)):
	        h1.Fill(pt[j])
	        h1.SetLineWidth(4)
	        h1.SetLineColor(kBlue)
	        h1.SetTitle("")
	        h1.SetXTitle("p_{T}(I)")
	        h1.SetYTitle("Events/bin")
	        h1.GetXaxis().CenterTitle()
	        h1.GetYaxis().CenterTitle()
	        h1.GetXaxis().SetLabelSize(.06)
	        h1.GetYaxis().SetLabelSize(.065)
	        h1.GetXaxis().SetTitleSize(.06)
	        h1.GetYaxis().SetTitleSize(.1)
	        h1.GetYaxis().SetTitleOffset(0.7)
	         
	legend.AddEntry(h1,"HIGAN","ls")        
#	h1.Scale(1/h1.Integral())
	
	inFile_real = ROOT.TFile.Open(real_filename,"READ")
	real_tree = inFile_real.Get("Hist")
	N = 10000
	for entrynum in range(0,N):
	    int_val = []
	    real_tree.GetEntry(entrynum)
	    int_val = getattr(real_tree,"value")
	    val = [] 
	    for i in range(400):
	        val.append(int_val[i])
	    val_arr = np.reshape(val,(20,20))
	    pt = []
	    for i_row in range(20):
	        for i_column in range(20):
	            temp = (val_arr[i_row][i_column]*math.cos(phi_bins[i_column]))**2 +  (val_arr[i_row][i_column]*math.sin(phi_bins[i_column]))**2
	            pt.append(np.sqrt(temp))

	    for j in range(len(pt)):
	        h2.Fill(pt[j])
	        h2.SetLineWidth(4)
	        h2.SetLineColor(kRed)
	        h2.SetTitle("")
	        h1.SetXTitle("p_{T}(I)")
	        h2.SetYTitle("Events/bin")
	        h2.GetXaxis().CenterTitle()
	        h2.GetYaxis().CenterTitle()
	        h2.GetXaxis().SetLabelSize(.06)
	        h2.GetYaxis().SetLabelSize(.065)
	        h2.GetXaxis().SetTitleSize(.06)
	        h2.GetYaxis().SetTitleSize(.1)
	        h2.GetYaxis().SetTitleOffset(0.7)
	         
	legend.AddEntry(h2,"JETSCAPE","ls")        
#	h2.Scale(1/h2.Integral())
	ROOT.gStyle.SetOptStat(0)
	pad1.cd()
	h1.Draw()
	h2.Draw("SAME")
	legend.Draw()
	pad1.SetLogy()
	c1.Update()
	
	pad2.cd()
	h3 = h2.Clone("h3")
	h3.SetYTitle("Ratio")
	h3.SetXTitle("p_{T}(I)")
	h3.SetNdivisions(7,"y")
	h3.GetXaxis().SetLabelOffset(0.01)
	h3.GetXaxis().SetLabelSize(0.13)
	h3.GetXaxis().SetTitleSize(0.18)
	h3.GetXaxis().SetTitleOffset(0.8)
	h3.GetYaxis().CenterTitle()
	h3.GetYaxis().SetLabelOffset(0.01)
	h3.GetYaxis().SetLabelSize(0.13)
	h3.GetYaxis().SetTitleSize(0.18)
	h3.GetYaxis().SetTitleOffset(0.35)
	h3.SetMarkerSize(1.9)
	h3.SetMarkerStyle(29)
	h3.SetMarkerColor(ROOT.kGreen+3)
	h3.SetLineStyle(1)
	h3.SetLineColor(2)
	h3.SetTitle("")
	h3.SetStats(0)
	h3.Divide(h1)
	h3.Draw("ep")
	
	
	
	f1r = ROOT.TF1("f1r","1",-3000,10000)
	f1r.SetLineStyle(2)
	f1r.SetLineWidth(1)
	f1r.Draw("Same")
	filename = '/home/datab/Trained_GAN/GAN_data/C70080/Pt_model%03d.pdf'%(epoch)
	c1.Print(filename)
	
	
	
for i in range(1,251):
    gan_filename = '/home/datab/Trained_GAN/GAN_data/C70080/model_%03d.root'%(i) #Gan generated Data
    real_filename = '/home/datab/Jetscape_centrality/Hist_C70080.root'  #Real Data
    print("Model_no",i)
    plot(gan_filename,real_filename,i) 
