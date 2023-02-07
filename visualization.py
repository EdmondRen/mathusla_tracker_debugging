import ROOT as root
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
from matplotlib import collections, colors, transforms


from detector import Detector
import physics




class Visualizer:
    DrawMomentum = True

    #writeDirectory = "../plots/"

    trackPtSize = 2

    def __init__(self):
        # print("Hi I'm a Visualizer")
        self.fig = plt.figure()
#        self.ax = Axes3D(self.fig)
        # self.ax = self.fig.gca(projection='3d')
        self.ax = self.fig.add_subplot(projection='3d')

        self.displayPoints = []
        self.unusedPoints = []
        self.vertices = []

        self.vert_vel = True

        self.writeDirectory = "../"


    def AddTrack(self, track):
        self.liveTracks.append(track)

    def AddPoint(self, point):
        self.displayPoints.append(point)

    def AddHit(self, point):
        self.unusedPoints.append(point)

    def AddVertex(self, point_dict):
        self.vertices.append(point_dict)

    def DetectorDisplay(self):
        det = Detector()
        xLims = det.xLims()
        zLims = det.zLims()


        self.ax.set_xlim(xLims[0], xLims[1])
        self.ax.set_ylim(det.yLims()[0], det.yLims()[1])
        self.ax.set_zlim(zLims[0], zLims[1])

        #constructing each layer for Poly3DCollection class

        layerX = [xLims[0], xLims[0], xLims[1], xLims[1]]
        layerZ = [zLims[0], zLims[1], zLims[1], zLims[0]]

        cols = []

        for layerN in range(det.numLayers()):
            layerY = [det.LayerYMid(layerN) for x in range(len(layerX))]
            verts = [list(zip(layerX, layerY, layerZ))]
            cols.append(Poly3DCollection(verts, alpha=0.10))
            cols[layerN].set_facecolor(det.DrawColor())
            self.ax.add_collection3d(cols[layerN])

        return det



    def TrackDisplay(self, ListOf_trackPtLists, Listof_colors, list_of_labels=None):
        xs, ys, zs, cs = [], [], [], []
        scatters = []
        for n, trackPtList in enumerate(ListOf_trackPtLists):
            xs += [trackPt.x for trackPt in trackPtList]
            ys += [trackPt.y for trackPt in trackPtList]
            zs += [trackPt.z for trackPt in trackPtList]
            cs += [Listof_colors[n] for trackPt in trackPtList]
            #print("cs is ",cs)
            #print("list of labels ",list_of_labels)
            scatters.append(self.ax.scatter(xs, ys, zs, s=self.trackPtSize, c=cs[n][0], label=list_of_labels[n]))
            xs, ys, zs, cs = [], [], [], []
        for pnt in self.displayPoints:
            scatters.append(self.ax.scatter(pnt[0],pnt[1],pnt[2],c=5))
        self.ax.set_xlabel("x [cm]")
        self.ax.set_ylabel("y [cm]")
        self.ax.set_zlabel("z [cm]")
        self.ax.legend(markerscale=3, loc=2)



    def TrackDisplayPoints(self, x, y, z, color=None, Label=None, opac=1):

        self.ax.plot(x, y, z, c=color, label=Label, alpha=opac)

        self.ax.set_xlabel("x [cm]")
        self.ax.set_ylabel("y [cm]")
        self.ax.set_zlabel("z [cm]")

        if Label != None:
            self.ax.legend(markerscale=3, loc=2)



    def PlotPoints(self):

        scatters = []

        for pnt in self.displayPoints:
            if len(pnt) == 3:
                scatters.append(self.ax.scatter(pnt[0],pnt[1],pnt[2],s=5,c="k",marker="*"))

            else:
                scatters.append(self.ax.scatter(pnt[0][0],pnt[0][1],pnt[0][2],s=5,c=pnt[1],marker="*"))

        for pnt in self.unusedPoints:
            if len(pnt) == 3:
                scatters.append(self.ax.scatter(pnt[0],pnt[1],pnt[2],s=10,c="k",marker="."))

            else:
                scatters.append(self.ax.scatter(pnt[0][0],pnt[0][1],pnt[0][2],s=5,c=pnt[1],marker="."))


#        print(type(self.vertices),' and ',type(dict()))
#        if type(self.vertices) == type(dict):

        #print(self.vert_vel)
        if self.vert_vel:
	        for dic in self.vertices:
	#            if len(pnt) == 3:
	             scatters.append(self.ax.scatter(dic['point'][0],dic['point'][1],dic['point'][2],s=20,c=dic['col'],marker="x"))
	             #print("with color {}".format(dic['col']))
	
	#             print(dic['vert vel'][0])
	#             print(dic['vert vel'])
	
	             if self.vert_vel:
	                   for n in range(len(dic['vert vel'][0])): # show velocity best estimates at vertex
	#                  print("velocities are {}, {}, {}".format(dic['vert vel'][0][n], dic['vert vel'][1][n], dic['vert vel'][2][n]))
	
		                   v = np.array([dic['vert vel'][0][n], dic['vert vel'][1][n], dic['vert vel'][2][n]])
		#                   print("square ",v * v)
		#                   print("initial ", v)
		                   print("Beta is ",np.sqrt(np.sum(v * v,axis=0)))
		                   v = 200 * v / np.sqrt(np.sum(v * v,axis=0))
		
		#                   print("final ",v)
		#                   print("sum ",np.sum(v * v,axis=0))
		
		
		#                                 dic['vert vel'][0][n], dic['vert vel'][1][n], dic['vert vel'][2][n],
		                   self.ax.quiver(dic['point'][0], dic['point'][1], dic['point'][2],
		                                 v[0], v[1], v[2],
		                                 color = dic['col'])# = 'k')

        else:
            for pnt in self.vertices:
                scatters.append(self.ax.scatter(pnt[0],pnt[1],pnt[2],s=20,c="r",marker="x"))

#        else:
#             v = np.array([dic['vert vel'][0], dic['vert vel'][1], dic['vert vel'][2]])
#             print("square ",v * v)
#             print("initial ", v)
#             v = 20 * v / np.sum(v * v,axis=0)
#
#             print("final ",v)
#             print("sum ",np.sum(v * v,axis=0))
#
#             self.ax.quiver(dic['point'][0], dic['point'][1], dic['point'][2],
#                                 v[0], v[1], v[2],
#                                 color = 'k')



    def Draw(self,outname='plot.pdf'):

        self.PlotPoints()

        self.DetectorDisplay()

        self.ax.view_init(elev=90,azim=-90)

        plt.savefig(self.writeDirectory + outname.split('.')[0]+'_x_.png')

        self.ax.view_init(elev=0,azim=0)

        plt.savefig(self.writeDirectory + outname.split('.')[0]+'_z_.png')


#        plt.show()





def Histogram(data, rng=None, Title=None, xaxis=None, log=False, fname='hist.png'):

    fig, ax = plt.subplots(figsize=(8,5))

    #ax.hist(data,100,range=rng)

    if log:
        ax.semilogy()


    if rng != None:
        arg_data = np.copy(data)

        data = np.array(data)

        before = np.count_nonzero(data)

        data[data < rng[0]] = 0
        data[data > rng[1]] = 0
        data[data == np.nan] = 0

        #above = len(data) - np.count_nonzero(data)
        above = before - np.count_nonzero(data) # how many got set to zero

        ax.hist(data,int(np.sqrt(len(arg_data)-above)),range=rng)

    else:
        ax.hist(data,int(np.sqrt(len(data))),range=rng)

    mean = np.mean(data)
    std = np.std(data)

    ax.set_title(Title)
    ax.set_xlabel(xaxis)

    if rng != None:
        ax.text(rng[1]*0.7,np.shape(data)[0]*5e-2,"Mean: {:.03g} \nSTD: {:0.3g} \nOverflow: {}".format(mean,std,above))

    else:
        ax.text(np.max(data)*0.7,np.shape(data)[0]*5e-2,"Mean: {:.03g} \nSTD: {:0.3g}".format(mean,std))

    plt.savefig(fname)
#    plt.show()



def root_Histogram(data, rng=None, ft_rng=None, bins=0, Title="Histogram", xaxis=None, logx=False, logy=False, fname='hist.png'):

    canv = root.TCanvas("canv","newCanvas")

    bins = int(np.sqrt(len(data))) + bins

    if rng != None:
        #hist = root.TH1F("hist",Title,bins,rng[0],rng[1])
        #hist = root.TH1F("hist",Title,bins,np.amin(data),np.max(data))
        #hist = root.TH1F("hist",Title,bins,-100,100)
        hist = root.TH1F("hist",Title+[';'+xaxis,''][xaxis==None],bins,rng[0],rng[1])

    else:
        hist = root.TH1F("hist",Title+[';'+xaxis,''][xaxis==None],bins,np.amin(data),np.max(data))

#    if len(ft_rng) != 0:
    if ft_rng != None:
        #hist.Fit('gaus','','',rng[0],rng[1])
        #fit = root.TF1("fit", "gaus", rng[0], rng[1])
        #fit = root.TF1("fit", "gaus", -3, 3)
        fit = root.TF1("fit", "gaus", ft_rng[0], ft_rng[1])

    else:
        #hist.Fit('gaus','','',np.amin(data),np.max(data))
        fit = root.TF1("fit", "gaus", np.amin(data), np.max(data))

    for elm in range(len(data)):
        hist.Fill(data[elm])

    root.gStyle.SetOptStat(111111)

    if ft_rng != None:
        hist.GetXaxis().SetRangeUser(ft_rng[0],ft_rng[1]);

        hist.Fit("fit")

        hist.GetXaxis().SetRangeUser(rng[0],rng[1]);
    
    if logx:
        canv.SetLogx()
        
    if logy:
        canv.SetLogy()

    hist.Draw()

    bin1 = hist.FindFirstBinAbove(hist.GetMaximum()/2)
    bin2 = hist.FindLastBinAbove(hist.GetMaximum()/2)
    fwhm = hist.GetBinCenter(bin2) - hist.GetBinCenter(bin1)
    hwhm = fwhm / 2

    print("fwhm is ", fwhm)
    print("hwhm is ", hwhm)

    canv.Update()
    canv.Draw()

    canv.SaveAs(fname)
    #canv.Print(fname.split('.')[0]+".pdf","PDF")



def root_2D_Histogram(data_x, data_z, xlims, zlims, Title='Plot', xbins=100, zbins=100, xlabel='x', zlabel='z', fname='plot.png'):

    if len(data_x) != len(data_z):
        return print("Length of data must be equal")

    canv = root.TCanvas("canv","newCanvas")

    hist = root.TH2F("hist",Title,xbins,xlims[0],xlims[1],zbins,zlims[0],zlims[1])

    root.gStyle.SetOptStat(111111)

#    hist.SetStats(0)
    hist.GetXaxis().SetTitle(xlabel)
    hist.GetYaxis().SetTitle(zlabel)

    for i in range(len(data_x)):
        hist.Fill(data_x[i],data_z[i])

    hist.Draw("colz")

    canv.Update()
    canv.Draw()

    canv.SaveAs(fname)
    
    
def drawdet_xz(use_cms=False, axis=None, layer_height_vis=0.2, alpha=0.1):
#     use_cms=False
#     axis=None
#     layer_height_vis=0.2
    if axis is None:
        axis=plt.gca()

    det=Detector() # Get detector geometry
    verts=[] # vertices of polygons

    # Loop 10 modules
    for ix in range(10): 
        layerX = [det.module_x_displacement[ix]-det.module_x_edge_length*0.5,
                  det.module_x_displacement[ix]-det.module_x_edge_length*0.5,
                  det.module_x_displacement[ix]+det.module_x_edge_length*0.5,
                  det.module_x_displacement[ix]+det.module_x_edge_length*0.5]
        # Loop 7 layers
        for iz in range(7):
            layerZ = [det.layer_z_displacement[iz]-layer_height_vis*0.5,
                     det.layer_z_displacement[iz]+layer_height_vis*0.5,
                     det.layer_z_displacement[iz]+layer_height_vis*0.5,
                     det.layer_z_displacement[iz]-layer_height_vis*0.5,]
            verts.append(np.transpose([layerX, layerZ]))

    col = collections.PolyCollection(verts, alpha=alpha)
    axis.add_collection(col)
    
drawdet_yz=drawdet_xz

def drawdet_xy(use_cms=False, axis=None, layer_height_vis=0.2, alpha=0.1):
    if axis is None:
        axis=plt.gca()

    det=Detector() # Get detector geometry
    verts=[] # vertices of polygons

    # Loop 10 modules
    for ix in range(10): 
        layerX = [det.module_x_displacement[ix]-det.module_x_edge_length*0.5,
                  det.module_x_displacement[ix]-det.module_x_edge_length*0.5,
                  det.module_x_displacement[ix]+det.module_x_edge_length*0.5,
                  det.module_x_displacement[ix]+det.module_x_edge_length*0.5]
        # Loop 10 layers
        for iy in range(10):
            layerY = [det.module_y_displacement[iy]-det.module_y_edge_length*0.5,
                      det.module_y_displacement[iy]+det.module_y_edge_length*0.5,
                      det.module_y_displacement[iy]+det.module_y_edge_length*0.5,
                      det.module_y_displacement[iy]-det.module_y_edge_length*0.5]
            verts.append(np.transpose([layerX, layerY]))

    col = collections.PolyCollection(verts, alpha=alpha)
    axis.add_collection(col)    

# Test:
# fig,ax1 = plt.subplots(1,1)
# drawdet_xz()
# plt.xlim(-50,50)
# plt.ylim(-1,12)    

