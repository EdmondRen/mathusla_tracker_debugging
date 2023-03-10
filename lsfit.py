import iminuit
from iminuit import Minuit, cost
import numpy as np
import scipy as sp


import cutflow,detector
cut=cutflow.sample_space("")
det=detector.Detector()

class Hit:
    def __init__(self, x, y, z, t):
        self.x=x
        self.y=y
        self.z=z
        self.t=t
        self.t_uncertainty=1
    def get_uncertainty(self):
        # Get the layer-dependent uncertainty of each hit
        self.hit_layer=cut.in_layer(self.y)
        hit_uncertainty = np.array(detector.Layer().uncertainty(self.hit_layer))
        self.x_uncertainty=hit_uncertainty[0]*100 # m->cm
        self.z_uncertainty=hit_uncertainty[2]*100 # m->cm
        self.y_uncertainty=2/np.sqrt(12)

def get_digi_hits(ev):
    ev.Tree.GetEntry(ev.EventNumber)
    hits=[]
    for ii in range(len(ev.Tree.Digi_y)):
        hit=Hit(ev.Tree.Digi_x[ii], ev.Tree.Digi_y[ii], ev.Tree.Digi_z[ii], ev.Tree.Digi_time[ii])
        hit.get_uncertainty()
        hits.append(hit)
    return hits

def get_event_truth(ev):
    ev.Tree.GetEntry(ev.EventNumber)
    dx=ev.Tree.Hit_x[1]-ev.Tree.Hit_x[0]
    dy=ev.Tree.Hit_y[1]-ev.Tree.Hit_y[0]
    dz=ev.Tree.Hit_z[1]-ev.Tree.Hit_z[0]
    dt=ev.Tree.Hit_time[1]-ev.Tree.Hit_time[0]
    truth=[ev.Tree.Hit_x[0], ev.Tree.Hit_y[0], ev.Tree.Hit_z[0], ev.Tree.Hit_time[0],dx/dt, dy/dt, dz/dt]
    return truth




# -------------------------------------
# LS fit
# ------------------------------------
class chi2_track:
    def __init__(self, hits):
        self.hits=hits
        self.func_code = iminuit.util.make_func_code(['x0', 'y0', 'z0', 't0', 'vx', 'vy', 'vz'])
    def __call__(self, x0, y0, z0, t0, vx, vy, vz):
        error=0
        for hit in self.hits:
            model_t = (hit.y - y0)/vy
            model_x = x0 + model_t*vx
            model_z = z0 + model_t*vz
            error+= np.sum(np.power([(model_t- (hit.t-t0))/hit.t_uncertainty, 
                                     (model_x-hit.x)/hit.x_uncertainty, 
                                     (model_z-hit.z)/hit.z_uncertainty],2))
        return error        

def guess_track(hits):
    # Guess initial value
    x0_init = hits[0].x
    y0_init = hits[0].y
    z0_init = hits[0].z
    t0_init = hits[0].t
    dt=hits[-1].t-hits[0].t
    vx_init = (hits[-1].x-hits[0].x)/dt
    vy_init = (hits[-1].y-hits[0].y)/dt
    vz_init = (hits[-1].z-hits[0].z)/dt
    v_mod = np.sqrt(vx_init**2+vy_init**2+vz_init**2)
    if v_mod>sp.constants.c*1e-7:
        vx_init = vx_init*0.99*sp.constants.c*1e-7/v_mod
        vy_init = vy_init*0.99*sp.constants.c*1e-7/v_mod
        vz_init = vz_init*0.99*sp.constants.c*1e-7/v_mod
    return  (x0_init,y0_init, z0_init,t0_init,vx_init,vy_init,vz_init)
    
def fit_track(hits, guess):
    x0_init,y0_init, z0_init,t0_init,vx_init,vy_init,vz_init = guess
    det=detector.Detector()

    m = Minuit(chi2_track(hits),x0=x0_init, y0=y0_init, z0=z0_init, t0=t0_init, vx=vx_init, vy=vy_init, vz=vz_init)
    m.fixed["y0"]=True
    m.limits["x0"]=(det.BoxLimits[0][0],det.BoxLimits[0][1])
    m.limits["z0"]=(det.BoxLimits[2][0],det.BoxLimits[2][1])
    m.limits["t0"]=(-100,1e5)
    m.limits["vx"]=(-sp.constants.c*1e-7, sp.constants.c*1e-7) # Other
    m.limits["vy"]=(-sp.constants.c*1e-7*2,0) if vy_init<0 else (0,sp.constants.c*1e-7*2) # Constrain the direction in Z(up) in real world
    m.limits["vz"]=(-sp.constants.c*1e-7, sp.constants.c*1e-7) # Beam direction; From MKS unit to cm/ns = 1e2/1e9=1e-7
    m.errors["x0"]=0.1
    m.errors["y0"]=0.1
    m.errors["z0"]=0.1
    m.errors["t0"]=0.3
    m.errors["vx"] = 0.01
    m.errors["vy"] = 0.1
    m.errors["vz"] = 0.01

    m.migrad()  # run optimiser
    m.hesse()   # run covariance estimator
    
    return m

def fit_track_scipy(hits, guess):
    # Using scipy?
    bounds=[]
    bounds.append((det.BoxLimits[0][0],det.BoxLimits[0][1]))
    bounds.append((det.BoxLimits[2][0],det.BoxLimits[2][1]))
    bounds.append((0,1e10))
    bounds.append((-sp.constants.c*1e-7, sp.constants.c*1e-7))
    lim_vy=(-sp.constants.c*1e-7,0) if guess[-2]<0 else (0,sp.constants.c*1e-7) # Constrain the direction in Z(up) in real world
    bounds.append(lim_vy)
    bounds.append((-sp.constants.c*1e-7, sp.constants.c*1e-7)) # Beam direction; From MKS unit to cm/ns = 1e2/1e9=1e-7    
    
    def ls(par): 
        return chi2_track(hits)(par[0], guess[1], *par[1:])
    guess_no_y=guess[:1]+guess[2:]
    res = sp.optimize.minimize(ls, x0=guess_no_y,method="powell",bounds=bounds)  
    
    return res