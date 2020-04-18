"""
Test 1D version of the Thacker test case as a validation of 
subgrid.

Notes on relative error:
  Error for eta is normalized by the high water elevation.
  Max(time) RMS(space) relative error is currently 0.0663.
  This is sensitive to timestep.  It is not sensitive to theta
  at the scale of 0.5 to 0.55.  It may be sensitive to resolution.
"""

import stompy.grid.unstructured_grid as ugrid
from stompy import utils
import matplotlib.pyplot as plt

# Rather than adding swampy to sys.path, assume this script is run
# from this folder and add the relative path
utils.path("../")
from swampy import swampy
import numpy as np
import six

##

class SwampyThacker1D(swampy.SwampyCore):
    W=20 # channel width, doesn't matter.
    L=1000 # channel length
    r0=330 # L/3 reasonable. scale of the parabolic bathymetry
    nx=100  # number of nodes in along-channel direction
    ny=11 # >=2, number of nodes in cross-channel direction
    h0=5  # verical scale of parabolic bathymetry
    eta=0.1 # perturbation for initial condition

    theta=0.55
    
    def __init__(self,**kw):
        utils.set_keywords(self,kw)
        self.omega=np.sqrt(2*9.8*self.h0)/self.r0
        self.period=2*np.pi/self.omega

        self.U=self.eta*self.r0*self.omega
        dx=self.L/float(self.nx)
        if 'dt' not in kw:
            dt=0.5*dx/self.U
            kw['dt']=dt

        super(SwampyThacker1D,self).__init__(**kw)

        # set a scale for eta, used to make eta errors relative
        # this is approximately the high water elevation
        self.eta_scale= self.eta * self.h0*(2 - self.eta)
        self.max_rel_error=0.0

    last_plot=-1000000
    plot_interval_per_period=1./20 # fraction of period
    def step_output(self,n,ei,**kwargs):
        plot_interval=self.period*self.plot_interval_per_period
        if self.t-self.last_plot<plot_interval:
            return
        self.last_plot=self.t

        eta_soln=self.eta_fn(self.grd.cells_center(),self.t)
        eta_errors=self.ei - eta_soln
        eta_stddev=np.std(eta_soln)
        rms_error=np.sqrt( np.mean(eta_errors**2) )
        print("Eta error rms: %.4f,  relative=%.4f"%(rms_error,rms_error/self.eta_scale))
        self.max_rel_error=max(self.max_rel_error, rms_error/self.eta_scale)
        
    def depth_fn(self,xy):
        return -self.h0 * (1-(xy[:,0]/self.r0)**2)
    def eta_fn(self,xy,t):
        """ 
        Analytical solution for freesurface, used for
        initial condition and validation.
        """
        fs= self.eta * self.h0/self.r0*(2*xy[:,0]*np.cos(self.omega*t)
                                        - self.eta*self.r0*np.cos(self.omega*t)**2)
        bed=self.depth_fn(xy)
        return np.maximum(fs,bed)
            
    def set_grid(self):
        g=ugrid.UnstructuredGrid(max_sides=4)
        g.add_rectilinear([-self.L/2,0],[self.L/2,self.W],
                          self.nx+1,self.ny)
        g.orient_edges()

        g.add_node_field('node_depth',self.depth_fn(g.nodes['x']))
        g.add_cell_field('cell_depth',self.depth_fn(g.cells_center()))

        super(SwampyThacker1D,self).set_grid(g)
    def set_initial_conditions(self):
        # allocate:
        super(SwampyThacker1D,self).set_initial_conditions()
        
        self.ic_zi[:]=-self.grd.cells['cell_depth']
        h=self.eta_fn(self.grd.cells_center(),t=0)
        self.ic_ei[:]=np.maximum(h,-self.ic_zi)
        
    def snapshot_figure(self,**kwargs):
        """
        Plot current model state with solution
        """
        fig=plt.figure(1)
        fig.clf()
        ax=fig.add_subplot(1,1,1)

        ax.plot(self.grd.cells_center()[:,0], -self.zi, 'b-o',ms=3)
        ax.plot(self.grd.cells_center()[:,0], self.ei, 'g-' )
        ax.plot(self.grd.cells_center()[:,0],
                self.eta_fn(self.grd.cells_center(),self.t),color='orange')

def test_1d_thacker():
    # basic run.
    sim=SwampyThacker1D(cg_tol=1e-10)
    sim.set_grid()
    sim.set_initial_conditions()
    (hi, uj, tvol, ei) = sim.run(tend=sim.period)
    assert sim.max_rel_error < 0.07, "Regression on relative error %.4f"%sim.max_rel_error

def test_1d_thacker_fine():
    sim=SwampyThacker1D(cg_tol=1e-10,nx=200)
    sim.set_grid()
    sim.set_initial_conditions()
    (hi, uj, tvol, ei) = sim.run(tend=sim.period)
    assert sim.max_rel_error < 0.07, "Regression on relative error %.4f"%sim.max_rel_error

def test_1d_thacker_coarse():
    sim=SwampyThacker1D(cg_tol=1e-10,nx=50)
    sim.set_grid()
    sim.set_initial_conditions()
    (hi, uj, tvol, ei) = sim.run(tend=sim.period)
    assert sim.max_rel_error < 0.15, "Regression on relative error %.4f"%sim.max_rel_error

def test_1d_thacker_coarse_finetime():
    # use a shorter timestep than the resolution
    # calls for
    sim=SwampyThacker1D(cg_tol=1e-10,nx=50,dt=5.0)
    sim.set_grid()
    sim.set_initial_conditions()
    (hi, uj, tvol, ei) = sim.run(tend=sim.period)
    assert sim.max_rel_error < 0.07, "Regression on relative error %.4f"%sim.max_rel_error
    

if 0:         
    sim=SwampyThacker1D(cg_tol=1e-10,dt=5.05,nx=70)
    sim.set_grid()
    sim.set_initial_conditions()
    (hi, uj, tvol, ei) = sim.run(tend=sim.period)
    
    sim.snapshot_figure()
    print("Max rms rel error: %.4f"%sim.max_rel_error)

        
