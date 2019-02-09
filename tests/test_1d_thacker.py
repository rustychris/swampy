"""
Test 1D version of the Thacker test case as a validation of 
subgrid.
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
    W=20
    L=1000
    r0=330 # L/3 reasonable.
    nx=100
    ny=21 # >=2
    h0=5
    eta=0.1

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

    def set_initial_conditions(self,h=0.0):
        # allocates the arrays
        super(SwampyThacker1D,self).set_initial_conditions()

        # bed elevation, positive-down
        self.ic_zi[:]=-self.grd.cells['cell_depth']
        # water surface, positive up
        self.ic_ei[:]=np.maximum(h,-self.ic_zi)

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
        print("Eta error rms: %.4f,  relative=%.4f"%(rms_error,rms_error/eta_stddev))
        
    def depth_fn(self,xy):
        return -self.h0 * (1-(xy[:,0]/self.r0)**2)
    def eta_fn(self,xy,t):
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

sim=SwampyThacker1D(cg_tol=1e-10)
sim.set_grid()
sim.set_initial_conditions()

(hi, uj, tvol, ei) = sim.run(tend=sim.period)
sim.snapshot_figure()
