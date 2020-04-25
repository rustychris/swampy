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

six.moves.reload_module(swampy)
##

class SubgridThacker1D(swampy.SwampyCore):
    W=20 # channel width, doesn't matter.
    L=1000 # channel length
    r0=330 # L/3 reasonable. scale of the parabolic bathymetry
    nx=100  # number of nodes in along-channel direction
    ny=11 # >=2, number of nodes in cross-channel direction
    h0=5  # verical scale of parabolic bathymetry
    eta=0.1 # perturbation for initial condition

    theta=0.55

    max_subcells=1
    max_subedges=1
    
    def __init__(self,**kw):
        utils.set_keywords(self,kw)
        self.omega=np.sqrt(2*9.8*self.h0)/self.r0
        self.period=2*np.pi/self.omega

        self.U=self.eta*self.r0*self.omega
        dx=self.L/float(self.nx)
        if 'dt' not in kw:
            dt=0.5*dx/self.U
            kw['dt']=dt

        super(SubgridThacker1D,self).__init__(**kw)

        # set a scale for eta, used to make eta errors relative
        # this is approximately the high water elevation
        self.eta_scale= self.eta * self.h0*(2 - self.eta)
        self.max_rel_error=0.0

        print("Testing with no advection")
        self.get_fu=self.get_fu_no_adv

    last_plot=-1000000
    plot_interval_per_period=1./20 # fraction of period
    def step_output(self,n,ei,**kwargs):
        plot_interval=self.period*self.plot_interval_per_period
        if self.t-self.last_plot<plot_interval:
            return
        self.last_plot=self.t

        # Have to account for subgrid here.
        eta_soln=self.eta_fn(self.grd.cells_center(),self.t)
        eta_soln=np.maximum(eta_soln, self.zi_agg['min'])
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
        return fs
            
    def set_grid(self):
        g=ugrid.UnstructuredGrid(max_sides=4)
        g.add_rectilinear([-self.L/2,0],[self.L/2,self.W],
                          self.nx+1,self.ny)
        g.orient_edges()

        #g.add_node_field('node_depth',self.depth_fn(g.nodes['x']))
        #g.add_cell_field('cell_depth',self.depth_fn(g.cells_center()))

        super(SubgridThacker1D,self).set_grid(g)
    def set_initial_conditions(self):
        super(SubgridThacker1D,self).set_initial_conditions()
        
        for i in range(self.grd.Ncells()):
            node_x=self.grd.nodes['x'][self.grd.cell_to_nodes(i),0]
            alpha=(np.arange(self.max_subcells)+0.5)/self.max_subcells
            x_sub = (1-alpha)*node_x.min() + alpha*node_x.max()
            y_sub = self.grd.cells_center()[i,1]*np.ones(self.max_subcells)

            self.ic_zi_sub['z'][i,:]=self.depth_fn(np.c_[x_sub,y_sub])
            self.ic_zi_sub['A'][i,:]=self.grd.cells_area()[i]/self.max_subcells

        # So that we have cell agg values for edges below
        self.prepare_cell_subgrid(self.ic_zi_sub)
        self.ic_ei[:]=self.eta_fn(self.grd.cells_center(),t=0)
        
        # Edges:
        e2c=self.grd.edge_to_cells().copy()
        missing=e2c.min(axis=1)<0
        e2c[missing,:] = e2c[missing,:].max()

        ltot=self.grd.edges_length()
        for j in range(self.grd.Nedges()):
            node_x=self.grd.nodes['x'][self.grd.edges['nodes'][j],0]
            if node_x[0] == node_x[1]:
                n_subedge=1
            else:
                n_subedge=self.max_subedges

            if self.max_subcells>1: # the real deal
                alpha=(np.arange(n_subedge)+0.5)/n_subedge
                x_sub = (1-alpha)*node_x[0] + alpha*node_x[1]
                y_sub = self.grd.edges_center()[j,1]*np.ones(n_subedge)

                # Extra logic to make sure that an edge is not deeper than
                # the deepest point in either neighboring cell.
                # Otherwise that neighbor can dry up, but the edge is still
                # wet.
                z_min=self.zi_agg['min'][e2c[j,:]].max() 
                self.ic_zj_sub['z'][j,:]=self.depth_fn(np.c_[x_sub,y_sub]).clip(z_min)
            else: # simplified case for testing basics
                self.ic_zj_sub['z'][j,:]=self.ic_zi_sub['z'][e2c[j,:],0].max()
            self.ic_zj_sub['l'][j,:]=ltot[j]/n_subedge

    def plot_cell_scalar(self,ax,scal,*a,**kw):
        srcs=[]
        segs=[]
        for i in range(self.grd.Ncells()):
            cell_points=self.grd.nodes['x'][self.grd.cell_to_nodes(i),:]
            
            srcs += [i,i,-1]
            segs += [cell_points[:,0].min(),cell_points[:,0].max(),
                     np.nan]
        segs=np.array(segs)
        srcs=np.array(srcs)
        return ax.plot(segs,scal[srcs],*a,**kw)
        
    def snapshot_figure(self,**kwargs):
        """
        Plot current model state with solution
        """
        fig=plt.figure(1)
        fig.clf()
        ax=fig.add_subplot(1,1,1)

        self.plot_cell_scalar(ax, self.zi_agg['min'],'b-')
        self.plot_cell_scalar(ax, self.ei, 'g-')
        #ax.plot(self.grd.cells_center()[:,0],
        # self.eta_fn(self.grd.cells_center(),self.t),color='orange')
        return fig


def test_1d_thacker_coarse_no_advection():
    """
    Thacker is not a strong test of advection -- here just confirm that
    the error even with no advection term is still well behaved. This
    is to (a) demonstrate that we need other tests to confirm advection,
    and (b) set a baseline for debugging when its helpful to disable advection
    but still run a wetting/drying case like Thacker.
    """
    sim=SubgridThacker1D(cg_tol=1e-10,nx=50)
    sim.get_fu=sim.get_fu_no_adv

    sim.set_grid()
    sim.set_initial_conditions()
    (hi, uj, tvol, ei) = sim.run(t_end=sim.period)
    assert sim.max_rel_error < 0.15, "Regression on relative error %.4f"%sim.max_rel_error


def test_1d_thacker_coarse_no_advection_subc10():
    """
    Most basic test with actual subgrid, just in cells.
    """
    sim=SubgridThacker1D(cg_tol=1e-10,nx=50)
    sim.get_fu=sim.get_fu_no_adv
    sim.max_subcells=10

    sim.set_grid()
    sim.set_initial_conditions()
    (hi, uj, tvol, ei) = sim.run(t_end=sim.period)
    # 2020-04-25: max_rel_error 0.0655 
    assert sim.max_rel_error < 0.07, "Regression on relative error %.4f"%sim.max_rel_error

if 0:
    sim=SubgridThacker1D(cg_tol=1e-10,nx=50,dt=10.0)
    sim.max_subcells=10
    sim.set_grid()
    sim.set_initial_conditions()

    sim.prepare_to_run()
    
    sim.snapshot_figure()
    plt.draw()
    plt.pause(0.01)

    while sim.t<sim.period:
        (_, uj, tvol, ei) = sim.run_until(sim.t+sim.dt)
        print("Max rms rel error: %.4f"%sim.max_rel_error)
        fig=sim.snapshot_figure()
        fig.canvas.draw()
        fig.canvas.start_event_loop(0.01)
        
# With no subgrid, and edges get shallower neighboring cell,
# this is stable, but has large errors.
# Max rms rel error is 2.3363.
# Note that there is no advection yet.
# The old code, with advection, got max_rel_error of <0.15

# Clone the old code, run it again to verify the 0.15,
# then run it without advection.
# => error with advection: 0.1055
#    error w/o advection:  0.1080


