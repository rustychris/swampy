"""
Test 2D version of the Thacker test case as a validation of 
subgrid.
"""
import os
from stompy.grid import unstructured_grid
from stompy import utils
import matplotlib.pyplot as plt
from matplotlib import gridspec

# Rather than adding swampy to sys.path, assume this script is run
# from this folder and add the relative path
utils.path("../")
from swampy import swampy
import numpy as np
import six
import nose.tools

six.moves.reload_module(swampy)
six.moves.reload_module(unstructured_grid)
## 

R=500e3

six.moves.reload_module(unstructured_grid)

def thacker_grid():
    thacker_2d_fn='thacker2d-%g.nc'%R
    if not os.path.exists(thacker_2d_fn):
        # Generate a radial grid
        g=unstructured_grid.UnstructuredGrid(max_sides=4)

        # Add guide rings
        scale=R/20.0 # nominal edge length

        nspoke0=4*8
        g.add_quad_ring(0*scale,1*scale,nrows=2,nspokes=nspoke0//4,sides=3,stagger0=0.75)
        g.add_quad_ring(1*scale,2*scale,nrows='stitch',nspokes=nspoke0//4,stagger0=0.25)
        g.add_quad_ring(2*scale,3*scale,nrows=2,nspokes=nspoke0/2,sides=3,stagger0=0.5) 
        g.add_quad_ring(3*scale,4*scale,nrows='stitch',nspokes=nspoke0/2) # row of quads
        g.add_quad_ring(4*scale,5*scale,nrows=2,nspokes=nspoke0) # row of quads
        g.add_quad_ring(5*scale,6*scale,nrows='stitch',nspokes=nspoke0)
        g.add_quad_ring(6*scale,10*scale,5,2*nspoke0)
        g.add_quad_ring(10*scale,11*scale,'stitch',2*nspoke0)
        g.add_quad_ring(11*scale,20*scale,11,4*nspoke0)

        g.make_cells_from_edges()
        g.edge_to_cells(recalc=True)
        g.renumber()
        g.write_ugrid(thacker_2d_fn,overwrite=True)
        
    g=unstructured_grid.UnstructuredGrid.read_ugrid(thacker_2d_fn)
    return g

## 
class Thacker2D(swampy.SwampyCore):
    r0=0.6*R
    h0=5  # verical scale of parabolic bathymetry
    eta=0.1 # perturbation for initial condition

    theta=0.55

    max_subcells=1
    max_subedges=1
    
    f=0 # no Coriolis yet
    d0=50 # m,aka h0?
    eta0=2 # m
    omega=2*np.pi/43200 # rad/s
    dt=112.5
    
    def __init__(self,**kw):
        utils.set_keywords(self,kw)

        self.A=( (self.d0+self.eta0)**2 - self.d0**2 )/( (self.d0+self.eta0)**2 + self.d0**2)
        self.L=np.sqrt( (8*9.8*self.d0) / ( self.omega**2-self.f**2) )
        self.period=2*np.pi/self.omega

        super().__init__(**kw)

        # set a scale for eta, used to make eta errors relative
        # this is approximately the high water elevation
        self.eta_scale=self.eta0
        self.max_rel_error=0.0

        self.get_fu=self.get_fu_orig

    last_plot=None
    plot_interval_per_period=1./20 # fraction of period
    def step_output(self,n,ei,**kwargs):
        plot_interval=self.period*self.plot_interval_per_period

        self.history=utils.array_append(self.history,None)
        self.history['t'][-1]=self.t

        self.history['eta_model'][-1,:] = self.ei[self.history_cells]
        # self.history['ur_model'][-1,:] = self.ei[self.history_cells]
        ur,ut,eta=self.analytical_ur_utan_eta(self.history_r,self.t)
        self.history['eta_ana'][-1]=eta
        self.history['ur_ana'][-1]=ur
        
        if (self.last_plot is not None) and (self.t-self.last_plot<plot_interval):
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

    def analytical_ur_utan_eta(self,r,t):
        w=self.omega
        A=self.A
        f=self.f
        d0=self.d0
        L=self.L
        
        ur=( (w*r*A*np.sin(w*t)) /
             (2*(1-A*np.cos(w*t))))
        utan= (f*r/( 2*(1-A*np.cos(w*t)))
               * ( np.sqrt(1-A**2) + A*np.cos(w*t) - 1) )
        eta=d0* (
            np.sqrt(1-A**2)/(1-A*np.cos(w*t))
            -1
            -r**2/L**2 * (
                (1-A**2) / (1-A*np.cos(w*t))**2 - 1)
            )
        return ur,utan,eta

    def depth_fn(self,xy):
        r=utils.mag(xy)
        return -self.d0 * (1-r**2/self.L**2)
    
    def eta_fn(self,xy,t):
        """ 
        Analytical solution for freesurface, used for
        initial condition and validation.
        """
        r=utils.mag(xy)
        _,_,eta = self.analytical_ur_utan_eta(r,t)
        return eta
            
    def set_grid(self):
        g=thacker_grid()
        g.orient_edges()

        super().set_grid(g)

        from shapely import geometry
        tran=geometry.LineString(np.array([ [-R,-0.001*R],[R,0.001*R]]))
        # precompute cells on a transect
        tran_cells=self.grd.select_cells_intersecting(tran,as_type='indices')
        cc=self.grd.cells_center()[tran_cells]
        order=np.argsort(cc[:,0])
        self.tran_cells=np.array(tran_cells)[order]
        self.tran_x=cc[order,0]

    def set_initial_conditions(self):
        super().set_initial_conditions()

        self.history_cells=[self.grd.select_cells_nearest(p)
                            for p in [ [1,1],
                                       [0.5*self.L,1],
                                       [0.9*self.L,1]] ]
        self.history_r=utils.mag( self.grd.cells_center()[self.history_cells,:] )
        Nh=len(self.history_cells)
        self.history=np.zeros( 0, [ ('t',np.float64),
                                    ('eta_model',np.float64,Nh),
                                    ('eta_ana', np.float64,Nh),
                                    ('ur_model',np.float64,Nh),
                                    ('ur_ana',np.float64,Nh)])
        
        # no subgrid yet
        cc=self.grd.cells_center()
        zi=self.depth_fn(cc)
        self.ic_zi_sub['z'][:,0]=zi
        self.ic_zi_sub['A'][:,0]=self.grd.cells_area()

        # So that we have cell agg values for edges below
        self.prepare_cell_subgrid(self.ic_zi_sub)
        self.ic_ei[:]=self.eta_fn(self.grd.cells_center(),t=0)
        
        # Edges:
        e2c=self.grd.edge_to_cells().copy()
        missing=e2c.min(axis=1)<0
        e2c[missing,:] = e2c[missing,:].max()

        ltot=self.grd.edges_length()
        n_subedge=1
        for j in range(self.grd.Nedges()):
            # simplified case for testing basics
            self.ic_zj_sub['z'][j,:]=self.ic_zi_sub['z'][e2c[j,:],0].max()
            self.ic_zj_sub['l'][j,:n_subedge]=ltot[j]/n_subedge

    def snapshot_figure(self,**kwargs):
        """
        Plot current model state with solution
        """
        fig=plt.figure(1)
        fig.clf()
        gs=gridspec.GridSpec(2,2)
        
        ax=fig.add_subplot(gs[0,0])

        ccoll=self.grd.plot_cells(ax=ax, values=self.ei,cmap='jet')
        ax.axis([-R,R,-R,R])
        plt.colorbar(ccoll)

        axt=fig.add_subplot(gs[1,:])
        axt.plot(self.tran_x,self.ei[self.tran_cells],label='Model')
        axt.plot(self.tran_x,self.zi_agg['min'][self.tran_cells])

        _,_,eta=self.analytical_ur_utan_eta(r=np.abs(self.tran_x),t=self.t)
        axt.plot(self.tran_x,np.maximum(eta,self.zi_agg['min'][self.tran_cells]),
                 'r-',label='Soln')
        axt.axis(ymin=-3,ymax=3)
        axt.legend(loc='lower right')

        axts=fig.add_subplot(gs[0,1])
        for i,r in enumerate(self.history_r):
            l=axts.plot(self.history['t'],
                      self.history['eta_model'][:,i],ls='-')
            axts.plot(self.history['t'],
                      self.history['eta_ana'][:,i],ls='--',color=l[0].get_color())
        
        
        return fig


def test_2d_thacker_no_advection():
    """
    Thacker is not a strong test of advection -- here just confirm that
    the error even with no advection term is still well behaved. This
    is to (a) demonstrate that we need other tests to confirm advection,
    and (b) set a baseline for debugging when its helpful to disable advection
    but still run a wetting/drying case like Thacker.
    """
    sim=Thacker2D(cg_tol=1e-10,nx=50)
    sim.get_fu=sim.get_fu_no_adv

    # With no subgrid, and edges get shallower neighboring cell,
    # this is stable and gets the same max_rel_error as previous code
    sim.set_grid()
    sim.set_initial_conditions()
    (hi, uj, tvol, ei) = sim.run(t_end=sim.period)

if 1:
    sim=Thacker2D(cg_tol=1e-10)
    sim.max_subcells=1
    sim.max_subedges=1
    sim.theta=0.55
    sim.set_grid()
    sim.set_initial_conditions()

    sim.prepare_to_run()

    assert np.allclose( sim.zj_sub['ltot'].max(axis=1), sim.grd.edges_length())
    assert np.allclose( sim.zi_sub['Atot'].max(axis=1), sim.grd.cells_area())
    
    sim.snapshot_figure()
    plt.draw()
    plt.pause(0.01)

    while sim.t<sim.period:
        (_, uj, tvol, ei) = sim.run_until(sim.t+sim.dt)
        #print("Max rms rel error: %.4f"%sim.max_rel_error)
        fig=sim.snapshot_figure()
        fig.canvas.draw()
        fig.canvas.start_event_loop(0.01)
        
