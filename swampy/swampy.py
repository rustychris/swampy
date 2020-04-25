"""
Created on Fri 24 Mar 2018

@author: Ed Gross, Steve Andrews
@organization: Resource Management Associates
@contact: ed@rmanet.com, steve@rmanet.com
@note: Highly simplified river model. Current assumptions:
    - circumcenters inside cell
"""
from __future__ import print_function
import six
import os

import time

from matplotlib.collections import PolyCollection
from scipy.spatial.distance import euclidean
from stompy import utils
from stompy.grid import unstructured_grid as ugrid
import logging as log

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sparse

# global variables
g = 9.8  # gravity
dc_min = 1.0
max_sides = 8
dzmin = 0.001

class SwampyError(Exception):
    """
    Superclass for swampy-related errors. Using this 
    for all errors makes testing more robust
    """
    pass

class SwampyInputError(SwampyError):
    "Inconsistent, invalid, incomplete input data"
    pass

class SwampyRuntimeError(SwampyError):
    "Failure to converge, values went nan, etc."
    pass

class SwampyCore(object):
    # tolerance for conjugate gradient
    cg_tol=1.e-12
    dt=0.025
    theta=1.0
    manning_n=0.0

    max_subcells=10 # can be changed up to set_initial_conditions
    max_subedges=10 # can be changed up to set_initial_conditions
    
    def __init__(self,**kw):
        utils.set_keywords(self,kw)
        self.bcs=[]
        # Default advection
        self.get_fu=self.get_fu_Perot

    def add_bc(self,bc):
        self.bcs.append(bc)

    def center_vel_perot(self, i, cell_uj):
        """
        Cell center velocity calculated with Perot dual grid interpolation
        return u center veloctiy vector, {x,y} components, based on
        edge-normal velocities in cell_uj.
        """
        acu = np.zeros(2, np.float64)
        uside = np.zeros_like(acu)
        ucen = np.zeros_like(acu)
        nsides = len(cell_uj)
        for l in range(nsides):
            j = self.grd.cells[i]['edges'][l]
            for dim in range(2):  # x and y
                uside[dim] = cell_uj[l] * self.en[j, dim]  # u vel component of normal vel
                acu[dim] += uside[dim] * self.len[j] * self.dist[i, l]
        for dim in range(2):  # x and y
            ucen[dim] = acu[dim] / self.grd.cells['_area'][i]

        return ucen

    def get_center_vel(self, uj):
        """
        Get center velocities for every cell in the grid
        """
        ui = np.zeros((self.ncells, 2), np.float64)
        for i in range(self.ncells):
            nsides = self.ncsides[i]
            cell_uj=uj[ self.grd.cells['edges'][i,:nsides] ]
            ui[i] = self.center_vel_perot(i, cell_uj)
            
        return ui

    def get_fu_no_adv(self, uj, fu, **kw):
        """
        Return fu term with no advection
        """
        # allocating fu and copying uj has been moved up and out of the advection
        # methods
        return
    
    def get_fu_orig(self, uj, fu, **kw):
        """
        Add to fu term the original advection method.
        """
        ui = self.get_center_vel(uj)
        for j in self.intern:  # loop over internal cells
            ii = self.grd.edges[j]['cells']  # 2 cells
            if uj[j] > 0.:
                i_up = ii[0]
            elif uj[j] < 0.:
                i_up = ii[1]
            else:
                fu[j] = 0.0
                continue
            # ui_norm = component of upwind cell center velocity perpendicular to cell face
            ui_norm = ui[i_up, 0] * self.en[j, 0] + ui[i_up, 1] * self.en[j, 1]
            # this has been updated to just add to fu, since fu already has explicit barotropic
            # and uj[j]
            fu[j] += uj[j] * (-2.*self.dt * np.sign(uj[j]) * (uj[j] - ui_norm) / self.dc[j])

        return fu

    def add_explicit_barotropic(self,hjstar,hjbar,hjtilde,ei,fu):
        """
        return the per-edge, explicit baroptropic acceleration.
        This used to be part of get_fu_Perot, but is split out since it is not 
        specifically advection.

        hjbar, hjtilde: Kramer and Stelling edge heights
        ei: cell-centered freesurface
        dest: edge-centered velocity array to which the term is added, typ. fu
        """
        if self.theta==1.0: # no explicit term
            return

        gDtThetaCompl = g * self.dt * (1.0 - self.theta)
        iLs=self.grd.edges['cells'][:,0]
        iRs=self.grd.edges['cells'][:,1]

        hterm=np.ones_like(self.aj)
        
        # dry=hjstar<=0
        dry=self.aj==0

        if 0: # Kramer and Stelling
            # three cases: deep enough to have the hjbar term, wet but not deep,
            # and dry.
            deep=hjbar>=dzmin
            hterm[deep]=hjtilde[deep]/hjbar[deep]
        else:
            pass
            
        hterm[dry]=0.0

        fu[self.intern] -= (gDtThetaCompl * hterm * (ei[iRs] - ei[iLs]) / self.dc[:])[self.intern]
        
    def get_fu_Perot(self, uj, alpha, sil, hi, hjbar, hjtilde, hjstar, fu):
        """
        Return fu term with Perot method.
        """
        ui = self.get_center_vel(uj)

        # some parts can be vectorized
        iLs=self.grd.edges['cells'][:,0]
        iRs=self.grd.edges['cells'][:,1]

        lhu=self.aj[:] * uj[:]

        # FIX: blindly using calc_wetarea result pi instead of
        # previous code that used self.area
        cell_area=self.pi

        for j in self.intern:
            iL = iLs[j]
            iR = iRs[j]

            # Skip dry edges
            # non-subgrid:
            # if hjstar[j]<=0: continue
            
            # FIX: does this properly allow for wetting?  Not sure whether
            # the goal is to even allow for mom-advection at wetting front.
            # But code below assumes area of cells>0.
            if ((cell_area[iL]==0.0)
                or (cell_area[iR]==0.0)
                or (self.aj[j]==0.0) ):
                continue 

            # code below uses hjbar, so better be sure it's nonzero,
            # though not sure the code below is still correct w/subgrid
            assert hjbar[j]>0
            
            # explicit barotropic moved out to separate function

            for lr,iLR in [(0,iL),
                           (1,iR)]:
                nsides = self.ncsides[iLR]
                sum1 = 0.
                for l in range(nsides):
                    k = self.grd.cells[iLR]['edges'][l]
                    Q = sil[iL, l] * lhu[k] # lhu[k]==self.len[k] * hjstar[k] * uj[k] == self.aj * uj
                    if Q >= 0.:  # ignore fluxes out of the cell
                        continue
                    iitmp = self.grd.edges[k]['cells']
                    i2 = iitmp[self.sil_idx[iLR,l]]
                    ui_norm = ui[i2, 0] * self.en[j, 0] + ui[i2, 1] * self.en[j, 1]
                    sum1 += Q * (ui_norm - uj[j])
                # in original code only difference is 2nd index to alpha
                fu[j] -= self.dt * alpha[j, lr] * sum1 / (cell_area[iLR] * hjbar[j])

        return fu

    # def calc_phi(r, limiter):
    #     """
    #     Calculate limiter phi
    #     """
    #     if limiter == 'vanLeer':
    #         phi = (r + abs(r)) / (1.0 + r)
    #     elif limiter == 'minmod':
    #         phi = max(0.0, min(r, 1.0))
    #     else:
    #         print('limiter not defined %s' % limiter)
    # 
    #     return phi

    def cell_to_edge_upwind(self, cell_data, u_j, tie='max'):
        """
        Upwind a cell-centered value on edges based on the velocity u_j.
         tie: 'max' For u_j==0 choose the greater of cell_data.
        """
        ii = self.edge_to_cells_reflect
        e_data = np.where( u_j>0, cell_data[ii[:,0]], cell_data[ii[:,1]] )
        ties=(u_j==0.0)
        if tie=='max':
            e_data[ties] = np.maximum( cell_data[ii[ties,0]],
                                       cell_data[ii[ties,1]] )
        else:
            raise Exception("Need to add support for other tie mechanisms")
        return e_data
        
    # def calc_hjstar(self, ei, uj):
    #     """
    #     Calculate hjstar from Kramer and Stelling
    #     """
    #     e_up=self.cell_to_edge_upwind(ei,uj,tie='max')
    # 
    #     # force non-negative here.
    #     # it is possible for zj+e_up<0.  this is not an error, just
    #     # an aspect of the discretization.
    #     # nonetheless hjstar should not be allowed to go negative
    #     # FIX - definition of hjstar unclear.
    #     hjstar = (e_up - self.zj_agg['min']).clip(0.0)
    #     return hjstar

    # def calc_hjbar(self, ei):
    #     """
    #     Calculate hjbar from Kramer and Stelling
    #     (linear interpolation from depth in adjacent cells)
    #     """
    #     hjbar = np.zeros(self.nedges,np.float64)
    #     for j in self.intern:  # loop over internal edges
    #         ii = self.grd.edges[j]['cells']  # 2 cells
    #         # Original code recalculated.
    #         # RH  - hi is already limited to non-negative
    #         # FIX - hi is not properly defined for subgrid, so
    #         # hjbar isn't either.
    #         hL=self.hi[ii[0]]
    #         hR=self.hi[ii[1]]
    #         hjbar[j] = self.alpha[j][0] * hL + self.alpha[j][1] * hR
    # 
    #     assert np.all(hjbar>=0)
    #     return hjbar

    # def calc_hjtilde(self, ei):
    #     """
    #     Calculate hjtilde from Kramer and Stelling
    #     (with average cell eta, and interpolated cell bed)
    #     """
    #     hjtilde = np.zeros(self.nedges,np.float64)
    #     for j in self.intern:  # loop over internal edges
    #         ii = self.grd.edges[j]['cells'] 
    #         eavg = 0.5 * (ei[ii[0]] + ei[ii[1]])
    #         # FIX: not sure of what the proper z is here.
    #         bL = self.zi_agg['min'][ii[0]]
    #         bR = self.zi_agg['min'][ii[1]]
    #         hjtilde[j] = eavg - (self.alpha[j][0] * bL + self.alpha[j][1] * bR)
    #     return hjtilde

    def calc_volume(self, eta):
        """
        Calculate cell volume give eta for cell
        With subgrid this gets more involved.
        """
        # non-subgrid:
        # return self.area * (eta+self.zi)

        # subgrid information for cells is a list of elevation and corresponding area.
        # the slow but clear way:
        vol=np.zeros(self.ncells,np.float64)
        for i in range(self.ncells):
            k=np.searchsorted(self.zi_sub['z'][i], # _elevations_ where areas are defined
                              eta[i])-1
            if k<0: 
                vol[i]=0.0
            else:
                dz_partial=(eta[i]-self.zi_sub['z'][i,k])
                assert dz_partial>=0.0
                vol[i]=self.zi_sub['Vtot'][i,k] + dz_partial*self.zi_sub['Atot'][i,k]
        return vol
        
    def calc_wetarea(self, eta):
        """
        Calculate cell wetted area give cell total water depth
        With subgrid this will be more involved.
        """
        # non-subgrid:
        # return np.where( eta>-self.zi, self.area, 0.0)

        # subgrid information for cells is a list of elevation and corresponding area.
        # the slow but clear way:
        A=np.zeros(self.ncells,np.float64)
        for i in range(self.ncells):
            k=np.searchsorted(self.zi_sub['z'][i], # _elevations_ where areas are defined
                              eta[i])-1
            if k<0: 
                A[i]=0.0
            else:
                A[i]=self.zi_sub['Atot'][i,k]
        return A

    def calc_edge_wetarea(self, eta_j):
        """
        Calculate wetted area for edges.
        eta_j: freesurface at the edge, presumably from upwinding cell
        eta (and max when u==0.0).

        This gets called with hjstar.  For subgrid, what is appropriate?

        hjstar is total depth with upwinded eta and shallower zi.

        Subgrid puts the onus of proper edge bathy on the user, rather
        than defaulting to shallower zi.
        """
        # non-subgrid:
        # return self.len * (self.zj + eta_j).clip(0.0)

        # subgrid information for cells is a list of elevation and corresponding area.
        # the slow but clear way:
        Aj=np.zeros(self.nedges,np.float64)
        for j in self.intern:
            k=np.searchsorted(self.zj_sub['z'][j], # _elevations_ where areas are defined
                              eta_j[j])-1
            if k<0: 
                Aj[j]=0.0
            else:
                dz_partial=(eta_j[j]-self.zj_sub['z'][j,k])
                assert dz_partial>=0.0
                Aj[j]=self.zj_sub['Atot'][j,k] + dz_partial*self.zj_sub['ltot'][j,k]
        return Aj

    def calc_edge_friction(self, uj, aj, hj):
        """
        Calculate friction coef at edge
        Cf = n^2*g*|u|/Rh^(4/3)
        """

        # FIX: Should update to deal with subedges
        cf = np.zeros(self.nedges)
        n = self.manning_n
        for j in self.intern:
            rh = hj[j]  # assume no side wall friction
            if rh < dzmin:
                cf[j] = 0.
            else:
                cf[j] = n * n * g * np.abs(uj[j]) / rh ** (4. / 3.)
        return cf

    def set_grid(self,ug):
        """
        Set grid from unstructured_grid.UnstructuredGrid instance
        """
        self.grd=ug
        self.set_topology()

    def set_topology(self):
        """
        Use functions of unstructured grid class for remaining topology
        """
        self.nedges = self.grd.Nedges()
        self.ncells = self.grd.Ncells()
        self.nnodes = self.grd.Nnodes()
        self.grd.update_cell_edges()
        self.grd.update_cell_nodes()
        self.grd.edge_to_cells()
        self.grd.cells_area()
        self.grd.cells['_center'] = self.grd.cells_centroid()
        self.grd.edges['mark'] = 0  # default is internal cell
        self.extern = np.where(np.min(self.grd.edges['cells'], axis=1) < 0)[0]
        self.grd.edges['mark'][self.extern] = 1  # boundary edge
        self.intern = np.where(self.grd.edges['mark'] == 0)[0]
        self.nedges_intern = len(self.intern)  # number of internal edges
        self.exy = self.grd.edges_center()
        self.en = self.grd.edges_normals()
        self.len = self.grd.edges_length()
        # Reflect edge neighbors at boundaries
        ii=self.grd.edge_to_cells().copy()
        nc1=ii[:,0]
        nc2=ii[:,1]
        ii[:,0]=np.where(ii[:,0]>=0,ii[:,0],ii[:,1])
        ii[:,1]=np.where(ii[:,1]>=0,ii[:,1],ii[:,0])
        self.edge_to_cells_reflect=ii
        
        # number of valid sides for each cell
        self.ncsides = np.asarray([sum(jj >= 0) for jj in self.grd.cells['edges']])

        return

    def get_cell_center_spacings(self):
        """
        Return cell center spacings
        Spacings for external edges are set to dc_min (should not be used)
        """
        dc = dc_min * np.ones(self.nedges, np.float64)
        # vectorize
        i0=self.grd.edges['cells'][self.intern,0]
        i1=self.grd.edges['cells'][self.intern,1]
        dc[self.intern] = utils.dist( self.grd.cells['_center'][i0],
                                      self.grd.cells['_center'][i1] )
        self.dc = dc
        return dc

    def edge_to_cen_dist(self):
        dist = np.zeros((self.ncells, max_sides), np.float64)
        for i in range(self.ncells):
            cen_xy = self.grd.cells['_center'][i]
            nsides = self.ncsides[i]
            for l in range(nsides):
                j = self.grd.cells[i]['edges'][l]
                side_xy = self.exy[j]
                # if grid is not 'cyclic', i.e. all nodes fall on circumcircle,
                # then this is not accurate, and according to Steve's code
                # it is necessary to actually intersect the line between cell
                # centers with the edge.
                # not sure that it wouldn't be better to use the perpendicular
                # distance, though.
                dist[i, l] = utils.dist(side_xy, cen_xy) # faster

        return dist

    def get_alphas_Perot(self):
        """
        Return Perot weighting coefficients
        Coefficients for external edges are set to zero (should not be used)
        """
        # should probably be refactored to used edge-center distances as above.
        # for cyclic grid should be fine.  for non-cyclic, need to verify what
        # the appropriate distance (center-to-intersection, or perpendicular)
        alpha = np.zeros((self.nedges, 2), np.float64)
        i0=self.grd.edges['cells'][self.intern,0]
        i1=self.grd.edges['cells'][self.intern,1]
        cen_xyL = self.grd.cells['_center'][i0]
        cen_xyR = self.grd.cells['_center'][i1]
        alpha[self.intern, 0] = utils.dist(self.exy[self.intern], cen_xyL) / self.dc[self.intern]
        alpha[self.intern, 1] = utils.dist(self.exy[self.intern], cen_xyR) / self.dc[self.intern]
        self.alpha = alpha

        return alpha

    def get_sign_array(self):
        """
        Sign array for all cells, for all edges
        Positive if face normal is pointing out of cell
        """
        sil = np.zeros((self.ncells, max_sides), np.float64)
        for i in range(self.ncells):
            for l in range(self.ncsides[i]):
                j = self.grd.cells[i]['edges'][l]
                ii = self.grd.edges[j]['cells']  # 2 cells
                sil[i, l] = (ii[1] - 2 * i + ii[0]) / (ii[1] - ii[0])

        # for indexing -- just 0,1
        self.sil_idx = ((sil+1)//2).astype(np.int64)

        return sil

    def get_side_num_of_cell(self):
        """
        Sparse matrix of local side number (0,1,2,or 3) for each cell, for each edge
        """
        lij = -1 * np.ones((self.ncells, self.nedges), np.int32)
        for i in range(self.ncells):
            for l in range(self.ncsides[i]):
                j = self.grd.cells[i]['edges'][l]
                lij[i, j] = l

        return lij

    def set_initial_conditions(self,t=0.0):
        """
        Subclasses or caller can specialize this.  Here we just allocate
        the arrays
        """
        self.t=t
        # Initial free surface elevation, positive up
        self.ic_ei = np.zeros(self.ncells, np.float64)
        # Bed elevation, positive down
        self.ic_zi_sub = self.allocate_cell_subgrid()
        self.ic_zj_sub = self.allocate_edge_subgrid()

        # other contenders:
        # self.ds_i  index array for cell BCs on downstream end
        # self.us_i  index array for cell BCs on upstream end
        # self.us_j  index array for edge BCs on upstream end
        # need to refactor BC code

    def run(self, t_end):
        """
        Initialize and run simulation until time reaches t_end.
        """
        self.prepare_to_run()
        return self.run_until(t_end)

    def prepare_to_run(self):
        # initialization of self.t moved to set_initial_condition()

        # PRECOMPUTE CONSTANTS
        self.gdt2theta2 = g * (self.dt*self.theta)**2 

        self.get_cell_center_spacings() # sets self.dc
        self.dist = self.edge_to_cen_dist()
        self.alpha = self.get_alphas_Perot()
        self.sil = self.get_sign_array()
        # lij = self.get_side_num_of_cell()

        # edge length divided by center spacing
        # len_dc_ratio = np.divide(self.len, dc)

        # cell center values
        # self.hi = np.zeros(self.ncells, np.float64)  # total depth
        # self.zi = np.zeros(self.ncells, np.float64)  # bed elevation, measured positive down. replace w/subgrid
        self.ei = np.zeros(self.ncells, np.float64)  # water surface elevation, cells
        self.vi = np.zeros(self.ncells, np.float64)  # cell volumes
        self.pi = np.zeros(self.ncells, np.float64)  # cell wetted areas

        # edge values
        self.uj = np.zeros(self.nedges, np.float64)  # normal velocity at side
        self.qj = np.zeros(self.nedges, np.float64)  # normal velocity*h at side
        self.aj = np.zeros(self.nedges, np.float64)  # edge wet areas
        self.cf = np.zeros(self.nedges, np.float64)  # edge friction coefs
        self.zj = np.zeros(self.nedges, np.float64)  # edge depth -- to replace w/subgrid
        self.cfterm = np.zeros(self.nedges, np.float64)  # edge friction coefs - term for matrices

        # Matrix
        # np.zeros((ncells, ncells), np.float64)  # inner iterations
        self.Ai = sparse.dok_matrix( (self.ncells,self.ncells),np.float64) 
        self.bi = np.zeros(self.ncells, np.float64)
        self.Ao = sparse.dok_matrix( (self.ncells, self.ncells), np.float64)  # outer iterations
        self.bo = np.zeros(self.ncells, np.float64)
        self.x0 = np.zeros(self.ncells, np.float64)

        # short vars for grid
        # self.area = self.grd.cells['_area']

        # set initial conditions
        self.prepare_cell_subgrid(self.ic_zi_sub)
        self.prepare_edge_subgrid(self.ic_zj_sub)
        
        self.ei[:] = np.maximum(self.ic_ei[:],self.zi_agg['min']) # RH: prop up to at least deepest point
        # FIX - unsure of correct definition of hi 
        # self.hi[:] = self.ei - self.zi_agg['min'] # update total depth. assume ei is valid, so no clipping here.
        # assert np.all(self.hi>=0.0)
        self.vi[:] = self.calc_volume(self.ei)
                      
        self.eta_cells={}
        self.eta_mask=np.zeros( self.ncells, np.bool8)
        for bc in self.bcs:
            if isinstance(bc,StageBC):
                for c in bc.cells(self):
                    self.eta_cells[c]=bc # doesn't matter what the value is
                    self.eta_mask[c]=True
        log.info("Total of %d stage BC cells"%len(self.eta_cells))

    zi_sub_dtype=[ ('z',np.float64), # positive UP
                   ('A',np.float64), # area exactly at this elevation
                   ('Atot',np.float64), # net wet area at this interface
                   ('Vtot',np.float64)  # cumulative volume below this interface
    ]
    zj_sub_dtype=[ ('z',np.float64), # positive UP
                   ('l',np.float64), # length exactly at this elevation
                   ('ltot',np.float64), # net length at this interface
                   ('Atot',np.float64), # net area below this interface
    ]
    
    def allocate_cell_subgrid(self):
        # scan to find max number of subgrid elements
        return np.zeros( (self.ncells, self.max_subcells), 
                         dtype=self.zi_sub_dtype)

    def allocate_edge_subgrid(self):
        # scan to find max number of subgrid elements
        return np.zeros( (self.nedges, self.max_subedges), 
                         dtype=self.zj_sub_dtype)

    def prepare_cell_subgrid(self,zi_sub_in):
        """
        Copy the subgrid data and standardize/preprocess
        as needed.
        Assume zi_sub_in already np.array( (Nc,max_subcells), zi_sub_dtype),
        with 'z' and 'A' fields populated
        """
        self.zi_sub=zi_sub_in.copy()
        
        self.zi_agg=np.zeros(self.ncells,[('min',np.float64), # deepest
                                          ('max',np.float64), # shallowest
                                          ('mean',np.float64)]) # area-weighted mean

        for i in range(self.ncells):
            pop=self.zi_sub['A'][i,:]>0.0
            Apop=self.zi_sub['A'][i,pop]
            zpop=self.zi_sub['z'][i,pop]
            self.zi_agg[i]['max']=zpop.max()
            self.zi_agg[i]['min']=zpop.min()
            self.zi_agg[i]['mean']=(zpop*Apop).sum() / Apop.sum()
            
            self.zi_sub['z'][i,~pop]=np.inf
            order=np.argsort(self.zi_sub['z'][i])
            self.zi_sub['z'][i,:]=self.zi_sub['z'][i,order]
            self.zi_sub['A'][i,:]=self.zi_sub['A'][i,order]
            self.zi_sub['Atot'][i,:]=np.cumsum(self.zi_sub['A'][i,:])
                    
            # V[i,k] is the volume up to that elevation
            # so V[i,0] is always 0.0
            self.zi_sub['Vtot'][i,0]=0.0
            dz=np.diff(self.zi_sub['z'][i])
            self.zi_sub['Vtot'][i,1:]=np.cumsum( dz*self.zi_sub['Atot'][i,:-1] )

    def prepare_edge_subgrid(self,zj_sub_in):
        """
        Copy the subgrid data and standardize/preprocess
        as needed.
        Assume zj_sub_in already np.array( (Ne,max_subedges), zj_sub_dtype),
        with 'z' and 'l' fields populated
        """
        if not np.allclose( zj_sub_in['l'].sum(axis=1), self.grd.edges_length() ):
            raise SwampyInputError("Sub-edge lengths do not sum to edge length")
        
        self.zj_sub=zj_sub_in.copy()
        self.zj_agg=np.zeros(self.nedges,[('min',np.float64), # deepest
                                          ('max',np.float64), # shallowest
                                          ('mean',np.float64)]) # length-weighted mean

        for j in range(self.nedges):
            pop=self.zj_sub['l'][j,:]>0.0
            lpop=self.zj_sub['l'][j,pop]
            zpop=self.zj_sub['z'][j,pop]
            self.zj_agg[j]['max']=zpop.max()
            self.zj_agg[j]['min']=zpop.min()
            self.zj_agg[j]['mean']=(zpop*lpop).sum() / lpop.sum()
            
            self.zj_sub['z'][j,~pop]=np.inf
            order=np.argsort(self.zj_sub['z'][j])
            npop=pop.sum()
            self.zj_sub['z'][j,:]=self.zj_sub['z'][j,order]
            self.zj_sub['l'][j,:]=self.zj_sub['l'][j,order]
            self.zj_sub['ltot'][j,:]=np.cumsum(self.zj_sub['l'][j,:])
                    
            # A[j,k] is the area up to the kth interface
            # so A[j,0] is always 0.0
            self.zj_sub['Atot'][j,0]=0.0
            dz=np.diff(self.zj_sub['z'][j,:].clip(-np.inf,zpop.max() ))
            self.zj_sub['Atot'][j,1:]=np.cumsum( dz*self.zj_sub['ltot'][j,:-1] )
        
    def run_until(self,t_end=None,n_steps=None):
        dt=self.dt
        if t_end is not None:
            self.t_end = t_end
            assert self.t<=t_end,"Request for t_end %s before t %s"%(t_end,t)
            n_steps = np.int(np.round((t_end - self.t) / dt))
        elif n_steps is not None:
            self.t_end = self.t+dt*n_steps

        uj=self.uj
        ei=self.ei
        alpha=self.alpha 
        sil=self.sil
        # hi=self.hi 
        ncells=self.ncells
        eta_cells=self.eta_cells
        cfterm=self.cfterm

        def update_geometry(ei,uj):
            """
            Update cell volume, area, edge area, friction terms
            based on freesurface and edge velocity
            """
            # hjstar = self.calc_hjstar(self.ei, self.uj)
            # hjbar = self.calc_hjbar(self.ei)
            # hjtilde = self.calc_hjtilde(self.ei)
            self.vi[:] = self.calc_volume(ei)
            self.pi[:] = self.calc_wetarea(ei)

            eta_j=self.cell_to_edge_upwind(cell_data=ei,
                                           u_j=uj,tie='max')
            self.aj[:] = self.calc_edge_wetarea(eta_j)
            self.cf[:] = self.calc_edge_friction(uj, self.aj, self.aj/self.len)
            self.cfterm[:] = 1. / (1. + self.dt * self.cf[:])

        update_geometry(ei,uj)
        
        # change code to vector operations at some point
        # time stepping loop
        tvol = np.zeros(n_steps)
        for n in range(n_steps):
            self.t+=dt # update model time
            log.info('step %d/%d  t=%.3fs'%(n+1,n_steps,self.t))

            # G: explicit baroptropic term, advection term
            fu=np.zeros_like(uj)
            fu[self.intern]=uj[self.intern]
            fu[self.aj<=0]=0.0
            
            self.add_explicit_barotropic(ei=ei,fu=fu,hjstar=None,hjbar=None,hjtilde=None)
            # shouldn't happen, but check to be sure
            assert np.all( fu[self.aj==0.0]==0.0 )
            
            self.get_fu(uj,fu=fu,alpha=alpha,sil=sil,
                        hi=None,hjstar=None,hjtilde=None,hjbar=None)
            
            # get_fu should avoid this, but check to be sure
            assert np.all( fu[self.aj==0.0]==0.0 )

            # Following Casulli 2009, eq 18
            # zeta^{m+1}=zeta^m-[P(zeta^m)+T]^{-1}[V(zeta^m)+T.zeta^m-b]
            # Rearranging...
            # [P(zeta^m)+T]  (zeta^{m+1}-zeta^m) = [V(zeta^m)+T.zeta^m-b]
            # where T holds the coefficients for the linear system.
            # note that aj goes into the T coefficients using the old freesurface
            # and is not updated as part of the linear system.
            # the matrix Ai in the code below corresponds to T 
            
            # set matrix coefficients
            # first put one on diagonal, and explicit stuff on rhs (b)
            # Ai=sparse.dok_matrix((ncells,ncells), np.float64)
            Ai_rows=[]
            Ai_cols=[]
            Ai_data=[]

            self.bi[:] = 0.

            for i in range(self.ncells):
                if i in self.eta_cells: # line profiling may show that this is slow
                    # Ai[i,i]=1.0
                    Ai_rows.append(i) 
                    Ai_cols.append(i)
                    Ai_data.append(1.0)
                    continue

                sum1 = 0.0
                for l in range(self.ncsides[i]):
                    j = self.grd.cells[i]['edges'][l]
                    if self.grd.edges[j]['mark'] == 0:  # if internal
                        sum1 += self.sil[i, l] * self.aj[j] * (self.theta * self.cfterm[j] * fu[j] +
                                                     (1-self.theta) * uj[j])
                        # Disabled during transition to subgrid
                        #if hjbar[j] > dzmin:
                        #    hterm = hjtilde[j] / hjbar[j]
                        #else:
                        hterm = 1.0
                        coeff = self.gdt2theta2 * self.aj[j] * self.cfterm[j] / self.dc[j] * hterm
                        # Ai[i, i] += coeff
                        Ai_rows.append(i)
                        Ai_cols.append(i)
                        Ai_data.append(coeff)

                        ii = self.grd.edges[j]['cells']  # 2 cells
                        i2 = ii[np.nonzero(ii - i)[0][0]]  # i2 = index of neighbor
                        #Ai[i, i2] = -coeff
                        Ai_rows.append(i)
                        Ai_cols.append(i2)
                        Ai_data.append(-coeff)
                # Ai is formulated in terms of just the change, and does not
                # have 1's on the diagonal, but don't be tempted to remove vi here.
                # it's correct.
                self.bi[i] = self.vi[i] - dt * sum1
            Ai=sparse.coo_matrix( (Ai_data,(Ai_rows,Ai_cols)), (ncells,ncells), dtype=np.float64 )
            Ai=Ai.tocsr()
            
            for iter in range(10):
                if 1:
                    # This block:
                    # Ao=Ai+diag(pi)
                    Ao_rows=[] # list(Ai_rows)
                    Ao_cols=[] # list(Ai_cols)
                    Ao_data=[] # list(Ai_data)

                    #for i in range(ncells):
                    #    Ao[i, i] += pi[i]
                    non_eta=np.nonzero(~self.eta_mask)[0]
                    Ao_rows.extend(non_eta)
                    Ao_cols.extend(non_eta)
                    Ao_data.extend(self.pi[~self.eta_mask])
                    # Ao[~eta_mask,~eta_mask] += pi[~eta_mask]

                    Ao=sparse.coo_matrix( (Ao_data,(Ao_rows,Ao_cols)), (ncells,ncells), dtype=np.float64)
                    Ao=Ao.tocsr()
                    Ao=Ai+Ao

                #for i in range(ncells):
                #    bo[i] = vi[i] + np.dot(Ai[i, :], ei[:]) - bi[i]
                self.bo[:] = self.vi + Ai.dot(self.ei) - self.bi

                self.bo[self.eta_mask]=0. # no correction

                for bc in self.bcs:
                    if isinstance(bc,StageBC):
                        c,e,h=bc.cell_edge_eta(self)
                        ei[c]=h
                    elif isinstance(bc,FlowBC):
                        # flow bcs:
                        c,e,Q=bc.cell_edge_flow(self)
                        self.bo[c] += -dt * Q

                # For thacker we spend a lot of time here (30%)
                # initial matrix has 680/2000 zeros on the diagonal --
                # seems like a lot.  Nonzero elements range from 56 to 7600.
                # also 680 zeros on the rhs.
                # So this thing is massively singular.

                wet_only=0
                if wet_only: # limit matrix solve to wet cells
                    # Probably there is a better way to construct the
                    # mask for wet cells.
                    wet=self.eta_mask | (self.pi>0.0) | (self.bo!=0.0)
                    # A[wet,:][:,wet] x[wet] = b[wet]
                    
                    bo_sel=self.bo[wet]
                    x0_sel=self.x0[wet]
                    # csr seems fastest for this step, at 225us for ncells=2000
                    # when wet is 2/3 full.
                    to_wet=sparse.identity(self.ncells,np.float64,format='csr')[wet,:]
                    # The goal is A[wet,:][:,wet]
                    # This approach gets there with matrix ops, but slicing may be
                    # faster.
                    Ao_sel=to_wet.dot(Ao).dot( to_wet.transpose() )
                    N_sel=len(bo_sel)
                else:
                    # Solve the full system
                    bo_sel=self.bo
                    x0_sel=self.x0
                    Ao_sel=Ao
                    N_sel=self.ncells

                # 1. Try a basic preconditioner
                #    diagonal entries are non-negative, limit to 1.0.
                #    roughly 20% speedup from preconditioner.  likely that the
                #    the zeros are the real cause of this being slow.
                if 1: # use preconditioner
                    Mdiag=1./( Ao_sel.diagonal().clip(1.0,np.inf) )
                    M=sparse.diags(Mdiag, format='csr')
                else:
                    M=None # no preconditioner
                    
                # invert matrix
                start = time.time()
                # CG
                ei_corr, success = sparse.linalg.cg(Ao_sel,
                                                    bo_sel, x0=x0_sel, M=M,
                                                    atol=0.0,
                                                    tol=self.cg_tol)  # Increase tolerance for faster speed
                # BiCGStab - not stable, gmres: not stable.
                end = time.time()
                # print('matrix solve took {0:0.4f} sec'.format(end - start))
                if success != 0:
                    raise SwampyRuntimeError('Error in convergence of conj grad solver')

                if wet_only:
                    # to keep the code below undisturbed, expand the wet-only results
                    # back to the full set of cells
                    full=np.zeros(ncells,np.float64)
                    full[wet]=ei_corr
                    ei_corr=full
                
                # better to clip ei and hi, not just correct hi.
                # FIX - hi not properly defined for subgrid
                #self.hi[:] = (self.ei - self.zi_agg['min'])
                # neg=self.hi<0
                # if np.any(neg):
                #     self.ei[neg]=self.zi_agg['min'][neg]
                #     self.hi[neg]=0
                self.ei[:] = np.maximum( self.ei[:] - ei_corr,
                                         self.zi_agg['min'] )
                
                # Update elevations
                # for subgrid, these should become functions of eta rather than
                # hi.
                self.pi[:] = self.calc_wetarea(self.ei)
                self.vi[:] = self.calc_volume(self.ei)
                rms = np.sqrt(np.sum(np.multiply(ei_corr, ei_corr)) / self.ncells)
                if rms < 0.01:
                    break

            # substitute back into u solution
            for j in self.intern:  # loop over internal cells
                ii = self.grd.edges[j]['cells']  # 2 cells
                # RH: don't accelerate dry edges
                hterm = 1.0*(self.aj[j]>0)
                term = g * dt * self.theta * (self.ei[ii[1]] - self.ei[ii[0]]) / self.dc[j]
                uj[j] = self.cfterm[j] * (fu[j] - term * hterm)

            # DBG: temporary check
            assert np.abs(self.uj).max() < 20.0

            update_geometry(ei=self.ei,uj=self.uj)

            self.postcalc_update_bc_velocity()
            
            # conservation properties
            tvol[n] = np.sum(self.vi)

            self.step_output(n=n,ei=ei,uj=uj)

        return None, uj, tvol, ei

    def postcalc_update_bc_velocity(self):
        """
        After the calculation of computational edges, update self.uj
        for BC edges where possible.  Currently leaves stage bcs
        alone, and just fills in velocity for flow bcs.
        """
        for bc in self.bcs:
            if isinstance(bc,StageBC):
                # not yet defined how edge adjacent to stage BC should be
                # handled.  Leave as zero.
                pass 
            elif isinstance(bc,FlowBC):
                c,e,Q=bc.cell_edge_flow(self)
                for j,q in zip(e,Q):
                    if Q==0.0: continue
                    a=self.aj[j]
                    if a<bc.small_area:
                        log.warning("Setting BC velocity, very small area will be clipped to %.4e"%bc.small_area)
                        a=bc.small_area
                    # the negative sign due to convention that boundary edges have outward pointing normal
                    # but boundary condition supplies inflow=positive q.
                    self.uj[j] = -q/a

    def get_edge_x_vel(self, uj):
        """
        Return x direction velocity and location at cell edge midpoints
        """
        xe = np.zeros(self.nedges_intern)
        uxe = np.zeros_like(xe)
        for idx, j in enumerate(self.intern):
            xe[idx] = self.exy[j][0]  # x coordinate
            uxe[idx] = uj[j] * self.en[j][0]

        return xe, uxe

    def get_xsect_avg_val(self, eta_i):
        """
        Return cross-sectionally averaged x velocities for non-uniform grids
        """
        xc = []
        uxc = []
        nx = self.domain_length / self.dx
        for i in range(int(nx)):
            startx = float(i) * self.dx
            endx = float(i + 1) * self.dx
            midx = (float(i) + 0.5) * self.dx
            d = (self.grd.cells['_center'][:, 0] > startx) & (self.grd.cells['_center'][:, 0] < endx)
            if np.sum(d) > 0:
                utmp = np.mean(eta_i[d])
                uxc.append(utmp)
                xc.append(midx)
        return (xc, uxc)

    def get_grid_poly_collection(self):
        """
        Return list of polygon points useful for 2D plotting
        """
        xy = []
        for i in range(self.ncells):
            tmpxy = []
            for l in range(self.ncsides[i]):
                n = self.grd.cells[i]['nodes'][l]
                tmp = self.grd.nodes[n]['x']
                tmpxy.append([tmp[0], tmp[1]])
            n = self.grd.cells[i]['nodes'][0]
            tmp = self.grd.nodes[n]['x']
            tmpxy.append([tmp[0], tmp[1]])
            xy.append(tmpxy)

        return xy

    def step_output(self,n,ei,**kwargs):
        pass

# Structuring BCs:
#   for bump case, currently the code replaces downstream matrix rows
#   and rhs values with ds_eta value.
#   and upstream, sets a flow boundary in the rhs.

class BC(object):
    _cells=None
    _edges=None
    geom=None
    def __init__(self,**kw):
        utils.set_keywords(self,kw)
    def update_matrix(self,model,A,b):
        pass
    def cells(self,model):
        if self._cells is None:
            self.set_elements(model)
        return self._cells
    def edges(self,model):
        if self._edges is None:
            self.set_elements(model)
        return self._edges
    def set_elements(self,model):
        cell_hash={}
        edges=[]
        for j in model.grd.select_edges_by_polyline(self.geom):
            c=model.grd.edges['cells'][j,0]
            if c in cell_hash:
                print("Skipping BC edge %d because its cell is already counted with a different edge"%j)
                continue
            cell_hash[c]=j
            edges.append(j)
        self._edges=np.array(edges)
        self._cells=model.grd.edges['cells'][self._edges,0]
        assert len(self._cells)==len(np.unique(self._cells))
    def plot(self,model):
        model.grd.plot_edges(color='k',lw=0.5)
        model.grd.plot_edges(color='m',lw=4,mask=self.edges(model))
        model.grd.plot_cells(color='m',alpha=0.5,mask=self.cells(model))

class FlowBC(BC):
    """
    Represents a flow in m3/s applied across a set of boundary
    edges.

    Positive Q is into the domain.  
    """
    Q=None
    ramp_time=0.0
    small_area=1e-6
    def cell_edge_flow(self,model):
        """
        return a tuple: ( array of cell indices,
                          array of edge indices,
                          array of inflows, m3/s )

        The sign of the flow is positive into the domain, even though
        the edge normals are pointing out of the domain.
        """
        c=self.cells(model)
        e=self.edges(model)
        # assumes that flow boundary eta is same as interior
        # cell
        if model.t<self.ramp_time:
            ramp=model.t/self.ramp_time
        else:
            ramp=1.0

        per_edge_A=model.aj[e]
        if per_edge_A.sum() < self.small_area:
            log.warning("FlowBC: area sum is 0.")
            
        per_edge_Q=self.Q*per_edge_A/max(self.small_area,per_edge_A.sum())
        return c,e,per_edge_Q*ramp
        
    def apply_bc(self,model,A,b):
        c,e,Q=self.cell_edge_flow(model)
        # used to be model.area[c]
        # TODO: now that wet area=pi is dynamic, is this still the correct
        # way to set the BC?  What happens when pi is updated?
        b[c] += model.dt / max(self.small_area,model.pi[c]) * Q

class StageBC(BC):
    h=None
    def apply_bc(self,model,A,b):
        c=self.cells(model)
        # changes to A are no in the loop above in SWAMPY
        #A[c, :] = 0.
        #A[c, c] = 1.
        b[c] = self.h
    def cell_edge_eta(self,model):
        c=self.cells(model)
        return c,None,self.h*np.ones_like(c)
