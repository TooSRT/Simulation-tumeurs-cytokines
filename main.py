"""
Simulation Main Module

This module provides the main function to run tumor simulation.

Author: LANGUE William, DUFRESNE Théo
Date: 17/03/2024
"""

import numpy as np
import pandas as pd
from Simulation.simulation import Simulation
from Simulation.simulation_cytokines import Simulation2
from O2_EDP.o2_edp import O2_EDP
from O2_EDP.cytokine_edp import cytokine_EDP
from O2_EDP.grid_O2 import O2_Grid
from O2_EDP.grid_cytokine import cytokine_Grid

def main():
    """
    Main function to run tumor simulation.

    It initializes a Simulation object with specified parameters and runs the simulation.
    """

    ####################
    # MODEL CONDITIONS #
    ####################

    nb_tumor = 10
    Nb_cells_cyt = 1
    unit = "cm"
    distrib = "gaussian"
    proliferation = True
    tol = 1e-8
    iter_max = 10
    iter_print = 10

    #############################
    # PARAMETERS INITIALIZATION #
    #############################

    assert distrib in ['uniform','gaussian']
    assert unit in ['cm','mm']
    if unit == "cm" :
        data = pd.DataFrame(pd.read_csv('parameters/parameters_cm.csv'))
    if unit == "mm" :
        data = pd.DataFrame(pd.read_csv('parameters/parameters_mm.csv'))

    Nx, delta_x, delta_t, Dn, n_max, rn, Dc, c, kappa, D_cytokine, Tau_p, Tau_c, P_prod, P_cons, w_max, D_tcells, alpha_c = data['Value']
    
    Nx = int(Nx)
    if(not proliferation) :
        rn = 0
    
    ##############    
    # SIMULATION #
    ##############

    #simulation = Simulation(nb_tumor, unit, distrib, tol, Nx, delta_x, delta_t, Dn, n_max, rn, Dc, c, kappa)
    simulation = Simulation2(nb_tumor, unit, distrib, tol, Nb_cells_cyt, Nx, delta_x, delta_t, Dn, D_cytokine, w_max, rn, Tau_p, Tau_c, P_prod, P_cons, D_tcells, alpha_c)
    simulation.load_simulation(iter_max, iter_print)
    
'''
    ###############
    # CRASH TESTS #
    ###############

    o2_edp = O2_EDP(100,np.zeros(100**2),np.array([0,5001]),1e-8, unit = "cm")
    o2_grid = O2_Grid(100**2, unit = 'cm')
    for i in range (100):
        o2_edp.O2_diffusion()
        #o2_grid.print(o2_edp.c,8.9,100,i*15)
'''

if __name__ == "__main__":
    main()



