# Modelling tumour evolution using an agent-based model

## Description

This code is a tool for simulating tumour evolution and the action of differents immune cells on them (T-CD4 and T-CD8). It must help biologists to verify and check experimental hypothesis about tumour evolution.

## Modules required

- numpy
- shutil
- time
- os
- re
- matplotlib

## Usage

The program consists of three parts and the main file.<br>
Simulation is the orchestra conductor file. It prepares the simulation nad contains the time loop of tumour evolution. It uses Grid and Density_EDP files.<br>
Density_EDP file comes from discretisation of partial differential equation for density. We found two importants algorithms : movement and proliferation of tumours.<br>
EDP file comes from discretisation of partial differential equation for O2 concentration diffusion, cytokines diffusion and the movement associated to T-cells.
Grid files are classes used to display results.<br>
After launch simulation in the main file, a plot repertory is created. In this repertory, we found a sub-repertory "pictures" in which there are all plots. Moreover, a graph of number of cells in relation of time and a video of tumour evolution  (based on pictures) are made. <br>


## Example

```
    ####################
    # MODEL CONDITIONS #
    ####################

    nb_tumor = 100
    Nb_cells_cyt = 50
    unit = "cm"
    distrib = "gaussian"
    proliferation = True
    tol = 1e-8
    iter_max = 100
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

    Nx, delta_x, delta_t, Dn, n_max, rn, Dc, c, pc = data['Value']
    Nx = int(Nx)
    if(not proliferation) :
        rn = 0
    
    ##############    
    # SIMULATION #
    ##############

    simulation = Simulation(nb_tumor, unit, distrib, tol, Nx, delta_x, delta_t, Dn, n_max, rn, Dc, c, pc)
    simulation.load_simulation(iter_max, iter_print)
```

## Authors

- William Langue
- Théo Servotte
- Théo Dufresne

### Research supervisor

M. Alexandre Poulain

## Project status

Under development.
