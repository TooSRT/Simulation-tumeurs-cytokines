# Modelling tumour evolution using an agent-based model

## Description

This code is a tool for simulating tumour evolution. It must help biologists to verify and check experimental hypothesis about tumour evolution.

Version 1.2

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
O2_EDP file comes from discretisation of partial differential equation for O2 concentration diffusion.
Grid files are classes used to display results.<br>
After launch simulation in the main file, a plot repertory is created. In this repertory, we found a sub-repertory "pictures" in which there are all plots. Moreover, a graph of number of cells in relation of time and a video of tumour evolution  (based on pictures) are made. <br>


## Example

```
    ####################
    # MODEL CONDITIONS #
    ####################

    nb_tumor = 100
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

Under development. For now biologic conditions are not impemented, everything is aleatory.
Next upgrades for version 2.0 :
* Link between cells density and O2 concentration