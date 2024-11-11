#import matplotlib.pyplot as plt
from src.invert_c_theta import Invert
import pandas as pd
import numpy as np
from firedrake import assemble
from concurrent import futures
import os
import firedrake
from firedrake import *

def compute_J1_J2(args):
    name, variable, reg_const, outline, mesh, modified_exists, invert_iter, gradient_tolerance, step_tolerance, lcar, nosigma_lossfcn, drichlet_ids, regularization_grad_fcn, constant_val = args
    print('\n function started for '+variable+' '+str(reg_const))
    
    if variable == 'theta':
        invert_pig = Invert(outline = outline, mesh_name =mesh,  reg_constant_theta = reg_const, read_mesh = False, drichlet_ids = drichlet_ids, lcar = lcar)
        invert_pig.import_velocity_data(name, constant_val=constant_val, modified_exists = modified_exists,  C = 'driving-stress')
        invert_pig.invert_theta(max_iterations=invert_iter, gradient_tolerance = gradient_tolerance, step_tolerance=step_tolerance, regularization_grad_fcn= regularization_grad_fcn)
        theta_optimized = invert_pig.get_theta()
        u_optimized = invert_pig.simulation_theta(theta_optimized)
        J1 = assemble(invert_pig.loss_functional(u_optimized))
        J2 = assemble(invert_pig.regularization_theta_grad(theta_optimized))*reg_const*reg_const

    elif variable == 'C':
        invert_pig = Invert(outline = outline, mesh_name =mesh,  reg_constant_c = reg_const, read_mesh = False, opts = None , drichlet_ids = drichlet_ids, lcar = lcar)
        invert_pig.import_velocity_data(name, constant_val=constant_val, modified_exists = modified_exists)
        invert_pig.import_geophysics_data(name_list=['data/geophysics/ADMAP_MagneticAnomaly_5km.tif', 
                                                'data/geophysics/ANTGG_BouguerAnomaly_10km.tif', 
                                                'data/geophysics/GeothermalHeatFlux_5km.tif',
                                                'data/geophysics/ALBMAP_SurfaceAirTemperature_5km.tif',
                                                'data/geophysics/EIGEN-6C4_GravityDisturbance_10km.tif',
                                                'data/geophysics/ALBMAP_SnowAccumulation_Arthern_5km.tif',
                                                'data/geophysics/Englacial_temp_Pattyn_2013.tif'])
        invert_pig.invert_C(max_iterations=invert_iter, gradient_tolerance = gradient_tolerance, step_tolerance=step_tolerance, regularization_grad_fcn= regularization_grad_fcn, loss_fcn_type = nosigma_lossfcn)
        C_optimized = invert_pig.get_C()
        u_optimized = invert_pig.simulation()
        J1 = assemble(invert_pig.loss_functional_nosigma(u_optimized))
        J2 = assemble(invert_pig.regularization_C_grad(C_optimized))*reg_const*reg_const

    elif variable == 'simultaneous':
        invert_pig = Invert(outline = outline, mesh_name = mesh, reg_constant_simultaneous = reg_const, read_mesh = False, drichlet_ids = drichlet_ids, lcar = lcar)       
        invert_pig.import_velocity_data(name, constant_val=constant_val, modified_exists = modified_exists)
        invert_pig.invert_C_theta_simultaneously(max_iterations=invert_iter, gradient_tolerance = gradient_tolerance, step_tolerance=step_tolerance, regularization_grad_fcn= regularization_grad_fcn, loss_fcn_type = nosigma_lossfcn)
        C_optimized = invert_pig.get_C()
        theta_optimized = invert_pig.get_theta()
        u_optimized = invert_pig.simulation()
        print("Using nosigma loss function")
        J1 = assemble(invert_pig.loss_functional_nosigma(u_optimized))       
        L = firedrake.Constant(7.5e3)
        J2 = assemble(0.5 / invert_pig.area * (L)**2 * (  firedrake.inner(firedrake.grad(theta_optimized),firedrake.grad(theta_optimized))+  firedrake.inner(firedrake.grad(C_optimized),firedrake.grad(C_optimized)) ) * firedrake.dx(invert_pig.mesh))

    print('function finished for '+variable+' '+str(reg_const),' J1: '+str(J1)+'   J2: '+str(J2))
    invert_pig = None
    u_optimized = None
    C_optimized = None
    theta_optimized = None
    return [J1, J2] 

def create_L_curve(name, variable,reg_const_list = [0.01, 0.1, 1, 10, 100],  outline='pine-island', mesh='pig', modified_exists=True, invert_iter=150, gradient_tolerance=1e-100, step_tolerance=1e-100, workers = 5, lcar=9e3, nosigma_lossfcn ='nosigma', drichlet_ids = [2,3,4], regularization_grad_fcn= True, constant_val = 0.01):
    
    J_list = []

    # for parallel execution uncomment the following lines
    with futures.ProcessPoolExecutor(max_workers=workers) as pool:
        args_list = [(name, variable, reg_const, outline, mesh, modified_exists, invert_iter, gradient_tolerance, step_tolerance, lcar, nosigma_lossfcn, drichlet_ids, regularization_grad_fcn, constant_val) for reg_const in reg_const_list]
        for J in pool.map(compute_J1_J2, args_list):
            print("IN POOL:", J)
            J_list.append(J)

    ## for serial execution uncomment the following lines
    # for reg_const in reg_const_list:
    #     print('reg_const: ',reg_const)
    #     J = compute_J1_J2((name, variable, reg_const, outline, mesh, modified_exists, invert_iter, gradient_tolerance, step_tolerance, lcar, nosigma_lossfcn, drichlet_ids, regularization_grad_fcn, constant_val))
    #     print('reg_const: ',reg_const, '  J: ',J)
    #     J_list.append(J)

    J_npy = np.array(J_list).T
    df = pd.DataFrame()
    df['Regularization Constant'] = reg_const_list
    df['J1'] = J_npy[0, :]
    df['J2'] = J_npy[1, :]
    return df