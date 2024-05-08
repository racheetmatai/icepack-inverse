import matplotlib.pyplot as plt
from src.invert_c_theta import Invert
import pandas as pd
import numpy as np
from firedrake import assemble
from concurrent import futures
import os
import firedrake
from firedrake import *

def compute_J1_J2(args):
    name, variable, reg_const, outline, mesh, modified_exists, invert_iter, gradient_tolerance, step_tolerance = args
    print('\n function started for '+variable+' '+str(reg_const))
    if variable == 'theta':
        invert_pig = Invert(outline = outline, mesh_name =mesh,  reg_constant_theta = reg_const, read_mesh = False, drichlet_ids = [2,3,4], lcar = 3.5e3)
        invert_pig.import_velocity_data(name, modified_exists = modified_exists,  C = 'driving-stress')
        invert_pig.invert_theta(max_iterations=invert_iter, gradient_tolerance = gradient_tolerance, step_tolerance=step_tolerance, regularization_grad_fcn= True)
        theta_optimized = invert_pig.get_theta()
        u_optimized = invert_pig.simulation_theta(theta_optimized)
        J1 = assemble(invert_pig.loss_functional(u_optimized))
        J2 = assemble(invert_pig.regularization_theta_grad(theta_optimized))*reg_const*reg_const
    elif variable == 'C':
        invert_pig = Invert(outline = outline, mesh_name =mesh,  reg_constant_c = reg_const, read_mesh = False, drichlet_ids = [2,3,4], lcar = 3.5e3)
        invert_pig.import_velocity_data(name, modified_exists = modified_exists)
        invert_pig.invert_C(max_iterations=invert_iter, gradient_tolerance = gradient_tolerance, step_tolerance=step_tolerance, regularization_grad_fcn= True)
        C_optimized = invert_pig.get_C()
        u_optimized = invert_pig.simulation_C(C_optimized)
        J1 = assemble(invert_pig.loss_functional(u_optimized))
        J2 = assemble(invert_pig.regularization_C_grad(C_optimized))*reg_const*reg_const
    elif variable == 'simultaneous':
        invert_pig = Invert(outline = 'pine-island', mesh_name = mesh, reg_constant_simultaneous = reg_const, read_mesh = False, drichlet_ids = [2,3,4], lcar = 3.5e3)       
        invert_pig.import_velocity_data(name, modified_exists = modified_exists)
        invert_pig.invert_C_theta_simultaneously(max_iterations=invert_iter, gradient_tolerance = gradient_tolerance, step_tolerance=step_tolerance, regularization_grad_fcn= True)
        C_optimized = invert_pig.get_C()
        theta_optimized = invert_pig.get_theta()
        u_optimized = invert_pig.simulation()
        J1 = assemble(invert_pig.loss_functional(u_optimized))
        # removed reg_constant so dont need to multiply by it
        L = firedrake.Constant(7.5e3)
        J2 = assemble(0.5 / invert_pig.area * (L)**2 * (  firedrake.inner(firedrake.grad(theta_optimized),firedrake.grad(theta_optimized))+  firedrake.inner(firedrake.grad(C_optimized),firedrake.grad(C_optimized)) ) * firedrake.dx(invert_pig.mesh))

    print('function finished for '+variable+' '+str(reg_const),' J1: '+str(J1)+'   J2: '+str(J2))
    invert_pig = None
    u_optimized = None
    C_optimized = None
    theta_optimized = None
    return [J1, J2]

def create_L_curve(name, variable, number_reg_constants=3, outline='pine-island', mesh='pig_coarsest_0_pw', modified_exists=True, invert_iter=10, gradient_tolerance=1e-100, step_tolerance=1e-100, workers = 5):
    reg_const_list =  [0.0001 ,0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.07, 0.1, 1, 10, 100, 1000] 
    J_list = []

    with futures.ProcessPoolExecutor(max_workers=workers) as pool:
        args_list = [(name, variable, reg_const, outline, mesh, modified_exists, invert_iter, gradient_tolerance, step_tolerance) for reg_const in reg_const_list]
        for J in pool.map(compute_J1_J2, args_list):
            print("IN POOL:", J)
            J_list.append(J)

    J_npy = np.array(J_list).T
    df = pd.DataFrame()
    df['Regularization Constant'] = reg_const_list
    df['J1'] = J_npy[0, :]
    df['J2'] = J_npy[1, :]
    return df

if __name__ == '__main__':
    folder_path = "completed/8/"

    # Get file names starting with 'ASE'
    file_names = [file for file in os.listdir(folder_path) if file.startswith('ASE')]
    
    name = file_names[-1][:40]
    variable = 'theta'
    l_curve_4 = create_L_curve(name, variable, number_reg_constants = 4,invert_iter = 100)
    plt.scatter(l_curve['J1'], l_curve['J2'])
    plt.savefig('C_22.png')