import matplotlib.pyplot as plt
from invert_c_theta import Invert
import pandas as pd
import numpy as np
from firedrake import assemble
from concurrent import futures
import os


#def compute_J1_J2(name, variable, reg_const, outline='pine-island', mesh='pig', modified_exists = True, invert_iter = 10, gradient_tolerance=2e-2, step_tolerance=1e-1):
def compute_J1_J2(args):
    name, variable, reg_const, outline, mesh, modified_exists, invert_iter, gradient_tolerance, step_tolerance = args
    print('\n function started for '+variable+' '+str(reg_const))
    if variable == 'theta':
        invert_pig = Invert(outline = outline, mesh_name =mesh,  reg_constant_theta = reg_const, read_mesh = True, drichlet_ids = [0])
        #invert_pig = Invert(outline = outline, mesh_name = mesh, reg_constant_theta = reg_const)
        invert_pig.import_velocity_data(name, modified_exists = modified_exists)
        invert_pig.invert_theta(max_iterations=invert_iter, gradient_tolerance = gradient_tolerance, step_tolerance=step_tolerance)
        theta_optimized = invert_pig.get_theta()
        u_optimized = invert_pig.simulation_theta(theta_optimized)
        J1 = assemble(invert_pig.loss_functional(u_optimized))
        J2 = assemble(invert_pig.regularization_theta(theta_optimized))*reg_const
        #J2 = invert_pig.get_norm(theta_optimized)
    elif variable == 'C':
        invert_pig = Invert(outline = outline, mesh_name =mesh,  reg_constant_c = reg_const, read_mesh = True, drichlet_ids = [0])
        #invert_pig = Invert(outline = outline, mesh_name = mesh, reg_constant_c = reg_const)
        invert_pig.import_velocity_data(name, modified_exists = modified_exists)
        invert_pig.invert_C(max_iterations=invert_iter, gradient_tolerance = gradient_tolerance, step_tolerance=step_tolerance)
        C_optimized = invert_pig.get_C()
        u_optimized = invert_pig.simulation_C(C_optimized)
        J1 = assemble(invert_pig.loss_functional(u_optimized))
        J2 = assemble(invert_pig.regularization_C(C_optimized))*reg_const
        #J2 = invert_pig.get_norm(C_optimized)
    
    # Save to Checkpoint File
    # mesh = invert_pig.get_mesh()
    # theta = invert_pig.get_theta()
    # C = invert_pig.get_C()
    # folder_path = "completed/16/"
    # save_name = 'variables_'+ variable + str(reg_const) + '.h5'
    # checkpoint_file_path = os.path.join(folder_path, save_name)
    # with firedrake.CheckpointFile(checkpoint_file_path, 'w') as afile:
    #     if afile is not None:
    #         afile.save_mesh(mesh)  # optional
    #         afile.save_function(u_optimized, name='u_optimized')
    #         afile.save_function(theta, name='theta')
    #         afile.save_function(C, name='C')
    #     else:
    #         print("Failed to create checkpoint file.")
    print('function finished for '+variable+' '+str(reg_const),' J1: '+str(J1)+'   J2: '+str(J2))
    invert_pig = None
    u_optimized = None
    C_optimized = None
    theta_optimized = None
    return [J1, J2]

def create_L_curve(name, variable, number_reg_constants=3, outline='pine-island', mesh='pig_coarsest_0_pw', modified_exists=True, invert_iter=10, gradient_tolerance=1e-100, step_tolerance=1e-100, workers = 5):
    reg_const_list = [0.01, 0.1, 1, 10] #list(np.linspace(0.2, 5, number_reg_constants))
    J_list = []

    #with futures.ProcessPoolExecutor() as pool:
    with futures.ProcessPoolExecutor(max_workers=workers) as pool:
    #with futures.ThreadPoolExecutor(max_workers=8) as pool:
        args_list = [(name, variable, reg_const, outline, mesh, modified_exists, invert_iter, gradient_tolerance, step_tolerance) for reg_const in reg_const_list]
        for J in pool.map(compute_J1_J2, args_list):
            print("IN POOL:", J)
            J_list.append(J)

    J_npy = np.array(J_list).T
    df = pd.DataFrame()
    df['regularization_const'] = reg_const_list
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