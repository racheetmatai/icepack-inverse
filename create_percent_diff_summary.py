from src.invert_c_theta import Invert
import pandas as pd
import firedrake
import numpy as np
import os
import copy
import pickle
import matplotlib.pyplot as plt

name_list=['data/geophysics/ADMAP_MagneticAnomaly_5km.tif', 
                                                'data/geophysics/ANTGG_BouguerAnomaly_10km.tif', 
                                                'data/geophysics/GeothermalHeatFlux_5km.tif',
                                                'data/geophysics/ALBMAP_SurfaceAirTemperature_5km.tif',
                                                'data/geophysics/EIGEN-6C4_GravityDisturbance_10km.tif',
                                                'data/geophysics/ALBMAP_SnowAccumulation_Arthern_5km.tif',]

def invert_dotson_fcn():
    drichlet_ids = [1,2,5,6,7,8,9,10,11]
    side_ids = []
    invert_dotson = Invert(outline = 'data/geojson/dotson-crosson.geojson', mesh_name = 'dotson', reg_constant_simultaneous = 1, read_mesh = False,opts = None, drichlet_ids = drichlet_ids , lcar = 9e3)
    invert_dotson.import_velocity_data(constant_val=0.01)
    invert_dotson.import_geophysics_data(name_list=['data/geophysics/ADMAP_MagneticAnomaly_5km.tif', 
                                                'data/geophysics/ANTGG_BouguerAnomaly_10km.tif', 
                                                'data/geophysics/GeothermalHeatFlux_5km.tif',
                                                'data/geophysics/ALBMAP_SurfaceAirTemperature_5km.tif',
                                                'data/geophysics/EIGEN-6C4_GravityDisturbance_10km.tif',
                                                'data/geophysics/ALBMAP_SnowAccumulation_Arthern_5km.tif',])
    u =  invert_dotson.simulation()
    invert_dotson.default_u = u
    invert_dotson.invert_C_theta_simultaneously(max_iterations=3, regularization_grad_fcn= True, loss_fcn_type = 'nosigma')
    u_optimized =  invert_dotson.simulation()
    invert_dotson.inverse_u = u_optimized
    theta = invert_dotson.θ
    C = invert_dotson.C
    return invert_dotson, theta, C

def invert_thwaites_fcn():
    drichlet_ids = [1,2,5,6]
    side_ids = []
    invert_thwaites = Invert(outline = 'data/geojson/thwaites.geojson', mesh_name = 'thwaites', reg_constant_simultaneous = 1, read_mesh = False,opts = None, drichlet_ids = drichlet_ids , lcar = 9e3)
    invert_thwaites.import_velocity_data(constant_val=0.01)
    invert_thwaites.import_geophysics_data(name_list=name_list)
    u =  invert_thwaites.simulation()
    invert_thwaites.default_u = u
    invert_thwaites.invert_C_theta_simultaneously(max_iterations=1, regularization_grad_fcn= True, loss_fcn_type = 'nosigma')
    u_optimized =  invert_thwaites.simulation()
    invert_thwaites.inverse_u = u_optimized
    theta = invert_thwaites.θ
    C = invert_thwaites.C
    return invert_thwaites, theta, C

def invert_pig_fcn():
    drichlet_ids = [2,3,4]
    side_ids = []
    invert_pig = Invert(outline = 'pine-island', mesh_name = 'pig', reg_constant_simultaneous = 1, read_mesh = False,opts = None, drichlet_ids = drichlet_ids , lcar = 9e3)
    invert_pig.import_velocity_data(constant_val=0.01)
    invert_pig.import_geophysics_data(name_list=name_list)
    u =  invert_pig.simulation()
    invert_pig.default_u = u
    invert_pig.invert_C_theta_simultaneously(max_iterations=3, regularization_grad_fcn= True, loss_fcn_type = 'regular')
    u_optimized =  invert_pig.simulation()
    invert_pig.inverse_u = u_optimized
    theta = invert_pig.θ
    C = invert_pig.C
    return invert_pig, theta, C

def process_csv(filename):
    df = pd.read_csv(filename, index_col=0)
    df = df.sample(frac=1).reset_index(drop=True)
    df['vel_mag'] = np.sqrt(df['x_velocity']**2 + df['y_velocity']**2)
    df['driving_stress'] = df['h'] * 9.8 * df['mag_s']
    return df

def compute_C_mean(select_dataset = 0):
    df_pig = process_csv('regularized_const_01C_simultaneous_pig_r1_geo_12.csv')
    df_thwaites = process_csv('regularized_const_01C_simultaneous_thwaites_r1_geo_12.csv')
    df_dotson = process_csv('regularized_const_01C_simultaneous_dotson_r1_geo_12.csv')

    if select_dataset == 0:
        df = pd.concat([df_dotson,df_dotson,df_thwaites], ignore_index=True)
        print("dataset selected: [df_dotson,df_dotson,df_thwaites]") 
    elif select_dataset == 1:
        df = pd.concat([df_pig,df_pig,df_thwaites], ignore_index=True)
        print("dataset selected: [df_pig,df_pig,df_thwaites]") 
    elif select_dataset == 2:
        df = pd.concat([df_dotson,df_pig], ignore_index=True)
        print("dataset selected: [df_dotson,df_pig]") 
    elif select_dataset == 3:
        df = pd.concat([df_dotson], ignore_index=True)
        print("dataset selected: [df_dotson]") 
    elif select_dataset == 4:
        df = pd.concat([df_pig], ignore_index=True)
        print("dataset selected: [df_pig]") 
    elif select_dataset == 5:
        df = pd.concat([df_thwaites], ignore_index=True)
        print("dataset selected: [df_thwaites]")
    
    C_mean = df['C'].mean()
    return C_mean

def compute_u_avg(invert_obj, C, theta, C_mean):
    invert_obj.C.dat.data[:] = np.ones(invert_obj.C.dat.data[:].shape) * C_mean
    u_avg=  invert_obj.simulation()
    invert_obj.default_u = u_avg

def eval_models(select_dataset, invert_obj):
    C_mean = compute_C_mean(select_dataset)
    compute_u_avg(invert_obj, invert_obj.C, invert_obj.theta, C_mean)
    if select_dataset == 0:
        base_folder = 'mlp_ensemble_0' 
    elif select_dataset == 1:
        base_folder = 'mlp_ensemble_1'
    elif select_dataset == 2:
        base_folder = 'mlp_ensemble_2' 
    elif select_dataset == 3:
        base_folder = 'mlp_ensemble_3'
    elif select_dataset == 4:
        base_folder = 'mlp_ensemble_4'
    elif select_dataset == 5:
        base_folder = 'mlp_ensemble_5'
    summary_list = []
    class_list = []
    u_optimized_list = []
    loss_function_list = []

    error_list = []
    columns_list = []
    df_summary_list = []

    # Collect all temp objects for later plotting
    temp_objects = []

    for folder in os.listdir(base_folder):
        if not os.path.isdir(os.path.join(base_folder, folder)):
            continue  # Skip if not a directory
        
        try:
            # Skip folders that don't contain numeric names (if needed)
            folder_num = int(folder)
        except ValueError:
            continue
        
        print('Processing folder:', folder)
        path = os.path.join(base_folder, folder)
        files = [f for f in os.listdir(path) if f.endswith('.pkl')]
        
        if not files:
            print(f"No .pkl files found in folder {folder}. Skipping.")
            continue
        
        temp_object = copy.copy(invert_obj)
        columns = None
        r2_list, r2_adjusted_list, mse_list = [], [], []
        
        for file in files:
            try:
                with open(os.path.join(path, file), "rb") as f:
                    model_bundle = pickle.load(f)
                    r2_list.append(model_bundle['r2_test'])
                    r2_adjusted_list.append(model_bundle['r2_adjusted_test'])
                    mse_list.append(model_bundle['mse_test'])
                    columns = model_bundle.get('input_columns', columns)
            except Exception as e:
                print(f"Error processing file {file} in folder {folder}: {e}")
                continue
        if r2_list and r2_adjusted_list and mse_list:
            r2_stats = pd.DataFrame(r2_list).describe()
            r2_adjusted_stats = pd.DataFrame(r2_adjusted_list).describe()
            mse_stats = pd.DataFrame(mse_list).describe()
            
            summary_list.append({
                'input_columns': columns,
                'r2_mean': r2_stats.loc['mean'].values[0],
                'r2_std': r2_stats.loc['std'].values[0],
                'r2_median': r2_stats.loc['50%'].values[0],
                'r2_adjusted_mean': r2_adjusted_stats.loc['mean'].values[0],
                'r2_adjusted_std': r2_adjusted_stats.loc['std'].values[0],
                'r2_adjusted_median': r2_adjusted_stats.loc['50%'].values[0],
                'mse_mean': mse_stats.loc['mean'].values[0],
                'mse_std': mse_stats.loc['std'].values[0],
                'mse_median': mse_stats.loc['50%'].values[0]
            })
        

        model_name = files[0][:-8]  # Assumes model files end with a fixed pattern (e.g., '_model.pkl')
        print(model_name)
        print(columns)
        columns_list.append(columns)
        
        # Ensure the model_name is properly constructed
        model_file_path = os.path.join(base_folder, folder, model_name)
        temp_object.compute_C_ML_regress(
            filename=model_file_path, 
            half=False, 
            flip=False, 
            use_driving_stress=False, 
            C_bounds=[-50, 55], 
            θ_bounds=[-102, 200],
            folder = '', 
            number_of_models=10
        )
        
        u_optimized = temp_object.simulation()
        # Store or process `u_optimized` as needed
        u_optimized_list.append(u_optimized)
        loss_val = firedrake.assemble(temp_object.loss_functional_nosigma(u_optimized))
        loss_function_list.append(loss_val)
        temp_object.ML_u = u_optimized
        # Collect the temp object for plotting later
        temp_objects.append(temp_object)
    return temp_objects, summary_list, columns_list, u_optimized_list, loss_function_list
    
def eval_pig_dotson_thwaites(select_dataset):
    # Get the evaluated objects and results for Dotson, PIG, and Thwaites
    inv_dotson, theta_dotson, C_dotson = invert_dotson_fcn()
    temp_objects_dotson, summary_list_dotson, columns_list_dotson, u_optimized_list_dotson, loss_function_list_dotson = eval_models(select_dataset, inv_dotson)
    
    inv_pig, theta_pig, C_pig = invert_pig_fcn()
    temp_objects_pig, summary_list_pig, columns_list_pig, u_optimized_list_pig, loss_function_list_pig = eval_models(select_dataset, inv_pig)
    
    inv_thwaites, theta_thwaites, C_thwaites = invert_thwaites_fcn()
    temp_objects_thwaites, summary_list_thwaites, columns_list_thwaites, u_optimized_list_thwaites, loss_function_list_thwaites = eval_models(select_dataset, inv_thwaites)
    
    # Since the lists have an equal number of objects, determine the number of rows
    n_rows = len(temp_objects_dotson)
    
    # Create subplots with 4 columns (1st column for summary, then 3 columns for plots) and n_rows rows
    fig, axes = plt.subplots(n_rows, 4, figsize=(20, 5 * n_rows))
    
    # Loop over the objects and plot summary in column 1 and images in columns 2-4
    for row in range(n_rows):
        # Summary column (column 0)
        summary_text = (
            f"{columns_list_dotson[row]}\n"
            f"R2 test: {summary_list_dotson[row]['r2_mean']:.4f}\n"
            f"R2_adjusted test: {summary_list_dotson[row]['r2_adjusted_mean']:.4f}\n"
            f"MSE test: {summary_list_dotson[row]['mse_mean']:.4f}\n"
            f"Loss function value: {loss_function_list_dotson[row]:.4f}"
        )
        axes[row, 0].text(0.1, 0.5, summary_text, fontsize=10, va="center", ha="left", wrap=True)
        axes[row, 0].axis('off')  # Turn off the axis for the text column
        
        # Dotson column (column 1)
        temp_objects_dotson[row].plot_percent_accounted(vmin=0, show_plot=False)
        axes[row, 1].set_title(f"Dotson {columns_list_dotson[row]}")
        
        # PIG column (column 2)
        temp_objects_pig[row].plot_percent_accounted(vmin=0, show_plot=False)
        axes[row, 2].set_title(f"PIG {columns_list_pig[row]}")
        
        # Thwaites column (column 3)
        temp_objects_thwaites[row].plot_percent_accounted(vmin=0, show_plot=False)
        axes[row, 3].set_title(f"Thwaites {columns_list_thwaites[row]}")
    
    # Adjust layout for better spacing
    plt.tight_layout()
    
    # Show the final figure
    plt.show()

