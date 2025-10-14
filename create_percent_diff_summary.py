from src.invert_c_theta import Invert
import pandas as pd
import firedrake
import numpy as np
import os
import copy
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colorbar as colorbar
import seaborn as sns

name_list=['data/geophysics/ADMAP_MagneticAnomaly_5km.tif', 
                                                'data/geophysics/ANTGG_BouguerAnomaly_10km_MF_combined_2.tif', # interpolated using multifidelity modeling
                                                'data/geophysics/GeothermalHeatFlux_5km.tif',
                                                'data/geophysics/ALBMAP_SurfaceAirTemperature_5km.tif',
                                                'data/geophysics/EIGEN-6C4_GravityDisturbance_10km.tif',
                                                'data/geophysics/ALBMAP_SnowAccumulation_Arthern_5km.tif',
                                                'data/geophysics/Englacial_temp_Pattyn_2013.tif',]

# Dataset labels
DATASET_NAMES = [
    "dotson, thwaites",
    "pig, thwaites",
    "dotson, pig",
    "dotson",
    "pig",
    "thwaites",
]

# Feature symbol mapping
FEATURE_SYMBOLS = {
    "s": r"$s$",
    "b": r"$b$",
    "h": r"$h$",
    "mag_s": r"$\|\nabla s\|$",
    "mag_b": r"$\|\nabla b\|$",
    "mag_h": r"$\|\nabla h\|$",
    "driving_stress": r"$\tau_d$",
    "gravity_disturbance": r"$g_d$",
    "heatflux": r"$Q_b$",
    "surface_air_temp": r"$T_s$",
    "snow_accumulation": r"$A_s$",
    "mag_anomaly": r"$g_b$",
}


# Training dataset mapping
CATCHMENT_NAMES = ["Dotson", "Pig", "Thwaites"]

TRAINING_MAPPING = {
    0: ["Dotson", "Dotson", "Thwaites"],
    1: ["Pig", "Pig", "Thwaites"],
    2: ["Dotson", "Pig"],
    3: ["Dotson"],
    4: ["Pig"],
    5: ["Thwaites"]
}

# feature_labels = [
#     's, b, h, mag_h, mag_s, mag_b, driving_stress, heatflux, snow_accumulation, surface_air_temp, gravity_disturbance',
#     's, h, mag_h, mag_s, driving_stress, snow_accumulation, surface_air_temp',
#     's, h, mag_h, mag_s, driving_stress',
#     's, h, mag_h, mag_s, snow_accumulation',
#     's, h, mag_h, mag_s, surface_air_temp',
#     's, h, mag_h, mag_s',
#     'mag_h, mag_s',
#     's, h',
#     'driving_stress, snow_accumulation, surface_air_temp',
#     'b, mag_b, gravity_disturbance',
#     'b, mag_b, heatflux, gravity_disturbance',
#     'b, mag_b, heatflux',
#     'heatflux, gravity_disturbance',
#     's, h, mag_h, mag_s, surface_air_temp, heatflux, gravity_disturbance'
# ]

# feature_labels = [
#     's, b, h, mag_h, mag_s, mag_b, driving_stress, heatflux, surface_air_temp, gravity_disturbance',
#     's, h, mag_h, mag_s, driving_stress, surface_air_temp',
#     's, h, mag_h, mag_s, driving_stress',
#     's, h, mag_h, mag_s, surface_air_temp',
#     's, h, mag_h, mag_s',
#     'mag_h, mag_s',
#     's, h',
#     'b, mag_b, gravity_disturbance',
#     'b, mag_b, heatflux, gravity_disturbance',
#     'b, mag_b, heatflux',
#     'heatflux, gravity_disturbance',
#     'b, mag_b, heatflux, s, h, mag_h, mag_s, surface_air_temp',
# ]

feature_labels = [
    's, b, h, mag_h, mag_s, mag_b, driving_stress, heatflux, surface_air_temp, gravity_disturbance, mag_anomaly',
    's, b, h, mag_h, mag_s, mag_b, driving_stress, heatflux, surface_air_temp, gravity_disturbance',
    's, b, h, mag_h, mag_s, mag_b, driving_stress, heatflux, surface_air_temp, mag_anomaly',
    's, b, h, mag_h, mag_s, mag_b, heatflux, surface_air_temp',
    's, h, mag_h, mag_s, driving_stress, surface_air_temp',
    's, h, mag_h, mag_s, driving_stress',
    's, h, mag_h, mag_s, surface_air_temp',
    's, h, mag_h, mag_s',
    'mag_h, mag_s',
    's, h',
    'b, mag_b, gravity_disturbance',
    'b, mag_b, heatflux, gravity_disturbance',
    'b, mag_b, heatflux',
    'heatflux, gravity_disturbance',
    'b, mag_b, mag_anomaly',
    'b, mag_b, heatflux, mag_anomaly',
    'heatflux, mag_anomaly',
]


# mapping_features = {
#     1: ['s', 'b', 'h', 'mag_h', 'mag_s', 'mag_b', 'driving_stress', 'heatflux', 'snow_accumulation', 'surface_air_temp', 'gravity_disturbance'],
#     2: ['s', 'h', 'mag_h', 'mag_s', 'driving_stress', 'snow_accumulation', 'surface_air_temp'],
#     3: ['s', 'h', 'mag_h', 'mag_s', 'driving_stress'],
#     4: ['s', 'h', 'mag_h', 'mag_s', 'snow_accumulation'],
#     5: ['s', 'h', 'mag_h', 'mag_s', 'surface_air_temp'],
#     6: ['s', 'h', 'mag_h', 'mag_s'],
#     7: ['mag_h', 'mag_s'],
#     8: ['s', 'h'],
#     9: ['driving_stress', 'snow_accumulation', 'surface_air_temp'],
#     10: ['b', 'mag_b', 'gravity_disturbance'],
#     11: ['b', 'mag_b', 'heatflux', 'gravity_disturbance'],
#     12: ['b', 'mag_b', 'heatflux'],
#     13: ['heatflux', 'gravity_disturbance'],
#     14: ['s',  'h', 'mag_h', 'mag_s', 'surface_air_temp', 'heatflux', 'gravity_disturbance'],
# }

# mapping_features = {
#     1: ['s', 'b', 'h', 'mag_h', 'mag_s', 'mag_b', 'driving_stress', 'heatflux', 'surface_air_temp', 'gravity_disturbance'],
#     2: ['s', 'h', 'mag_h', 'mag_s', 'driving_stress', 'surface_air_temp'],
#     3: ['s', 'h', 'mag_h', 'mag_s', 'driving_stress'],
#     4: ['s', 'h', 'mag_h', 'mag_s', 'surface_air_temp'],
#     5: ['s', 'h', 'mag_h', 'mag_s'],
#     6: ['mag_h', 'mag_s'],
#     7: ['s', 'h'],
#     8: ['b', 'mag_b', 'gravity_disturbance'],
#     9: ['b', 'mag_b', 'heatflux', 'gravity_disturbance'],
#     10: ['b', 'mag_b', 'heatflux'],
#     11: ['heatflux', 'gravity_disturbance'],
#     12: ['b', 'mag_b', 'heatflux', 's', 'h', 'mag_h', 'mag_s', 'surface_air_temp']
# }

mapping_features = {
    1: ['s', 'b', 'h', 'mag_h', 'mag_s', 'mag_b', 'driving_stress', 'heatflux', 'surface_air_temp', 'gravity_disturbance', 'mag_anomaly'],
    2: ['s', 'b', 'h', 'mag_h', 'mag_s', 'mag_b', 'driving_stress', 'heatflux', 'surface_air_temp', 'gravity_disturbance'],
    3: ['s', 'b', 'h', 'mag_h', 'mag_s', 'mag_b', 'driving_stress', 'heatflux', 'surface_air_temp', 'mag_anomaly'],
    4: ['s', 'b', 'h', 'mag_h', 'mag_s', 'mag_b', 'heatflux', 'surface_air_temp'],
    5: ['s', 'h', 'mag_h', 'mag_s', 'driving_stress', 'surface_air_temp'],
    6: ['s', 'h', 'mag_h', 'mag_s', 'driving_stress'],
    7: ['s', 'h', 'mag_h', 'mag_s', 'surface_air_temp'],
    8: ['s', 'h', 'mag_h', 'mag_s'],
    9: ['mag_h', 'mag_s'],
    10: ['s', 'h'],
    11: ['b', 'mag_b', 'gravity_disturbance'],
    12: ['b', 'mag_b', 'heatflux', 'gravity_disturbance'],
    13: ['b', 'mag_b', 'heatflux'],
    14: ['heatflux', 'gravity_disturbance'],
    15: ['b', 'mag_b', 'mag_anomaly'],
    16: ['b', 'mag_b', 'heatflux', 'mag_anomaly'],
    17: ['heatflux', 'mag_anomaly'],
}

def invert_amundsen_fcn():
    drichlet_ids = [1,2,4,6,7,8,9,10,11]
    side_ids = []
    invert_amundsen = Invert(outline = 'data/geojson/amundsen_v1.geojson', mesh_name = 'amundsen', reg_constant_c  = 0.05, reg_constant_simultaneous = 1, read_mesh = False,opts = None, drichlet_ids = drichlet_ids , lcar = 9e3)
    invert_amundsen.import_velocity_data(constant_val=0.01)
    invert_amundsen.import_geophysics_data(name_list=name_list)
    # u =  invert_amundsen.simulation()
    # invert_amundsen.default_u = u
    # invert_amundsen.invert_C_theta_simultaneously(max_iterations=200, regularization_grad_fcn= True, loss_fcn_type = 'nosigma')
    invert_amundsen.invert_C(max_iterations=300, regularization_grad_fcn= True, loss_fcn_type = 'nosigma')
    u_optimized =  invert_amundsen.simulation()
    invert_amundsen.inverse_u = u_optimized
    # theta = invert_amundsen.θ.copy(deepcopy=True)
    # C = invert_amundsen.C.copy(deepcopy=True)
    return invert_amundsen

def invert_dotson_fcn():
    drichlet_ids = [1,2,4,6,7,8,9,10,11]
    side_ids = []
    invert_dotson = Invert(outline = 'data/geojson/dotson-crosson.geojson', mesh_name = 'dotson',reg_constant_c  = 0.05, reg_constant_simultaneous = 1, read_mesh = False,opts = None, drichlet_ids = drichlet_ids , lcar = 9e3)
    invert_dotson.import_velocity_data(constant_val=0.01)
    invert_dotson.import_geophysics_data(name_list=name_list)
    # u =  invert_dotson.simulation()
    # invert_dotson.default_u = u
    # invert_dotson.invert_C_theta_simultaneously(max_iterations=200, regularization_grad_fcn= True, loss_fcn_type = 'nosigma')
    invert_dotson.invert_C(max_iterations=300, regularization_grad_fcn= True, loss_fcn_type = 'nosigma')
    u_optimized =  invert_dotson.simulation()
    invert_dotson.inverse_u = u_optimized
    # theta = invert_dotson.θ.copy(deepcopy=True)
    # C = invert_dotson.C.copy(deepcopy=True)
    return invert_dotson

def invert_thwaites_fcn():
    drichlet_ids = [1,2,5,6]
    side_ids = []
    invert_thwaites = Invert(outline = 'data/geojson/thwaites.geojson', mesh_name = 'thwaites', reg_constant_c  = 0.05, reg_constant_simultaneous = 1, read_mesh = False,opts = None, drichlet_ids = drichlet_ids , lcar = 9e3)
    invert_thwaites.import_velocity_data(constant_val=0.01)
    invert_thwaites.import_geophysics_data(name_list=name_list)
    # u =  invert_thwaites.simulation()
    # invert_thwaites.default_u = u
    # invert_thwaites.invert_C_theta_simultaneously(max_iterations=170, regularization_grad_fcn= True, loss_fcn_type = 'nosigma')
    invert_thwaites.invert_C(max_iterations=170, regularization_grad_fcn= True, loss_fcn_type = 'nosigma')
    u_optimized =  invert_thwaites.simulation()
    invert_thwaites.inverse_u = u_optimized
    # theta = invert_thwaites.θ.copy(deepcopy=True)
    # C = invert_thwaites.C.copy(deepcopy=True)
    return invert_thwaites

def invert_pig_fcn():
    drichlet_ids = [2,3,4]
    side_ids = []
    invert_pig = Invert(outline = 'pine-island', mesh_name = 'pig', reg_constant_c  = 0.05, reg_constant_simultaneous = 1, read_mesh = False,opts = None, drichlet_ids = drichlet_ids , lcar = 9e3)
    invert_pig.import_velocity_data(constant_val=0.01)
    invert_pig.import_geophysics_data(name_list=name_list)
    # u =  invert_pig.simulation()
    # invert_pig.default_u = u
    # invert_pig.invert_C_theta_simultaneously(max_iterations=200, regularization_grad_fcn= True, loss_fcn_type = 'regular')
    invert_pig.invert_C(max_iterations=300, regularization_grad_fcn= True, loss_fcn_type = 'nosigma')
    u_optimized =  invert_pig.simulation()
    invert_pig.inverse_u = u_optimized
    # theta = invert_pig.θ.copy(deepcopy=True)
    # C = invert_pig.C.copy(deepcopy=True)
    return invert_pig

def get_phi(h, s, ramp_power=1, g=9769603575225600.0, ρ_W=1.0282341471330407e-18, ρ_I=9.207917118369125e-19):
    p_W = ρ_W * g * np.maximum(0, h - s)
    p_I = ρ_I * g * h
    if p_I == 0:
        return 0
    return max((1 - (p_W / p_I)**ramp_power), 0)

def process_csv(filename):
    df = pd.read_csv(filename, index_col=0)
    df = df.sample(frac=1).reset_index(drop=True)
    df['vel_mag'] = np.sqrt(df['x_velocity']**2 + df['y_velocity']**2)
    df['driving_stress'] = df['h'] * 9.8 * df['mag_s']
    df['phi'] = df.apply(lambda row: get_phi(row['h'], row['s']), axis=1)
    return df

def compute_C_mean(select_dataset = 0, phi_threshold=0.1):
    # df_pig = process_csv('regularized_const_01C_simultaneous_pig_r1_geo_12.csv')
    # df_thwaites = process_csv('regularized_const_01C_simultaneous_thwaites_r1_geo_12.csv')
    # df_dotson = process_csv('regularized_const_01C_simultaneous_dotson_r1_geo_12.csv')

    # df_pig = process_csv('regularized_const_01C_C_only_englacial_temp_pig_r01_geo_12.csv')
    # df_thwaites = process_csv('regularized_const_01C_C_only_englacial_temp_thwaites_r01_geo_12.csv')
    # df_dotson = process_csv('regularized_const_01C_C_only_englacial_temp_dotson_r01_geo_12.csv')

    df_pig = process_csv('regularized_const_01C_C_only_englacial_temp_pig_r005_geo_12.csv')
    df_thwaites = process_csv('regularized_const_01C_C_only_englacial_temp_thwaites_r005_geo_12.csv')
    df_dotson = process_csv('regularized_const_01C_C_only_englacial_temp_dotson_r005_geo_12.csv')

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
    
    df['phi'] = df.apply(lambda row: get_phi(row['h'], row['s']), axis=1)
    #C_mean = df['C'].mean()
    C_mean = df.loc[df['phi'] > phi_threshold, 'C'].mean()
    return C_mean

def compute_u_avg(invert_obj, C_mean, phi_threshold=0.1):
    temp_C = np.ones(invert_obj.C.dat.data[:].shape) * C_mean
    invert_obj.compute_features() #u=u)
    phi = firedrake.interpolate(invert_obj.get_phi(invert_obj.h, invert_obj.s), invert_obj.Q)

    # Create a mask where phi exceeds the threshold
    mask = phi.dat.data[:] > phi_threshold

    if np.any(mask):
        # Apply the regression results only to the masked locations
        invert_obj.C.dat.data[mask] = temp_C[mask]
    else:
        print(f"No values in phi exceed the threshold of {phi_threshold}. Skipping regress computation.")

    u_avg=  invert_obj.simulation()
    invert_obj.default_u = u_avg

def regression(temp_object, model_file_path, C_bounds = [-50, 55]):
    temp_object.compute_C_ML_regress(
            filename=model_file_path, 
            half=False, 
            flip=False, 
            use_driving_stress=False, 
            C_bounds=C_bounds, 
            folder = '', 
            number_of_models=10
        )
    return temp_object


def get_models_summary(select_dataset):
    if select_dataset == 0:
        base_folder = 'mlp_ensemble_englacial_0' 
    elif select_dataset == 1:
        base_folder = 'mlp_ensemble_englacial_1'
    elif select_dataset == 2:
        base_folder = 'mlp_ensemble_englacial_2' 
    elif select_dataset == 3:
        base_folder = 'mlp_ensemble_englacial_3'
    elif select_dataset == 4:
        base_folder = 'mlp_ensemble_englacial_4'
    elif select_dataset == 5:
        base_folder = 'mlp_ensemble_englacial_5'
    summary_list = []


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
        
        
        columns = None
        r2_list, r2_adjusted_list, mse_list, history_list = [], [], [], []
        
        for file in files:
            try:
                with open(os.path.join(path, file), "rb") as f:
                    model_bundle = pickle.load(f)
                    r2_list.append(model_bundle['r2_test'])
                    r2_adjusted_list.append(model_bundle['r2_adjusted_test'])
                    mse_list.append(model_bundle['mse_test'])
                    columns = model_bundle.get('input_columns', columns)
                    history_list.append(model_bundle['history_list'])
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
                'mse_median': mse_stats.loc['50%'].values[0],
                'history_list': history_list
            })
    return summary_list

def collect_r2_scores():
    all_r2_scores = []  # To store scores for all datasets
    #feature_labels = []  # To store feature subset names

    for select_dataset in range(6):  # Loop through all dataset combinations
        summary_list = get_models_summary(select_dataset)

        # Extract r2_mean and corresponding feature subsets
        dataset_r2_scores = []
        dataset_feature_labels = []

        for entry in summary_list:
            dataset_r2_scores.append(entry['r2_mean'])
            dataset_feature_labels.append(", ".join(entry['input_columns']))  # Combine feature names as labels

        # # Reorder the R² scores based on the feature labels of the first dataset
        # if select_dataset == 0:
        #     # Initialize the feature_labels list with the first dataset's feature labels
        #     feature_labels = dataset_feature_labels
        # else:
            # Reorder the dataset's r2_mean values based on the feature labels from the first dataset
        reordered_r2_scores = reorder_list(dataset_r2_scores, dataset_feature_labels, feature_labels)
        all_r2_scores.append(reordered_r2_scores)
        #     continue
        # print(feature_labels)

        # # For the first dataset, directly add the R² scores
        # all_r2_scores.append(dataset_r2_scores)

    return all_r2_scores, feature_labels


def reorder_list(r2_scores, current_labels, reference_labels):
    """
    Reorders the r2_scores based on the reference_labels order.
    Assumes that current_labels and reference_labels are lists of the same length,
    and that both lists contain the same set of feature names (just in different orders).
    """
    # Create a mapping of label to its index in reference_labels
    label_index_map = {label: idx for idx, label in enumerate(reference_labels)}

    # Reorder the r2_scores based on the reference_labels
    reordered_scores = [None] * len(reference_labels)

    for score, label in zip(r2_scores, current_labels):
        if label == 'b, mag_b, heatflux, s, h, mag_h, mag_s, surface_air_temp':
            label = 's, b, h, mag_h, mag_s, mag_b, heatflux, surface_air_temp'
        index_in_reference = label_index_map[label]
        reordered_scores[index_in_reference] = score

    return reordered_scores

# Replace feature subset labels with symbols
def replace_features_with_symbols(feature_labels):
    symbol_labels = []
    for label in feature_labels:
        # Convert feature subsets to symbols, preserving order
        symbols = [FEATURE_SYMBOLS.get(feature, feature) for feature in label.split(", ")]
        symbol_labels.append(", ".join(symbols))
    return symbol_labels

def change_feature_label(feature_labels):
    new_order = [r"$s$", r"$b$", r"$h$", r"$\|\nabla h\|$", r"$\|\nabla s\|$", r"$\|\nabla b\|$",    r"$Q_b$",    r"$T_s$"]
    for idx, features in enumerate(feature_labels):
        if set(features) == set(new_order):
            break
    #print(idx)
    feature_labels[idx] = new_order

# Visualize R2 scores using a heatmap with symbols and fixed cbar range
def plot_r2_heatmap_with_symbols(r2_scores, feature_labels):
    feature_symbols = replace_features_with_symbols(feature_labels)
    change_feature_label(feature_labels)
    r2_df = pd.DataFrame(np.array(r2_scores).T, columns=DATASET_NAMES, index=feature_symbols)

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        r2_df,
        annot=True,
        fmt=".2f",
        cmap="YlGn",  # Green for high values, red for low values
        cbar=True,
        vmin=0, vmax=1  # Set bounds from 0 to 100%
    )
    
    plt.title("Validation $R^2$ Scores for Feature Subsets and Datasets")
    plt.xlabel("Training Dataset Combinations")
    plt.ylabel("Feature Subsets")
    #plt.xticks(ha='right')
    plt.tight_layout()
    plt.show()


def eval_models(select_dataset, invert_obj):
    C_mean = compute_C_mean(select_dataset)
    compute_u_avg(invert_obj, C_mean)
    if select_dataset == 0:
        base_folder = 'mlp_ensemble_englacial_0' 
    elif select_dataset == 1:
        base_folder = 'mlp_ensemble_englacial_1'
    elif select_dataset == 2:
        base_folder = 'mlp_ensemble_englacial_2' 
    elif select_dataset == 3:
        base_folder = 'mlp_ensemble_englacial_3'
    elif select_dataset == 4:
        base_folder = 'mlp_ensemble_englacial_4'
    elif select_dataset == 5:
        base_folder = 'mlp_ensemble_englacial_5'
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
        temp_object.C = firedrake.Function(temp_object.Q)
        temp_object = regression(temp_object, model_file_path, C_bounds = [-50, 55])
        
        try:
            u_optimized = temp_object.simulation()
        except:
            print('Error in simulation')
            try:
                temp_object.opts = {"dirichlet_ids": temp_object.drichlet_ids,
                                "side_wall_ids": temp_object.side_ids,
                            "diagnostic_solver_type": "icepack",
                            "diagnostic_solver_parameters": {
                                "max_iterations":50,},}
                temp_object.create_model_weertman()
                u_optimized = temp_object.simulation()
            except:
                print('Error in simulation again reducing C bounds')
                try:
                    temp_object = regression(temp_object, model_file_path, C_bounds = [-24, 45])
                    u_optimized = temp_object.simulation()
                except:
                    print('Error in simulation again reducing C bounds 2nd time')
                    try:
                        temp_object = regression(temp_object, model_file_path, C_bounds = [-11, 2])
                        u_optimized = temp_object.simulation()
                    except:
                        print('Error in simulation assigning u_optimized')
                        temp_object.C = firedrake.Function(temp_object.Q)
                        u_optimized =  temp_object.simulation()

        temp_object.opts = {"dirichlet_ids": temp_object.drichlet_ids,
                    "side_wall_ids": temp_object.side_ids,
                   "diagnostic_solver_type": "petsc",
                "diagnostic_solver_parameters": {
                    "snes_type": "newtontr",
                    "ksp_type": "gmres",
                    "pc_type": "lu",
                    "pc_factor_mat_solver_type": "mumps",},}

            

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
    inv_dotson = invert_dotson_fcn()
    temp_objects_dotson, summary_list_dotson, columns_list_dotson, u_optimized_list_dotson, loss_function_list_dotson = eval_models(select_dataset, inv_dotson)
    
    inv_pig = invert_pig_fcn()
    temp_objects_pig, summary_list_pig, columns_list_pig, u_optimized_list_pig, loss_function_list_pig = eval_models(select_dataset, inv_pig)
    
    inv_thwaites = invert_thwaites_fcn()
    temp_objects_thwaites, summary_list_thwaites, columns_list_thwaites, u_optimized_list_thwaites, loss_function_list_thwaites = eval_models(select_dataset, inv_thwaites)

    return temp_objects_dotson, summary_list_dotson, columns_list_dotson, u_optimized_list_dotson, loss_function_list_dotson, temp_objects_pig, summary_list_pig, columns_list_pig, u_optimized_list_pig, loss_function_list_pig, temp_objects_thwaites, summary_list_thwaites, columns_list_thwaites, u_optimized_list_thwaites, loss_function_list_thwaites

def plot_percent_diff(select_dataset, temp_objects_dotson, summary_list_dotson, columns_list_dotson, u_optimized_list_dotson, loss_function_list_dotson, temp_objects_pig, summary_list_pig, columns_list_pig, u_optimized_list_pig, loss_function_list_pig, temp_objects_thwaites, summary_list_thwaites, columns_list_thwaites, u_optimized_list_thwaites, loss_function_list_thwaites ):   
    # Determine the dataset used based on select_dataset
    if select_dataset == 0:
        dataset_used = "[df_dotson, df_dotson, df_thwaites]"
    elif select_dataset == 1:
        dataset_used = "[df_pig, df_pig, df_thwaites]"
    elif select_dataset == 2:
        dataset_used = "[df_dotson, df_pig]"
    elif select_dataset == 3:
        dataset_used = "[df_dotson]"
    elif select_dataset == 4:
        dataset_used = "[df_pig]"
    elif select_dataset == 5:
        dataset_used = "[df_thwaites]"

    # Since the lists have an equal number of objects, determine the number of rows
    n_rows = len(temp_objects_dotson)

    # Increase figure size to make the rows larger
    fig = plt.figure(figsize=(22, 6 * n_rows))

    # Create a gridspec layout with an extra row for the title
    # Reduce hspace to decrease vertical gaps between rows
    gs = gridspec.GridSpec(n_rows + 1, 5, width_ratios=[4, 4, 4, 4, 0.2], wspace=0.4, hspace=0.2)

    # Add a new axes for the title with the dynamic dataset info
    ax_title = fig.add_subplot(gs[0, :])
    title_text = f"Percent Difference Accounted for by ML\nDataset used for training: {dataset_used}"
    ax_title.text(0.5, 0.5, title_text, fontsize=16, weight='bold', ha='center', va='center')
    ax_title.axis('off')  # Turn off the axis for the title

    # Loop over the objects and plot in the corresponding grid positions
    for row in range(n_rows):
        # Summary column (column 0)
        ax_summary = fig.add_subplot(gs[row + 1, 0])
        summary_text = (
            f"{columns_list_dotson[row]}\n"
            f"R2 test: {summary_list_dotson[row]['r2_mean']:.4f}\n"
            f"R2_adjusted test: {summary_list_dotson[row]['r2_adjusted_mean']:.4f}\n"
            f"MSE test: {summary_list_dotson[row]['mse_mean']:.4f}\n"
            f"Loss function value dotson: {loss_function_list_dotson[row]:.4f}\n"
            f"Loss function value pig: {loss_function_list_pig[row]:.4f}\n"
            f"Loss function value thwaites: {loss_function_list_thwaites[row]:.4f}\n"
        )
        ax_summary.text(0.1, 0.5, summary_text, fontsize=10, va="center", ha="left")
        ax_summary.axis('off')  # Turn off the axis for the text column
        
        # Dotson column (column 1)
        ax_dotson = fig.add_subplot(gs[row + 1, 1])
        _, ax_dotson = temp_objects_dotson[row].plot_percent_accounted(vmin=0, axes=ax_dotson)
        if row == 0:
            ax_dotson.set_title("Dotson")
        
        # PIG column (column 2)
        ax_pig = fig.add_subplot(gs[row + 1, 2])
        _, ax_pig = temp_objects_pig[row].plot_percent_accounted(vmin=0, axes=ax_pig)
        if row == 0:
            ax_pig.set_title("PIG")
        
        # Thwaites column (column 3)
        ax_thwaites = fig.add_subplot(gs[row + 1, 3])
        _, ax_thwaites = temp_objects_thwaites[row].plot_percent_accounted(vmin=0, axes=ax_thwaites)
        if row == 0:
            ax_thwaites.set_title("Thwaites")

        # Create a color bar at the end of the row (column 4)
        cax = fig.add_subplot(gs[row + 1, 4])
        fig.colorbar(ax_dotson.collections[0], cax=cax)

    # Show the final figure
    plt.show()