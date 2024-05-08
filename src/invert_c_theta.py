import firedrake
import icepack
import xarray
import pandas
import matplotlib.pyplot as plt
import numpy as np

from src.create_mesh import fetch_outline, create_mesh
from src.helper_functions import get_min_max_coords, plot_bounded_antarctica
from src.data_preprocessing import clean_imported_data, get_windowed_velocity_file, create_vertex_only_mesh_for_sparse_data, interpolate_data_onto_vertex_only_mesh
from firedrake import assemble, Constant, inner, grad, dx
import icepack.models.friction
from icepack.constants import ice_density as ρ_I, water_density as ρ_W, gravity as g
from icepack.statistics import StatisticsProblem, MaximumProbabilityEstimator
from sklearn.cluster import KMeans
import joblib
from scipy.ndimage import gaussian_filter
from firedrake import *
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler
import pickle
from keras.models import model_from_json
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import tensorflow as tf
import keras

class Invert:
    """Class for solving inverse problems related to Antarctic ice flow.

    Args:
        outline (str): Name of the outline (e.g., 'pine-island').
        mesh_name (str): Name of the mesh.
        δ (float): Mesh resolution.
        temperature (float): Ice temperature.
        m (float): Friction exponent.

    Attributes:
        outline (GeoJSON): GeoJSON outline data.
        mesh (firedrake.Mesh): Computational mesh.
        area (firedrake.Constant): Mesh area.
        thickness (xarray.DataArray): Ice thickness data.
        surface (xarray.DataArray): Ice surface data.
        bed (xarray.DataArray): Bedrock elevation data.
        Q (firedrake.FunctionSpace): Scalar function space.
        h (firedrake.Function): Ice thickness field.
        s (firedrake.Function): Ice surface field.
        b (firedrake.Function): Bedrock elevation field.
        V (firedrake.VectorFunctionSpace): Vector function space.
        δ (float): Mesh resolution.
        A0 (firedrake.Constant): Rate factor.
        m (firedrake.Constant): Friction exponent.
        θ_default (firedrake.Constant): Default fluidity.

    Methods:
        plot_bounded_antarctica(): Plot Antarctica with bounding box.
        compute_C(): Compute friction coefficient field.
        import_velocity_data(name, modified_exists=True): Import velocity data.
        weertman_friction_with_ramp(**kwargs): Weertman friction model.
        viscosity(**kwargs): Depth-averaged viscosity model.
        loss_functional(u): Compute the loss functional.
        simulation_theta(θ): Simulate ice flow with given fluidity.
        simulation_C(C): Simulate ice flow with given friction coefficient.
        regularization_theta(θ): Regularization term for fluidity.
        regularization_C(C): Regularization term for friction coefficient.
        invert_theta(gradient_tolerance=1e-8, step_tolerance=1e-1, max_iterations=50): Invert for fluidity.
        invert_C(gradient_tolerance=1e-8, step_tolerance=1e-1, max_iterations=50): Invert for friction coefficient.
        get_C(): Get the computed friction coefficient field.
        get_theta(): Get the computed fluidity field.
        get_outline(): Get the outline of glacier/ice-sheet.
        get_mesh(): Get the computational mesh.
    """
    def __init__(
        self,
        read_mesh = False,
        outline = 'pine-island', 
        mesh_name = 'pig', 
        δ = 5e3,
        temperature = 260,
        m = 3.0,
        reg_constant_c = 1.0,
        reg_constant_theta = 1.0,
        reg_constant_simultaneous = 1.0,
        lcar = 5e3,
        drichlet_ids = [1,2,3,4],
        side_ids = [],
        accumulation_rate_vs_elevation_file = 'PIG_elevation_vs_accumulation.csv',
        opts = None):
        """Initialize the Invert instance."""
        self.outline = fetch_outline(outline)
        if not read_mesh:
            create_mesh(outline = self.outline, name= mesh_name, lcar = lcar)
        self.mesh = firedrake.Mesh(mesh_name+'.msh')
        self.area = Constant(assemble(Constant(1.0) * dx(self.mesh)))
        thickness_filename = icepack.datasets.fetch_bedmachine_antarctica()
        thickness_data = xarray.open_dataset(thickness_filename)
        self.thickness = thickness_data["thickness"]
        self.surface = thickness_data["surface"]
        self.bed = thickness_data["bed"]
        self.Q = firedrake.FunctionSpace(self.mesh, family="CG", degree=2)
        self.h = icepack.interpolate(self.thickness, self.Q)
        self.h0 = self.h.copy(deepcopy=True)
        self.s = icepack.interpolate(self.surface, self.Q)
        self.b = icepack.interpolate(self.bed, self.Q)
        self.V = firedrake.VectorFunctionSpace(self.mesh, "CG", 2)
        self.δ = δ
        T = Constant(temperature)
        self.A0 = icepack.rate_factor(T)
        self.m = Constant(m)
        self.reg_C = Constant(reg_constant_c)
        self.reg_theta = Constant(reg_constant_theta)
        self.reg_C_theta = Constant(reg_constant_simultaneous)
        self.θ = firedrake.Function(self.Q)
        self.C = firedrake.Function(self.Q)

        model_weertman = icepack.models.IceStream(friction=self.weertman_friction_with_ramp, viscosity = self.viscosity)
        if opts is None:
            opts = {"dirichlet_ids": drichlet_ids,
                    "side_wall_ids": side_ids,
                   "diagnostic_solver_type": "petsc",
                "diagnostic_solver_parameters": {
                    "snes_type": "newtontr",
                    "ksp_type": "gmres",
                    "pc_type": "lu",
                    "pc_factor_mat_solver_type": "mumps",
                },} 
        self.solver_weertman = icepack.solvers.FlowSolver(model_weertman, **opts)

        if accumulation_rate_vs_elevation_file is not None:
            self.set_accumulation_rate(accumulation_rate_vs_elevation_file)

    def smooth_h(self):
        """
        smooth h if inversions are failing
        """
        α = Constant(2e3)
        J = 0.5 * ((self.h - self.h0) ** 2 + α ** 2 * inner(grad(self.h), grad(self.h))) * dx
        F = firedrake.derivative(J, self.h)
        firedrake.solve(F == 0, self.h)

    def set_accumulation_rate(self, accumulation_rate_vs_elevation_file):
        # Load dataset from CSV file
        data = np.genfromtxt(accumulation_rate_vs_elevation_file, delimiter=',', names=True)
        # Extract data columns
        elevation = data['Elevation']
        accumulation = data['Accumulation']
        # Create interpolation function
        interpolation_function = interp1d(elevation, accumulation, kind='linear', fill_value='extrapolate')
        # Interpolate accumulation values corresponding to the surface elevation data
        accumulation_values = interpolation_function(self.surface.values)
        thickness_filename = icepack.datasets.fetch_bedmachine_antarctica()
        thickness_data = xarray.open_dataset(thickness_filename)
        thickness_data['accumulation'] = (('y', 'x'), accumulation_values)
        self.a = icepack.interpolate(thickness_data['accumulation'], self.Q)

    def __del__(self):
        # Close the raster files
        self.vx_file.close()
        self.vy_file.close()
        self.stdx_file.close()
        self.stdy_file.close()

    def get_norm(self, field):
        return firedrake.norm(field)

    def plot_bounded_antarctica(self):
        """Plot Antarctica with bounding box.

        Returns:
            fig (matplotlib.figure.Figure): Matplotlib figure.
            axes (matplotlib.axes._axes.Axes): Matplotlib axes.
        """
        fig, axes = plot_bounded_antarctica(self.outline, self.δ)
        return fig, axes

    def plot_mesh(self):
        fig, axes = self.plot_bounded_antarctica()
        axes.set_xlabel("meters")
        kwargs = {
            "interior_kw": {"linewidth": 0.25},
            "boundary_kw": {"linewidth": 2},
        }
        firedrake.triplot(self.get_mesh(), axes=axes, **kwargs)
        axes.legend();

    def plot_u_error(self, u, vmin=0, vmax=50):
        """Plot error in u compared to u_initial."""
        fig, axes = self.plot_bounded_antarctica()
        δu = firedrake.interpolate((u - self.u_initial)**2/ (2 * self.σ**2), self.Q)
        colors = firedrake.tripcolor(
            δu, vmin=vmin, vmax=vmax, cmap="Reds", axes=axes
        )
        fig.colorbar(colors);
        plt.title("Error in U")
        plt.show()

    def plot_theta(self, vmin=-10, vmax=5):
        """Plot theta"""
        fig, axes = self.plot_bounded_antarctica()
        colors = firedrake.tripcolor(
            self.θ, axes=axes, vmin=vmin, vmax=vmax
        )
        fig.colorbar(colors)
        plt.title("Theta")
        plt.show()

    def plot_accumulation(self, vmin=None, vmax=None):
        """Plot accumulation"""
        fig, axes = self.plot_bounded_antarctica()
        colors = firedrake.tripcolor(
            self.a, axes=axes, vmin=vmin, vmax=vmax
        )
        fig.colorbar(colors)
        plt.title("Accumulation")
        plt.show()

    def plot_thickness(self, vmin=None, vmax=None):
        """Plot thickness"""
        fig, axes = self.plot_bounded_antarctica()
        colors = firedrake.tripcolor(
            self.h, axes=axes, vmin=vmin, vmax=vmax
        )
        fig.colorbar(colors)
        plt.title("Thickness")
        plt.show()

    def plot_surface(self, vmin=None, vmax=None):
        """Plot surface elevation"""
        fig, axes = self.plot_bounded_antarctica()
        colors = firedrake.tripcolor(
            self.s, axes=axes, vmin=vmin, vmax=vmax
        )
        fig.colorbar(colors)
        plt.title("surface elevation")
        plt.show()

    def plot_C(self, vmin=-2, vmax=2):
        """Plot C"""
        fig, axes = self.plot_bounded_antarctica()
        colors = firedrake.tripcolor(
            self.C, axes=axes, vmin=vmin, vmax=vmax
        )
        fig.colorbar(colors);
        plt.title("C")
        plt.show()

    def plot_C_total(self, vmin=None, vmax=None):
        """Plot C0*exp(C)"""
        expr = self.C0*firedrake.exp(self.C)
        total_C = firedrake.interpolate(expr, self.Q)
        fig, axes = self.plot_bounded_antarctica()
        colors = firedrake.tripcolor(
            total_C, axes=axes, vmin=vmin, vmax=vmax
        )
        plt.title("Basal Friction Coefficient")
        fig.colorbar(colors);
        plt.show()

    def plot_streamline_u(self, u, resolution = 2500):
        """Plot u streamlines"""
        fig, axes = self.plot_bounded_antarctica()
        kwargs = {"resolution": resolution}
        streamlines = firedrake.streamplot(u, axes=axes, **kwargs)
        fig.colorbar(streamlines, label="meters/year");
        plt.title("Surface velocity streamlines")

    def compute_C_old(self):
        """Compute the friction coefficient field (C)."""
        expr = ρ_I * g * firedrake.sqrt(inner(grad(self.s), grad(self.s))) / (firedrake.sqrt(inner(grad(self.u_initial), grad(self.u_initial))) ** (1/self.m))
        self.C0 = firedrake.interpolate(expr, self.Q)

    def cell_areas(self, mesh):
        area = Function(self.Q)
        area.interpolate(CellSize(self.mesh))
        return area
    
    def compute_C_driving_stress(self):
        d_areas = self.cell_areas(self.mesh)
        area = assemble(d_areas*dx)
        expr = (ρ_I * g * self.h * firedrake.sqrt(inner(grad(self.s), grad(self.s))) / (firedrake.sqrt(inner(self.u_initial, self.u_initial)) ** (1/self.m)))
        self.C0 = icepack.interpolate(expr, self.Q)     
        # Compute percentiles
        percentiles = np.percentile(self.C0.dat.data, [5, 95])
        # Clip values
        C0_clipped = np.clip(self.C0.dat.data, percentiles[0], percentiles[1])
        # Assign clipped values back to the function
        self.C0.dat.data[:] = C0_clipped

    def compute_C(self, constant_val = 1e-3):
        self.C0 = firedrake.Constant(constant_val)

    def compute_features(self):
        u = self.simulation()
        u1, u2 = firedrake.split(u)
        vel_mag = firedrake.sqrt(u1**2 + u2**2)
        grad_u_1 = firedrake.grad(u1)  # Gradient of u1  field
        grad_u_2 = firedrake.grad(u2)  # Gradient of u2  field
        grad_h = firedrake.grad(self.h)  # Gradient of ice thickness
        #grad_h_1, grad_h_2 = firedrake.split(grad_h)
        grad_s = firedrake.grad(self.s)  # Gradient of surface
        #grad_s_1, grad_s_2 = firedrake.split(grad_s)
        grad_b = firedrake.grad(self.b)  # Gradient of bed
        #grad_b_1, grad_b_2 = firedrake.split(grad_b)

        grad_u11, grad_u12 = firedrake.split(firedrake.interpolate(grad_u_1, self.V))
        grad_u21, grad_u22 = firedrake.split(firedrake.interpolate(grad_u_2, self.V))

        inv1 = grad_u11 + grad_u22
        inv2 = 0.5*(grad_u11**2 + grad_u22**2 - grad_u11*grad_u22 + grad_u12*grad_u21)

        magh = firedrake.sqrt(firedrake.dot(grad_h, grad_h))
        mags = firedrake.sqrt(firedrake.dot(grad_s, grad_s))
        magb = firedrake.sqrt(firedrake.dot(grad_b, grad_b))

        inv1_fcn = firedrake.interpolate(inv1, self.Q)
        inv2_fcn = firedrake.interpolate(inv2, self.Q)
        magh_fcn = firedrake.interpolate(magh, self.Q)
        mags_fcn = firedrake.interpolate(mags, self.Q)
        magb_fcn = firedrake.interpolate(magb, self.Q)
        vel_mag_fcn = firedrake.interpolate(vel_mag, self.Q)

        inv1_fcn_npy = inv1_fcn.dat.data[:]
        inv2_fcn_npy = inv2_fcn.dat.data[:]
        magh_fcn_npy = magh_fcn.dat.data[:]
        mags_fcn_npy = mags_fcn.dat.data[:]
        magb_fcn_npy = magb_fcn.dat.data[:]
        vel_mag_fcn_npy = vel_mag_fcn.dat.data[:]

        h_npy = self.h.dat.data[:]
        s_npy = self.s.dat.data[:]
        b_npy = self.b.dat.data[:]
        

        # Create a new DataFrame with the new data
        cluster = np.array([inv1_fcn_npy, inv2_fcn_npy, magh_fcn_npy, mags_fcn_npy, magb_fcn_npy, h_npy, s_npy, b_npy, vel_mag_fcn_npy]).T
        cluster_df_full = pandas.DataFrame(cluster, columns=['invariant1', 'invariant2', 'mag_h', 'mag_s', 'mag_b', 'h', 's', 'b', 'vel_mag'])
        cluster_df_full['driving_stress'] = cluster_df_full['h']*9.8*cluster_df_full['mag_s']
        self.cluster_df_full = cluster_df_full

    def regress(self, filename = 'model.pkl'):
        # To load everything back from the file
        with open(filename+'.pkl', "rb") as f:
            loaded_model_bundle = pickle.load(f)
        
        # Extract components
        loaded_model_architecture = loaded_model_bundle['model_architecture']
        loaded_model_weights = loaded_model_bundle['model_weights']
        loaded_input_scaler = loaded_model_bundle['input_scaler']
        loaded_output_scaler = loaded_model_bundle['output_scaler']
        loaded_input_columns = loaded_model_bundle['input_columns']
        loaded_output_columns = loaded_model_bundle['output_columns']

        self.compute_features()
        
        
        loaded_model = keras.models.load_model(filename+'.h5')

        df = self.cluster_df_full[loaded_input_columns].copy()
        df_scaled = loaded_input_scaler.transform(df.to_numpy())
        prediction = loaded_output_scaler.inverse_transform(loaded_model.predict(df_scaled).reshape(-1,1)).reshape(-1,)
        return prediction

    def compute_C_theta_ML_regress(self,   filename = 'model'):
        self.C.dat.data[:] = self.regress(filename+'_C')
        self.θ.dat.data[:] = self.regress(filename+'_theta')

    def classify_regress(self, filename = 'C_6'):
        """Load pre-trained classification and regression models from files specified by the given filename, and use them to classify and regress on the data stored in the object.
    
        Args:
            filename (str, optional): The base filename for the classifier and regressor models. Default is 'C_6'.
        """

        variable_name = filename.split('_')[0]
        
        classifier_name = filename+'_classifier.joblib'
        classifier, meta_classifier = joblib.load(classifier_name)

        input_columns_cl = meta_classifier['input_columns']
        input_scaler_cl = meta_classifier['input_scaler']
        output_scaler_cl = meta_classifier['output_scaler']
 
        regressor_name = filename+'_regressor.joblib'
        regressor, meta_regressor = joblib.load(regressor_name)

        input_columns_reg = meta_regressor['input_columns']
        input_scaler_reg = meta_regressor['input_scaler']
        output_scaler_reg = meta_regressor['output_scaler']

        if input_columns_reg != input_columns_cl:
            raise ValueError("Input columns of classifier and regressor are not the same")


        df = self.cluster_df_full[input_columns_cl].copy()
        df_cl_scaled = input_scaler_cl.transform(df.to_numpy())
        importance = output_scaler_cl.inverse_transform(classifier.predict(df_cl_scaled).reshape(-1, 1))
        df['importance'] = importance

        df_reg = df[df['importance'] > 0.5]
        df_reg_scaled = input_scaler_reg.transform(df_reg[input_columns_reg].to_numpy())
        #print(df_reg_scaled.shape)
        variable_values = output_scaler_reg.inverse_transform(regressor.predict(df_reg_scaled).reshape(-1,1))

        # Create an array of zeros with the same length as df
        df[variable_name] = np.zeros(len(df))
        
        # Conditionally assign variable_values where importance > 0.5
        df.loc[df['importance'] > 0.5, variable_name] = variable_values
                
        self.cluster_df_full[['importance', variable_name]] = df[['importance', variable_name]]
        
    def compute_C_theta_ML(self, C_name = 'C_6', theta_name = 'theta_6'):
        """Compute values for the variables C and theta using classification and regression models.

        Args:
            C_name (str, optional): The base filename for the C classifier and regressor models. Default is 'C_6'.
            theta_name (str, optional): The base filename for the theta classifier and regressor models. Default is 'theta_6'.
        """
        self.classify_regress(C_name)
        self.classify_regress(theta_name)
        self.C.dat.data[:] = self.cluster_df_full['C'].to_numpy()
        self.θ.dat.data[:] = self.cluster_df_full['theta'].to_numpy()

    def load_kMeans(self, filename = 'kmeans_model.joblib', cluster_val = 'avg', cluster_values = None, verbose = False):
        """Load a pre-trained k-Means clustering model and use it to predict cluster values for the data stored in the object.

        Args:
            filename (str, optional): The filename for the pre-trained k-Means clustering model. Default is 'kmeans_model.joblib'.
            cluster_val (str, optional): The method to determine cluster values, either 'avg' (average) or 'median'. Default is 'avg'.
            cluster_values (list, optional): Precomputed cluster values to use instead of those calculated from the model. Default is None.
            verbose (bool, optional): If True, print additional information during execution. Default is False.
    
        Returns:
            numpy.ndarray: An array of predicted cluster values for the data.
        """
        kmeans , loaded_averages_dict = joblib.load(filename)
        # Access the input columns used for clustering
        input_columns = loaded_averages_dict['input_columns']
        if verbose:
            print("Input columns used for clustering:", input_columns)
        
        
        # Access the averages dictionary
        average_c_per_cluster = loaded_averages_dict['average_c_per_cluster']
        std_c_per_cluster = loaded_averages_dict['std_c_per_cluster']
        median_c_per_cluster = loaded_averages_dict['median_c_per_cluster']


        cluster_df = self.cluster_df_full[input_columns].copy()
        cluster_df['predicted_cluster'] = kmeans.predict(cluster_df)
        if cluster_values is None:
            if cluster_val == 'avg':
                cluster_values = list(average_c_per_cluster.values())
            else:
                cluster_values = list(median_c_per_cluster.values())

        if verbose:
            print("Number of clusters:", len(cluster_values))
        # Cap or threshold the cluster values
        #cluster_values = np.clip(cluster_values, min_threshold, max_threshold)
        print(cluster_values)
        cluster_df['predicted_cluster_value'] = cluster_df['predicted_cluster'].map(lambda x: cluster_values[x])
        return cluster_df['predicted_cluster_value'].to_numpy()

    def compute_C_kMeans(self, filename = 'kmeans_model.joblib', cluster_val = 'avg', cluster_values = None, verbose = False):
        """Compute values for the variable C using a pre-trained k-Means clustering model.

        Args:
            filename (str, optional): The filename for the pre-trained k-Means clustering model. Default is 'kmeans_model.joblib'.
            cluster_val (str, optional): The method to determine cluster values, either 'avg' (average) or 'median'. Default is 'avg'.
            cluster_values (list, optional): Precomputed cluster values to use instead of those calculated from the model. Default is None.
            verbose (bool, optional): If True, print additional information during execution. Default is False.
        """
        self.C.dat.data[:] = self.load_kMeans(filename, cluster_val, cluster_values, verbose)

    def compute_theta_kMeans(self, filename = 'kmeans_model.joblib', cluster_val = 'avg', cluster_values = None, verbose = False):
        """Compute values for the variable theta using a pre-trained k-Means clustering model.

        Args:
            filename (str, optional): The filename for the pre-trained k-Means clustering model. Default is 'kmeans_model.joblib'.
            cluster_val (str, optional): The method to determine cluster values, either 'avg' (average) or 'median'. Default is 'avg'.
            cluster_values (list, optional): Precomputed cluster values to use instead of those calculated from the model. Default is None.
            verbose (bool, optional): If True, print additional information during execution. Default is False.
        """
        self.θ.dat.data[:] = self.load_kMeans(filename, cluster_val, cluster_values, verbose)
        
    def import_velocity_data(self, name = None, modified_exists = True, C = 'constant', constant_val = 1e-3):
        """Import velocity data and preprocess.

        Args:
            name (str): Name of the velocity data.
            modified_exists (bool): Flag indicating whether modified data exists.

        Returns:
            None
        """
        self.vx, self.vx_file, self.vy, self.vy_file, self.stdx, self.stdx_file, self.stdy, self.stdy_file, self.window, self.transform = get_windowed_velocity_file(name, self.outline, self.δ, modified_exists = modified_exists)
        self.Δ, self.indices= create_vertex_only_mesh_for_sparse_data(self.mesh, self.window, self.transform, self.stdx)
        self.N = len(self.indices)
        self.u_o = interpolate_data_onto_vertex_only_mesh(self.Δ, self.vx, self.indices)
        self.v_o = interpolate_data_onto_vertex_only_mesh(self.Δ, self.vy, self.indices)
        self.σ_x = interpolate_data_onto_vertex_only_mesh(self.Δ, self.stdx, self.indices)
        self.σ_y = interpolate_data_onto_vertex_only_mesh(self.Δ, self.stdy, self.indices)
        self.u_initial = icepack.interpolate((self.vx_file, self.vy_file), self.V)
        if C == 'constant':
            print('C0 is constant:',constant_val)
            self.compute_C(constant_val = constant_val)
        elif C == 'driving-stress':
            print('C0 is computed using driving stress')
            self.compute_C_driving_stress()
        else:
            print('C0 is constant:',constant_val)
            self.compute_C(constant_val = constant_val)
        self.σx = icepack.interpolate(self.stdx_file, self.Q)
        self.σy = icepack.interpolate(self.stdy_file, self.Q)
        self.σ = firedrake.interpolate(firedrake.sqrt(self.σx**2 + self.σy**2), self.Q)
    
    def weertman_friction_with_ramp(self, **kwargs):
        """Weertman friction model with a ramp. The ramp ensures a smooth transition of C from hard bed to over water.

        Args:
            **kwargs: Keyword arguments including velocity, thickness, surface, and friction.

        Returns:
            firedrake.Function: Friction term.
        """
        u = kwargs["velocity"]
        h = kwargs["thickness"]
        s = kwargs["surface"]
        C = kwargs["log_friction"]
    
        p_W = ρ_W * g * firedrake.max_value(0, h - s)
        p_I = ρ_I * g * h
        ϕ = 1 - p_W / p_I

        friction = self.C0 * ϕ * firedrake.exp(C)
        return icepack.models.friction.bed_friction(
            velocity=u,
            friction=friction,
        ) 
        
    def viscosity(self, **kwargs):
        """Depth-averaged viscosity model.

        Args:
            **kwargs: Keyword arguments including velocity, thickness, and log_fluidity.

        Returns:
            firedrake.Function: Depth-averaged viscosity.
        """
        u = kwargs["velocity"]
        h = kwargs["thickness"]
        θ = kwargs["log_fluidity"]
    
        A = self.A0 * firedrake.exp(θ)
        return icepack.models.viscosity.viscosity_depth_averaged(
            velocity=u, thickness=h, fluidity=A
        )

    def loss_functional_scipy(self, u): ## Doesnt work
        """Compute the loss for optimization.

        Args:
            u (firedrake.Function): Velocity field.

        Returns:
            firedrake.Constant: Loss functional.
        """
        d_areas = self.cell_areas(self.mesh)
        
        area = assemble(d_areas*dx)
               
        u_interp = firedrake.interpolate(u[0], self.Δ)
        v_interp = firedrake.interpolate(u[1], self.Δ)
        δu, δv = u_interp - self.u_o, v_interp - self.v_o
        return 0.5 / Constant(self.N) * ((δu / self.σ_x)**2 + (δv / self.σ_y)**2) * d_areas

    def loss_functional(self, u):
        """Compute the loss for optimization.

        Args:
            u (firedrake.Function): Velocity field.

        Returns:
            firedrake.Constant: Loss functional.
        """
               
        u_interp = firedrake.interpolate(u[0], self.Δ)
        v_interp = firedrake.interpolate(u[1], self.Δ)
        δu, δv = u_interp - self.u_o, v_interp - self.v_o
        return 0.5 / Constant(self.N) * ((δu / self.σ_x)**2 + (δv / self.σ_y)**2) * dx

    def loss_functional_nosigma(self, u):
        """Compute the loss for optimization.

        Args:
            u (firedrake.Function): Velocity field.

        Returns:
            firedrake.Constant: Loss functional.
        """
               
        u_interp = firedrake.interpolate(u[0], self.Δ)
        v_interp = firedrake.interpolate(u[1], self.Δ)
        δu, δv = u_interp - self.u_o, v_interp - self.v_o
        return 0.5 / Constant(self.N) * ((δu)**2 + (δv)**2) * dx
    
    def simulation_theta(self, θ):
        """Simulate ice flow with a given fluidity.

        Args:
            θ (firedrake.Function): Fluidity field.

        Returns:
            firedrake.Function: Simulated velocity field.
        """
        return self.solver_weertman.diagnostic_solve(
        velocity=self.u_initial, thickness=self.h, surface=self.s,  log_fluidity=θ, log_friction=self.C
        )

    def simulation_C(self, C):
        """Simulate ice flow with a given friction coefficient.

        Args:
            C (firedrake.Function): Friction coefficient field.

        Returns:
            firedrake.Function: Simulated velocity field.
        """
        
        return self.solver_weertman.diagnostic_solve(
        velocity=self.u_initial, thickness=self.h, surface=self.s,  log_fluidity=self.θ, log_friction=C
        )

    def simulation(self, u = None):
        """Simulate ice flow.

        Returns:
            firedrake.Function: Simulated velocity field.
        """
        if u is None:
            u = self.u_initial
        return self.solver_weertman.diagnostic_solve(
        velocity= u, thickness=self.h, surface=self.s,  log_fluidity=self.θ, log_friction=self.C
        )

    def simulation_C_theta(self, C_θ):
        """Simulate ice flow with a given friction coefficient.

        Args:
            C_θ (firedrake.Function): Friction coefficient and fluidity field.

        Returns:
            firedrake.Function: Simulated velocity field.
        """
        
        return self.solver_weertman.diagnostic_solve(
        velocity=self.u_initial, thickness=self.h, surface=self.s,  log_fluidity=C_θ[1], log_friction=C_θ[0]
        )

    
    def simulation_prognostic(self, u, δt):
        """Simulate thickness evolution.

        Returns:
            firedrake.Function: Simulated thickness and surface field.
        """
        self.h = self.solver_weertman.prognostic_solve(
            δt,
            thickness=self.h,
            velocity=u,
            accumulation=self.a,
            thickness_inflow=self.h0,
        )
        self.s = icepack.compute_surface(thickness=self.h, bed=self.b)
        return self.h, self.s
    
    def regularization_C_grad(self, C):
        """Regularization term for friction coefficient.

        Args:
            C (firedrake.Function): Friction coefficient field.

        Returns:
            firedrake.Constant: Regularization term.
        """
        
        L = Constant(7.5e3)
        return 0.5 / self.area * (L / self.reg_C)**2 * inner(grad(C), grad(C)) * dx
    
    
    def regularization_theta_grad(self, θ):
        """Regularization term for fluidity.

        Args:
            θ (firedrake.Function): Fluidity field.

        Returns:
            firedrake.Constant: Regularization term.
        """
        L = Constant(7.5e3)
        return 0.5 / self.area * (L / self.reg_theta)**2 * inner(grad(θ), grad(θ)) * dx

    def regularization_C(self, C):
        """Regularization term for friction coefficient.

        Args:
            C (firedrake.Function): Friction coefficient field.

        Returns:
            firedrake.Constant: Regularization term.
        """
        
        L = Constant(7.5e3)
        return 0.5 / self.area * (L / self.reg_C)**2 * (C*C) * dx

    def regularization_theta(self, θ):
        """Regularization term for fluidity.

        Args:
            θ (firedrake.Function): Fluidity field.

        Returns:
            firedrake.Constant: Regularization term.
        """
        L = Constant(7.5e3)
        return 0.5 / self.area * (L / self.reg_theta)**2 * (θ*θ) * dx

    def regularization_C_theta(self, C_θ):
        """Regularization term for fluidity.

        Args:
            C_θ (firedrake.Function): Friction and Fluidity field.

        Returns:
            firedrake.Constant: Regularization term.
        """
        L = Constant(7.5e3)
        return 0.5 / self.area * (L / self.reg_C_theta)**2 * (  (C_θ[1]*C_θ[1]) +   (C_θ[0]*C_θ[0]) ) * dx 

    def regularization_C_theta_grad(self, C_θ):
        """Regularization term for fluidity.

        Args:
            C_θ (firedrake.Function): Friction and Fluidity field.

        Returns:
            firedrake.Constant: Regularization term.
        """
        L = Constant(7.5e3)
        return 0.5 / self.area * (L / self.reg_C_theta)**2 * (  inner(grad(C_θ[1]),grad(C_θ[1]))+  inner(grad(C_θ[0]),grad(C_θ[0])) ) * dx 

    def invert_C_theta_simultaneously(self, gradient_tolerance=1e-100, step_tolerance=1e-100, max_iterations=50, nosigma_lossfcn = False, regularization_grad_fcn= False, **kwargs):
        """Invert for C and fluidity simultaneously.

        Args:
            gradient_tolerance (float): Tolerance for the gradient norm.
            step_tolerance (float): Tolerance for the step size.
            max_iterations (int): Maximum number of iterations.

        Returns:
            None
        """
        if nosigma_lossfcn:
            loss_fcn = self.loss_functional_nosigma
        else:
            loss_fcn = self.loss_functional

        if regularization_grad_fcn:
            reg_fcn = self.regularization_C_theta_grad
        else:
            reg_fcn = self.regularization_C_theta
        self.θ = firedrake.Function(self.Q)
        self.C = firedrake.Function(self.Q)
        problem = StatisticsProblem(
            simulation=self.simulation_C_theta,
            loss_functional=loss_fcn,
            regularization=reg_fcn,
            controls=(self.C , self.θ),)
        estimator = MaximumProbabilityEstimator(
                problem,
                gradient_tolerance=gradient_tolerance,
                step_tolerance=step_tolerance,
                max_iterations=max_iterations,
                **kwargs)
        self.C, self.θ = estimator.solve()
    
    def invert_theta(self, gradient_tolerance=1e-100, step_tolerance=1e-100, max_iterations=50, nosigma_lossfcn = False, regularization_grad_fcn= False, **kwargs):
        """Invert for fluidity.

        Args:
            gradient_tolerance (float): Tolerance for the gradient norm.
            step_tolerance (float): Tolerance for the step size.
            max_iterations (int): Maximum number of iterations.

        Returns:
            None
        """
        if nosigma_lossfcn:
            loss_fcn = self.loss_functional_nosigma
        else:
            loss_fcn = self.loss_functional

        if regularization_grad_fcn:
            reg_fcn = self.regularization_theta_grad
        else:
            reg_fcn = self.regularization_theta
        self.θ = firedrake.Function(self.Q)
        problem = StatisticsProblem(
            simulation=self.simulation_theta,
            loss_functional=loss_fcn,
            regularization=reg_fcn,
            controls=self.θ,)
        estimator = MaximumProbabilityEstimator(
                problem,
                gradient_tolerance=gradient_tolerance,
                step_tolerance=step_tolerance,
                max_iterations=max_iterations,
                **kwargs
            )
        self.θ = estimator.solve()

    def invert_C(self, gradient_tolerance=1e-100, step_tolerance=1e-100, max_iterations=50, nosigma_lossfcn = False, regularization_grad_fcn= False, **kwargs):
        """Invert for friction coefficient.

        Args:
            gradient_tolerance (float): Tolerance for the gradient norm.
            step_tolerance (float): Tolerance for the step size.
            max_iterations (int): Maximum number of iterations.

        Returns:
            None
        """
        if nosigma_lossfcn:
            loss_fcn = self.loss_functional_nosigma
        else:
            loss_fcn = self.loss_functional

        if regularization_grad_fcn:
            reg_fcn = self.regularization_C_grad
        else:
            reg_fcn = self.regularization_C
        self.C = firedrake.Function(self.Q)
        problem = StatisticsProblem(
            simulation=self.simulation_C,
            loss_functional=loss_fcn,
            regularization=reg_fcn,
            controls=self.C,)
        estimator = MaximumProbabilityEstimator(
                problem,
                gradient_tolerance=gradient_tolerance,
                step_tolerance=step_tolerance,
                max_iterations=max_iterations,
                **kwargs
            )
        self.C = estimator.solve()    
    
    def get_C(self):
        """Get the computed friction coefficient field.

        Returns:
            firedrake.Function: Friction coefficient field.
        """
        return self.C

    def get_theta(self):
        """Get the computed fluidity exponent field.

        Returns:
            firedrake.Function: Fluidity exponent coefficient field.
        """
        return self.θ

    def set_C(self, C):
        self.C = C
    
    def set_theta(self, θ):
        self.θ = θ

    def set_mesh(self, mesh):
        self.mesh = mesh

    def invert_C_theta_alternate(self, gradient_tolerance=1e-100, step_tolerance=1e-100, max_iterations=50, per_variable_iteration = 1, **kwargs):
        self.θ = firedrake.Function(self.Q)
        self.C = firedrake.Function(self.Q)
        
        for _ in range(int(max_iterations/2)):
            problem_theta = StatisticsProblem(
            simulation=self.simulation_theta,
            loss_functional=self.loss_functional,
            regularization=self.regularization_theta,
            controls=self.θ,)
            estimator_theta = MaximumProbabilityEstimator(
                problem_theta,
                gradient_tolerance=gradient_tolerance,
                step_tolerance=step_tolerance,
                max_iterations=per_variable_iteration,
                **kwargs,
            )
            problem_C = StatisticsProblem(
            simulation=self.simulation_C,
            loss_functional=self.loss_functional,
            regularization=self.regularization_C,
            controls=self.C,)
            estimator_C = MaximumProbabilityEstimator(
                problem_C,
                gradient_tolerance=gradient_tolerance,
                step_tolerance=step_tolerance,
                max_iterations=per_variable_iteration,
                **kwargs,
            )
            self.C = estimator_C.solve()
            self.θ = estimator_theta.solve()
        
    
    def get_outline(self):
        """Get the GeoJSON outline data.

        Returns:
            GeoJSON: Return GeoJSON outline data of selected glacier/ice-sheet.
        """
        return self.outline

    def get_mesh(self):
        """Get the mesh.

        Returns:
            firedrake.Mesh: Return mesh.
        """
        return self.mesh

    def get_dataframe(self, u):
        """Get a dataframe with variable values

        Args:
            u (firedrake.Function): velocity field that needs to be included in the dataframe.

        Returns:
            pandas.DataFrame: Return DataFrame.
        """
        u1, u2 = firedrake.split(u)
        u1_initial, u2_initial = firedrake.split(self.u_initial)
        grad_u_1 = firedrake.grad(u1)  # Gradient of u1  field
        grad_u_2 = firedrake.grad(u2)  # Gradient of u2  field
        grad_h = firedrake.grad(self.h)  # Gradient of ice thickness
        grad_s = firedrake.grad(self.s)  # Gradient of surface
        grad_b = firedrake.grad(self.b)  # Gradient of bed

        # Create a function to represent the gradients
        grad_u_1_func = firedrake.Function(self.V, name='grad_u_1')
        grad_u_2_func = firedrake.Function(self.V, name='grad_u_2')
        grad_h_func = firedrake.Function(self.V, name='grad_h')
        grad_s_func = firedrake.Function(self.V, name='grad_s')
        grad_b_func = firedrake.Function(self.V, name='grad_b')
        # Interpolate the gradients to obtain functions
        firedrake.interpolate(grad_u_1, grad_u_1_func)
        firedrake.interpolate(grad_u_2, grad_u_2_func)
        firedrake.interpolate(grad_h, grad_h_func)
        firedrake.interpolate(grad_s, grad_s_func)
        firedrake.interpolate(grad_b, grad_b_func)

        C_1 = icepack.interpolate(self.C, self.Δ)
        θ_1 = icepack.interpolate(self.θ, self.Δ)
        grad_u_1, grad_u_2 = firedrake.split(grad_u_1_func)
        grad_v_1, grad_v_2 = firedrake.split(grad_u_2_func)
        grad_h_1, grad_h_2 = firedrake.split(grad_h_func)
        grad_s_1, grad_s_2 = firedrake.split(grad_s_func)
        grad_b_1, grad_b_2 = firedrake.split(grad_b_func)
        grad_u_1_func = firedrake.Function(self.Q, name='grad_u_1')
        grad_u_2_func = firedrake.Function(self.Q, name='grad_u_2')
        grad_v_1_func = firedrake.Function(self.Q, name='grad_v_1')
        grad_v_2_func = firedrake.Function(self.Q, name='grad_v_2')

        grad_h_1_func = firedrake.Function(self.Q, name='grad_h_1')
        grad_h_2_func = firedrake.Function(self.Q, name='grad_h_2')
        grad_s_1_func = firedrake.Function(self.Q, name='grad_s_1')
        grad_s_2_func = firedrake.Function(self.Q, name='grad_s_2')
        grad_b_1_func = firedrake.Function(self.Q, name='grad_b_1')
        grad_b_2_func = firedrake.Function(self.Q, name='grad_b_2')
        
        firedrake.interpolate(grad_u_1, grad_u_1_func)
        firedrake.interpolate(grad_u_2, grad_u_2_func)
        firedrake.interpolate(grad_v_1, grad_v_1_func)
        firedrake.interpolate(grad_v_2, grad_v_2_func)

        firedrake.interpolate(grad_h_1, grad_h_1_func)
        firedrake.interpolate(grad_h_2, grad_h_2_func)
        firedrake.interpolate(grad_s_1, grad_s_1_func)
        firedrake.interpolate(grad_s_2, grad_s_2_func)
        firedrake.interpolate(grad_b_1, grad_b_1_func)
        firedrake.interpolate(grad_b_2, grad_b_2_func)
        
        gu1 = icepack.interpolate(grad_u_1_func, self.Δ)
        gu2 = icepack.interpolate(grad_u_2_func, self.Δ)
        gv1 = icepack.interpolate(grad_v_1_func, self.Δ)
        gv2 = icepack.interpolate(grad_v_2_func, self.Δ)

        gh1 = icepack.interpolate(grad_h_1_func, self.Δ)
        gh2 = icepack.interpolate(grad_h_2_func, self.Δ)
        gs1 = icepack.interpolate(grad_s_1_func, self.Δ)
        gs2 = icepack.interpolate(grad_s_2_func, self.Δ)
        gb1 = icepack.interpolate(grad_b_1_func, self.Δ)
        gb2 = icepack.interpolate(grad_b_2_func, self.Δ)

        gu1_npy = gu1.dat.data[:]
        gu2_npy = gu2.dat.data[:]
        gv1_npy = gv1.dat.data[:]
        gv2_npy = gv2.dat.data[:]

        gh1_npy = gh1.dat.data[:]
        gh2_npy = gh2.dat.data[:]
        gs1_npy = gs1.dat.data[:]
        gs2_npy = gs2.dat.data[:]
        gb1_npy = gb1.dat.data[:]
        gb2_npy = gb2.dat.data[:]
        
        theta_npy = θ_1.dat.data[:]
        C_npy = C_1.dat.data[:]
        x = θ_1.function_space().mesh().coordinates.dat.data_ro[:,1]
        y = θ_1.function_space().mesh().coordinates.dat.data_ro[:,0]
        u1 =  icepack.interpolate(u1, self.Δ)
        u2 =  icepack.interpolate(u2, self.Δ)
        u1_npy = u1.dat.data[:]
        u2_npy = u2.dat.data[:]
        u1_initial =  icepack.interpolate(u1_initial, self.Δ)
        u2_initial =  icepack.interpolate(u2_initial, self.Δ)
        u1_initial_npy = u1_initial.dat.data[:]
        u2_initial_npy = u2_initial.dat.data[:]
        h_npy = icepack.interpolate(self.h, self.Δ).dat.data[:]
        s_npy = icepack.interpolate(self.s, self.Δ).dat.data[:]
        b_npy = icepack.interpolate(self.b, self.Δ).dat.data[:]
        err_x = icepack.interpolate(self.σx, self.Δ)
        err_y = icepack.interpolate(self.σy, self.Δ)
        err_x_npy = err_x.dat.data[:]
        err_y_npy = err_y.dat.data[:]

        
        df = pandas.DataFrame({
            'theta': theta_npy,
            'C':C_npy,
            'x': x,
            'y': y,
            's11': gu1_npy,
            's12': gu2_npy,
            's21': gv1_npy,
            's22': gv2_npy,
            'grad_h_1':gh1_npy,
            'grad_h_2':gh2_npy,
            'grad_s_1':gs1_npy,
            'grad_s_2':gs2_npy,
            'grad_b_1':gb1_npy,
            'grad_b_2':gb2_npy,
            'x_velocity': u1_npy,
            'y_velocity': u2_npy,
            'x_velocity_initial': u1_initial_npy,
            'y_velocity_initial': u2_initial_npy,
            'err_x': err_x_npy,
            'err_y': err_y_npy,
            'h': h_npy,
            's': s_npy,
            'b': b_npy, 
        })
        df['invariant1'] = df['s11'] + df['s22']
        # Calculate the second invariant
        df['invariant2'] = 0.5 * (df['s11']**2 + df['s22']**2 - df['s11']*df['s22'] + df['s12']*df['s21'])
        df['mag_h'] = np.sqrt(df['grad_h_1']**2 + df['grad_h_2']**2)
        df['mag_s'] = np.sqrt(df['grad_s_1']**2 + df['grad_s_2']**2)
        df['mag_b'] = np.sqrt(df['grad_b_1']**2 + df['grad_b_2']**2)
        df['total_u_error'] = np.sqrt((((df['x_velocity'] -df['x_velocity_initial'])/df['err_x'])**2) + (((df['y_velocity'] -df['y_velocity_initial'])/df['err_y'])**2))
        return df


    
