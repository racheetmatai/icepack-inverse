FROM firedrakeproject/firedrake-vanilla:2023-09

# Install system dependencies
RUN sudo apt update && sudo apt install patchelf

# Activate Firedrake environment and install Icepack and other dependencies
RUN source firedrake/bin/activate && \
    pip install git+https://github.com/icepack/Trilinos.git && \
    pip install git+https://github.com/icepack/pyrol.git && \
    git clone https://github.com/icepack/icepack.git && \
    cd icepack && git checkout 28eed36fe652da79769d0130822037f903b23ed3 && \
    pip install --editable . && \
    cd .. && \
    pip install jupyter lab && \
    pip install scikit-learn && \
    pip install tensorflow && \
    pip install h5py && \
    pip install xarray && \
    pip install --upgrade xarray && \
    pip install h5netcdf

