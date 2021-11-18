FROM mcr.microsoft.com/planetary-computer/python:latest

# NB_USER is an owner of the filesystem so installs must be performed
# by this user.
RUN echo "Running as ${NB_USER} user..."

# Inject welcome file into image to greet users upon startup
ADD --chown=${NB_USER} welcome.txt /scratch/welcome.txt

# Add environment extension configuration file into Image for including any additional packages into notebook environment
ADD --chown=${NB_USER} notebook-env-extension.yml /evn-extension/notebook-env-extension.yml

# Use Conda to install any environment extensions into Conda notebook environment
# So it becomes available on Dask workers.
RUN conda env update -n notebook --file /evn-extension/notebook-env-extension.yml

# Inject the dev_concept library src code into the planetary compute image
ADD --chown=${NB_USER} dev_concept /app/dev_concept

WORKDIR /app/dev_concept

# Pip Install the dev_concept library into Conda notebook environment
# So it becomes available on Dask workers.
RUN /srv/conda/envs/notebook/bin/pip install -e .

WORKDIR ${HOME}