#!/bin/bash

source ./buildtools.sh

msg "\nStarting Your Connected local development environemnt..."

# This will ask you for your Jupyter Hub Token to make sure you can utilize planetary compute resources
get_jupyterhub_token

# Running the docker-compose composition.
docker-compose up 