#!/bin/bash

RED="\033[1;31m"
GREEN="\033[0;32m"
YELLOW="\033[1;33m"
BLUE="\033[1;34m"
PURPLE="\033[1;35m"
CYAN="\033[1;36m"
WHITE="\033[1;37m"
RESET="\033[0m"

function get_jupyterhub_token {

	msg "\nThe setup script needs your token for https://https://pccompute.westeurope.cloudapp.azure.com/compute/hub/token:"

    msg "\n\tJUPYTERHUB_API_TOKEN:"
    read JUPYTERHUB_API_TOKEN
    exit_if_empty $JUPYTERHUB_API_TOKEN

	export JUPYTERHUB_API_TOKEN
}

function exit_if_empty {
    if [ -z "$1" ];
    then
        printf "\n\n${PURPLE}Exiting.${RESET}"
        exit 0
    fi
}

function msg {
	printf "\n${BLUE}$1${RESET}"
}

exit_if_error() {
	if [ $(($(echo "${PIPESTATUS[@]}" | tr -s ' ' +))) -ne 0 ]; then
		if [ ! -z "$1" ];
		then
			echo ""
			echo -e "${RED}ERROR: ${1}$RESET"
			echo ""
		fi
		exit 1
	fi
}

exit_with_error() {
	echo ""
	echo -e "${RED}ERROR: ${1}$RESET"
	echo ""
	exit 1
}

# Configurable Properties for perosnal development environment
DOCKER_REPOSITORY=pbooth01
DOCKER_IMAGE_NAME=pb-hub
BASE_HUB_IMAGE_TAG=1.0.0

# Based on your Docker Configuration Information your image is automatically generated for you.
DOCKER_IMAGE_FULL=$DOCKER_REPOSITORY/$DOCKER_IMAGE_NAME:$BASE_HUB_IMAGE_TAG
export DOCKER_IMAGE_FULL
