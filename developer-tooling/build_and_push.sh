#!/bin/bash

source ./buildtools.sh


printf "\nBuilding repository: $DOCKER_IMAGE_FULL"
printf "\n"

docker build -t $DOCKER_IMAGE_FULL .
exit_if_error

printf "\nPushing repository: $DOCKER_IMAGE_FULL"
printf "\n"

docker push $DOCKER_IMAGE_FULL
exit_if_error