# Improved Local Development Utilizing Planetary Computer's Scalable Compute Resources



The developer tooling section of this repo is dedicated to improving the local development experience with planetary computer. The work inside of this section builds on top of the great work that has been completed by the Open Source pangeo-data team: https://github.com/pangeo-data/pangeo-docker-images#pangeo-docker-images

The developer-tooling section takes their exiting planetary-computer docker image and creates a wrapper around it that allows individuals to customize the image and have increased control over their local development experience.

### Architectural Diagram

![local compute architecture](./local-compute-architecture.png)

The architectural diagram is from the planetary computer website. https://planetarycomputer.microsoft.com/docs/concepts/computing/ and displays what a local devlopment environemnt consists of. As a developer you are running code found locally on your machine, but accessing planetary computes computational resources to allow your code to run much faster.

This provides many benefits, but the out of the box behavior can be limiting in that you are not in control of the libraries that end up in your Dask cluster. You can view them



The benefits of this tooling are:

- Ability to easily configure the workers on your dask cluster with the libraries you need.
- Provide an easily reproducible development  environment that can be shared across your team.
- Develop your own libraries on the fly and integrate them into your devlopment process.
- Automatic integration with Source control.