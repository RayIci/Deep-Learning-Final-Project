# Deep Learning Final Project: Text to Image Synthesis with Generative Adversarial Networks

This is the final project for the Deep Learning exam for University of Genova. This projects
aim to get a better and deeper understanding on how text to image synthesis with Generative Adversarial Networks works.

The paper from which this laboratory is inspired is [Generative Adversarial Text to Image Synthesis](https://paperswithcode.com/paper/generative-adversarial-text-to-image).

## What do you find in this project

In this project you will find the final implementation of the networks presented in the paper and a notebook file explaining the done experiments.

## How to run the project

To run the app make sure that you have `Docker` installed on you computer (if not [Docker link here](https://www.docker.com/)).
With docker engine up and running execute the following command:

``` bash
    docker compose up --build
```

## Project Structure
Data Folder : data captions in .t7 format and images in the images folder organized in subdirectories based on species.
examples Folder : ipynb files useful to understand how the net and the data are pre-processed and trained.
Paper Folder : It containes the original Paper file and the short Report  
src Folder : it contains all the code that has been produced to train the Network(s)