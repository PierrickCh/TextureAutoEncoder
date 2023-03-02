

# A geometrically aware auto-encoder for multi-texture synthesis



[Arxiv](https://arxiv.org/pdf/2302.01616.pdf) 

### Official pytorch implementation of the paper: "A geometrically aware auto-encoder for multi-texture synthesis"





![](imgs/teaser.PNG)




### Citation

If you use this code for your research, please cite our paper:

```

@misc{https://doi.org/10.48550/arxiv.2302.01616,
  doi = {10.48550/ARXIV.2302.01616},
  url = {https://arxiv.org/abs/2302.01616},
  author = {Chatillon, Pierrick and Gousseau, Yann and Lefebvre, Sidonie},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), Artificial Intelligence (cs.AI), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {A geometrically aware auto-encoder for multi-texture synthesis},
  publisher = {arXiv},
  year = {2023},
  copyright = {arXiv.org perpetual, non-exclusive license}

}


```



## Code



### Installation



These commands will create a conda environment called TextureAE with the required dependencies, then place you in it :


```
conda env create -f requirements.yml
conda activate TextureAE
```

Download and unzip the weights for the VGG19 network from Gatys et. al. by running:

```
bash get_vgg_weights.sh
```


###  Train




```
python code/train.py --name <name_of_the_experiment> --dataset_folder <path_to_dataset>
```


Please refer to code/config.py for described additional arguments.
All the models, arguments and tensorboard logs for an experiments are stored under the same folder ./runs/name_of_the_experiment/

### Inference



```
python code/inference.py --name <name_of_the_experiment> 
```

This will fill the folder ./runs/name_of_the_experiment/inference/ with inference results
Additional parameters are:
--text   is a string that will be used as guide to perform spacial editing.
--n_gif  denotes the lenght of gifs animations.
