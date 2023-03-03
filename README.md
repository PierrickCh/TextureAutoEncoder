

# A geometrically aware auto-encoder for multi-texture synthesis
[Pierrick Chatillon]() | [Yann Gousseau](https://perso.telecom-paristech.fr/gousseau/) | [Sidonie Lefebvre]()


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



### Installation

The 'Macro500' dataset provided by the authors of [Towards Universal Texture Synthesis by Combining Spatial Noise Injection with Texton Broacasting in StyleGAN-2](https://github.com/JueLin/textureSynthesis-stylegan2-pytorch) can be downloaded [here](https://drive.google.com/file/d/15tM8vlc-ZnYVQpyjf63QyQQ9inqtijmt/view?usp=sharing).

These commands will create a conda environment called TextureAE with the required dependencies, then place you in it :
```
conda env create -f requirements.yml
conda activate TextureAE
```

Download and unzip the weights for the VGG19 network from [Gatys et. al.](https://arxiv.org/abs/1505.07376) by running, the following command:

```
bash get_vgg_weights.sh
```

Alternatively you can download them directly from [here](httx ps://drive.google.com/file/d/1tdfMcwSogBfAkMcLVJd9z_frsEg8fxAB/view?usp=sharing) and unzip in the main directory.



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
