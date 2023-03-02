

# A geometrically aware auto-encoder for multi-texture synthesis



[Arxiv](https://arxiv.org/pdf/2302.01616.pdf) 

### Official pytorch implementation of the paper: "A geometrically aware auto-encoder for multi-texture synthesis"




## Random samples from a *single* image

With SinGAN, you can train a generative model from a single natural image, and then generate random samples from the given image, for example:



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



This command will create a conda environment called TextureAE with the required dependencies:



```

source install_env.sh

```



Download the weights for the VGG19 network at,





###  Train




```

python code/train.py --name <name_of_the_experiment> 

```



Please refer to code/config.py for described additional arguments.





### Inference



```

python code/inference.py --name <name_of_the_experiment> 

```

This will fill the folder ./runs/name_of_the_experiment/inference/ with inference results



'--name',type=str, default='test',help='name of your experiment, as seen in the run folder')

    parser.add_argument('--text',type=str, default='yourtext')

    parser.add_argument('--n_gif',type=int, default=50,help='lenght of gifs animations')




###  Random samples of arbitrary sizes

To generate random samples of arbitrary sizes, please first train SinGAN model on the desired image (as described above), then run 



```

python random_samples.py --input_name <training_image_file_name> --mode random_samples_arbitrary_sizes --scale_h <horizontal scaling factor> --scale_v <vertical scaling factor>

```



###  Animation from a single image



To generate short animation from a single image, run



```

python animation.py --input_name <input_file_name> 

```



This will automatically start a new training phase with noise padding mode.



###  Harmonization



To harmonize a pasted object into an image (See example in Fig. 13 in [our paper](https://arxiv.org/pdf/1905.01164.pdf)), please first train SinGAN model on the desired background image (as described above), then save the naively pasted reference image and it's binary mask under "Input/Harmonization" (see saved images for an example). Run the command



```

python harmonization.py --input_name <training_image_file_name> --ref_name <naively_pasted_reference_image_file_name> --harmonization_start_scale <scale to inject>



```



Please note that different injection scale will produce different harmonization effects. The coarsest injection scale equals 1. 



###  Editing



To edit an image, (See example in Fig. 12 in [our paper](https://arxiv.org/pdf/1905.01164.pdf)), please first train SinGAN model on the desired non-edited image (as described above), then save the naive edit as a reference image under "Input/Editing" with a corresponding binary map (see saved images for an example). Run the command



```

python editing.py --input_name <training_image_file_name> --ref_name <edited_image_file_name> --editing_start_scale <scale to inject>



```

both the masked and unmasked output will be saved.

Here as well, different injection scale will produce different editing effects. The coarsest injection scale equals 1. 



###  Paint to Image



To transfer a paint into a realistic image (See example in Fig. 11 in [our paper](https://arxiv.org/pdf/1905.01164.pdf)), please first train SinGAN model on the desired image (as described above), then save your paint under "Input/Paint", and run the command



```

python paint2image.py --input_name <training_image_file_name> --ref_name <paint_image_file_name> --paint_start_scale <scale to inject>



```

Here as well, different injection scale will produce different editing effects. The coarsest injection scale equals 1. 



Advanced option: Specify quantization_flag to be True, to re-train *only* the injection level of the model, to get a on a color-quantized version of upsampled generated images from the previous scale. For some images, this might lead to more realistic results.



### Super Resolution

To super resolve an image, please run:

```

python SR.py --input_name <LR_image_file_name>

```

This will automatically train a SinGAN model correspond to 4x upsampling factor (if not exist already).

For different SR factors, please specify it using the parameter `--sr_factor` when calling the function.

SinGAN's results on the BSD100 dataset can be download from the 'Downloads' folder.



## Additional Data and Functions



### Single Image Fréchet Inception Distance (SIFID score)

To calculate the SIFID between real images and their corresponding fake samples, please run:

```

python SIFID/sifid_score.py --path2real <real images path> --path2fake <fake images path> 

```  

Make sure that each of the fake images file name is identical to its corresponding real image file name. Images should be saved in `.jpg` format.



### Super Resolution Results

SinGAN's SR results on the BSD100 dataset can be download from the 'Downloads' folder.



### User Study

The data used for the user study can be found in the Downloads folder. 



real folder: 50 real images, randomly picked from the [places database](http://places.csail.mit.edu/)



fake_high_variance folder: random samples starting from n=N for each of the real images 



fake_mid_variance folder: random samples starting from n=N-1 for each of the real images 



For additional details please see section 3.1 in our [paper](https://arxiv.org/pdf/1905.01164.pdf)


