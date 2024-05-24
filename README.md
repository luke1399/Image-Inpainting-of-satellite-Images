# Image-Inpainting-of-satellite-Images

The project is about image inpainting, which consists of filling deteriorated or missing parts of a picture to reconstruct a complete image.

The dataset taken into consideration is the TensorFlow EuroSAT dataset based on Sentinel-2 satellite images, in the the rgb version. This consists of 27000 images, at resolution 64x64.

A portion of the image is randomly masked according to the procedure described in the attached notebook. The goal was to reconstruct the full image.

After several attempts, the best architecture that I came up with is a classical UNET.
The attempts I made include a GAN, which was obtaining promising results (around 0.88 accuracy on test set), but with training time and complexity not making it worth, with a single epoch lasting up to 4 minute, with a minum need of training the network for 15-20 epochs; a WNET, where basically I thought of using two UNETs instead of one, because the results the single UNET produces look like blurry images, so I thought of training a second one, but with focus on making sharper images from prediction of the first net. The results were slightly better, but the training became much slower and complex.

This network architecture is designed following a classical UNET structure, with a focus on hierarchical feature extraction through a deep encoder-decoder framework.

The network begins with a series of convolutional layers in the encoder part, gradually reducing spatial dimensions while increasing feature depth. This hierarchy allows the network to capture increasingly abstract representations of the input image. 

Pooling layers are interspersed throughout the encoder, strategically placed to downsample feature maps, reducing computational complexity while retaining important spatial information

In the decoder section, upsampling operations are employed to gradually reconstruct the spatial dimensions of the encoded features. These upsampling layers help recover spatial details lost during the encoding process, ultimately producing a high-resolution output.

Skip connections, as usually in a UNET, are incorporated to facilitate information flow between corresponding encoder and decoder layers. These connections help alleviate the vanishing gradient problem and promote feature reuse, enabling the network to reconstruct fine details accurately.

Each decoder block consists of convolutional layers followed by ReLU activations, similar to the encoder. These layers refine the reconstructed features, gradually bringing them closer to the ground truth image.

Therefore, a Lambda layer is applied at the very end. It takes two inputs: the masked image given in input to the network and the output of the last layer of the network (note that this last layer uses a **sigmoid** activation function to squash the values within the range [0,1]). So, the Lambda layer is in charge of replacing the masked input region with the new generated values in the respective position. 
