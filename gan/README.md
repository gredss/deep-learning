# Synthetic Image Generation with GAN and FID Evaluation

This project focuses on developing and evaluating Generative Adversarial Network (GAN) architectures for synthetic image generation. The primary goal is to enhance the quality of generated images so that they resemble real-world data as closely as possible. Evaluation is conducted using the Frechet Inception Distance (FID), a widely adopted metric that quantifies the similarity between generated and real image distributions.

## Objective

To build a robust GAN capable of synthesizing realistic images and to assess the improvements through systematic architectural modifications and empirical FID evaluations.

---

## Project Structure

### 1. Baseline Model

The baseline GAN is composed of the following components:

#### Generator

* **Architecture**: Sequential model with three convolutional layers using 3x3 kernels and stride 1. Each layer uses ReLU activations, except for the output layer, which uses Tanh. Upsampling is performed using `UpSampling2D` followed by `Cropping2D` to achieve the target resolution of 100x100 pixels.
* **Latent Input**: Random noise vectors sampled from a standard Gaussian distribution.

#### Discriminator

* **Architecture**: A three-layer convolutional network using ReLU activation functions, followed by a flattening operation and a single dense output neuron with Sigmoid activation.

### 2. Modified Model

To improve upon the baseline, several modifications were introduced:

#### Modified Generator

* **Deeper Network**: Utilizes `Conv2DTranspose` for learned upsampling.
* **Stabilization**: Batch normalization and LeakyReLU activations are applied throughout the network.
* **Output Size**: Consistently outputs RGB images of shape 100x100x3.

#### Modified Discriminator

* **Enhanced Depth**: Incorporates additional convolutional layers.
* **Regularization**: Uses `GaussianNoise` to mitigate overfitting, particularly effective on limited datasets.

---

## Model Integration

Both generator and discriminator components are integrated into a complete GAN architecture. During training, the discriminator’s weights are frozen when updating the generator to maintain training stability.

---

## Evaluation Methodology

### Frechet Inception Distance (FID)

FID is computed to measure the similarity between real and generated images:

* **Feature Extraction**: InceptionV3 is used to extract features from 299x299 resized images.
* **Distribution Comparison**: The mean and covariance of extracted features are used to calculate the FID score between two Gaussian distributions.

---

## Data Preparation

### Dataset

* Images are extracted from a ZIP archive and resized to 100x100 pixels.
* Pixel values are normalized to the range \[-1, 1].

### Augmentation

To enrich the training data and reduce overfitting, basic augmentation techniques are applied to real images only. These include:

* Horizontal flipping
* Random rotations
* Minor zoom transformations

---

## Training Procedure

The training pipeline includes:

1. **Discriminator Training**: Real and fake images are used to train the discriminator using noisy labels to improve robustness.
2. **Generator Training**: The generator is updated via the combined GAN model.
3. **FID Monitoring**: Every 200 epochs, the generator is evaluated by computing FID against real images to monitor progress.

Training was conducted for 1,000 epochs with a batch size of 32.

---

## Implementation Notes

* Optimizers: Adam with learning rate 0.0002 and beta\_1=0.5 is used for both generator and discriminator in the baseline. For the modified model, exponential learning rate decay is implemented.
* Loss Function: Binary cross-entropy is used for both discriminator and GAN losses.
* Trainable Parameters: Proper model freezing was verified to ensure only generator parameters were updated during GAN training.

---

## Results

| Model    | FID @ Epoch 1000 |
| -------- | ---------------- |
| Baseline | 513.41           |
| Modified | 474.33           |

A reduction of approximately 39.08 in FID demonstrates the improved capability of the modified generator to produce images that are perceptually closer to real ones.

---

## Limitations

Training was limited to 1,000 epochs due to computational constraints on Google Colab. Nonetheless, the consistent decline in FID across evaluation intervals indicates that both models were still converging and had not reached their performance ceilings.

---

## Conclusion

The architectural modifications, including deeper layers, batch normalization, and noise-based regularization, significantly improved the visual fidelity of the generated images as confirmed by FID scores. The modified GAN demonstrates a more stable and effective training dynamic compared to the baseline.

### Potential Extensions

Further improvements could be explored through:

* Adjusting the latent space dimensionality.
* Implementing advanced normalization techniques such as Spectral Normalization.
* Exploring alternative upsampling mechanisms, including PixelShuffle or a hybrid of `UpSampling2D` and `Conv2D`.


---

## Citation

This work was conducted as part of an experimental study in generative modeling and is subject to ongoing development. Please cite appropriately when referencing or adapting the codebase.
