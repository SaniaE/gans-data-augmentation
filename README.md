# Synthetic Data Generation for Petrol Pump Satellite Imagery with GANs and VAEs 

This repository explores various generative models, focusing on Generative Adversarial Networks (GANs) and Variational Autoencoders (VAEs), to generate realistic synthetic satellite imagery of petrol pumps. By leveraging different GAN architectures, such as Simple GAN, DCGAN, and WPGAN, along with a VAE model, the project explores the ability of generative models to expand both the diversity and volume of a dataset. These synthetic images are aimed at overcoming data scarcity, thereby improving the robustness and generalization of machine learning models trained on the augmented dataset.

<p align="center">
  <img src="images/data_sample.png" alt="dataset-sample"/>
</p>

The experiments conducted examine different generative model architectures, loss functions, and optimizations, aiming to identify the best approaches for stabilizing training and producing high-quality synthetic samples.

## Models Implemented
- **Simple GAN (Generative Adversarial Network):** A basic GAN model consisting of a generator and a discriminator. The generator creates synthetic images, while the discriminator classifies them as real or fake.

- **DCGAN (Deep Convolutional GAN):** An extension of GANs utilizing convolutional layers for both the generator and discriminator. 

- **WPGAN (Wasserstein GAN):** A variant of GANs designed to improve stability and convergence through the Wasserstein loss function with gradient penalty and the addition of a critic instead of a discriminator.

- **Variational Autoencoder (VAE):** A probabilistic generative model that learns a latent space for data generation, leveraging a combination of reconstruction loss and KL divergence to regularize the model.

## Key Features

- **GPU Implementation:** The models are implemented with GPU support for faster training.

- **Training Loops:** Custom training loops for each model type to ensure effective training and convergence.

- **Weights Initialization:** Proper initialization techniques to help stabilize training and avoid vanishing/exploding gradients.

- **Checkpointing:** Save and load model checkpoints during training to ensure that the model can resume from the last state in case of interruptions.

- **Visualizations:** Training progress visualizations for both the generator’s outputs and the loss curves.

## Results 

Expanding a scarce dataset of high-resolution satellite imagery comes with unique challenges: aerial data features complex multi-scale textures, high-frequency spatial details (like roads, rooftops, and parking bays), and severe vulnerability to mode collapse when trained on limited samples.

<p align="center">
  <table align="center" style="border-collapse: collapse; border: none;">
    <tr>
      <td align="center" style="border: none;"><img src="images/step_500.png" width="100%" alt="Step 500"/><br><sub><b>Step 500</b> (Early Collapse)</sub></td>
      <td align="center" style="border: none;"><img src="images/step_2500.png" width="100%" alt="Step 2500"/><br><sub><b>Step 2500</b> (Structural Emergence)</sub></td>
      <td align="center" style="border: none;"><img src="images/step_5500.png" width="100%" alt="Step 5500"/><br><sub><b>Step 5500</b> (Geometry Refinement)</sub></td>
    </tr>
    <tr>
      <td align="center" style="border: none;"><img src="images/step_8500.png" width="100%" alt="Step 8500"/><br><sub><b>Step 8500</b> (Feature Stabilization)</sub></td>
      <td align="center" style="border: none;"><img src="images/step_10500.png" width="100%" alt="Step 10500"/><br><sub><b>Step 10500</b> (High-Frequency Details)</sub></td>
      <td align="center" style="border: none;"><img src="images/step_12500.png" width="100%" alt="Step 12500"/><br><sub><b>Step 12500</b> (Converged Fidelity)</sub></td>
    </tr>
  </table>
</p>


**1. Moving Beyond Standard GANs & DCGANs:** 

- Early experiments with Simple GANs and standard DCGANs suffered from unstable training dynamics, vanishing gradients, and structural blurriness. Cross-entropy losses often led to the discriminator prematurely overpowering the generator, resulting in flat, unrecoverable color blobs.

**2. The WGAN-GP Advantage:**

- To establish stable convergence, the architecture transitioned to a Wasserstein GAN with Gradient Penalty (WGAN-GP). By replacing the traditional discriminator with a Critic that optimizes the Wasserstein distance (Earth Mover's Distance) alongside a gradient penalty constraint, the model gained a smooth, continuous loss landscape. This allowed the generator to steadily learn meaningful structural gradients rather than stalling out mid-training.

**3. Stabilizing High-Resolution Training with DiffAugment:**

- Even with WGAN-GP, training on limited high-resolution satellite tiles risked critic overfitting—where the model memorizes the small training set rather than learning true data distribution. Integrating applied transformations (color jitter, spatial translation, and cutout) directly to both real and fake images without disrupting backpropagation. This acted as a powerful regularizer, allowing the generator to achieve sharp, high-fidelity spatial coherence at $256 \times 256$ resolution without leaking artifacts into downstream tasks.

### Installation

1. Clone the repository and change directory

    ```bash
    git clone https://github.com/SaniaE/gans-data-augmentation.git
    cd path/to/repo/clone
    ```
2. Create and activate a new virtual enviornment

    ```bash
    python -m venv <environment_name>
    .\<environment_name>\Scripts\activate   # For Windows
    source <environment_name>/bin/activate  # For macOS/Linux
    ```
3. Add virtual environment to the Python Kernel 

    ```bash
    python -m pip install --upgrade pip 
    pip install ipykernel 
    python -m ipykernel install --user --name=<kernel_name>
    ```
4. Install dependencies

    ```bash
    pip install -r requirements.txt
    ```
5. Launch Jupyter Notebook 

    ```bash
    jupyter notebook