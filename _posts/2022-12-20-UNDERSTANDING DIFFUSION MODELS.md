# UNDERSTANDING DIFFUSION MODELS 

1. __DIFFUSION MODELS + DENOISING DIFFUSION MODEL EXPLAINED
2. __3 POPULAR SOTA DIFFUSION MODELS
3. __SOURCE CODE FOR IMPLEMENTATION
4. __CONCLUSION
5. __REFERENCES

It is no news that Diffusion models are the best of the best Generative models that have surpassed GANS in Computer Vision and other Multi-Modal Learning tasks. These models have created significant feats in arts, generative NFT's, Interior design, personalized avatar creations, 3d modelling, urban design etc.

This post is going to be a two part series where I first of all expose how Denoising Diffusion models work which is the foundation upon which all these SOTA models (GLIDE, DALLE.2, SD) are being built. 

### DIFFUSION MODELS

DIffusion model are a class of state-of-the art deep generative models that have shown impressive results on various tasks ranging from multi-modal modelling Vision to NLP to waveform signal Processing even to 3d object generation (but not comparable to humans). These models have achieved very impressive quality and diversity of sample synthesis than other state-of-the art genrative models like GANS, VAEs, Normalizing Flows etc.

Diffusion model are generative models meaning it is a model used to generate data similar to the data which it was trained on. A generative model attempts to learn the data distribution over $p(x|y)$ where $x$ is the image and $y$ the labels in order to generate novel samples. Onced trained, a generative model can generate novel samples from an approximation of $p(x|y)$, denoted $p_\theta({x|y})$ 


### Areas Where Diffusion Models have been applied to
1. Image Super Resolution
2. Image Inpainting
3. Image Outpainting
4. Image to Image generation
5. Semantic Segmentation
6. Point Cloud Completion and Generation
7. Text-to-Image Generation
8. Text-to-Audio Generation


## STATE OF THE ART DIFFUSION MODELS (cherrypicked)
  1. DALLE 2
  2. STABLE DIFFUSION 

## UNDERSTANDING THE FOUNDATIONS (DDPMs) 
Originally introduced by Sohl-Dickstein et al. (2015) in his paper "Deep unsupervised learning using nonequilibrium thermodynamics" which was followed up by Ho et al. (2020) in his paper "Denoising diffusion probabilistic models" (DDPMs), as well every other paper that has built on it. It is a latent variable generative model inspired by non-equilibrum thermodynamics in physics where samples generated is made possible by denoising process. 


1. Denoising Diffusion models is a two step model consisting of a forward process also called noising process and a backward process also know called denoisng or reverse noising process.
2. It works by adding gaussian noise to input data which could be an image following a Markov Chain. 
3. The forward process follows a Markov chain where a little bit of gaussian noise is added to the input image which progressively disturbs the data distribution untill it is completely destroyed or unrecognised from gaussian noise.
4. The backward or reverse process learns to restore the data to its original form.
5. Reversing or removing the noise means recovering the values of the pixels, so that the resulting image will be similar to the original image. The reverse diffusion process takes a noisy image and learns to generate a less noisy version of that image, this process will be repeated until noise is converted to data. **OR**
6. In order to generate new data, standard gaussian noise is used to perform the  denoising. This process will be repeated until noise is converted to data.
7. The recovering or noise reversal is parameterized because it uses a neural network. The task of the Neural Network is to predict the noise that was added in a given image.
8. The objective function of this model was simplified to a MSE Loss i.e given a noised image, the noise added is predicted and then this predicted noise is subtracted from the noise to get the real image. This is basically what is happenning when training a Diffusion Model, It is learning to denoise.
9. In DDPMs the forward process is fixed while the reverse process is what needs learning meaning we need to train only a single neural network.
10. The important components of a denoising diffusion mdoels include:
	1. UNET Architecture
	2. positional emebedidng
	3. noise scheduler
	4. attention mechanism
	 

## MATHEMATICAL EXPLANATION

### A. NOISING: 
#### DEFINING THE PROCESS FORMALLY
###### FORWARD PROCESS:
The forward diffusion process starts from data (image) and generates this intermediate noisy images by simply adding noise one step at a time
At every step, a normal distribution will be used to generate an image conditioned on the previous image.

the normal distribution which is represented as $q(x_t|x_{t-1}) = \mathcal{N(x_t; \sqrt{1-\beta_t{x_{t-1}}\beta_t}I)}$ is going to take $x_{t-1}$ the prevoius step and generate $x_t$ the current step. It takes $x_0$ and it generates $x_1$ 

A normal distribution over the current step $x_t$ where the mean is $(\mathcal{\sqrt{1-\beta_t}})$ times the image at the prevoius time step which is ${x_{t-1}}$ and ${\beta_t}I$ represents the variance scheduler which in the real sense is a very small positive scalar value $0.001$  

This normal distribution, $\mathcal{N(x_t; {\sqrt{1-\beta_t}{x_{t-1}}, \beta_t}I)}$  takes the image at the previous step, rescales the pixel values in this image and then adds tiny bit of noise via the variance scheduler "per time step"

###### JOINT DISTRIBUTION:
A joint distibution can also be defined for all the samples generated in the forward process starting from $x_1$ all the way to $x_T$.  The joint distribution which is the samples conditioned on $x_0$ is the cumulateive product of the conditionals that are formed at each step as such $q(x_1,...,x_T|x_0)$ defines the joint distribution of all the samples that will be generated in the forward markov process
$$q(x_1,...,x_T|x_0) = \prod^T_{t=1}   {q(x_t|x_{t-1})}$$
###### Speed?
Why can't we use $x_0$ our input image to generate noisy samples at any time step say $x_{10}$. Simply put can't we use $x_0$ to generate $x_{10}$ ?. 
We can do that by making
${\alpha}_t$ = $1-\beta_t$ , 
then $\bar{\alpha}_t$ which is the cumulative product of ${\alpha}_t$ now becomes
$$\bar{\alpha}_t = \prod^{t}_{s=1}(1-\beta_s)$$
In order to answer the speed question we can then rewrite the original formular as follows:
		$q(x_t|x_0) = \mathcal{N(x_t;\sqrt{\bar{\alpha_t}}x_0,(1-{\bar{\alpha_t}})I)}$
Using the reparameterization trick we can sample $x_t$ as follows
$x_t$ = $\sqrt{\bar{\alpha}}\space x_0 + \sqrt{1-\bar{\alpha_t}}\space \epsilon$  where $\epsilon \sim{\mathcal{N(0,1)}}$  and ${1-\bar{\alpha_t}}$ is our noise schedule at any time step, as such given $x_0$ we can draw samples at any time step $t$. 

It should also be noted that the forward diffusion process is defined such that as  $(x_T\mid{x_0})$ approaches infinity it becomes indistinguishable from standard normal distribution $\mathcal{N({x_T;(0,1)})}$. 
___
1. The forward chain pertubs the data distribution by gradually adding Gaussian noise to the ground truth image with a pre-designed schedule until the data distribution converges to a given prior, i.e., a standard Gaussian Distribution -- (Isometric Gaussian).
$$ q(x_1,...,x_T|x_0) = \prod^T_{t=1}   {q(x_t|x_{t-1})} ---(1)$$
$$ q(x_t|x_{t-1}) = \mathcal{N(x_t; \sqrt{1-\beta_t{x_{t-1}}\beta_t}I)}   ---(2)$$ 
$q(x_t)$ is used to denote the distributions of latent variables $x_t$ in the forward process.

The noising process defined in Eq.(2) allows us to sample an arbitrary step of the noised latents directly conditioned on the input $x_o$.
	Where $\alpha_t$ = $1-\beta_t$ and $\bar{\alpha_t}$ = $\prod^t_{s=0}$ $\bar{\alpha_s}$, we can wite the marginal as:
	$$ q(x_t|x_0) = \mathcal{N(x_t;\sqrt{\bar{\alpha_t}}x_0,(1-{\bar{\alpha_t}})I)} $$
$$ x_t = \sqrt{\bar{\alpha_t}}x_0 + \sqrt{1-{\bar{\alpha_t}}}\epsilon $$
When $\bar{\alpha}_t$ approximates 0, $x_T$ is practically indistinguisahble from pure Gaussian noise: $p(x_T)\approx$ $\mathcal{N}(x_T;0,1)$.



### B. DENOISING:  What is Means to Reverse The Noise?

#### DENOISING: DEFINING THE GENERATIVE MODEL BY DENOISING

In order to generate data from a diffusion model, we will start from pure noise which is a standard normal distribution with zero mean and unit variance and generates data by denoising one step at a time.
					
					
As such $p(x_T)$ = $\mathcal{N(x_T;(0,1))}$  is the distribution of data at the end of the forward diffusion process.
the parametric denoising distribution can be defined as follows $p_\theta(x_{t-1}\mid{x_t})$ = $\mathcal{N({x_{t-1};{\mu}_\theta{(x_t,t)},{\Sigma_\theta({x_t,{t}}})})}$ apart from the sample $x_t$ at time $t$ the model also takes $t$ as input in order to account for the different noise levels at different time steps in the forward process noise schedule so that the model can learn to undo this individually

#### Joint Distribution

The joint distribution can be written as 
It is the product of the base distibution $p{(x_T)}$ and the product of the conditionals which still follows a markov process 

		$x_0\Leftarrow \cdots  \Leftarrow \cdots  \Leftarrow x_{T-1}\cdots  \Leftarrow x_T$

$$p_\theta(x_{0:T})=p(x_T)\prod^T_{t=1}p_\theta(x_{t-1}|x_t)$$

The model is now tasked with learning the probability density of an earlier time step given the current timestep. During this process iteration is done from pure noise $x_T$ to $x_0$ our final image 

Starting from sampled noise, the diffusion model performs $T$ denoising steps until a sharp image is formed. 

The denosing process produces a series of intermediate images with decreasing levels of noise, denote as $x_T, x_{T-1},...x_0$,

Given only $x_T$ which is indistinguishable from gaussian noise we can get $x_0$ an output image. 

The reverse process takes the completely noised image and learns to gradually revert the Markov chain of noise corruption to the ground truth. The reversed process is then written as follows: 
$$ p_\theta({x_{t-1}|x_t}) = \mathcal{N}(x_{t-1};\mu(x_t,t, {\Sigma}_\theta(x_t,t)) $$
___
In Denoising Diffusion Models,  The Noise $\epsilon$ is what is Predicted and this is done by Optimizing the variational upper bound on the negative log-likelihood.
___
$$
\\E\ [-\log{p_{\theta}}(x_0)]< \\ E_q[-\log\frac{p_\theta({x_{0:T}})}{q(x_{1:T}|x_0)}]--(7) $$
$$ \\ E_q[-\log{p{(x_T)}}-\sum_{t>=1}{\log\frac{p_\theta({x_{0:T}})}{q(x_{1:T}|x_0)}} ]-- (8)$$
$$= -L_{VLB}$$
___
Reparametrization have been appplied to Eq. (8), which results in the general objective below:
$$  E_{t\sim{\mathcal{U(0,T),x_0{\sim{q{(x_0),\epsilon\sim{\mathcal{N(0,1)}}}}}}}}[\lambda{(t)}||\epsilon-{\epsilon_\theta}(x_t,t)||^2]   --(10)$$
___
The neural network ${\epsilon_\theta}(x_t,t)$ predicts $\epsilon$ by minimizing the loss = ${||\epsilon-\epsilon_\theta}(x_t,t)||^2$ which is the $L_2$ Loss 
	INTUITIVELY: Given a noised image, the noise added is predicted and then this predicted noise is subtracted from the noise to get the real image. This is basically what is happenning when training a Diffusion Model, It is learning to denoise.

____
___
Both the forward Diffusion processes $q(x_t|x_{t-1})$ and the backward or reconstruction process $q(x_{t-1}|x_t)$ are modelled as the products of Markov transition probabilities:
$$q(x_{0:T}) = q(x_0)\prod_{t=1}^T{q(x_t|x_{t-1})}, p_\theta(x_{T:0}) = p(x_{T})\prod_{t=T}^1{p_{\theta}(x_{t-1}|x_t)},$$
$q(x_0)$ is the real data distribution


#### IMPORTant
Diffusion models are latent variable models
	Latent variables$:$ = $x_1,x_{2},x_3,x_4,\cdots x_T$  
	Observed variables$:$ $x_0$
	

#### TRAINING A DENOSING DIFFUSION PROBABILISTIC MODEL
The reverse step process is only tasked with learning the mean while its variance is set to a constant
______
##### OBJECTIVE FUNCTION OF A DDPM
$$E_{x_0{\sim{q{(x_0),\epsilon\sim{\mathcal{N(0,1)t\sim{\mathcal{U(0,T),}}}}}}}}[||\epsilon-{\epsilon_\theta}(\sqrt{\bar{\alpha}}\space x_0 + \sqrt{1-\bar{\alpha_t}}\space \epsilon),t||^2]$$
	Where $x_t = \sqrt{\bar{\alpha}}\space x_0 + \sqrt{1-\bar{\alpha_t}}\space \epsilon)$

                         $E_{x_0{\sim{q{(x_0),\epsilon\sim{\mathcal{N(0,1)t\sim{\mathcal{U(0,T),}}}}}}}}[||\epsilon-{\epsilon_\theta}(x_t,t)||^2]$

Our loss funtion finally looks like this$:$
			                $L_{simple}={E_{x_0,t,\epsilon}}[||\epsilon_-\epsilon_{\theta}(x_t,t)||^2]$
________
###### The Training Algorithm looks like this: ![[trainDIFF.PNG]]

#### IMAGE GENERATION or SAMLING FROM A DENOSING DIFFUSION PROBABILISTIC MODEL 

_______
###### The Sampling Algorithm looks like this: ![[sampleDIFF.PNG]]



## IMPORTANT COMPONENTS OF A DENOISING DIFFUSION MODEL
1. UNET ARCHITECTURE: The UNET Architecture is a Convolutional Network originally developed for biomedical image segmentation. It has a U Shape and it does downsampling on one part and upsampling on the other side and all its operation are convolution based.

2. POSITIONAL EMBEDDING: Positional embedding as originally used in the Attention is all you need paper was utilized for train the neural network has shared parameters across time which means it can't distinguish between various timesteps as such it needs to filter the noise across images with different noise intensities.                    
    
    PE is added as a way of encoding positional information into the UNET Model to help distinguish the various noise intensities within the markov chain process

	The positional embedding is added as additional information in the downsample, middle and upsample block of the UNET Model.
		
	PE are wave frequencies used to capture positonal information at both odd $sin()$ and even $cosine()$ positions and this embeddings can be calculated as follows.
	
	$PE(_{pos,2i}) = sin(pos\div{1000^{2i/d}})$
	$PE(_{pos,2i+1}) = cos(pos\div{1000^{2i/d}})$


3. NOISE SCHEDULER: This is just a technique that makes possible an iterative addition of noise to an image of adding noise.

4. ATTENTION MECHANISM: It is a mechanism that makes a model selectively focus on the varying parts of the inputs. It assigns weights to the different positions which indicates the importance of the input sequence for generating outputs by calculating attention scores. Originally in the transformers it was used both in the encoder and decoder parts of the Model.



## IMPLEMENTING DENOISNG DIFFUSION MODEL WITH PYTORCH
It should be noted that I will not run training because it is needless and basically useless to train this kind of model because it is not state of the art as compared to the other models briefly explained above except you have a gpu compute where you can run it for over 1000 epochs. In **part 2** of this article you will see a detailed handson code for finetuning stable diffusion using dreambooth.

```jupyter
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from torch.utils.data import Subset
import copy
```

```jupyter
def show_images(images, title=""):
    """Shows the provided images as sub-pictures in a square"""
    images = [np.clip(im.permute(1,2,0).numpy(),0,1) for im in images]

  

    # Defining number of rows and columns
    fig = plt.figure(figsize=(8, 8))
    rows = int(len(images) ** (1 / 2))
    cols = round(len(images) / rows) 

    # Populating figure with sub-plots
    idx = 0
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, idx + 1)

            if idx < len(images):
                plt.imshow(images[idx])
                plt.axis('off')
                idx += 1

    fig.suptitle(title, fontsize=30)
    # Showing the figure
    plt.show()
```

```jupyter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device
```

```jupyter
def sinusoidal_embedding(n, d):
    # Returns the standard positional embedding
    embedding = torch.tensor([[i / 10_000 ** (2 * j / d) for j in range(d)] for i in range(n)])
    sin_mask = torch.arange(0, n, 2)
    embedding[sin_mask] = torch.sin(embedding[sin_mask])
    embedding[1 - sin_mask] = torch.cos(embedding[sin_mask])
    return embedding
```

```jupyter
class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

  
    def forward(self, x):
        x = self.conv(x)
        return x


class down_layer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down_layer, self).__init__()
        self.pool = nn.MaxPool2d(2, stride=2, padding=0)
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(self.pool(x))
        return x
```

```jupyter
class up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up, self).__init__()
        self.up_scale = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
  
    def forward(self, x1, x2): # x1 (bs,out_ch,w1,h1) x2 (bs,in_ch,w2,h2)
        x2 = self.up_scale(x2) # (bs,out_ch,2*w2,2*h2)
        diffY = x1.size()[2] - x2.size()[2]
        diffX = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2]) # (bs,out_ch,w1,h1)
        x = torch.cat([x2, x1], dim=1) # (bs,2*out_ch,w1,h1)
        return x

class up_layer(nn.Module):
    def __init__(self, in_ch, out_ch): # !! 2*out_ch = in_ch !!
        super(up_layer, self).__init__()
        self.up = up(in_ch, out_ch)
        self.conv = double_conv(in_ch, out_ch)  

    def forward(self, x1, x2): # x1 (bs,out_ch,w1,h1) x2 (bs,in_ch,w2,h2)
        a = self.up(x1, x2) # (bs,2*out_ch,w1,h1)
        x = self.conv(a) # (bs,out_ch,w1,h1) because 2*out_ch = in_ch
        return x
```


```jupyter
class UNet(nn.Module):
    def __init__(self, in_channels=1, n_steps=1000, time_emb_dim=100):
        super(UNet, self).__init__()
        self.conv1 = double_conv(in_channels, 64)
        self.down1 = down_layer(64, 128)
        self.down2 = down_layer(128, 256)
        self.down3 = down_layer(256, 512)
        self.down4 = down_layer(512, 1024)
        self.up1 = up_layer(1024, 512)
        self.up2 = up_layer(512, 256)
        self.up3 = up_layer(256, 128)
        self.up4 = up_layer(128, 64)
        self.last_conv = nn.Conv2d(64, in_channels, 1)

        # Time embedding
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)

        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)

        self.time_embed.requires_grad_(False)
        self.te1 = self._make_te(time_emb_dim, in_channels)
        self.te2 = self._make_te(time_emb_dim, 64)
        self.te3 = self._make_te(time_emb_dim, 128)
        self.te4 = self._make_te(time_emb_dim, 256)
        self.te5 = self._make_te(time_emb_dim, 512)
        self.te1_up = self._make_te(time_emb_dim, 1024)
        self.te2_up = self._make_te(time_emb_dim, 512)
        self.te3_up = self._make_te(time_emb_dim, 256)
        self.te4_up = self._make_te(time_emb_dim, 128)
  

    def _make_te(self, dim_in, dim_out):
        return nn.Sequential(nn.Linear(dim_in, dim_out), nn.SiLU(), nn.Linear(dim_out, dim_out))

    def forward(self, x , t): # x (bs,in_channels,w,d)
        bs = x.shape[0]
        t = self.time_embed(t)
        x1 = self.conv1(x+self.te1(t).reshape(bs, -1, 1, 1)) # (bs,64,w,d)
        x2 = self.down1(x1+self.te2(t).reshape(bs, -1, 1, 1)) # (bs,128,w/2,d/2)
        x3 = self.down2(x2+self.te3(t).reshape(bs, -1, 1, 1)) # (bs,256,w/4,d/4)
        x4 = self.down3(x3+self.te4(t).reshape(bs, -1, 1, 1)) # (bs,512,w/8,h/8)
        x5 = self.down4(x4+self.te5(t).reshape(bs, -1, 1, 1)) # (bs,1024,w/16,h/16)
        x1_up = self.up1(x4, x5+self.te1_up(t).reshape(bs, -1, 1, 1)) # (bs,512,w/8,h/8)
        x2_up = self.up2(x3, x1_up+self.te2_up(t).reshape(bs, -1, 1, 1)) # (bs,256,w/4,h/4)
        x3_up = self.up3(x2, x2_up+self.te3_up(t).reshape(bs, -1, 1, 1)) # (bs,128,w/2,h/2)
        x4_up = self.up4(x1, x3_up+self.te4_up(t).reshape(bs, -1, 1, 1)) # (bs,64,w,h)
        output = self.last_conv(x4_up) # (bs,in_channels,w,h)
        return output
```


```jupyter
bs = 3
x = torch.randn(bs,1,32,32)
n_steps=1000
timesteps = torch.randint(0, n_steps, (bs,)).long()
unet = UNet()
```

```jupyter
y = unet(x,timesteps)
y.shape
```

```jupyter
class DDPM(nn.Module):
    def __init__(self, network, num_timesteps, beta_start=0.0001, beta_end=0.02, device=device) -> None:
        super(DDPM, self).__init__()
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.network = network
        self.device = device
        self.sqrt_alphas_cumprod = self.alphas_cumprod ** 0.5 # used in add_noise
        self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod) ** 0.5 # used in add_noise and step

  

    def add_noise(self, x_start, x_noise, timesteps):
        # The forward process
        # x_start and x_noise (bs, n_c, w, d)
        # timesteps (bs)
        s1 = self.sqrt_alphas_cumprod[timesteps] # bs
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps] # bs
        s1 = s1.reshape(-1,1,1,1) # (bs, 1, 1, 1) for broadcasting
        s2 = s2.reshape(-1,1,1,1) # (bs, 1, 1, 1)
        return s1 * x_start + s2 * x_noise

    def reverse(self, x, t):
        # The network return the estimation of the noise we added
        return self.network(x, t)
        
    def step(self, model_output, timestep, sample):
        # one step of sampling
        # timestep (1)
        t = timestep
        coef_epsilon = (1-self.alphas)/self.sqrt_one_minus_alphas_cumprod
        coef_eps_t = coef_epsilon[t].reshape(-1,1,1,1)
        coef_first = 1/self.alphas ** 0.5
        coef_first_t = coef_first[t].reshape(-1,1,1,1)
        pred_prev_sample = coef_first_t*(sample-coef_eps_t*model_output)

        variance = 0
        if t > 0:
            noise = torch.randn_like(model_output).to(self.device)
            variance = ((self.betas[t] ** 0.5) * noise)
        pred_prev_sample = pred_prev_sample + variance
        return pred_prev_sample
```

```jupyter
def training_loop(model, dataloader, optimizer, num_epochs, num_timesteps, device=device):
    """Training loop for DDPM"""
    global_step = 0
    losses = []
    for epoch in range(num_epochs):
        model.train()
        progress_bar = tqdm(total=len(dataloader))
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(dataloader):
            batch = batch[0].to(device)
            noise = torch.randn(batch.shape).to(device)
            timesteps = torch.randint(0, num_timesteps, (batch.shape[0],)).long().to(device)

  

            noisy = model.add_noise(batch, noise, timesteps)
            noise_pred = model.reverse(noisy, timesteps)
            loss = F.mse_loss(noise_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "step": global_step}
            losses.append(loss.detach().item())
            progress_bar.set_postfix(**logs)
            global_step += 1

        progress_bar.close()
```

```jupyter
root_dir = './'

transforms01 = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        #torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

dataset = torchvision.datasets.CIFAR10(root=root_dir, train=True, transform=transforms01, download=True)

dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=512, shuffle=True,num_workers=10)
```

```jupyter
for b in dataloader:
    batch = b[0]
    break
    
bn = [b for b in batch[:100]]
show_images(bn, "origin")
```

![[res.png]]

```jupyter
learning_rate = 1e-3
num_epochs = 15
num_timesteps = 1000
network = UNet(in_channels=3)
network.to(device)
model = DDPM(network, num_timesteps, beta_start=0.0001, beta_end=0.02, device=device)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
training_loop(model, dataloader, optimizer, num_epochs, num_timesteps, device=device)
```

Model will start training for about 15 epochs

```jupyter
def generate_image(ddpm, sample_size, channel, size):
    """Generate the image from the Gaussian noise"""
  
    frames = []
    frames_mid = []
    ddpm.eval()
    with torch.no_grad():
        timesteps = list(range(ddpm.num_timesteps))[::-1]
        sample = torch.randn(sample_size, channel, size, size).to(device)
        for i, t in enumerate(tqdm(timesteps)):
            time_tensor = (torch.ones(sample_size) * t).long().to(device)
            residual = ddpm.reverse(sample, time_tensor).to(device)

            sample = ddpm.step(residual, time_tensor[0], sample)

            if t==500:
                #sample_squeezed = torch.squeeze(sample)
                for i in range(sample_size):
                    frames_mid.append(sample[i].detach().cpu())

        #sample = torch.squeeze(sample)
        for i in range(sample_size):
            frames.append(sample[i].detach().cpu())
    return frames, frames_mid
```

```jupyter
def make_dataloader(dataset, class_name ='ship'):
    s_indices = []
    s_idx = dataset.class_to_idx[class_name]
    for i in range(len(dataset)):
        current_class = dataset[i][1]
        if current_class == s_idx:
            s_indices.append(i)
    s_dataset = Subset(dataset, s_indices)
    return torch.utils.data.DataLoader(dataset=s_dataset, batch_size=512, shuffle=True)
```

```jupyter
ship_dataloader = make_dataloader(dataset)
```

```jupyter
ship_network = copy.deepcopy(network)
ship_model = DDPM(ship_network, num_timesteps, beta_start=0.0001, beta_end=0.02, device=device)
num_epochs = 10
num_timesteps = model.num_timesteps
learning_rate = 1e-3
ship_model.train()
optimizer = torch.optim.Adam(ship_model.parameters(), lr=learning_rate)
training_loop(ship_model, ship_dataloader, optimizer, num_epochs, num_timesteps, device=device)
```

```jupyter
generated, generated_mid = generate_image(ship_model, 100, 3, 32)
```

```jupyter
show_images(generated, "Generated ships")
```

![[ddpm.png]]

This training was only done for very few epochs thats why we still do not have a very detailed generated result.

## CONCLUSIONS
This article covered the foundations of Diffusion model by going deeply into Denoising diffusion models, the concepts, the maths and the code. There's actually no point training this kind of an AI model because it wont result in any generation that is as good as DALL-E or Stable Diffusion in terms of Fidelity and Diveristy. As such we have SOTA Models that can perform text to image generation better than this one even though the purpose of this article was just to give an overview of the building blocks of a diffusion model.
____
## REFERENCES
1. Karsten Kreis; Ruigui Gao; Arash Vahdat (2022-5-4): "Denoising Diffusion-based Generative Modelling: Foundations and Applications (CVPR 2002 Worskshop) "
2. Ho, J., Jain, A., and Abbeel, P. Denoising diffusion probabilistic models, 2020.
3. Sohl-Dickstein, J., Weiss, E. A., Maheswaranathan, N., and Ganguli, S. Deep unsupervised learning using nonequilibrium thermodynamics, 2015.
4. https://github.com/dataflowr/notebooks/blob/master/Module18/ddpm_micro_sol.ipynb
6. Karsten Kreis; Ruigui Gao; Arash Vahdat (2022-5-4): "Denoising Diffusion-based Generative Modelling: Foundations and Applications (CVPR 2002 Worskshop) "