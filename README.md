# DiffCrysGen

DiffCrysGen is a score-based diffusion model. It treats the entire materials representation with a single, unified diffusion process, allowing a single denosing neural network to predict a holistic score for the entire noisy crystal data. This unified treatment significantly simplifies the architecture and improves the computational efficiency.


<img src="images/logo-DiffCrysGen.png" alt="DiffCrysGen Logo" width="350"/>

## Generative diffusion framework in DiffCrysGen
<img src="images/diffusion-schematic.png" alt="DiffCrysGen Schematic" width="550">

## Architecture of the denoising neural network
<img src="images/architecture.png" alt="DiffCrysGen Architecture" width="750">


## Quick Start
For a simple walkthrough of generating materials and analyzing them, see the [DiffCrysGen Demo Notebook](./notebooks/DiffCrysGen-demo.ipynb).

## License

This project is licensed under the **MIT License**.

See the [LICENSE](LICENSE) file for details.
Developed by: [Sourav Mal](https://github.com/SouravMal) at Harish-Chandra Research Institute (HRI) (https://www.hri.res.in/), Prayagraj, India.


## Citation

Please consider citing our work if you find it helpful:

```bibtex
@misc{mal2025generativediffusionmodeldiffcrysgen,
      title={Generative Diffusion Model DiffCrysGen Discovers Rare Earth-Free Magnetic Materials}, 
      author={Sourav Mal and Nehad Ahmed and Subhankar Mishra and Prasenjit Sen},
      year={2025},
      eprint={2510.12329},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci},
      url={https://arxiv.org/abs/2510.12329}, 
}
