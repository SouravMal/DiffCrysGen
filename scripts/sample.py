from diffcrysgen.sampler import generate_samples
import numpy as np

samples = generate_samples(num_samples=200)
np.save("generated_ircr.npy", samples)
