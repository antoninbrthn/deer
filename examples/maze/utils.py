import numpy as np


def noise(obs, strength = 0.5):
    b = 1 * strength
    a = -1 * strength
    n = (b - a) * np.random.random(obs.shape) + a
    noisy_obs = np.clip(obs + n, a_min = -1, a_max = 1)
    return noisy_obs


def big_noise(obs, strength = 0.6, size = 3):
    b = 1 * strength
    a = -1 * strength
    noise = (b - a) * np.random.random(obs.shape[-6:-2] + (obs.shape[-2] // size, obs.shape[-1] // size)) + a
    noise = noise.repeat(size, axis = -1).repeat(size, axis = -2)
    noisy_obs = np.clip(obs + noise, a_min = -1, a_max = 1)
    return noisy_obs


def efficient_noise(obs, strength=0.6, size=3):
    return np.array([big_noise(o, strength=strength, size=size) for o in obs])