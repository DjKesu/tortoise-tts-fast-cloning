import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F

import IPython

from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_audio, load_voice, load_voices

# This will download all the models used by Tortoise from the HuggingFace hub.
tts = TextToSpeech()

# This is the text that will be spoken.
text = "Joining two modalities results in a surprising increase in generalization! What would happen if we combined them all?" #@param {type:"string"}
"""
Then took the other, as just as fair,
And having perhaps the better claim,
Because it was grassy and wanted wear;
Though as for that the passing there
Had worn them really about the same,"""

preset = "ultra_fast" #@param ["ultra_fast", "fast", "standard", "high_quality", "very_fast"]

voice = 'krish' #@param {type:"string"}

#@markdown Load it and send it through Tortoise.
voice_samples, conditioning_latents = load_voice(voice)
print(voice_samples)
# conditioning_latents = tts.get_conditioning_latents(
#     voice_samples,
#     return_mels=False,  # Set to True if you want mel spectrograms to be returned
#     latent_averaging_mode=1,  # Choose the mode (0, 1, or 2) as needed
#     original_tortoise=False,  # Set to True or False as needed
# )
gen = tts.tts_with_preset(text, voice_samples=voice_samples, conditioning_latents=conditioning_latents, 
                          preset=preset)
torchaudio.save('generated.wav', gen.squeeze(0).cpu(), 24000)
IPython.display.Audio('generated.wav')

