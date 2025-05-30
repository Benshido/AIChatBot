from TTS.api import TTS
from pydub import AudioSegment
from pydub.playback import play

model_name = "tts_models/nl/mai/tacotron2-DDC"

# Load the Dutch TTS model
# Load a Dutch TTS model (this is the correct way to instantiate the model)
tts = TTS(model_name="tts_models/nl/mai/tacotron2-DDC")

# Generate speech and save it to a file
tts.tts_to_file(
    text="Hallo, dit is een test van het Nederlandse spraaksysteem.",
    file_path="test.wav"
)

sound = AudioSegment.from_wav("test.wav")
play(sound)
