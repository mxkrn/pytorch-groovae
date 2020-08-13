MIN_MIDI_PITCH = 0
MAX_MIDI_PITCH = 127
MIDI_PITCHES = MAX_MIDI_PITCH - MIN_MIDI_PITCH + 1

DRUM_PITCH_CLASSES = [
    [36], # kick drum
    [38, 37, 40], # snare drum
    [42, 22, 44], # closed hi-hat
    [46, 26], # open hi-hat
    [43, 58], # low tom
    [47, 45], # mid tom
    [50, 48], # high tom
    [49, 52, 55, 57], # crash cymbal
    [51, 53, 59] # ride cymbal
]

SEQUENCE_LENGTH = 32
