from data.base import BaseSequenceDataset


def test_base_sequence_dataset():
    dataset = BaseSequenceDataset()
    sample = dataset._track_from_midi(dataset.loaded_files[0])
    assert sample.shape[1] == 1
    assert sample.shape[2] == 9
    assert sample.shape[0] > 0
    # print('shape:', sample.shape)
    # print('===================')
    # print(sample)
    # assert(type(sample) == str)
