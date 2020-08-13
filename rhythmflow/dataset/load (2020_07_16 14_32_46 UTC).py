from .base import *


def load_data(args):
    ref_split = args.path + '/reference_split_' + args.dataset+ "_" + args.data + '.th'
    if (args.train_type == 'random' or (not os.path.exists(ref_split))):
        train_loader, valid_loader, test_loader, args = load_dataset(args)
        if (args.train_type == 'fixed'):
            torch.save([train_loader, valid_loader, test_loader], ref_split)
        # Take fixed batch
        fixed_data, fixed_params, fixed_meta, fixed_audio = next(iter(test_loader))
        fixed_data, fixed_params, fixed_meta, fixed_audio = fixed_data.to(args.device), fixed_params.to(args.device), fixed_meta, fixed_audio
        fixed_batch = (fixed_data, fixed_params, fixed_meta, fixed_audio)
    else:
        data = torch.load(ref_split)
        train_loader, valid_loader, test_loader = data[0], data[1], data[2]
        fixed_data, fixed_params, fixed_meta, fixed_audio = next(iter(test_loader))
        fixed_data, fixed_params, fixed_meta, fixed_audio = fixed_data.to(args.device), fixed_params.to(args.device), fixed_meta, fixed_audio
        fixed_batch = (fixed_data, fixed_params, fixed_meta, fixed_audio)
        args.output_size = train_loader.dataset.output_size
        args.input_size = train_loader.dataset.input_size
    if (args.latent_dims == 0): # Set latent dims to output dims
        args.latent_dims = args.output_size
    return train_loader, valid_loader, test_loader, args, fixed_batch

def load_dataset(args, **kwargs):
    if (args.dataset in ['toy'], ["32par"], ["64par"], ["64par_aug"], ["128par"]):
        params = {'32par':'32contparams.txt', '64par':'64contparams.txt', '64par_aug':'64contparams.txt', '128par':'128contparams.txt'}
        with open('synth/params/' + params[args.dataset]) as f: # load list of parameters to not fix
            use_params = [line.strip() for line in f]
        dset_train = CompSynthesizerDataset(args.path + '/' + args.dataset, use_params, data=args.data, **kwargs)
        dset_valid = copy.deepcopy(dset_train).switch_set('valid')
        dset_test = copy.deepcopy(dset_train).switch_set('test')
        dset_train = dset_train.switch_set('train')
    else:
        raise Exception('Wrong name of the dataset!')
    args.input_size = dset_train.input_size
    args.output_size = dset_train.output_size
    train_loader = DataLoader(dset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.nbworkers, pin_memory=False, **kwargs)
    valid_loader = DataLoader(dset_valid, batch_size=args.batch_size, shuffle=(args.train_type == 'random'), num_workers=args.nbworkers, pin_memory=False, **kwargs)
    test_loader = DataLoader(dset_test, batch_size=args.batch_size, shuffle=(args.train_type == 'random'), num_workers=args.nbworkers, pin_memory=False, **kwargs)
    return train_loader, valid_loader, test_loader, args

def get_external_sounds(path, ref_loader, args, **kwargs):
    dset = AudioDataset(path, data=args.data, mean=ref_loader.means, var=ref_loader.vars, **kwargs)
    loader = DataLoader(dset, batch_size=args.batch_size, shuffle=False, num_workers=args.nbworkers, pin_memory=True, **kwargs)
    dset.final_params = ref_loader.final_params
    return loader

if __name__ == '__main__':
    # Define arguments
    parser = argparse.ArgumentParser()
    # Data arguments
    parser.add_argument('--path', type=str, default='/Users/esling/Datasets/diva_dataset', help='')
    parser.add_argument('--dataset', type=str, default='toy', help='')
    parser.add_argument('--data', type=str, default='mel', help='')
    parser.add_argument('--batch_size', type=int, default=64, help='')
    parser.add_argument('--epochs', type=int, default=100, help='')
    args = parser.parse_args()
    train_loader, valid_loader, test_loader, args = load_dataset(args)
    # Take fixed batch (train)
    data, params, meta, audio = next(iter(train_loader))
    plot_batch(data[:16].unsqueeze(1))
    plot_batch_detailed(data[:5], params[:5])
    # Take fixed batch (train)
    data, params, meta = next(iter(test_loader))
    plot_batch(data[:16].unsqueeze(1))
    plot_batch_detailed(data[:5], params[:5])