import argparse
import os
import math
import data
import torch
from utils import utils

from utils.progress import Progress
import numpy as np
from funcmodel import VQ
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def func_net_loss(args, x, y, vq_loss, model):
    """
    Compute combined loss
    """
    # Use Discretized Logistic as an alternative to MSE, see [1]
    # log_pxz = utils.discretized_logistic(x_prime, model.dec_log_stdv,
                                                    # sample=x).mean()


    mse = nn.MSELoss()(x,y)
    loss = mse + args.commitment_cost * vq_loss

    return loss

def train_epoch(args, loss_func, pbar, train_loader, model, optimizer,
                            train_bpd, train_recon_error , train_perplexity):
    """
    Train for one epoch
    """
    model.train()
    # Loop data in epoch
    for x, y in train_loader:

        # This break used for debugging
        if args.max_iterations is not None:
            if args.global_it > args.max_iterations:
                break

        x = x.to(args.device)

        # Get reconstruction and vector quantization loss
        # `x_prime`: reconstruction of `input`
        # `vq_loss`: MSE(encoded embeddings, nearest emb in codebooks)
        x_prime, vq_loss, perplexity = model(x)

        loss, log_pxz, bpd = loss_func(args, x_prime, x, vq_loss, model)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_bpd.append((-1)*bpd.item())
        train_recon_error.append((-1)*log_pxz.item())
        train_perplexity.append(perplexity.item())

        # Print Average every 100 steps
        if (args.global_it+1) % 100 == 0:
            av_bpd = np.mean(train_bpd[-100:])
            av_rec_err = np.mean(train_recon_error[-100:])
            av_ppl = np.mean(train_perplexity[-100:])
            if args.model == 'vqvae':
                pbar.print_train(bpd=float(av_bpd), rec_err=float(av_rec_err),
                                        ppl=float(perplexity), increment=100)
            elif args.model == 'diffvqvae':
                pbar.print_train(bpd=float(av_bpd), temp=float(model.temp),
                                        ppl=float(perplexity), increment=100)
        args.global_it += 1

def main(args):
    ###############################
    # TRAIN PREP
    ###############################
    print("Loading data")
    train_loader, valid_loader, test_loader, d_settings = \
                                data.get_toy_data(args.batch_size)

    args.input_size = [d_settings["seq_len"]]
    args.downsample = args.input_size[-1]
    # args.downsample = args.input_size[-1] // args.enc_height
    # args.data_variance = data_var
    print(f"Training set size {len(train_loader.dataset)}")
    print(f"Validation set size {len(valid_loader.dataset)}")
    print(f"Test set size {len(test_loader.dataset)}")

    print("Loading model")
    model = VQ(args).to(device)
    print(f'The model has {utils.count_parameters(model):,} trainable params')

    optimizer = optim.Adam(model.parameters(),lr=args.learning_rate,
                                                                amsgrad=False)

    print(f"Start training for {args.num_epochs} epochs")
    num_batches = math.ceil(len(train_loader.dataset)/train_loader.batch_size)
    pbar = Progress(num_batches, bar_length=10, custom_increment=True)

    # Needed for bpd
    # args.KL = args.enc_height * args.enc_height * args.num_codebooks * \
                                                # np.log(args.num_embeddings)
    # args.num_pixels  = np.prod(args.input_size)

    ###############################
    # MAIN TRAIN LOOP
    ###############################
    best_valid_loss = float('inf')
    train_bpd = []
    train_recon_error = []
    train_perplexity = []
    args.global_it = 0

    loss_func = func_net_loss
    for epoch in range(args.num_epochs):
        pbar.epoch_start()
        train_epoch(args, loss_func, pbar, train_loader, model, optimizer,
                                train_bpd, train_recon_error, train_perplexity)
        # loss, _ = test(valid_loader, model, args)
        # pbar.print_eval(loss)
        valid_loss = evaluate(args, loss_func, pbar, valid_loader, model)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_valid_epoch = epoch
            torch.save(model.state_dict(), args.save_path)
        pbar.print_end_epoch()

    print("Plotting training results")
    utils.plot_results(train_recon_error, train_perplexity,
                                                        "results/train.png")

    print("Evaluate and plot validation set")
    generate_samples(model, valid_loader)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Vector Quantization')
    add = parser.add_argument

    # Data and training settings
    add('--data_folder', type=str, default=".data/cifar10",
            help='Location of data (will download data if does not exist)')
    add('--batch_size', type=int, default=32)
    add('--max_iterations', type=int, default=None,
            help="Max it per epoch, for debugging (default: None)")
    add('--num_epochs', type=int, default=40,
            help='number of epochs (default: 40)')
    add('--learning_rate', type=float, default=3e-4)
    add('--dataset', type=str, choices=['toy'], default='toy')

    # Quantization settings
    add('--num_codebooks', type=int, default=1,
            help='Number of codebooks')
    add('--embed_dim', type=int, default=64,
            help='Embedding size, `D` in paper')
    add('--num_embeddings', type=int, default=512,
            help='Number of embeddings to choose from, `K` in paper')
    add('--commitment_cost', type=float, default=0.25,
            help='Beta in the loss function')
    add('--decay', type=float, default=0.99,
            help='Moving av decay for codebook update')

    # VQVAE model, defaults like in paper
    add('--model', type=str, choices=['vqvae'], default='vqvae')
    add('--enc_height', type=int, default=8,
            help="Encoder output size, used for downsampling and KL")
    add('--num_hiddens', type=int, default=10,
            help="Number of channels for Convolutions, not ResNet")
    add('--num_residual_hiddens', type=int, default = 32,
            help="Number of channels for ResNet")
    add('--num_residual_layers', type=int, default=2)
    add('--embed_grad_update', action='store_true', default=False,
            help="If True, update Embed with gradient instead of EMA")
    add('--ent_coef', type=float, default=0.01, metavar='M',
            help='Entropy reg coef (default: 0.01)')
    add('--orth_coef', type=float, default=1e-6, metavar='M',
            help='Orthogonal regularization coef (default: 1e-6)')

    # Diff NearNeigh settings
    add('--nn_temp', type=float, default=20.0, metavar='M',
            help='Starting diff. nearest neighbour temp (default: 1.0)')
    add('--min_temp', type=float, default=1.01, metavar='M',
            help='Minimum Nearest neighbour temp (default: 0.01)')
    add('--temp_decay_rate', type=float, default=0.9, metavar='M',
            help='Nearest neighbour temp decay rate (default: 0.9)')
    add('--temp_decay_schedule', type=float, default=100, metavar='M',
            help='How many batches before decay (default: 100)')
    add('--temp_grad_update', action='store_true', default=False,
            help="Update temp only with gradient")

    # Func mod settings
    add('--emb_chunks', type=int, default=8,
            help="Split embedding into how many chunks")
    add('--dec_h_size', type=int, default=20,
            help="Size of hidden decoder layer")

    # Misc
    add('--saved_model_name', type=str, default='func_net.pt')
    add('--saved_model_dir', type=str, default='saved_models/')
    add('--seed', type=int, default=521)

    args = parser.parse_args()

    # Extra args
    args.device = device
    args.save_path = os.path.join(args.saved_model_dir, args.saved_model_name)
    utils.maybe_create_dir(args.saved_model_dir)

    # Seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    main(args)

