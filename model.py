import utils
import torch
from torch import nn
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Quantize(nn.Module):
    def __init__(self, dim, num_embeddings, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.num_embeddings = num_embeddings
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, num_embeddings)
        self.register_buffer('embed', embed)
        self.register_buffer('cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('embed_avg', embed.clone())

    def forward(self, x):
        flatten = x.reshape(-1, self.dim)
        # Dist: squared-L2(p,q) = ||p||^2 + ||q||^2 - 2pq
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        encodings = F.one_hot(embed_ind, self.num_embeddings)
        encodings = encodings.type(flatten.dtype) # cast
        embed_ind = embed_ind.view(*x.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            self.cluster_size.data.mul_(self.decay).add_(
                1 - self.decay, encodings.sum(0)
            )
            embed_sum = flatten.transpose(0, 1) @ encodings
            self.embed_avg.data.mul_(self.decay).add_(1 - self.decay, embed_sum)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.num_embeddings * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - x).pow(2).mean()
        # The +- `x` is the "straight-through" gradient trick!
        quantize = x + (quantize - x).detach()

        return quantize, diff, embed_ind, encodings

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))

class FuncNet():
    """
    Contains the function selector network and sequence processing network
    """
    pass

class VQ(nn.Module):
    def __init__(self, args):
        super().__init__()

        in_channel = args.input_size[0]
        channel    = args.num_hiddens
        num_residual_layers = args.num_residual_layers
        num_residual_hiddens = args.num_residual_hiddens
        embed_dim = args.embed_dim
        num_embeddings = args.num_embeddings
        decay = args.decay
        downsample = args.downsample
        num_codebooks = args.num_codebooks

        assert embed_dim % num_codebooks == 0, ("you need that last dimension"
                            " to be evenly divisible by the amt of codebooks")

        self.enc = SmallEnc(in_channel, channel)

        self.quantize_conv = nn.Conv2d(channel, embed_dim, 1)
        self.dec = Decoder(embed_dim, in_channel, channel, num_residual_layers,
                                    num_residual_hiddens, stride=downsample)

        # build the codebooks
        self.quantize = nn.ModuleList([Quantize(embed_dim // num_codebooks,
                                num_embeddings) for _ in range(num_codebooks)])

        self.register_parameter('dec_log_stdv', torch.nn.Parameter(\
                                                        torch.Tensor([0.])))

    def forward(self, x):
        """
        Args:
            x (Tensor): shape BCHW
        """
        # `diff`: MSE(embeddings in z_e_s, closest in codebooks)
        # `z_q`, shape B*EMB_DIM*CHW, is neirest neigh embeddings to x
        z_q, diff, emb_idx, ppl = self.encode(x)

        # `dec`: decode `z_q` to `x` size, it is the image reconstruction
        dec = self.decode(z_q)

        return dec, diff, ppl

    def encode(self, x):
        # Encode x to continuous space
        pre_z_e = self.enc(x)
        # Project that space to the proper size for embedding comparison
        z_e = self.quantize_conv(pre_z_e)

        # Divide into multiple chunks to fit each codebook
        z_e_s = z_e.chunk(len(self.quantize), 1)

        z_q_s, enc_indices, encodings = [], [], []
        diffs = 0.

        # `argmin`: the indices corresponding to closest embedding in codebook
        # `z_q`: same size as z_e_s but now holds the vectors from codebook
        # `diff`: MSE(embeddings in z_e_s, closest in codebooks)
        for z_e, quantize in zip(z_e_s, self.quantize):
            # z_e, change shape form  BCHW to BHWC
            z_q, diff, enc_ind, enc = quantize(z_e.permute(0, 2, 3, 1))
            z_q_s   += [z_q]
            encodings += [enc]
            enc_indices += [enc_ind]
            diffs   += diff

        # TODO: print ppl, this should be concat
        # concat avg_probs, then calc ppl
        encoding_indices = torch.cat(enc_indices, dim=-1)
        encodings_cat = torch.cat(encodings, dim=-1)
        avg_probs = torch.mean(encodings_cat, dim=0)
        perplexity = torch.exp(-torch.sum(\
                                    avg_probs * torch.log(avg_probs + 1e-10)))

        # Stack the z_q_s and permute, now `z_q` has the same shape as the
        # first z_e
        z_q = torch.cat(z_q_s, dim=-1)
        z_q = z_q.permute(0, 3, 1, 2)

        return z_q, diffs, encoding_indices, perplexity

    def decode(self, quant):
        return self.dec(quant)

class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, x):
        out = self.conv(x)
        out += x

        return out

class SmallEnc(nn.Module):
    """ Small conv encoder for toy data """
    def __init__(self, in_size, h_size):
        super().__init__()

        self.lin = nn.Sequential(
            nn.Linear(input_size, h_size, bias=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        r = self.lin(x)
        return r

class Encoder(nn.Module):
    def __init__(self, in_channel, channel, num_residual_layers,
            num_residual_hiddens, stride):
        super().__init__()
        if stride == 8:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]

        if stride == 4:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]

        elif stride == 2:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 3, padding=1),
            ]

        elif stride == 1:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 5, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel // 2, 3, padding=1)
            ]

        for i in range(num_residual_layers):
            blocks += [ResBlock(channel, num_residual_hiddens)]

        blocks += [nn.ReLU(inplace=True)]

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class Decoder(nn.Module):
    def __init__(self, in_channel, out_channel, channel, num_residual_layers, num_residual_hiddens, stride):
        super().__init__()

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        for i in range(num_residual_layers):
            blocks += [ResBlock(channel, num_residual_hiddens)]

        blocks += [nn.ReLU(inplace=True)]

        if stride == 8:
            blocks += [
                nn.ConvTranspose2d(channel, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(channel // 2, out_channel, 4, stride=2, padding=1),
            ]

        if stride == 4:
            blocks += [
                nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(channel // 2, out_channel, 4, stride=2, padding=1),
            ]

        elif stride == 2:
            blocks += [nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1)]

        elif stride == 1:
            blocks += [nn.Conv2d(channel, out_channel, 3, padding=1)]

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


def vq_vae_loss(args, x_prime, x, vq_loss, model):
    """
    Compute discretized recon loss, combined loss for training and bpd.

    Bits per dimension (bpd) is simply nats per pixel converted to base 2.
    -(NLL / num_pixels) / np.log(2.)
    """
    # Use Discretized Logistic as an alternative to MSE, see [1]
    log_pxz = utils.discretized_logistic(x_prime, model.dec_log_stdv,
                                                    sample=x).mean()
    # recon_error = torch.mean((data_recon - data)**2)/args.data_variance
    # loss = recon_error + vq_loss

    loss = -1 * (log_pxz / args.num_pixels) + args.commitment_cost * vq_loss
    elbo = - (args.KL - log_pxz) / args.num_pixels
    bpd  = elbo / np.log(2.)

    return loss, log_pxz, bpd

