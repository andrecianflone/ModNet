import utils
import torch
from torch import nn
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Quantize(nn.Module):
    """
    Returns the embedding from the codebook
    """
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
        """
        Returns:
            quantize  : closest embedding in codebook
            diff      : difference between embedding and prediction
            embed_ind : embedding index
            encodings : one-hot of embed_in

        """
        flatten = x.reshape(-1, self.dim)
        # Get distance between x and all vectors in codebook
        # Dist: squared-L2(p,q) = ||p||^2 + ||q||^2 - 2pq
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        # Return index of closest neighbour
        _, embed_ind = (-dist).max(1)
        encodings = F.one_hot(embed_ind, self.num_embeddings)
        encodings = encodings.type(flatten.dtype) # cast
        # embed_ind = embed_ind.view(*x.shape[:-1])
        # Get corresponding embeddings from codebook
        quantize = self.embed_code(embed_ind)

        # Move average encodings
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

        x = x.squeeze()
        diff = (quantize.detach() - x).pow(2).mean()
        # The +- `x` is the "straight-through" gradient trick!
        quantize = x + (quantize - x).detach()

        return quantize, diff, embed_ind, encodings

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))

class FuncMod(nn.Module):
    """
    Contains the function selector network and sequence processing network
    """
    def __init__(self, args):
        super().__init__()

        in_channel = args.input_size[0]
        channel    = args.num_hiddens
        num_residual_layers = args.num_residual_layers
        num_residual_hiddens = args.num_residual_hiddens
        embed_dim = args.embed_dim
        num_embeddings = args.num_embeddings
        decay = args.decay
        emb_chunks = args.emb_chunks # split emb into how many layers
        downsample = args.downsample
        num_codebooks = args.num_codebooks
        dec_h_size = args.dec_h_size
        dec_input_size = args.dec_input_size
        data_y_size = args.data_y_size

        assert embed_dim % num_codebooks == 0, ("you need that last dimension"
                            " to be evenly divisible by the amt of codebooks")

        # Encodes input to choose function
        self.enc_f = SmallEnc(in_channel, channel)

        # Projects input to a suitable size to send to function
        self.enc_x = SmallishEnc(in_channel, channel, dec_input_size)

        # self.quantize_conv = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_conv = nn.Conv1d(channel, embed_dim, 1)
        # self.dec = Decoder(embed_dim, in_channel, channel, num_residual_layers,
                                    # num_residual_hiddens, stride=downsample)
        self.dec = Decoder(dec_input_size, dec_h_size, embed_dim,data_y_size)

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

        # X embedded to be used for retrieved function
        embedded_x = self.enc_x(x)

        # `dec`: decode `z_q` to `x` size, it is the image reconstruction
        dec = self.decode(z_q, embedded_x)

        return dec, diff, ppl

    def encode(self, x):
        # Encode x to continuous space
        pre_f_e = self.enc_f(x)
        # Project that space to the proper size for embedding comparison
        z_f = self.quantize_conv(pre_f_e.unsqueeze(-1))

        # Divide into multiple chunks to fit each codebook
        z_e_s = z_f.chunk(len(self.quantize), 1)

        z_q_s, enc_indices, encodings = [], [], []
        diffs = 0.

        # `enc_ind`: the indices corresponding to closest embedding in codebook
        # `z_q`: same size as z_e_s but now holds the vectors from codebook
        # `diff`: MSE(embeddings in z_e_s, closest in codebooks)
        for z_e, quantize in zip(z_e_s, self.quantize):
            # z_e, change shape form  BCHW to BHWC
            # z_q, diff, enc_ind, enc = quantize(z_e.permute(0, 2, 3, 1))
            z_q, diff, enc_ind, enc = quantize(z_e)
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
        # z_q = z_q.permute(0, 3, 1, 2)

        return z_q, diffs, encoding_indices, perplexity

    def decode(self, quant, embedded_x):
        return self.dec(quant, embedded_x)

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
            nn.Linear(in_size, h_size, bias=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        r = self.lin(x)
        return r

class SmallishEnc(nn.Module):
    """ Small conv encoder for toy data """
    def __init__(self, in_size, h_size, out_size):
        super().__init__()

        self.lin = nn.Sequential(
            nn.Linear(in_size, h_size, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(h_size, out_size, bias=True),
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
    """
    Decoder which is basically the embedding reshape
    """
    def __init__(self, x_size, h_size, embed_dim, out_size):
        super().__init__()
        self.embed_dim  = embed_dim # size of input layer

        in_size = x_size
        self.net = nn.Sequential(
            nn.Linear(in_size, h_size, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(h_size, out_size, bias=True),
        )

        # Assert codebook embedding is the same size as decoder total params
        # TODO
        self.param_count_ls = [p.numel() for p in self.net.parameters()]
        param_count = sum(self.param_count_ls)

        assert param_count == embed_dim, "Embedding and net size don't match"

    def forward(self, quant_fn, x):

        # TODO: below only for batch size 1
        quant_fn = quant_fn.squeeze()
        idx = 0
        # Load decoder params from embedding
        for p in self.net.parameters():
            # Get appropriate num of params from embedding
            end_idx = idx + p.numel()
            new_vec = quant_fn[idx:end_idx]
            # Reshape to match
            new_vec = torch.reshape(new_vec, p.shape)
            # Override with codebook params
            p = new_vec
            idx = end_idx

        result = self.net(x)

        return result


class oldDecoder(nn.Module):
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
