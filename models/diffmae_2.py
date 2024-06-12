import torch
import torch.nn as nn
import numpy as np

class DiffMAE2(nn.Module):
    def __init__(self, dataset, args, njoints=22, num_features=263, latent_dim=512, ff_size=1024, num_layers_E=24, num_layers_D=8, num_heads=4, dropout=0.1, activation="gelu", clip_dim=512, ) -> None:
        super().__init__()
        self.dataset = dataset
        self.args = args
        self.njoints = njoints
        self.num_features = num_features 
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_heads = num_heads
        self.num_layers_E = num_layers_E
        self.num_layers_D = num_layers_D
        self.dropout = dropout
        self.activation = activation
        self.clip_dim = clip_dim

        self.cond_mask_prob = 0.2
        self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)
        self.position_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.position_encoder)
        self.input_process = InputProcess(self.num_features, self.latent_dim)

        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)

        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                        num_layers=self.num_layers_E)
        if self.num_layers_D !=0:
            seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
                                                                nhead=self.num_heads,
                                                                dim_feedforward=self.ff_size,
                                                                dropout=self.dropout,
                                                                activation=activation)
            self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer,
                                                                num_layers=self.num_layers_D)
        
        self.output_process = OutputProcess(self.num_features, self.latent_dim)
        
    
    def forward(self, x, t, clip_emb, word_emb=None):
        '''
        x: (b,196,263)
        clip_emb: (b,512)
        word_emb: (b,77,512)
        '''
        emb = self.embed_timestep(t)  # [1, bs, 512]
        masked_clip_emb = self.random_mask_text(clip_emb, force_mask=False)
        text_emb = self.embed_text(masked_clip_emb)[None, ...] # 有概率随机让某个batch文本为全0
        emb += text_emb

        x = self.input_process(x)
        x = torch.cat((emb, x), axis=0) # (197,b,d)
        seq_in = self.position_encoder(x)
        z = self.seqTransEncoder(seq_in)[1:]

        if self.num_layers_D==0:
            out = self.output_process(z)
            return out

        seq_out = self.seqTransDecoder(z, memory=emb)
        out = self.output_process(seq_out)
        return out
    
    def random_mask_text(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond
        
    
class InputProcess(nn.Module):
    def __init__(self, input_feats=263, latent_dim=512):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim) # (263,512)

    def forward(self, x):
        '''
        x: (b,196,263)
        '''
        x = x.permute(1,0,2)
        x = self.poseEmbedding(x)  # [seqlen, bs, d] 论文图2下面的Linear
        return x

class OutputProcess(nn.Module):
    def __init__(self, out_feats=263, latent_dim=512):
        super().__init__()
        self.out_feats = out_feats
        self.latent_dim = latent_dim
        self.poseFinal = nn.Linear(self.latent_dim, self.out_feats)

    def forward(self, x):
        x = self.poseFinal(x)  # [seqlen, bs, 150]
        x = x.permute(1,0,2)  # [b,L,263]
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)
    
class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)
