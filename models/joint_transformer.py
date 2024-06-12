import torch
import torch.nn as nn
import numpy as np

class JointTransformer(nn.Module):
    def __init__(self, args, num_features=263, latent_dim=512, ff_size=1024, num_layers_E=24, num_layers_D=8, num_heads=4, dropout=0.1, activation="gelu") -> None:
        super().__init__()
        self.args = args
        self.num_features = num_features 
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_heads = num_heads
        self.num_layers_E = num_layers_E
        self.num_layers_D = num_layers_D
        self.dropout = dropout
        self.activation = activation

        self.position_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.input_process = InputProcess(self.num_features, self.latent_dim)
        self.root_process = InputProcess(7, self.latent_dim)
        self.joint_process = InputProcess(6, self.latent_dim)
        self.foot_process = InputProcess(4, self.latent_dim)

        self.down_conv1 = nn.Conv2d(self.latent_dim, self.latent_dim, kernel_size=(3,1), stride=2, padding=1)
        self.down_conv2 = nn.Conv2d(self.latent_dim, self.latent_dim, kernel_size=(3,1), stride=2, padding=1)

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
        
    
    def forward(self, x, text_emb, word_emb=None, attn_mask=None):
        '''
        x: (b,196,263)
        text_emb: (b,512)
        word_emb: (b,77,512)
        '''
        root, joint, foot = self.preprocess(x)  
        root_emb = self.root_process(root)[:, :, None] # (b,196,1,d)
        joint_emb = self.joint_process(joint) # (b,196,21,d)
        foot_emb = self.foot_process(foot)[:, :, None] # (b,196,1,d)
        motion_emb = torch.cat([root_emb, joint_emb, foot_emb], dim=2) #(b,196,24,d)
        motion_emb = motion_emb.permute(0,3,1,2) #(b,d,196,24) 

        motion_emb = self.down_conv1(motion_emb)
        motion_emb = self.down_conv2(motion_emb) #(b,d,49,24) 
        x = motion_emb.flatten(2,3).permute(2,0,1) # b,49*24,d

        # 文本
        emb = text_emb[None, ...]
        if word_emb is not None:
            emb = torch.cat([emb, word_emb.permute(1,0,2)], dim=0)
        
        # transformer encoder
        seq_in = self.position_encoder(x) # 49*24,b,d
        z = self.seqTransEncoder(seq_in) # 49*24,b,d

        if self.num_layers_D==0:
            out = self.output_process(z)
            return out

        seq_out = self.seqTransDecoder(z, memory=emb)
        out = self.output_process(seq_out)
        return out
    
    def preprocess(self, x):
        B, L, num_features = x.shape
        if num_features == 263: # 如果是HumanML3D 
            n_joints = 22
            root = torch.cat([x[..., :4], x[..., 193:196]], dim=-1)
            ric = x[..., 4:67].view(B, L, n_joints, 3)
            rot = x[..., 67:193].view(B, L, n_joints, 6)
            vel = x[..., 193:259].view(B, L, n_joints, 3)
            foot = x[..., 259:]
            joint = torch.cat([ric, rot, vel], dim=2) # (B,L,n_joints,12)
        elif num_features == 251:
            pass
        return root, joint, foot

    def postprocess(self, x):
        pass
    
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
    
