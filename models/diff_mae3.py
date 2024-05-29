import torch
import torch.nn as nn
import numpy as np
import math

from utils.transformer_utils import Block, CrossAttentionBlock

class WeightedFeatureMaps(nn.Module):
    def __init__(self, k, embed_dim, *, norm_layer=nn.LayerNorm, decoder_depth):
        super(WeightedFeatureMaps, self).__init__()
        self.linear = nn.Linear(k, decoder_depth, bias=False)
        
        std_dev = 1. / math.sqrt(k)
        nn.init.normal_(self.linear.weight, mean=0., std=std_dev)

    def forward(self, feature_maps):
        # Ensure the input is a list
        assert isinstance(feature_maps, list), "Input should be a list of feature maps"
        # Ensure the list has the same length as the number of weights
        assert len(feature_maps) == (self.linear.weight.shape[1]), "Number of feature maps and weights should match"
        stacked_feature_maps = torch.stack(feature_maps, dim=-1)  # shape: (B, L, C, k)
        # compute a weighted average of the feature maps
        # decoder_depth is denoted as j
        output = self.linear(stacked_feature_maps)
        return output

class JointTransformer(nn.Module):
    def __init__(self, args, num_features=263, latent_dim=32, ff_size=1024, num_layers_E=3, num_layers_D=3, num_heads=8, dropout=0.1, activation="gelu",
                weight_fm=True, use_input=True) -> None:
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
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.position_encoder)
        
        self.input_process = InputProcess(self.num_features, self.latent_dim)
        self.root_process = InputProcess(7, self.latent_dim)
        self.joint_process = InputProcess(3, self.latent_dim)
        self.foot_process = InputProcess(4, self.latent_dim)

        self.output_process = OutputProcess(self.num_features, self.latent_dim*22)

        self.norm_layer = nn.LayerNorm

        use_fm = [-1]
        self.weight_fm = weight_fm
        self.use_input = use_input # use input as one of the feature maps
        if len(use_fm) == 1 and use_fm[0] == -1:
            self.use_fm = list(range(num_layers_E))
        else:
            self.use_fm = [i if i >= 0 else num_layers_E + i for i in use_fm]
        if self.weight_fm:
            # print("Weighting feature maps!")
            # print("using feature maps: ", self.use_fm)
            dec_norms = []
            for i in range(num_layers_D):
                norm_layer_i = self.norm_layer(latent_dim)
                dec_norms.append(norm_layer_i)
            self.dec_norms = nn.ModuleList(dec_norms)

            # feature weighting
            self.wfm = WeightedFeatureMaps(len(self.use_fm) + (1 if self.use_input else 0), latent_dim, norm_layer=self.norm_layer, decoder_depth=num_layers_D)

        self.self_attn = True #use self attention in decoder

        self.encoder = nn.ModuleList([
            Block(self.latent_dim, self.num_heads, mlp_ratio=2, qkv_bias=True, qk_scale=None, norm_layer=self.norm_layer)
            for i in range(self.num_layers_E)])

        self.decoder = nn.ModuleList([
            CrossAttentionBlock(self.latent_dim, self.latent_dim, self.num_heads, mlp_ratio=2, qkv_bias=True, qk_scale=None, norm_layer=self.norm_layer, self_attn=self.self_attn)
            for i in range(self.num_layers_D)])
        self.decoder_norm = self.norm_layer(self.latent_dim)

    
    def forward(self, x, t, text_emb=None, word_emb=None, attn_mask=None):
        '''
        x: (b,196,263), gt with mask noise here
        text_emb: (b,512)
        word_emb: (b,77,512)
        attn_mask: mask indicating valid control signal(b,196,J,3)
        '''
        # root, joint, foot = self.preprocess(x)  
        # root_emb = self.root_process(root)[:, :, None] # (b,196,1,d)
        # joint_emb = self.joint_process(joint) # (b,196,21,d)
        # foot_emb = self.foot_process(foot)[:, :, None] # (b,196,1,d)
        # motion_emb = torch.cat([root_emb, joint_emb, foot_emb], dim=2) #(b,196,J,d)
        # motion_emb = motion_emb.permute(0,3,1,2) #(b,d,196,J) 
        # # motion_emb = self.down_conv1(motion_emb)
        # # motion_emb = self.down_conv2(motion_emb) #(b,d,49,24) 
        # x_emb = motion_emb.flatten(2,3).permute(2,0,1) # b,196*J,d, this is [B N C] now

        root, joint = self.preprocess(x)  
        root_emb = self.root_process(root)[:, :, None] # (b,196,1,d)
        joint_emb = self.joint_process(joint) # (b,196,21,d)
        motion_emb = torch.cat([root_emb, joint_emb], dim=2) #(b,196,J,d)

        '''patchify this embedding'''
        # motion_emb = motion_emb.permute(0,3,1,2) #(b,d,196,J) 
        x_emb = motion_emb.flatten(1,2) # b,196*J,d, this is [B N C] now

        vis_mask = convert_mask(attn_mask)

        vis_emb = x_emb[vis_mask,:]
        vis_emb = self.position_encoder(vis_emb)

        enc_feats = []
        if self.use_input:
            enc_feats.append(vis_emb)
        for idx,blk in enumerate(self.encoder):
            vis_emb = blk(vis_emb)
            if self.weight_fm:
                enc_feats.append(vis_emb)

        mask_emb = x_emb[~vis_mask,:] #already add t step noise outside
        mask_emb = self.position_encoder(mask_emb)
        t_emb = self.embed_timestep(t)
        mask_emb += t_emb
        
        if self.weight_fm:
        # y input: a list of Tensors (B, C, D)
            enc_feats = self.wfm(enc_feats)

        for idx, blk in enumerate(self.decoder):
            if self.weight_fm:
                mask_emb = blk(mask_emb, self.dec_norms[idx](enc_feats[..., idx]))
            else:
                mask_emb = blk(mask_emb, vis_emb)
                
        mask_emb = self.decoder_norm(mask_emb)
        x_emb[~vis_mask,:] = mask_emb
        x_out = x_emb.reshape(x.shape[0],x.shape[1],-1)
        x_out = self.output_process(x_out)
        #calc loss in diffusion loop
        return x_out 
    
    def preprocess(self, x):
        B, L, num_features = x.shape
        if num_features == 263: # 如果是HumanML3D 
            n_joints = 22
            root = torch.cat([x[..., :4], x[..., 193:196]], dim=-1)
            
            ric = x[..., 4:67].view(B, L, n_joints-1, 3)
            # rot = x[..., 67:193].view(B, L, n_joints-1, 6)
            # vel = x[..., 196:259].view(B, L, n_joints-1, 3)
            
            # ric = torch.cat([x[..., 4:35],x[...,35:67]], dim=-1).view(B, L, n_joints-2, 3)
            # rot = torch.cat([x[..., 67:127],x[...,139:193]], dim=-1).view(B, L, n_joints-2, 3)
            # vel = torch.cat([x[..., 193:223],x[...,229:259]], dim=-1).view(B, L, n_joints-2, 3)
            
            # LFrot = x[..., 127:133].view(B, L, 1, 6)
            # RFrot = x[..., 133:139].view(B, L, 1, 6)

            # LFric = x[..., 35:39].view(B, L, 1, 3)
            # RFric = x[..., 39:43].view(B, L, 1, 3)
        
            # RFvel = x[..., 223:226].view(B, L, 1, 3)
            # RFvel = x[..., 226:229].view(B, L, 1, 3)

            # Lfoot = x[..., 259:261].view(B, L, 1, 2)
            # Rfoot = x[..., 261:263].view(B, L, 1, 2)

            # joint = torch.cat([ric, rot, vel], dim=-1) # (B,L,n_joints,12)
            # joint = torch.cat([ric, rot, vel], dim=-1) # (B,L,n_joints,12)
            joint = ric # (B,L,n_joints,3)
        elif num_features == 251:
            pass
        return root, joint#, foot
    
def convert_mask(attn_mask):
    '''convert joint mask into embedding mask         
    attn_mask:B 196 22 3 mask exported by dataset
    '''
    #ric only: 67->latent_dim
    vis_mask = attn_mask[...,0].flatten(1,2)
    # vis_mask = vis_mask[...,None].repeat(1,1,32)
    return vis_mask

class InputProcess(nn.Module):
    def __init__(self, input_feats=263, latent_dim=512):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim) # (263,512)

    def forward(self, x):
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
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
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
    
if __name__ == '__main__':
    x = torch.randn(1,196,263)
    
