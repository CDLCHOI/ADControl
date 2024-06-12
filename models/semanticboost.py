import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
# from models.clip import clip
import clip
import json
from .semboost.base_transformer import RefinedLayer, Refined_Transformer
from .semboost.Encode_Full import Encoder_Block

class SemanticBoost(nn.Module):
    def __init__(self, njoints, nfeats, latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 activation="gelu", dataset='amass', clip_dim=512,
                 arch='trans_enc', clip_version=None, **kargs):
        super().__init__()

        self.local = kargs["local"]
        self.encode_full = kargs.get("encode_full", 0)      #### encode_full = 1 add tokens  & encode_full = 2 model compress tokens
        self.txt_tokens = kargs.get("txt_tokens", 0)    #### txt_tokens = 1 add tokens  & txt_tokens = 2 model compress tokens
        self.dataset = dataset
        self.condition_length = 77
        self.num_frames = kargs.get("num_frames", 196)
        self.json_dict = kargs.get("json_dict")

        if arch.endswith("static"): # arch = 'llama_decoder_static'
            self.position_type = "static"     #### [static or rope]  only for llama arch
            self.arch = arch.replace("_static", "") # self.arch = 'llama_decoder_static'
        elif arch.endswith("rope"):
            self.position_type = "rope"
            self.arch = arch.replace("_rope", "")
        else:
            self.position_type = "static"
            self.arch = arch

        if isinstance(self.num_frames, list) or isinstance(self.num_frames, tuple):
            self.num_frames = self.num_frames[0]

        self.njoints = njoints # 269
        self.nfeats = nfeats # 1

        self.latent_dim = latent_dim # 512

        self.ff_size = ff_size # 1024
        self.num_layers = num_layers # 8
        self.num_heads = num_heads # 4
        self.dropout = dropout # 0.1

        self.activation = activation # 'swiglu'
        self.clip_dim = clip_dim # 512
        self.action_emb = kargs.get('action_emb', None)

        self.input_feats = self.njoints * self.nfeats

        self.cond_mode = kargs.get('cond_mode', 'no_cond') # 'text'
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.) # 0.1


        self.input_process = InputProcess(self.input_feats, self.latent_dim)    #### 输入 x 的 linear
        self.output_process = OutputProcess(self.input_feats, self.latent_dim, self.njoints,
                                            self.nfeats)

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        if self.arch == 'trans_enc':
            print("TRANS_ENC init")
            seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)
            self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer, num_layers=self.num_layers)

        elif self.arch == "llama_encoder":
            TransLayer = RefinedLayer(self.latent_dim, self.num_heads, self.ff_size, self.dropout, self.activation, max_seq_len=self.num_frames, position_type=self.position_type, norm_type="rmsnorm")
            self.seqTransEncoder = Refined_Transformer(TransLayer, self.num_layers)

        elif self.arch == "llama_decoder": # 这
            TransLayer = RefinedLayer(self.latent_dim, self.num_heads, self.ff_size, self.dropout, self.activation, max_seq_len=self.num_frames, position_type=self.position_type, word_tokens=True, norm_type="rmsnorm")
            self.seqTransEncoder = Refined_Transformer(TransLayer, self.num_layers) # 里面有cross attention

        else:
            raise ValueError('Please choose correct architecture')

        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        if self.cond_mode != 'no_cond':
            if 'text' in self.cond_mode:
                self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)
                print('EMBED TEXT')
                print('Loading CLIP...')
                self.clip_version = clip_version
                self.clip_model = self.load_and_freeze_clip(clip_version)

                if self.txt_tokens == 2:
                    if self.arch in ["trans_enc", "llama_encoder"]:
                        scale = 3
                    elif self.arch in ["llama_decoder"]: # 这
                        scale = 2

                    encode_compress_layer = RefinedLayer(d_model=self.latent_dim * scale,
                                                                    nhead=self.num_heads,
                                                                    dim_feedforward=self.ff_size,
                                                                    dropout=self.dropout,
                                                                    activation=self.activation, norm_type="rmsnorm")
                    self.condition_compress = nn.Sequential(
                        Refined_Transformer(encode_compress_layer, num_layers=1),
                        nn.Linear(self.latent_dim * scale, self.latent_dim, )
                    )       

        if self.encode_full != 0: ####  [1, bs, 512] -> [seq, bs, 1024] -> [seq, bs, 512]
            self.code_full = Encoder_Block(begin_channel=self.input_feats, latent_dim=self.latent_dim, num_layers=6, TN=1, bias=kargs["conv_bias"], norm_type=kargs["conv_norm"], activate_type=kargs["conv_activate"])      

            if self.encode_full == 2:
                encode_compress_layer = RefinedLayer(d_model=self.latent_dim * 2,
                                                                nhead=self.num_heads,
                                                                dim_feedforward=self.ff_size,
                                                                dropout=self.dropout,
                                                                activation=self.activation, norm_type="rmsnorm")
                # dynamic-enrich模块最后的transformer encoder和linear
                self.encode_compress = nn.Sequential(
                    Refined_Transformer(encode_compress_layer, num_layers=1),
                    nn.Linear(self.latent_dim * 2, self.latent_dim, )
                )

        print(" =========================", self.cond_mode, "===================================")

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu', jit=False)  # Must set jit=False for training
        clip_model.float()
    
        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    def mask_cond(self, cond, force_mask=False):
        bs = cond.shape[0]
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob)  # 1-> use null_cond, 0-> use real cond
            if len(cond.shape) == 3:
                mask = mask.view(bs, 1, 1)
            else:
                mask = mask.view(bs, 1)
            return cond * (1. - mask)
        else:
            return cond

    def clip_text_embedding(self, raw_text):
        device = self.clip_model.ln_final.weight.device
        default_context_length = self.condition_length
        # 得到的texts为 [bs, context_length]，每一个值的含义是该单词在词汇表里的index，标点符号如句号也会得到一个index
        # 最前面和最后面的word index默认为49406和49407，猜测为SOS(start of sentence)和EOS(end of sentence)
        # 还有默认长度default_context_length设为77，说是所有clipmodel都用的是77，所以如果超过77的会截断，不到77的会补0
        texts = clip.tokenize(raw_text, context_length=default_context_length, truncate=True).to(device) # （b, 77）
        if self.txt_tokens == 0:   
            clip_feature = self.clip_model.encode_text(texts)
        else:
            with torch.no_grad():
                x = self.clip_model.token_embedding(texts)  # [batch_size, n_ctx, d_model] 将word index转化为embedding
                x = x + self.clip_model.positional_embedding
                x = x.permute(1, 0, 2)  # NLD -> LND
                x = self.clip_model.transformer(x)
                x = x.permute(1, 0, 2)  # LND -> NLD
                x = self.clip_model.ln_final(x) # (b,77,512) 每个word的单独特征
                # 每个batch都取出从0开始到EOS的位置（由argmax获取，上面提到EOS的index为49407），得到（b, 512）
                # 跟后面的投影矩阵(512,512)做乘法提取得到 整个句子的 全局特征
                clip_feature = x[torch.arange(x.shape[0]), texts.argmax(dim=-1)] @ self.clip_model.text_projection
            clip_feature = clip_feature.unsqueeze(1)
            clip_feature = torch.cat([clip_feature, x], dim=1)     #### [bs, T, 512]
        return clip_feature
        
    def get_mask(self, sz1, sz2):
        mask = (torch.triu(torch.ones(sz1, sz2)) == 1).transpose(0, 1)
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask.requires_grad = False
        return mask

    def forward(self, x, timesteps, y=None):
        """
        x: 即论文中的xt, (b, 269, 1, 120) [batch_size, njoints, nfeats, max_frames]
        timesteps: (b,)  999~0
        """
        
        results = {}
        emb = self.embed_timestep(timesteps)  # [1, bs, d] 包含了位置编码
        x = x.to(emb.dtype)

        real_length = x.shape[-1] # 真实动作长度，即输入的x中动作长度

        '''
        真是动作长度小于预设的最大动作长度, 补0
        x.shape从(b,269,1,120) -> (b,269,1,196)
        '''
        if self.encode_full != 0 and x.shape[-1] < self.num_frames:
            extension = torch.zeros([x.shape[0], x.shape[1], x.shape[2], self.num_frames - x.shape[-1]], device=x.device, dtype=x.dtype)
            x = torch.cat([x, extension], dim=-1) # 得到补0扩展后的动作

        if self.encode_full == 1:
            latent = self.code_full(x) ### [seq, bs, 512]
            current = self.input_process(x)       
            latent = latent.repeat(current.shape[0], 1, 1)
            current = current + latent
        elif self.encode_full == 2:
            latent = self.code_full(x) ### [seq, bs, 512] 文中提到的动作的global information 即 XG
            current = self.input_process(x)                      #### [seq, bs, 512]
            latent = latent.repeat(current.shape[0], 1, 1)
            current = torch.cat([current, latent], dim=2) # (196,1,1024) 这里对应的就是论文图中Xt和XG的concat
            ''' 此处current就是dynamic-enrich模块的最终输出 '''
            current = self.encode_compress(current) # (196,1,512) dynamic-enrich模块最后的transformer encoder和linear
        else:
            current = self.input_process(x)                      #### [seq, bs, 512]

        force_mask = y.get('uncond', False)
        if 'text' in self.cond_mode:
            enc_text = self.clip_text_embedding(y['text']).to(emb.dtype) # (b,78,512)  ### MASK_COND 
            txt_emb = self.embed_text(enc_text) # (b,78,512)
            txt_emb = self.mask_cond(txt_emb, force_mask=force_mask) # 会按照一定的比例把 batch_size 中的一部分文本句整句换成 [0, 0, ... 0]
            
            if len(txt_emb.shape) == 3:
                txt_emb = txt_emb.permute(1, 0, 2) # (78,b,512)
            else:
                txt_emb = txt_emb.unsqueeze(0)
        else:
            txt_emb = None

        if txt_emb is not None:
            all_emb = txt_emb
        else:
            all_emb = torch.zeros_like(emb)

        if self.arch in ["trans_enc", "llama_encoder"] and txt_emb is not None:
            if self.txt_tokens == 1:
                word_embedding = all_emb[1::, :, :]
                global_embedding = all_emb[0:1, :, :].repeat(word_embedding.shape[0], 1, 1)
                all_emb = word_embedding + global_embedding
                emb = emb.repeat(all_emb.shape[0], 1, 1)
                emb += all_emb
            elif self.txt_tokens == 2: # 这
                word_embedding = all_emb[1::, :, :]
                global_embedding = all_emb[0:1, :, :].repeat(word_embedding.shape[0], 1, 1)
                emb = emb.repeat(word_embedding.shape[0], 1, 1)
                concat_embedding = torch.cat([emb, global_embedding, word_embedding], dim=2)
                emb = self.condition_compress(concat_embedding)
            else:
                emb += all_emb
        elif txt_emb is not None: 
            if self.txt_tokens == 1:
                emb = emb.repeat(all_emb.shape[0], 1, 1)
                emb += all_emb
            elif self.txt_tokens == 2: # 这里
                emb = emb.repeat(all_emb.shape[0], 1, 1) # emb是timestep的特征: (1,b,512) -> (78,b,512)
                concat_embedding = torch.cat([emb, all_emb], dim=2) # 对应论文图把timestep特征和文本特征concat 
                emb = self.condition_compress(concat_embedding) # 和dynamic-enrich模块一样的结构，包含transformer encoder和linear  
            else:
                emb += all_emb 
        else:
            emb = emb.repeat(all_emb.shape[0], 1, 1)
            emb += all_emb

        if self.arch in ["trans_enc", "llama_encoder"]:
            real_token_length = emb.shape[0]           ######### 用来截断输出，只保留真正的output
        elif self.arch in ["llama_decoder"]: # 这里
            real_token_length = 1

        if self.arch in ["trans_enc", "llama_encoder"]:
            xseq = torch.cat([emb, current], dim=0)

            if self.arch in ["trans_enc"] or self.position_type == "static":
                xseq = self.sequence_pos_encoder(xseq)

            output = self.seqTransEncoder(xseq)

        elif self.arch in ["llama_decoder"]: # 这里
            if emb.shape[0] == 1:
                emb = emb.repeat(1+self.condition_length, 1, 1)

            # emb即为论文图中Tg,Tw1,Tw2,...,TwK
            xseq = torch.cat([emb[0:1], current], dim=0) # 论文网络图中的Tg，即 text global
            word_tokens = emb[1::]

            if self.position_type == "static":
                xseq = self.sequence_pos_encoder(xseq)
                
            '''
            论文主体Context-Attuned Motion Denoiser中的米黄色背景那部分
            接受word token即论文图中Tw, 以及dynamic-enrich模块输出的motion token
            '''
            output = self.seqTransEncoder(xseq, word_tokens=word_tokens) 

        output = output[real_token_length:]
        output = self.output_process(output)  # [bs, njoints, nfeats, nframes] 输出的Linear
        output = output[:, :, :, :real_length] # HumanML3D长度是196，实际是120，做个切片
        # results["output"] = output
        return output
  
    def _apply(self, fn):
        super()._apply(fn)

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):  
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)      ###### max_len 是 T_steps 长度， d_model 是嵌入特征的维度
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_parameter('pe', nn.Parameter(pe, requires_grad=False))

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
            nn.Linear(self.latent_dim, time_embed_dim, ),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim, ),
        )

    def forward(self, timesteps):       #### timesteps 也是按照 position 的方式编码的 [times, 1, latent] -> [1, times, latent] ?
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class InputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
      
    def forward(self, x):
        bs, njoints, nfeats, nframes = x.shape          ### [B,263, nframes] -> [B, nframes, 263]
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints*nfeats) 
        x = self.poseEmbedding(x)  # [seqlen, bs, d]
        return x

     
class OutputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim, njoints, nfeats):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats

        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
    

    def forward(self, output):
        nframes, bs, d = output.shape
        output = self.poseFinal(output)  # [seqlen, bs, 150]
        output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        output = output.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]

        return output


class EmbedAction(nn.Module):
    def __init__(self, num_actions, latent_dim):
        super().__init__()
        self.action_embedding = nn.Parameter(torch.randn(num_actions, latent_dim))

    def forward(self, input):
        idx = input[:, 0].to(torch.long)  # an index array must be long
        output = self.action_embedding[idx]
        return output