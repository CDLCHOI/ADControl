def get_semboost_args(args):

    clip_version = 'ViT-B/32'
    args.arch = "llama_decoder_rope"
    # cond_mode = 'no_cond'
    cond_mode = "text"
     
    activation = "swiglu"

    if args.dataname == 't2m':
        njoints = 263
        nfeats = 1
        dataset = "humanml"
    elif args.dataname == 'kit':
        njoints = 251
        nfeats = 1
        dataset = "kit"

    return {'njoints': njoints, 'nfeats': nfeats, 'latent_dim': 512, 'ff_size': 1024, 'num_layers': 8, 'num_heads': 4,
            'dropout': 0.1, 'activation': activation, 'cond_mode': cond_mode, 'cond_mask_prob': 0.1, 'arch': args.arch,
            'clip_version': clip_version, 'dataset': dataset, "local":False, "encode_full":2, "txt_tokens":2,
            "num_frames":196, "conv_bias":True, "conv_activate":"relu", 
            "conv_norm":"layernorm"}