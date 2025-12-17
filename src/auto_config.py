def auto_config(profile):
    # Nâng cấp: latent_dim = min(16, số feature/4)
    input_dim = profile['num_features'] + profile['cat_features']
    latent_dim = min(16, max(4, input_dim//4))
    return {
        "latent_dim": latent_dim,
        "epochs": 50,
        "contamination": 0.01,
        "use_global": True,
        "use_local": True,
        "use_sparse": True,
        "early_stopping": True,
        "patience": 5
    }
