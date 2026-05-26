import torch
from models.koopman_arch import KoopmanModel

def load_trained_model(model_path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    latent_dim = config["latent_dim"]

    x_dim = 15
    u_dim = 4

    model = KoopmanModel(x_dim=x_dim, z_dim=latent_dim, u_dim=u_dim, dt=config["dt"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    print("Model Loaded.")
    print(f"Device: {device}")
    print(f"Latent Dimension: {latent_dim}")
    return model, config