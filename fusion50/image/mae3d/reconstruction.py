"""MAE reconstruction utilities copied from FOCUS3D/help_function.py."""
import torch

def get_masked_sample(model, dataset, idx, mask_ratio=0.75, device='cuda'):
    """
    Load a sample, apply masking, and get model reconstruction.
    Returns:
        original: (C, D, H, W) tensor on CPU
        masked_input: (C, D, H, W) tensor (what the encoder sees after masking)
        reconstruction: (C, D, H, W) tensor (model output)
        mask: (N_patches,) binary mask where 1 = removed patches
    """
    model.eval()
    with torch.no_grad():
        # Load a single sample and add batch dimension
        sample, _ = dataset[idx]                     # (C, D, H, W)
        sample = sample.unsqueeze(0).to(device)      # (1, C, D, H, W)

        # Forward through encoder to get latent, mask, ids_restore
        latent, mask, ids_restore = model.forward_encoder(sample, mask_ratio)
        # Decoder prediction (patch-level)
        pred = model.forward_decoder(latent, ids_restore)  # (1, num_patches, p^3 * C)
        # Reconstruct full volume
        rec = model.unpatchify(pred)                  # (1, C, D, H, W)

        # Create a masked version of the input for visualization:
        # For patches that are masked (mask=1), we can set them to zero or a constant.
        # Here we set them to zero to visualize what the encoder sees.
        # First, get the patchified version
        patches = model.patchify(sample)               # (1, num_patches, p^3 * C)
        # Mask patches: set removed patches to zero
        patches_masked = patches * (1 - mask.unsqueeze(-1).float())  # mask=1 -> set to zero
        # Unpatchify to get the "visible" volume
        visible_vol = model.unpatchify(patches_masked) # (1, C, D, H, W)

        return (sample.squeeze(0).cpu(),
                visible_vol.squeeze(0).cpu(),
                rec.squeeze(0).cpu(),
                mask.squeeze(0).cpu())

def visualize_3d_reconstruction(original, masked, reconstructed, mask, title_prefix=""):
    """
    original, masked, reconstructed: (C, D, H, W) tensors (C=1)
    mask: (num_patches,) mask (optional, not used directly here)
    Display middle slices along depth, height, width.
    """
    import matplotlib.pyplot as plt
    # Remove channel dimension for plotting
    orig = original.squeeze(0).numpy()      # (D, H, W)
    masked_vol = masked.squeeze(0).numpy()  # (D, H, W)
    recon = reconstructed.squeeze(0).numpy()# (D, H, W)

    D, H, W = orig.shape
    mid_d = D // 2
    mid_h = H // 2
    mid_w = W // 2

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle(title_prefix + " | Left: Original | Middle: Masked Input | Right: Reconstruction", fontsize=14)

    # Row 0: Depth slice (XY plane at mid depth)
    axes[0, 0].imshow(orig[mid_d, :, :], cmap='gray')
    axes[0, 0].set_title(f'Original (depth {mid_d})')
    axes[0, 1].imshow(masked_vol[mid_d, :, :], cmap='gray')
    axes[0, 1].set_title('Masked Input')
    axes[0, 2].imshow(recon[mid_d, :, :], cmap='gray')
    axes[0, 2].set_title('Reconstruction')

    # Row 1: Coronal slice (XZ plane at mid height)
    axes[1, 0].imshow(orig[:, mid_h, :], cmap='gray', aspect='auto')
    axes[1, 0].set_title(f'Original (height {mid_h})')
    axes[1, 1].imshow(masked_vol[:, mid_h, :], cmap='gray', aspect='auto')
    axes[1, 1].set_title('Masked Input')
    axes[1, 2].imshow(recon[:, mid_h, :], cmap='gray', aspect='auto')
    axes[1, 2].set_title('Reconstruction')

    # Row 2: Sagittal slice (YZ plane at mid width)
    axes[2, 0].imshow(orig[:, :, mid_w], cmap='gray', aspect='auto')
    axes[2, 0].set_title(f'Original (width {mid_w})')
    axes[2, 1].imshow(masked_vol[:, :, mid_w], cmap='gray', aspect='auto')
    axes[2, 1].set_title('Masked Input')
    axes[2, 2].imshow(recon[:, :, mid_w], cmap='gray', aspect='auto')
    axes[2, 2].set_title('Reconstruction')

    for ax_row in axes:
        for ax in ax_row:
            ax.axis('off')
    plt.tight_layout()
    plt.show()

def load_model(checkpoint_path, device, model_kwargs=None):
    """
    Load a trained MaskedAutoencoderViT3D model from a checkpoint.

    Args:
        checkpoint_path (str): Path to the .pth checkpoint file.
        device (torch.device): Device to load the model on.
        model_kwargs (dict): Arguments for initializing the model (same as training).
                             If None, uses the default large 3D MAE parameters.

    Returns:
        model (nn.Module): Loaded model in eval mode.
    """
    if model_kwargs is None:
        # Use the same parameters as in the training script
        model_kwargs = {
            'img_size': (32, 96, 96),
            'patch_size': (8, 12, 12),
            'in_chans': 1,
            'norm_pix_loss': False,
            'embed_dim': 1008,
            'depth': 24,
            'num_heads': 16,
            'decoder_embed_dim': 768,
            'decoder_depth': 8,
            'decoder_num_heads': 16,
            'mlp_ratio': 4,
        }

    # Instantiate the model
    from .models_mae import MaskedAutoencoderViT3D
    from functools import partial
    model_kwargs = dict(model_kwargs)
    model_kwargs.setdefault("norm_layer", partial(torch.nn.LayerNorm, eps=1e-6))
    model = MaskedAutoencoderViT3D(**model_kwargs)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Handle possible DataParallel wrapping (keys starting with 'module.')
    if isinstance(checkpoint, dict):
        # 按优先级尝试常见键名
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            # 如果没有这些键，假设整个 checkpoint 就是 state_dict
            state_dict = checkpoint
    else:
        # 如果不是字典，直接当作 state_dict
        state_dict = checkpoint

    # 处理 DataParallel 包装（移除 'module.' 前缀）
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    print(f"Model loaded from {checkpoint_path}")
    return model

load_mae_model = load_model
