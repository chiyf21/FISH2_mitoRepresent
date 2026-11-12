"""MAE patch preparation copied from FOCUS3D/help_function.py."""
import random
from pathlib import Path
import numpy as np
import tifffile
from scipy.ndimage import zoom

def read_tiff_as_zyx(path):
    """
    Robust TIFF reader.

    Supports:
    1. Normal 3D TIFF: directly returns (Z, Y, X)
    2. Multi-page TIFF where each page is YX: stacks pages into (Z, Y, X)
       Example: IDR068 image.ome_T118_C003.tiff
    """
    path = str(path)

    with tifffile.TiffFile(path) as tif:
        # First try normal reading.
        arr = tif.asarray()
        arr = np.squeeze(arr)

        # Case 1: normal 3D volume
        if arr.ndim == 3:
            return arr

        # Case 2: tifffile only returns one YX page,
        # but the file actually contains many YX pages.
        if arr.ndim == 2 and len(tif.pages) > 1:
            first_shape = tif.pages[0].shape
            first_dtype = tif.pages[0].dtype

            same_shape = all(page.shape == first_shape for page in tif.pages)

            if same_shape:
                vol = np.empty((len(tif.pages), *first_shape), dtype=first_dtype)

                for z, page in enumerate(tif.pages):
                    vol[z] = page.asarray()

                return vol

        raise ValueError(
            f"Expected 3D TIFF or multi-page YX TIFF, "
            f"but got arr.shape={arr.shape}, pages={len(tif.pages)}"
        )

def percentile_normalize(img, p_low=1.0, p_high=99.0, return_stats=False):
    lo = np.percentile(img, p_low)
    hi = np.percentile(img, p_high)

    if hi <= lo:
        out = np.zeros_like(img, dtype=np.float32)
        if return_stats:
            return out, lo, hi
        return out

    img = np.clip(img, lo, hi)
    img = (img - lo) / (hi - lo)
    img = img.astype(np.float32)

    if return_stats:
        return img, lo, hi
    return img

def random_crop_3d(img, patch_size):
    """
    Randomly crop a 3D patch from img.

    Args:
        img: 3D numpy array with shape (D, H, W)
        patch_size: tuple/list (pd, ph, pw)

    Returns:
        patch: 3D numpy array
    """
    pd, ph, pw = patch_size
    d, h, w = img.shape

    if d < pd or h < ph or w < pw:
        return None

    z0 = random.randint(0, d - pd)
    y0 = random.randint(0, h - ph)
    x0 = random.randint(0, w - pw)

    return img[z0:z0 + pd, y0:y0 + ph, x0:x0 + pw]

def resize_3d_by_radius(img, cell_radius, target_radius, order=1, scale_z=True):
    """
    Resize a 3D image according to target_radius / cell_radius.
    """
    scale = float(target_radius) / float(cell_radius)

    if scale_z:
        zoom_factors = (scale, scale, scale)
    else:
        zoom_factors = (1.0, scale, scale)

    resized = zoom(img, zoom=zoom_factors, order=order)
    return resized

def force_min_shape_by_global_scaling(img, min_shape, order=1):
    """
    If any axis of img is smaller than min_shape, scale the entire 3D volume
    isotropically with one global scale factor.

    Example:
        img.shape = (31, 307, 427), min_shape = (32, 96, 96)
        global_scale = 32 / 31
        output shape ~= (32, 317, 441)

        img.shape = (19, 307, 427), min_shape = (32, 96, 96)
        global_scale = 32 / 19
        output shape ~= (32, 517, 719)
    """
    img = np.asarray(img)

    if img.ndim != 3:
        raise ValueError(f"Expected 3D image, got shape={img.shape}")

    d, h, w = img.shape
    pd, ph, pw = min_shape

    global_scale = max(
        float(pd) / float(d),
        float(ph) / float(h),
        float(pw) / float(w),
        1.0,
    )

    if global_scale <= 1.0:
        return img.astype(np.float32, copy=False), 1.0

    out = zoom(
        img,
        zoom=(global_scale, global_scale, global_scale),
        order=order,
    ).astype(np.float32, copy=False)

    # 极少数情况下，zoom 的四舍五入可能仍导致某个轴差 1 个 voxel。
    # 这种情况下再补一次整体 scale，而不是只补单轴。
    for _ in range(2):
        od, oh, ow = out.shape
        if od >= pd and oh >= ph and ow >= pw:
            break

        extra_scale = max(
            float(pd) / float(od),
            float(ph) / float(oh),
            float(pw) / float(ow),
            1.0,
        )

        if extra_scale <= 1.0:
            break

        global_scale *= extra_scale
        out = zoom(
            out,
            zoom=(extra_scale, extra_scale, extra_scale),
            order=order,
        ).astype(np.float32, copy=False)

    return out, global_scale

def sample_patches_from_one_volume(
    img,
    patch_size,
    intensity_quantile=0.3,
    max_trials=300,
    black_eps=0.0,
):
    candidate_patches = []
    candidate_means = []

    d, h, w = img.shape
    pd, ph, pw = patch_size

    if d < pd or h < ph or w < pw:
        print(
            f"[SAMPLE_IMPOSSIBLE_SHAPE] "
            f"img_shape={img.shape} patch_size={patch_size}",
            flush=True,
        )
        return []

    black_patch_count = 0

    for _ in range(max_trials):
        patch = random_crop_3d(img, patch_size)
        if patch is None:
            break

        # 关键改动：切出来之后先判断是不是全黑块
        patch_max = float(np.max(patch))

        if patch_max <= black_eps:
            black_patch_count += 1
            continue

        candidate_patches.append(patch)
        candidate_means.append(float(patch.mean()))

    if len(candidate_patches) == 0:
        print(
            f"[SAMPLE_NO_CANDIDATES] "
            f"img_shape={img.shape} patch_size={patch_size} "
            f"max_trials={max_trials} "
            f"black_patch_count={black_patch_count}",
            flush=True,
        )
        return []

    threshold = float(np.quantile(candidate_means, intensity_quantile))

    kept_patches = [
        patch for patch, mean_val in zip(candidate_patches, candidate_means)
        if mean_val >= threshold
    ]

    print(
        f"[SAMPLE_STATS] "
        f"img_shape={img.shape} patch_size={patch_size} "
        f"candidate_count={len(candidate_patches)} "
        f"kept_count={len(kept_patches)} "
        f"black_patch_count={black_patch_count} "
        f"mean_min={min(candidate_means):.6f} "
        f"mean_max={max(candidate_means):.6f} "
        f"mean_threshold={threshold:.6f}",
        flush=True,
    )

    return kept_patches

def build_mae_patches(
    input_dir,
    output_dir,
    cell_radius,
    patch_size,
    target_radius,
    num_patches_needed,
    intensity_quantile=0.2,
    max_trials_per_volume=3000,
    valid_suffixes=(".tif", ".tiff"),
    random_seed=12,
    scale_z=True,
):
    """
    Randomly read 3D images from input_dir, resize them according to the
    radius ratio, normalize them using percentiles, sample patches,
    filter low-intensity patches, and save patches to output_dir.

    This version logs all issues that may affect patch generation.
    """
    import traceback

    random.seed(random_seed)
    np.random.seed(random_seed)

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    print("=" * 100, flush=True)
    print(f"[BUILD_START] input_dir={input_dir}", flush=True)
    print(f"[BUILD_OUTPUT] output_dir={output_dir}", flush=True)
    print(
        f"[BUILD_PARAMS] "
        f"cell_radius={cell_radius} "
        f"target_radius={target_radius} "
        f"patch_size={patch_size} "
        f"num_patches_needed={num_patches_needed} "
        f"intensity_quantile={intensity_quantile} "
        f"max_trials_per_volume={max_trials_per_volume} "
        f"scale_z={scale_z}",
        flush=True,
    )

    if not input_dir.exists():
        print(f"[INPUT_DIR_NOT_FOUND] {input_dir}", flush=True)
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)

    file_list = [
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in valid_suffixes
    ]

    print(f"[FILE_COUNT] input_dir={input_dir} n_files={len(file_list)}", flush=True)

    if len(file_list) == 0:
        print(f"[NO_TIFF_FOUND] input_dir={input_dir}", flush=True)
        return 0

    random.shuffle(file_list)

    saved_count = 0
    read_error_count = 0
    not_3d_count = 0
    too_small_after_resize_count = 0
    no_valid_patch_count = 0
    write_error_count = 0
    processed_volume_count = 0

    pd, ph, pw = patch_size
    scale = float(target_radius) / float(cell_radius)
    zoom_factors = (scale, scale, scale) if scale_z else (1.0, scale, scale)

    for img_path in file_list:
        if saved_count >= num_patches_needed:
            print(
                f"[TARGET_REACHED] saved_count={saved_count} "
                f"needed={num_patches_needed}",
                flush=True,
            )
            break

        print("-" * 100, flush=True)
        print(f"[FILE_START] {img_path}", flush=True)

        try:
            img = read_tiff_as_zyx(img_path)
        except Exception as e:
            read_error_count += 1
            print(f"[READ_ERROR] file={img_path} error={repr(e)}", flush=True)
            continue

        img = np.asarray(img)

        if img.ndim != 3:
            not_3d_count += 1
            print(
                f"[NOT_3D_SKIP] file={img_path} shape={img.shape} ndim={img.ndim}",
                flush=True,
            )
            continue

        raw_shape = tuple(int(x) for x in img.shape)
        raw_dtype = str(img.dtype)

        print(
            f"[RAW_VOLUME] file={img_path.name} "
            f"raw_shape={raw_shape} raw_dtype={raw_dtype} "
            f"raw_min={float(np.min(img)):.6f} raw_max={float(np.max(img)):.6f}",
            flush=True,
        )

        try:
            img = img.astype(np.float32, copy=False)

            resized_img = resize_3d_by_radius(
                img,
                cell_radius=cell_radius,
                target_radius=target_radius,
                order=1,
                scale_z=scale_z,
            )
        except Exception as e:
            print(
                f"[RESIZE_ERROR] file={img_path} "
                f"raw_shape={raw_shape} zoom_factors={zoom_factors} "
                f"error={repr(e)}",
                flush=True,
            )
            traceback.print_exc()
            continue

        resized_shape = tuple(int(x) for x in resized_img.shape)

        print(
            f"[RESIZED_VOLUME] file={img_path.name} "
            f"raw_shape={raw_shape} resized_shape={resized_shape} "
            f"zoom_factors={zoom_factors}",
            flush=True,
        )

        if (
            resized_shape[0] < pd
            or resized_shape[1] < ph
            or resized_shape[2] < pw
        ):
            before_force_shape = resized_shape

            try:
                resized_img, force_global_scale = force_min_shape_by_global_scaling(
                    resized_img,
                    min_shape=patch_size,
                    order=1,
                )
            except Exception as e:
                too_small_after_resize_count += 1
                print(
                    f"[FORCE_GLOBAL_SCALE_ERROR] file={img_path.name} "
                    f"raw_shape={raw_shape} resized_shape={before_force_shape} "
                    f"patch_size={patch_size} error={repr(e)}",
                    flush=True,
                )
                traceback.print_exc()
                continue

            resized_shape = tuple(int(x) for x in resized_img.shape)

            print(
                f"[FORCE_GLOBAL_SCALE] file={img_path.name} "
                f"raw_shape={raw_shape} "
                f"before_shape={before_force_shape} "
                f"after_shape={resized_shape} "
                f"patch_size={patch_size} "
                f"force_global_scale={force_global_scale:.6f}",
                flush=True,
            )

            if (
                resized_shape[0] < pd
                or resized_shape[1] < ph
                or resized_shape[2] < pw
            ):
                too_small_after_resize_count += 1
                print(
                    f"[TOO_SMALL_AFTER_FORCE_GLOBAL_SCALE_SKIP] file={img_path.name} "
                    f"raw_shape={raw_shape} resized_shape={resized_shape} "
                    f"patch_size={patch_size}",
                    flush=True,
                )
                continue

        try:
            p1 = float(np.percentile(resized_img, 1.0))
            p995 = float(np.percentile(resized_img, 99.5))

            print(
                f"[NORMALIZE_STATS] file={img_path.name} "
                f"p1={p1:.6f} p99_5={p995:.6f}",
                flush=True,
            )

            if p995 <= p1:
                print(
                    f"[NORMALIZE_DEGENERATE_WARNING] file={img_path.name} "
                    f"p1={p1:.6f} p99_5={p995:.6f}",
                    flush=True,
                )

            resized_img = percentile_normalize(
                resized_img,
                p_low=1.0,
                p_high=99.5,
            )

        except Exception as e:
            print(
                f"[NORMALIZE_ERROR] file={img_path} "
                f"resized_shape={resized_shape} error={repr(e)}",
                flush=True,
            )
            traceback.print_exc()
            continue

        try:
            kept_patches = sample_patches_from_one_volume(
                resized_img,
                patch_size=patch_size,
                intensity_quantile=intensity_quantile,
                max_trials=max_trials_per_volume,
            )
        except Exception as e:
            print(
                f"[SAMPLE_ERROR] file={img_path} "
                f"resized_shape={resized_shape} patch_size={patch_size} "
                f"error={repr(e)}",
                flush=True,
            )
            traceback.print_exc()
            continue

        if len(kept_patches) == 0:
            no_valid_patch_count += 1
            print(
                f"[NO_VALID_PATCHES] file={img_path.name} "
                f"raw_shape={raw_shape} resized_shape={resized_shape} "
                f"patch_size={patch_size} "
                f"intensity_quantile={intensity_quantile} "
                f"max_trials={max_trials_per_volume}",
                flush=True,
            )
            continue

        random.shuffle(kept_patches)

        saved_before_file = saved_count

        for patch in kept_patches:
            if saved_count >= num_patches_needed:
                break

            if patch.shape != tuple(patch_size):
                print(
                    f"[BAD_PATCH_SHAPE_SKIP] file={img_path.name} "
                    f"patch_shape={patch.shape} expected={patch_size}",
                    flush=True,
                )
                continue

            out_name = f"{img_path.stem}_patch_{saved_count:06d}.tif"
            out_path = output_dir / out_name

            try:
                tifffile.imwrite(str(out_path), patch.astype(np.float32))
                saved_count += 1
            except Exception as e:
                write_error_count += 1
                print(
                    f"[WRITE_ERROR] out_path={out_path} "
                    f"patch_shape={patch.shape} error={repr(e)}",
                    flush=True,
                )
                continue

        saved_this_file = saved_count - saved_before_file
        processed_volume_count += 1

        print(
            f"[FILE_DONE] file={img_path.name} "
            f"kept_patches={len(kept_patches)} "
            f"saved_this_file={saved_this_file} "
            f"total_saved={saved_count}",
            flush=True,
        )

    print("=" * 100, flush=True)
    print(f"[BUILD_END] input_dir={input_dir}", flush=True)
    print(f"[BUILD_SUMMARY] input_dir={input_dir}", flush=True)
    print(f"  total_files={len(file_list)}", flush=True)
    print(f"  processed_volume_count={processed_volume_count}", flush=True)
    print(f"  saved_count={saved_count}", flush=True)
    print(f"  num_patches_needed={num_patches_needed}", flush=True)
    print(f"  read_error_count={read_error_count}", flush=True)
    print(f"  not_3d_count={not_3d_count}", flush=True)
    print(f"  too_small_after_resize_count={too_small_after_resize_count}", flush=True)
    print(f"  no_valid_patch_count={no_valid_patch_count}", flush=True)
    print(f"  write_error_count={write_error_count}", flush=True)

    if saved_count == 0:
        print(
            f"[BUILD_WARNING_ZERO_PATCH] input_dir={input_dir} "
            f"No patches were saved.",
            flush=True,
        )
    elif saved_count < num_patches_needed:
        print(
            f"[BUILD_WARNING_INSUFFICIENT_PATCHES] input_dir={input_dir} "
            f"saved_count={saved_count} needed={num_patches_needed}",
            flush=True,
        )

    return saved_count
