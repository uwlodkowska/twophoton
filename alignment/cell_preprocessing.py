# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from skimage.draw import disk
import numpy as np
import pandas as pd
from skimage import io
from concurrent.futures import ProcessPoolExecutor
from scipy.spatial import cKDTree
from scipy.stats import median_abs_deviation as mad
from statsmodels.nonparametric.smoothers_lowess import lowess

import utils
from constants import ICY_COLNAMES
import constants

def calculate_disk(coords, radius, disk_no, img):
    center_z = coords[ICY_COLNAMES['zcol']]+disk_no
    if (center_z <0 or center_z>=img.shape[0]):
        return [0,0] #spr czy to potrzebne
    disk_ = disk((coords[ICY_COLNAMES['ycol']],coords[ICY_COLNAMES['xcol']]), radius,shape = img[0].shape) 
    sum_int = np.sum(img[center_z][disk_])
    area_int = len(img[center_z][disk_])
    return [sum_int, area_int]

def calculate_intensity_old(coords, img):
    center_coords_df = coords[constants.COORDS_3D]
    center_coords_df = center_coords_df.round().astype(int)
    sum_int = 0
    area_int = 0
    for i in range(-2,3): #going through 5 flat slices making up the 3d cell
        diameter = constants.ROI_DIAMETER[abs(i)]
        rad = diameter//2
        res = calculate_disk(center_coords_df,rad, i, img)
        sum_int += res[0]
        area_int += res[1]
    if area_int == 0:
        return 0
    return sum_int/area_int

def calculate_circ_slice(x,y,rad, img_s):
    xmin = max(x-rad, 0)
    ymin = max(y-rad, 0)
    xmax = min(x+rad, img_s.shape[1]-1)
    ymax = min(y+rad, img_s.shape[0]-1)
    brightness = int(img_s[ymax][xmax])+int(img_s[ymin][xmin])-int(img_s[ymin][xmax])-int(img_s[ymax][xmin])
    area =  (xmax-xmin)*(ymax-ymin)
    return brightness, area

def calculate_intensity(x, y, z, img):
    sum_int = 0
    area_int = 0
    z_slices = np.arange(z - 2, z + 3)  # 5 slices centered at z
    valid_slices = z_slices[(z_slices >= 0) & (z_slices < img.shape[0])]  # Keep valid z-indices

    for z_slice in valid_slices:
        diameter = constants.ROI_DIAMETER[abs(z_slice - z)]
        rad = diameter // 2

        xmin = max(x - rad, 0)
        xmax = min(x + rad, img.shape[2] - 1)
        ymin = max(y - rad, 0)
        ymax = min(y + rad, img.shape[1] - 1)

        img_s = img[z_slice]

        # Efficient NumPy sum within the bounding box
        brightness = int(img_s[ymax][xmax])+int(img_s[ymin][xmin])-int(img_s[ymin][xmax])-int(img_s[ymax][xmin])
        area =  (xmax-xmin)*(ymax-ymin)

        sum_int += brightness
        area_int += area

    return 0 if area_int == 0 else sum_int / area_int



def get_brightest_cells(mouse, region, session, percentage):
    df = pd.read_csv(constants.dir_path +"m" + str(mouse)+"_r"+str(region)+"_"+ 
                     session +"_optimized.csv")
    df = df.loc[df.int_optimized > df.int_optimized.quantile(1-percentage)]
    df = df[constants.COORDS_3D]
    return df
 
def optimize_centroid_position_old(row, img, suff):
    current_max = 0
    best_coords = [0,0,0]
    for x in range(-constants.TOLERANCE, constants.TOLERANCE+1):
        for y in range(-constants.TOLERANCE, constants.TOLERANCE+1):
            for z in range(-1, 2):
                tst_coords = {
                    ICY_COLNAMES['xcol'] : int(row[ICY_COLNAMES['xcol']].round())+x,
                    ICY_COLNAMES['ycol'] : int(row[ICY_COLNAMES['ycol']].round())+y,
                    ICY_COLNAMES['zcol'] : int(row[ICY_COLNAMES['zcol']].round())+z,
                    }
                mean_calculated = calculate_intensity_old(pd.Series(tst_coords), img)
                if mean_calculated > current_max:
                    current_max = mean_calculated
                    best_coords = [x,y,z]
    row['shift_x'+suff] = best_coords[0]
    row['shift_y'+suff] = best_coords[1]
    row['shift_z'+suff] = best_coords[2]
    row['int_optimized'+suff] = current_max
    
    return row

def optimize_centroids_old(df, img, suff=""):
    for col in ['shift_x','shift_y','shift_z', 'int_optimized']:
        df[col+suff] = 0
    df = df.apply(optimize_centroid_position_old,img =img, suff=suff, axis=1)
    return df


def calculate_integral_image(stack):
    """Compute integral image for efficient brightness calculations."""
    integral_volume = np.array([img.cumsum(axis=0, dtype=np.int64).cumsum(axis=1, dtype=np.int64) for img in stack])
    return integral_volume

def find_active_cells(df, bgr, k_std):
    for i in range(3):
        b_mean = bgr[i][0]
        b_std = bgr[i][1]
        threshold = b_mean + k_std * b_std
        df[f'active{i}'] = df[f"int_optimized{i}"] > threshold
    return df


def optimize_centroid_position(args):
    """Optimize the position of a single centroid using vectorized operations."""
    row, img, suff, tolerance = args
    if "close_neighbour" in row.index and row["close_neighbour"]:
        tolerance = tolerance//2
    
    x_center = int(np.round(row[ICY_COLNAMES['xcol']]))
    y_center = int(np.round(row[ICY_COLNAMES['ycol']]))
    z_center = int(np.round(row[ICY_COLNAMES['zcol']]))
    
    # Generate all possible shifts
    x_shifts = np.arange(-tolerance, tolerance + 1)
    y_shifts = np.arange(-tolerance, tolerance + 1)
    z_shifts = np.array([-1, 0, 1])

    # Generate all possible coordinate combinations
    x_offsets, y_offsets, z_offsets = np.meshgrid(x_shifts, y_shifts, z_shifts, indexing='ij')
    
    new_x = x_center + x_offsets.ravel()
    new_y = y_center + y_offsets.ravel()
    new_z = z_center + z_offsets.ravel()

    # Keep only valid coordinates
    valid_mask = (0 <= new_x) & (new_x < img.shape[2]) & \
                 (0 <= new_y) & (new_y < img.shape[1]) & \
                 (0 <= new_z) & (new_z < img.shape[0])

    new_x, new_y, new_z = new_x[valid_mask], new_y[valid_mask], new_z[valid_mask]

    # Compute brightness in one batch
    brightness_values = np.array([calculate_intensity(x, y, z, img) for x, y, z in zip(new_x, new_y, new_z)])
    # added for the purpose of method validation tests
    if len(brightness_values) == 0:
        return (0,)*4
    
    
    # Find the best shift
    best_idx = np.argmax(brightness_values)
    best_coords = (new_x[best_idx] - x_center, new_y[best_idx] - y_center, new_z[best_idx] - z_center)

    return (*best_coords, brightness_values[best_idx])

def update_df_coords(df, suff):
    shift_arr = ["shift_z", 'shift_y', 'shift_x']
    for i, cname in enumerate(constants.COORDS_3D):
        df[cname] = df[cname] + df[shift_arr[i]+suff]
    return df

def optimize_centroids(df, img, suff="", tolerance=constants.TOLERANCE, update_coords=True):
    """Optimize the positions of centroids using parallel processing."""
    integral_img = calculate_integral_image(img)

    args_list = [(row, integral_img, suff, tolerance) for _, row in df.iterrows()]
    
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(optimize_centroid_position, args_list))

    # Convert results back to DataFrame
    df[[f'shift_x{suff}', f'shift_y{suff}', f'shift_z{suff}', f'int_optimized{suff}']] = results

    if update_coords:
        df = update_df_coords(df, suff)

    return df


def filter_border_cells(df, sessions, img_shape, nucl_size_half = 3):
    z, y, x = img_shape
    for sid in sessions:
        if (f'int_optimized_{sid}' in df.columns):
            df = df[
                ((df[ICY_COLNAMES['xcol']] + df[f'shift_x_{sid}']) > nucl_size_half) &
                ((df[ICY_COLNAMES['xcol']] + df[f'shift_x_{sid}']) < x - nucl_size_half) &
                ((df[ICY_COLNAMES['ycol']] + df[f'shift_y_{sid}']) > nucl_size_half) &
                ((df[ICY_COLNAMES['ycol']] + df[f'shift_y_{sid}']) < y - nucl_size_half) &
                ((df[ICY_COLNAMES['zcol']] + df[f'shift_z_{sid}']) > 2) &
                ((df[ICY_COLNAMES['zcol']] + df[f'shift_z_{sid}']) < z - 2)
                ]
    return df

def calculate_background_for_cell(row, img, distance, suff):
    distances = [distance, distance, distance//2]

    shifts = [distances[i] * np.eye(3, dtype=int)[i] for i in range(3)] +\
        [-distances[i] * np.eye(3, dtype=int)[i] for i in range(3)] 
        
    base_coords = [int(row[ICY_COLNAMES[f'{x}col']] + row[f'shift_{x}{suff}']) for x in ['x', 'y', 'z']]
    
    shift_variants = [base_coords + shift for shift in shifts]
    
    zmax, ymax, xmax = img.shape

    bounds = [xmax, ymax, zmax]
    valid_variants = [ coords for coords in shift_variants if all(0 <= c < bounds[i] for i, c in enumerate(coords))]

    brightness_values = np.array([calculate_intensity(x, y, z, img) for x, y, z in valid_variants])
    bg_min = brightness_values.min()
    bg_med = np.median(brightness_values)
    bg_rstd = 1.4826 * np.median(np.abs(brightness_values - bg_med))
    cell_val = calculate_intensity(*base_coords, img)
    
    if bg_rstd == 0:
        snr_robust = np.nan
    else:
        snr_robust = (cell_val - bg_med) / bg_rstd
    return bg_min, snr_robust

def find_background(df, img, suff="", distance = 10):
    """Optimize the positions of centroids using parallel processing."""
    integral_img = calculate_integral_image(img)
    df[[f"background{suff}", f"snr_robust{suff}"]] = df.apply(
        calculate_background_for_cell,
        img=integral_img,
        distance=distance,
        suff=suff,
        axis=1,
        result_type="expand"
    )

    return df




def standarize_intensity(df, img):
    df["intensity_standarized"] = df.apply(calculate_intensity,img = img, axis = 1)
    return df[df['Mean Intensity (ch 0)']/df['intensity_standarized']<=1.5]


def find_quantile_threshold(df, img):
    df = standarize_intensity(df, img)
    return df["intensity_standarized"]

def pixels_to_um(df):
    df[ICY_COLNAMES['xcol']] = df[ICY_COLNAMES['xcol']]*constants.XY_SCALE
    df[ICY_COLNAMES['ycol']] = df[ICY_COLNAMES['ycol']]*constants.XY_SCALE
    df[ICY_COLNAMES['zcol']] = df[ICY_COLNAMES['zcol']]*constants.Z_SCALE
    return df

def calculate_intensity_row(args):
    """Calculate intensity for a given row and image."""
    row, img = args  # Unpack arguments
    integral_img = calculate_integral_image(img)
    x_center = int(row[ICY_COLNAMES['xcol']].round())
    y_center = int(row[ICY_COLNAMES['ycol']].round())
    z_center = int(row[ICY_COLNAMES['zcol']].round())

    if not (0 <= x_center < img.shape[2] and 0 <= y_center < img.shape[1] 
            and 0 <= z_center < img.shape[0]):
        return np.NaN
    return calculate_intensity(x_center, y_center, z_center, integral_img)

def calculate_background_intensity(df, img):
    """Calculate background intensity for a DataFrame using parallel processing."""
    
    bg_df = df[[ICY_COLNAMES[cname] for cname in ICY_COLNAMES if 'col' in cname]].copy()
    bg_df[ICY_COLNAMES['zcol']] = df[ICY_COLNAMES['zcol']] - constants.TOLERANCE - 0.1



    df_coords = df[constants.COORDS_3D].values
    bg_coords = bg_df[constants.COORDS_3D].values

    tree = cKDTree(df_coords)

    # Identify background points too close to foreground
    close_indices = tree.query_ball_point(bg_coords, r=constants.TOLERANCE)
    too_close = [i for i, neighbors in enumerate(close_indices) if len(neighbors) > 0]

    bg_df.drop(index=bg_df.index[too_close], inplace=True)

    # Prepare data for parallel processing
    args_list = [(row, img) for _, row in bg_df.iterrows()]


    with ProcessPoolExecutor() as executor:
        bg_df['bg_intensity'] = list(executor.map(calculate_intensity_row, args_list))



    bg_df.dropna(inplace=True)

    return bg_df, None
    
def z_score_intensity(df, session_ids):
    """Per-session, per-mouse robust z-score: (x - median) / (1.4826*MAD)."""
    out = df.copy()
    for sid in session_ids:
        intensity_col = f'int_optimized_{sid}'
        if intensity_col not in df.columns:
            continue
        intensity_s = out[intensity_col]
        out[f'{intensity_col}_z'] = (intensity_s - intensity_s.median())/(1.4826*mad(intensity_s, nan_policy='omit') + 1e-9)

    return out

def intensity_depth_detrend(
    df: pd.DataFrame,
    session_ids,
    z_col: str = ICY_COLNAMES['zcol'],
    frac: float = 0.3,        # LOWESS span
    bin_width: float = 5.0,   # depth-bin width (same units as z_col)
    use_log1p: bool = False,
    eps: float = 1e-9,
    intensity_prefix: str = "int_optimized_",
    background_prefix: str = "background_",
    k_dim: float = 4.0,       # “mean − k·MAD of background residual” threshold
) -> pd.DataFrame:
    """
    For each session:
      - Build intensity signal S_I = intensity - background; background S_B = background.
      - Optional log1p to both.
      - LOWESS detrend both vs depth.
      - Per depth bin (shared bins across both signals):
          * sigma_I = 1.4826*MAD(resid_I)   -> intensity scale
          * sigma_B = 1.4826*MAD(resid_B)   -> background scale (SNR)
          * mu_B    = mean(resid_B)
      - Outputs:
          * int_z  = resid_I / sigma_I   (your current GEE “SD” scale)
          * bg_z   = resid_I / sigma_B   (SNR-like)
          * bg_only_z = resid_B / sigma_B (diagnostics)
      - Optional filter flag: resid_I < mu_B - k_dim*sigma_B (dimmer than background).
    """
    out = df.copy()

    # Depth and bins
    z = pd.to_numeric(out[z_col], errors="coerce").astype(float).values
    z_min, z_max = np.nanmin(z), np.nanmax(z)
    edges = np.arange(np.floor(z_min), np.ceil(z_max) + bin_width, bin_width)
    z_bin = pd.cut(z, bins=edges, right=False, include_lowest=True, labels=False)

    def _lowess(y, x):
        order = np.argsort(x)
        xs, ys = x[order], y[order]
        ok = np.isfinite(xs) & np.isfinite(ys)
        trend_s = np.full_like(ys, np.nan, dtype=float)
        if ok.any():
            trend_s[ok] = lowess(ys[ok], xs[ok], frac=frac, it=2, return_sorted=False)
        trend = np.full_like(y, np.nan, dtype=float)
        finite = np.isfinite(xs) & np.isfinite(trend_s)
        if finite.sum() >= 2:
            trend[:] = np.interp(x, xs[finite], trend_s[finite])
        return trend

    def _robust_scale(s):
        # returns 1.4826*MAD per bin (robust sd)
        ser = pd.Series(s)
        sig = ser.groupby(z_bin).transform(lambda r: 1.4826 * mad(r, nan_policy="omit")).astype(float)
        if sig.isna().any():
            med = np.nanmedian(sig.values)
            sig = sig.fillna(med if np.isfinite(med) else 1.0)
        sig = sig.replace(0.0, eps)
        return sig

    # optional global dim mask (OR across sessions)
    dim_mask_global = np.zeros(len(out), dtype=bool)

    for sid in session_ids:
        Icol = f"{intensity_prefix}{sid}"
        Bcol = f"{background_prefix}{sid}"
        if Icol not in out.columns or Bcol not in out.columns:
            continue

        I = pd.to_numeric(out[Icol], errors="coerce").values
        B = pd.to_numeric(out[Bcol], errors="coerce").values

        SI = I - B     # intensity signal
        SB = B         # background signal

        if use_log1p:
            SI = np.log1p(SI)
            SB = np.log1p(SB)

        muI = _lowess(SI, z); 
        residI = SI - muI
        muB = _lowess(SB, z); 
        residB = SB - muB

        # per-bin scales
        sigma_I = _robust_scale(residI)  # intensity MAD
        sigma_B = _robust_scale(residB)  # background MAD

        # standardized outputs
        int_z  = residI / sigma_I.values     # your existing SD scale
        bg_z   = residI / sigma_B.values     # SNR-like
        bg_only_z = residB / sigma_B.values  # diagnostics

        out[f"{Icol}_rstd"] = int_z
        
        out[f"{Icol}_bgz"]  = bg_z
        out[f"{Bcol}_rstd"]  = bg_only_z

        # optional prefilter (dimmer than background)
        #out[f"is_dim_by_bg_{sid}"] = residI < (mean_B.values - k_dim * sigma_B.values)

        bg_mean = df.groupby(z_bin)[f"background_{sid}"].transform("mean")

        bg_std  = df.groupby(z_bin)[f"background_{sid}"].transform("std")
        threshold = bg_mean + 3.0 * bg_std
        out[f"is_dim_by_bg_{sid}"] = (out[Icol] < threshold)



    return out




def intensity_depth_detrend_pooled_mad(
    df: pd.DataFrame,
    session_ids,
    z_col: str = ICY_COLNAMES['zcol'],
    frac: float = 0.3,        # LOWESS span
    bin_width: float = 5.0,   # depth-bin width (same units as z_col)
    use_log1p: bool = False,
    eps: float = 1e-9,
    intensity_prefix: str = "int_optimized_",
    background_prefix: str = "background_",
    k_dim: float = 4.0,       # “mean − k·MAD of background residual” threshold (kept as in your code)
) -> pd.DataFrame:
    """
    Single-mouse version.
    For each session:
      - Build SI = intensity - background; SB = background.
      - Optional log1p.
      - LOWESS detrend both vs depth (per session).
    Then (pooled across sessions, per depth bin):
      - sigma_I = 1.4826 * MAD(resid_I) about the pooled median of resid_I
      - sigma_B = 1.4826 * MAD(resid_B) about the pooled median of resid_B
    Finally (per session):
      - int_z     = resid_I / sigma_I      (common per-bin scale)
      - bg_z      = resid_I / sigma_B
      - bg_only_z = resid_B / sigma_B
    """
    out = df.copy()

    # Depth bins for this mouse
    z = pd.to_numeric(out[z_col], errors="coerce").astype(float).values
    z_min, z_max = np.nanmin(z), np.nanmax(z)
    edges = np.arange(np.floor(z_min), np.ceil(z_max) + bin_width, bin_width)
    z_bin = pd.cut(z, bins=edges, right=False, include_lowest=True, labels=False)

    def _lowess(y, x):
        order = np.argsort(x)
        xs, ys = x[order], y[order]
        ok = np.isfinite(xs) & np.isfinite(ys)
        trend_s = np.full_like(ys, np.nan, dtype=float)
        if ok.any():
            trend_s[ok] = lowess(ys[ok], xs[ok], frac=frac, it=2, return_sorted=False)
        trend = np.full_like(y, np.nan, dtype=float)
        finite = np.isfinite(xs) & np.isfinite(trend_s)
        if finite.sum() >= 2:
            trend[:] = np.interp(x, xs[finite], trend_s[finite])
        return trend

    # 1) Residuals per session (LOWESS is per session)
    residI_by_sid = {}
    residB_by_sid = {}

    for sid in session_ids:
        Icol = f"{intensity_prefix}{sid}"
        Bcol = f"{background_prefix}{sid}"
        if Icol not in out.columns or Bcol not in out.columns:
            continue

        I = pd.to_numeric(out[Icol], errors="coerce").values
        B = pd.to_numeric(out[Bcol], errors="coerce").values

        SI = I - B
        SB = B

        if use_log1p:
            SI = np.log1p(SI)
            SB = np.log1p(SB)

        muI = _lowess(SI, z)
        muB = _lowess(SB, z)

        residI_by_sid[sid] = SI - muI
        residB_by_sid[sid] = SB - muB

        # (kept) optional dim-by-background flag (original threshold style)
        bg_mean = out.groupby(z_bin)[Bcol].transform("mean")
        bg_std  = out.groupby(z_bin)[Bcol].transform("std")
        threshold = bg_mean + 3.0 * bg_std
        out[f"is_dim_by_bg_{sid}"] = (out[Icol] < threshold)

    # 2) Pooled per-bin scales (MAD after LOWESS, pooled across sessions)
    if residI_by_sid:
        present_sids = [sid for sid in session_ids if sid in residI_by_sid]

        pool_I = pd.DataFrame({
            "z_bin": np.repeat(z_bin, len(present_sids)),
            "resid": np.concatenate([residI_by_sid[s] for s in present_sids])
        })
        pool_B = pd.DataFrame({
            "z_bin": np.repeat(z_bin, len(present_sids)),
            "resid": np.concatenate([residB_by_sid[s] for s in present_sids])
        })

        med_I_by_bin = pool_I.groupby("z_bin")["resid"].median()
        med_B_by_bin = pool_B.groupby("z_bin")["resid"].median()

        sig_I_by_bin = pool_I.groupby("z_bin")["resid"].apply(
            lambda r: 1.4826 * np.nanmedian(np.abs(r - med_I_by_bin.loc[r.name]))
        )
        sig_B_by_bin = pool_B.groupby("z_bin")["resid"].apply(
            lambda r: 1.4826 * np.nanmedian(np.abs(r - med_B_by_bin.loc[r.name]))
        )

        sigma_I = pd.Series(z_bin).map(sig_I_by_bin).astype(float).replace(0.0, eps).values
        sigma_B = pd.Series(z_bin).map(sig_B_by_bin).astype(float).replace(0.0, eps).values

        # 3) Standardize per session with the pooled scales; write outputs
        for sid in present_sids:
            Icol = f"{intensity_prefix}{sid}"
            Bcol = f"{background_prefix}{sid}"

            int_z     = residI_by_sid[sid] / sigma_I
            bg_z      = residI_by_sid[sid] / sigma_B
            bg_only_z = residB_by_sid[sid] / sigma_B

            out[f"{Icol}_rstd_common"] = int_z
            out[f"{Icol}_resid"] = residI_by_sid[sid]
            #out[f"{Icol}_bgz"]  = bg_z
            #out[f"{Bcol}_rstd"] = bg_only_z

    return out





def _winsor_per_mouse(df, col, mouse_col="mouse", lo=0.5, hi=99.5):
    qlo, qhi = lo/100.0, hi/100.0

    def _clip(s: pd.Series) -> pd.Series:
        v = s.dropna()
        if v.empty:                
            return s               
        lo_v, hi_v = v.quantile([qlo, qhi])
        return s.clip(lower=lo_v, upper=hi_v)
    return df.groupby(mouse_col)[col].transform(_clip)

def classify_by_rstd(df, id_pairs, tau=0.9):
    res = df.copy()
    for id1, id2 in id_pairs:
        a = f'int_optimized_{id1}_rstd'
        b = f'int_optimized_{id2}_rstd'
        outcol = f'{id1}_to_{id2}'
        d = res[b] - res[a]
        res[outcol] = 'stable'
        res.loc[d >  tau, outcol] = 'up'
        res.loc[d < -tau, outcol] = 'down'
    return res


def test_fun(mouse, region, s_idxses, session_order):
    old_method_df = pd.read_csv(constants.dir_path + constants.FILENAMES["cell_data_fn_template"]
                .format(mouse, region, session_order[s_idxses[1]]+"_"+session_order[s_idxses[0]]))
    df = pd.read_csv(constants.dir_path + constants.FILENAMES["cell_data_fn_template"]
                     .format(mouse, region, session_order[s_idxses[0]]), "\t", header=1)

    img_ref = io.imread(constants.dir_path + constants.FILENAMES["img_fn_template"]
                    .format(mouse, region, session_order[s_idxses[0]])).astype("uint8")

    img_comp = io.imread(constants.dir_path + constants.FILENAMES["img_fn_template"]
                    .format(mouse, region, session_order[s_idxses[1]])).astype("uint8")
    df["int1"] = df.apply(calculate_intensity, img = img_ref, axis = 1)
    df["int2"] = df.apply(calculate_intensity, img = img_comp, axis = 1)
    joined = old_method_df.join(df, on='idx2', how='right')
    return joined
    #plt.plot(joined.intensity2)
    #plt.plot(joined.int1, alpha=0.5)
    

    