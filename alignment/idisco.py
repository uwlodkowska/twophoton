#%% setup
import pandas as pd
from pathlib import Path
import numpy as np

ctx_idx = [1,5,7,8,10,14,16,20,21]
landmark_idx = [2,3,4,6,11,12,13,17,22]
mouse2group = ({m: "ctx" for m in ctx_idx}
               | {m: "landmark" for m in landmark_idx})
cells_dir = Path("/mnt/data/idisco/all_cells/")   # adjust to folder with all cell CSVs
atlas_file = "/mnt/data/idisco/all_cells/ABA_brain_regions.csv"

voxel_um = 25.0
voxel_mm3 = (voxel_um**3) * 1e-9

performance = {1:74, 2:74,3:88, 4:74, 5:72, 6:88, 7:72, 8:56,10:86,11:80, 12:94,
               13:78, 14:88, 16:52, 17:82, 20:82, 21:84, 22:78, 23:92}

performance_gap_norm = {1:0.124, 2:0.049,3:-0.045, 4:0.159, 5:0, 6:0.097, 7:0.176, 8:0.359,10:0.127,11:0.1, 12:0.162,
               13:0, 14:-0.075, 16:0.293, 17:0.229, 20:0.074, 21:-0.103, 22:-0.044, 23:0.231}
performance_training_len = {1:9, 2:9,3:9, 4:12, 5:8, 6:7, 7:12, 8:7,10:9,11:11, 12:12,
               13:7, 14:11, 16:10, 17:6, 20:8, 21:10, 22:8, 23:10}

perf_land = {1:84,2:74,3:88,4:74,5:72,6:88,7:86,8:84,10:96,11:80,12:94,13:78,14:82,16:76,17:82, 20:88,21:76,22:78}
#%%
p_df = pd.DataFrame(performance.items(),columns = ["mouse", "test_score"])

#%%
p_df["gap_lmk_ctx"] = p_df["mouse"].apply(lambda x : performance_gap_norm[x])
p_df["training_len"] = p_df["mouse"].apply(lambda x : performance_training_len[x])
print(p_df)

#%%
lev=6
#%%helper defs

def count_cells_for_mouse(file, mouse_id, vols, L):
    df = pd.read_csv(file, sep=";")
    df.columns = df.columns.str.strip()
    to_delete = df[df.acronym.isin(['NoL', 'brain', 'universe'])].index
    
    
    
    df.drop(to_delete, inplace=True)
    df = df.rename(columns={"id":"region_id"})
    for c in ["xt","yt","zt","region_id"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["region_id"]).copy()
    df["region_id"] = df["region_id"].astype(int)
    
    df["mouse"] = mouse_id
    
    counts = (df.groupby(["mouse","region_id"], as_index=False)
               .size()
               .rename(columns={"size":"count"}))
    
    base = counts.merge(vols, on="region_id", how="left")
    #base["density_per_voxel"] = base["count"] / base["volume_vox"]
    #base["density_per_mu"] = base["count"] / base["volume_mu"]
    return base

def get_level_id(acr, L, level_maps):
    tmpacr = str(acr.get(f"L{L}"))
    for Lr in reversed(range(1,L+1)):
        if tmpacr in level_maps[Lr]:
            acr[f"level{L}_id"] = level_maps[Lr][tmpacr]
            return acr
    acr[f"level{L}_id"] = np.nan#acr["region_id"]
    return acr

def aggregate_to_level_simple(base, atlas, L):
    """
    For each base.region_id:
      - Prefer the id of its level-L structure (choosing the *leaf at level L* if duplicates).
      - If that's unavailable, step upwards (L-1, L-2, ...) to the first level where it appears,
        and take the *leaf at that level*.
    Then aggregate counts/volumes by that chosen id.
    """
    # --- Prep atlas with level columns present
    A = atlas.copy()
      # assumes this ensures L1..L exist up to L
    cols = [f"L{x}" for x in range(L+2)]
    A[cols] = A[cols].replace(".", np.nan)
    # Helper: build acronym->id map for each level, preferring leaves at that level
    def _leaf_map_for_level(A, k):
        col = f"L{k}"
        # rows that have a value at this level
        X = A[A[col].notna()].copy()
        # "leaf at this level" = next level is NA (if next level exists)
        nxt = f"L{k+1}"
        if nxt in A.columns:
            X = X[X[nxt].isna()]
        # if duplicates remain, keep first deterministically
        X = (X[[col, "id"]]
             .dropna()
             .assign(**{col: X[col].astype(str).str.strip()}))
        X = X.sort_values(["id"]).drop_duplicates(subset=[col], keep="first")
        return dict(zip(X[col], X["id"].astype(int)))
    
    # Precompute maps for all levels 1..L
    level_maps = {k: _leaf_map_for_level(A, k) for k in range(1, L + 1)}
    A = guarded_fill_level(A, L).rename(columns={"id": "region_id"})
    d = base.merge(A, on="region_id", how="left")
    d[f"L{L}"] = d[f"L{L}"].fillna(d['acr_atl'])
    d = d.apply(lambda x: get_level_id(x, L, level_maps), axis=1)
    d = d.copy()
    d[f"level{L}_id"]=d[f"level{L}_id"].astype("Int64")
    level_ac = list(d[f"L{L}"])
    to_delete = []
    for Lr in reversed(range(1,L)):
        for lid in level_ac:
            parents = d[d[f"L{Lr}"]==lid]
            if parents.shape[0]>1:
                to_delete+=[lid]
    mask = d[f"L{L}"].isin(to_delete)
    d = d[~mask].copy()
    
    vols["region_id"]=vols["region_id"].astype("Int64")
    g = d.groupby(["mouse",f"level{L}_id"], as_index=False).agg(count=("count","sum"), LC=(f"L{L}","first"))
    g = g.rename(columns={"LC":f"L{L}" })
    print(g.columns)
    out = g.merge(vols.rename(columns={"region_id":f"level{L}_id"}), on=f"level{L}_id", how="left")
    out = out[["mouse",f"level{L}_id",f"L{L}","count","volume_mu","acr_atl"]]
    out = out.loc[out["count"]>500].copy()
    out = out.loc[out["volume_mu"]>0.01].copy()
    
    return out

#%%




def guarded_fill_level(atlas: pd.DataFrame, L: int) -> pd.DataFrame:
    """
    Propagate L{L-1} into L{L} only for parent acronyms that have NO children at L.
    Mutates the L{L} column (returns a copy).
    """
    A = atlas.copy()
    cols = [f"L{x}" for x in range(L+2)]
    colpairs=list(zip(cols[0:], cols[1:]))
    for c1,c2 in colpairs:
        mask = A[c2].isna() & A[c1].notna()
        A.loc[mask, c2] = A.loc[mask, c1]
    return A




#%%

atlas = pd.read_csv(atlas_file)

#%%
vols = pd.read_excel(
    "/mnt/data/idisco/all_cells/new_atl_.xlsx",
    header=1,
    engine="openpyxl",
    keep_default_na=True
)
#%%
atlas = atlas.rename(columns={"idx":"id"})
atlas["id"]    = pd.to_numeric(atlas["id"],    errors="coerce").astype("Int64")
atlas["depth"] = pd.to_numeric(atlas["depth"], errors="coerce").astype("Int64")


#%%
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(atlas.head(20))


#%% cerebellum ids to exclude
cerebellum_ids = list(np.array(atlas[atlas["L1"]=='CB']["id"]))
#%% olfactory ids
olf5_acr = ["AOB", "MOB"]
olf3_acr = ["lotg"]
olf5_mask = atlas["L5"].isin(olf5_acr)
olf3_mask = atlas["L3"].isin(olf3_acr)
olf5_ids = np.array(atlas[olf5_mask]["id"])
olf3_ids = np.array(atlas[olf3_mask]["id"])


olfactory_nerve_ids = [1016,840]

olfactory_ids = list(olf3_ids) + list(olf5_ids) + olfactory_nerve_ids
#%% ventricular system and fiber tracts ids

grey_ids = list(np.array(atlas[atlas["L0"]=='grey']["id"]))




#%% spatial memory circuit ids
spatial5_acr = ["MBO", "RSP", "ATN", "LAT"]
spatial4_acr = ["HPF"]#, "CLA"]
mask4 = atlas["L4"].isin(spatial4_acr)
mask5 = atlas["L5"].isin(spatial5_acr)
spatial4_ids = np.array(atlas[mask4]["id"])
spatial5_ids = np.array(atlas[mask5]["id"])


#%%

vols = vols.rename(columns={"id":"region_id", "voxels":"volume_vox", "abbreviation":"acr_atl"})
vols["region_id"]  = pd.to_numeric(vols["structure ID"], errors="coerce").astype("Int64")
vols["volume_mu"] = pd.to_numeric(vols["Mean Volume (m)"], errors="coerce")
vols = vols[["region_id", "volume_mu", "acr_atl"]]


# atlas["parent_id"] = atlas.apply(compute_parent_id, axis=1).astype("Int64")
# parent = dict(zip(atlas["id"].dropna().astype(int), atlas["parent_id"].dropna().astype(int)))
# depth  = dict(zip(atlas["id"].dropna().astype(int), atlas["depth"].dropna().astype(int)))

#%% Collect from all cell CSVs
all_counts = []
L=lev

for m in ctx_idx + landmark_idx:
    f = f"/mnt/data/idisco/all_cells/cells_{m}.csv"
    
    base = count_cells_for_mouse(f, m, vols, L)
    
    counts_tmp = aggregate_to_level_simple(base, atlas, L)
    grey_mask = counts_tmp[f'level{lev}_id'].isin(grey_ids)
    
    tot = counts_tmp["count"].sum()
    counts_tmp = counts_tmp.loc[grey_mask]
    keep = counts_tmp["count"].sum()
    print(m, keep/tot)
    all_counts.append(counts_tmp)

counts = pd.concat(all_counts, ignore_index=True)


#%% add group column
counts_group = counts.copy()
counts_group['group'] = (counts_group["mouse"].map(mouse2group).fillna("other"))

#%% exclude cerebellum
cerebellum_mask = counts_group[f'level{lev}_id'].isin(cerebellum_ids)
counts_group = counts_group.loc[~cerebellum_mask].copy()

#%%keep grey only

grey_mask = counts_group[f'level{lev}_id'].isin(grey_ids)
tot = counts_group["count"].sum()
counts_group = counts_group.loc[grey_mask]
keep = counts_group["count"].sum()

print(keep/tot)

#%%exclude olfactory
olfactory_mask = counts_group[f'level{lev}_id'].isin(olfactory_ids)
counts_group = counts_group.loc[~olfactory_mask].copy()


#%%only include spatial memory circuit
mask = counts_group[f'level{lev}_id'].isin(list(spatial5_ids) + list(spatial4_ids))
counts_group = counts_group.loc[mask].copy()


#%% remove zero volume  and calculate density
counts_group = counts_group.dropna(subset=["volume_mu"]).copy()
counts_group.loc[:,"density_per_mu"] = counts_group["count"]/counts_group["volume_mu"]


#%%
tst = counts_group["acr_atl"].unique()
tst.sort()
print(tst, len(tst))
#%%
print(counts.head())
print(counts.columns)

#%%


import numpy as np
import pandas as pd
from skbio.diversity import beta_diversity
from skbio.stats.distance import permanova

def run_permanova(counts, L=5, value_col="density_per_mm3",
                  ctx_idx=None, landmark_idx=None,
                  metric="braycurtis", normalize="hellinger",
                  permutations=999, seed=0):
    """
    counts: long DF from your aggregate step with columns:
            ['mouse', f'level{L}_id', value_col]
    normalize: 'none' | 'rel' (row-proportions) | 'hellinger' (sqrt of proportions)
    metric: e.g. 'braycurtis' (good default) or 'euclidean' (often with hellinger)
    """
    # 1) samples × features matrix
    wide = (counts.pivot_table(index="mouse",
                               columns=f"level{L}_id",
                               values=value_col,
                               aggfunc="sum",
                               fill_value=0)
                  .sort_index())
    # drop all-zero features
    print(wide.shape)
    wide = wide.loc[:, (wide > 0).any(axis=0)]
    print(wide.shape)

    # 2) (optional) normalization to compositional scale
    if normalize in ("rel", "hellinger"):
        row_sums = wide.sum(axis=1).replace(0, np.nan)
        wide = wide.div(row_sums, axis=0).fillna(0)
        if normalize == "hellinger":
            wide = np.sqrt(wide)

    # 3) group labels (ctx vs landmark)
    mice = wide.index.astype(int)
    groups = (counts[["mouse","group"]]
                .drop_duplicates()
                .set_index("mouse")
                .reindex(wide.index)["group"]
                .fillna("other"))


    # 4) distance matrix + PERMANOVA
    dm = beta_diversity(metric, wide.values, ids=wide.index)
    meta = pd.DataFrame({"group": groups})
    res = permanova(dm, meta, column="group",
                    permutations=permutations, seed=seed)
    return res  # pandas DataFrame with F, p-value, R^2, etc.

# === use it ===
#%%
counts_group = counts_group.loc[counts_group["volume_mu"]>0].copy()
print(counts_group.head())
#counts_clean = counts_group[counts_group["mouse"]!=7]
res = run_permanova(counts_group, L=lev, value_col="density_per_mu",
                    ctx_idx=ctx_idx, landmark_idx=landmark_idx,
                    metric="euclidean", normalize="hellinger",
                    permutations=10000, seed=1)

print(res)




#%%

import matplotlib.pyplot as plt
from skbio.stats.ordination import pcoa

def plot_dispersion_box(permdisp_distances_df, title="Odległość od centroidu grupy", ax=None):
    d = permdisp_distances_df.copy()
    if ax is None:
        fig, ax = plt.subplots(figsize=(5,4.5))
    else:
        fig = ax.figure

    groups = [g for g,_ in d.groupby("group")]
    glabels = {"ctx": "kontekst", "landmark": "landmark"}
    data = [d.loc[d["group"]==g, "dist_to_centroid"].to_numpy() for g in groups]
    ax.boxplot(data, labels=[glabels[str(g)] for g in groups])
    ax.tick_params(axis='both', which='minor', labelsize=16)
    ax.set_title(title, fontsize=18)
    plt.tight_layout()
    ax.xaxis.set_tick_params(labelsize=16)
    return fig, ax

def distances_to_centroid(dm, groups):
    coords = pcoa(dm).samples.values
    labels = groups.rename(index=str).reindex(dm.ids).to_numpy()
    d = []
    for g in np.unique(labels):
        mask = labels == g
        cg = coords[mask].mean(axis=0)
        for sid, dist in zip(np.array(dm.ids)[mask], np.linalg.norm(coords[mask]-cg, axis=1)):
            d.append({"sample": sid, "group": g, "dist_to_centroid": float(dist)})
    return pd.DataFrame(d)

def permanova_null_hist(dm, groups, permutations=1999, seed=7, title="PERMANOVA null vs observed"):
    rng = np.random.default_rng(seed)
    meta = pd.DataFrame({"group": groups.rename(index=str).reindex(dm.ids).values}, index=dm.ids)

    # observed F (no extra permutations inside call)
    F_obs = float(permanova(dm, meta, column="group", permutations=0)["test statistic"])

    labs = meta["group"].to_numpy()
    F_null = []
    for _ in range(permutations):
        meta["group"] = rng.permutation(labs)
        Fp = float(permanova(dm, meta, column="group", permutations=0)["test statistic"])
        F_null.append(Fp)
    F_null = np.array(F_null)

    fig, ax = plt.subplots(figsize=(6,4.5))
    ax.hist(F_null, bins=30, density=True)
    ax.axvline(F_obs, linestyle="--")
    ax.set_xlabel("pseudo-F (permuted labels)")
    ax.set_ylabel("density")
    ax.set_title(title)
    plt.tight_layout()
    return fig, ax, F_obs

from matplotlib.lines import Line2D
def plot_pcoa_scatter(dm, groups, ids = [0,1], title="PCoA (axes 1–2)", ax=None):
    coords = pcoa(dm)
    X = coords.samples.iloc[:, [ids[0],ids[1]]].copy()
    X.index = pd.Index(dm.ids, name="mouse")
    X["group"] = groups.rename(index=str).reindex(X.index)

    uniq = list(pd.Series(X["group"].unique()).astype(str))
    color_map = {g: f"C{i}" for i, g in enumerate(uniq)}

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5,5))
    else:
        fig = ax.figure


    # scatter + centroids in the SAME color (no centroid label here)
    for g, sub in X.groupby("group"):
        c = color_map[str(g)]
        ax.scatter(sub.iloc[:,0], sub.iloc[:,1], label=str(g), color=c, alpha=0.9)
        cx, cy = sub.iloc[:,0].mean(), sub.iloc[:,1].mean()
        ax.scatter(cx, cy, marker="X", s=120, color=c, zorder=3)

    # axis labels with variance explained
    vx = coords.proportion_explained

    glabels = {"ctx": "kontekst", "landmark": "landmark"}
    ax.set_xlabel(f"PCoA{ids[0]+1} ({vx.iloc[ids[0]]*100:.1f}%)", fontsize=16)
    ax.set_ylabel(f"PCoA{ids[1]+1} ({vx.iloc[ids[1]]*100:.1f}%)", fontsize=16)
    ax.set_title(title, fontsize=18)
    ax.set_aspect("equal", adjustable="datalim")

    # custom legend: groups + ONE black centroid entry
    group_handles = [Line2D([0],[0], marker='o', linestyle='None',
                            color=color_map[g], label=glabels[str(g)]) for g in uniq]
    centroid_handle = Line2D([0],[0], marker='X',ms = 10, linestyle='None',
                             color='k', label='centroid')
    ax.legend(group_handles + [centroid_handle], [*uniq, "centroid"], frameon=False)
    ax.tick_params(axis='both', which='minor', labelsize=12)

    plt.tight_layout()
    return fig, ax

from skbio.diversity import beta_diversity

def make_dm_from_counts(counts: pd.DataFrame,
                        L: int = 5,
                        value_col: str = "density_per_mm3",
                        normalize: str = "hellinger",   # 'rel' or 'hellinger'
                        metric: str = "euclidean",      # use 'braycurtis' if normalize='rel'
                        keep_cols=None,
                        mice_keep=None):
    """
    Returns: dm (DistanceMatrix), groups (Series indexed by mouse ids)
    """
    # 1) wide matrix: rows=mice, cols=level-L regions
    X = (counts.pivot_table(index="mouse",
                            columns=f"level{L}_id",
                            values=value_col,
                            aggfunc="sum",
                            fill_value=0)
               .sort_index())

    # lock feature set if provided (avoid geometry drift across subsets)
    if keep_cols is None:
        X = X.loc[:, (X > 0).any(axis=0)]
        keep_cols = X.columns
    X = X.reindex(columns=keep_cols, fill_value=0)

    # optional subset of mice
    if mice_keep is not None:
        X = X.loc[X.index.isin(mice_keep)]

    # 2) per-mouse normalization
    if normalize in ("rel", "hellinger"):
        rs = X.sum(axis=1).replace(0, np.nan)
        X = X.div(rs, axis=0).fillna(0)
        if normalize == "hellinger":
            X = np.sqrt(X)

    # 3) build DistanceMatrix
    ids = X.index.astype(str)                # dm ids are strings
    dm  = beta_diversity(metric, X.values, ids=ids)

    # 4) groups aligned to dm ids
    groups = (counts[["mouse","group"]]
                .drop_duplicates("mouse")
                .set_index("mouse")
                .reindex(X.index)["group"])

    return dm, groups, keep_cols





#%%
#counts_clean = counts_group[counts_group["mouse"]!=7]
dm, groups, keep_cols = make_dm_from_counts(counts_group, L=lev,
                                            value_col="density_per_mu",
                                            normalize="hellinger",
                                            metric="euclidean")
#%%
# Now you can call the visualization helpers:
# 1) PCoA scatter
fig, ax = plot_pcoa_scatter(dm, groups, title="PCoA dla gęstości cFos")
fig, ax = plot_pcoa_scatter(dm, groups,ids=[0,2], title="PCoA dla gęstości cFos")
fig, ax = plot_pcoa_scatter(dm, groups,ids=[1,2], title="PCoA dla gęstości cFos")

# 2) Dispersion boxplot
permdisp_distances = distances_to_centroid(dm, groups)
fig, ax = plot_dispersion_box(permdisp_distances)

# 3) Permutation null histogram
fig, ax, F_obs = permanova_null_hist(dm, groups, permutations=1999)

#%%
import matplotlib.pyplot as plt

def plot_pcoa_plus_dispersion(dm, groups, ids=[1,2],
                              figsize=(12,5), width_ratios=(2,1)):
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs  = fig.add_gridspec(1, 2, width_ratios=width_ratios)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    # left: PCoA scatter
    plot_pcoa_scatter(dm, groups, ids=ids, ax=ax0)

    # right: dispersion boxplot
    permdisp_distances = distances_to_centroid(dm, groups)
    plot_dispersion_box(permdisp_distances,ax=ax1)

    return fig, (ax0, ax1)

fig, (ax0, ax1) = plot_pcoa_plus_dispersion(dm, groups, ids=[0,1])


#%%
d = distances_to_centroid(dm, groups)  # your function

iqr = d.groupby('group')['dist_to_centroid'].quantile([.25,.75]).unstack()
thr = iqr[0.75] + 1.5*(iqr[0.75] - iqr[0.25])
outliers = d[d['dist_to_centroid'] > d['group'].map(thr)] \
             .sort_values(['group','dist_to_centroid'], ascending=[True, False])

print(outliers)
#%%

import numpy as np
import pandas as pd
from skbio.diversity import beta_diversity
from skbio.stats.distance import permanova, DistanceMatrix
from skbio.stats.ordination import pcoa

def _wide_and_groups(counts, L=5, value_col="density_per_mm3", normalize="hellinger"):
    """
    Build samples×features matrix (rows=mice, cols=level-L regions) and group labels from `counts`.
    Expects columns: ['mouse','group', f'level{L}_id', value_col]
    """
    # samples × features
    wide = (counts.pivot_table(index="mouse",
                               columns=f"level{L}_id",
                               values=value_col,
                               aggfunc="sum",
                               fill_value=0)
                  .sort_index())

    # drop features absent everywhere
    wide = wide.loc[:, (wide > 0).any(axis=0)]

    # per-mouse normalization (row-wise)
    if normalize in ("rel", "hellinger"):
        row_sums = wide.sum(axis=1).replace(0, np.nan)
        wide = wide.div(row_sums, axis=0).fillna(0)
        if normalize == "hellinger":
            wide = np.sqrt(wide)

    # groups from counts (preferred)
    groups = (counts[["mouse","group"]]
                .drop_duplicates("mouse")
                .set_index("mouse")
                .reindex(wide.index)["group"]
                .fillna("other"))
    return wide, groups

def medoid_outlier_df(wide, groups, metric="braycurtis"):
    """
    Robust outlier check: distance-to-group-medoid within each group (label-safe).
    Returns a DataFrame with per-sample distances and an outlier flag (Tukey high fence).
    """
    from skbio.diversity import beta_diversity

    ids = wide.index.astype(str)
    dm = beta_diversity(metric, wide.values, ids=ids)

    rows = []
    for grp in groups.unique():
        idx = groups[groups == grp].index.astype(str).tolist()
        if len(idx) < 2:
            for sid in idx:
                rows.append({"mouse": int(sid), "group": grp,
                             "dist_to_medoid": np.nan, "outlier": False})
            continue

        sub = dm.filter(idx)
        sumd = sub.data.sum(axis=1)
        medoid_i = int(np.argmin(sumd))
        d_to_medoid = sub.data[:, medoid_i]

        q1, q3 = np.percentile(d_to_medoid, [25, 75])
        iqr = q3 - q1
        high = q3 + 1.5 * iqr

        for sid, d in zip(sub.ids, d_to_medoid):
            rows.append({"mouse": int(sid), "group": grp,
                         "dist_to_medoid": float(d),
                         "outlier": bool(d > high)})
    return pd.DataFrame(rows).sort_values(["group","dist_to_medoid"], na_position="last")

def permdisp(dm: DistanceMatrix, groups, permutations=999, seed=0):
    """
    PERMDISP (betadisper-like): tests homogeneity of multivariate dispersion between groups.
    `groups` is a Series indexed by mouse IDs (ints). `dm.ids` are strings of those IDs.
    """
    coords = pcoa(dm).samples.values
    labels = groups.rename(index=str).reindex(dm.ids).to_numpy()

    # distances to group centroids in PCoA space
    d_to_centroid = np.empty(len(dm.ids))
    uniq = np.unique(labels)
    for g in uniq:
        mask = labels == g
        Xg = coords[mask]
        cg = Xg.mean(axis=0)
        d_to_centroid[mask] = np.linalg.norm(Xg - cg, axis=1)

    # ANOVA F on distances
    y = d_to_centroid
    grand_mean = y.mean()
    ssb = 0.0
    k = len(uniq)
    for g in uniq:
        mask = labels == g
        ssb += mask.sum() * (y[mask].mean() - grand_mean) ** 2
    dfb = k - 1

    ssw = 0.0
    for g in uniq:
        mask = labels == g
        ssw += ((y[mask] - y[mask].mean()) ** 2).sum()
    dfw = len(y) - k
    F = (ssb / dfb) / (ssw / dfw) if dfb > 0 and dfw > 0 else np.nan

    # permutations on group labels
    rng = np.random.default_rng(seed)
    perm_F = []
    for _ in range(permutations):
        lab_p = rng.permutation(labels)
        ssb_p = 0.0
        for g in uniq:
            mask = lab_p == g
            ssb_p += mask.sum() * (y[mask].mean() - grand_mean) ** 2
        Fp = (ssb_p / dfb) / (ssw / dfw) if dfb > 0 and dfw > 0 else np.nan
        perm_F.append(Fp)
    perm_F = np.asarray(perm_F)
    pval = (np.sum(perm_F >= F) + 1) / (permutations + 1)

    out = pd.DataFrame({
        "method": ["PERMDISP"],
        "statistic": [F],
        "p_value": [pval],
        "df_between": [dfb],
        "df_within": [dfw]
    })
    dist_df = pd.DataFrame({"sample": dm.ids, "dist_to_centroid": y, "group": labels})
    return out, dist_df

def loo_permanova(dm: DistanceMatrix, meta: pd.DataFrame, column="group",
                  permutations=999, seed=0):
    """
    Leave-one-out PERMANOVA influence analysis.
    `meta` must be a DataFrame indexed by the SAME string ids as `dm.ids`.
    """
    # align metadata strictly to string IDs & order
    meta = meta.rename(index=str).reindex(dm.ids)
    if meta[column].isna().any():
        bad = meta.index[meta[column].isna()].tolist()
        raise ValueError(f"Grouping missing for IDs: {bad}")

    full = permanova(dm, meta, column=column,
                     permutations=permutations, seed=seed)

    rows = []
    ids = list(dm.ids)
    for sid in ids:
        keep_ids = [s for s in ids if s != sid]
        dm_i   = dm.filter(keep_ids)
        meta_i = meta.loc[keep_ids]
        res_i  = permanova(dm_i, meta_i, column=column,
                           permutations=permutations, seed=seed)
        rows.append({
            "left_out": sid,
            "F_full": full["test statistic"],
            "p_full": full["p-value"],
            "F_loo": res_i["test statistic"],
            "p_loo": res_i["p-value"],
            "delta_F": res_i["test statistic"] - full["test statistic"],
            "delta_p": res_i["p-value"] - full["p-value"]
        })
    return full, pd.DataFrame(rows).sort_values("p_loo")

def outlier_diagnostics(counts, L=5, value_col="density_per_mm3",
                        normalize="hellinger", metric="braycurtis",
                        permutations=999, seed=42):
    """
    Runs:
      - medoid outlier distances within each group,
      - PERMDISP test,
      - leave-one-out PERMANOVA.
    Returns a dict of results. Only uses groups 'ctx'/'landmark'.
    """
    wide, groups = _wide_and_groups(counts, L=L, value_col=value_col, normalize=normalize)

    # keep only ctx/landmark; maintain index alignment
    mask = groups.isin(["ctx", "landmark"])
    wide   = wide.loc[mask]
    groups = groups.loc[mask]

    # Distance matrix with string IDs; metadata aligned to EXACT same ids
    ids = wide.index.astype(str)
    dm  = beta_diversity(metric, wide.values, ids=ids)
    meta = pd.DataFrame({"group": groups.reindex(wide.index).values}, index=ids)

    # 1) robust outliers
    med = medoid_outlier_df(wide, groups, metric=metric)

    # 2) dispersion equality
    disp_res, disp_df = permdisp(dm, groups, permutations=permutations, seed=seed)

    # 3) leave-one-out permanova
    perma_full, perma_loo = loo_permanova(dm, meta, column="group",
                                          permutations=permutations, seed=seed)

    return {
        "wide": wide,
        "groups": groups,
        "dm": dm,
        "medoid_outliers": med,
        "permdisp_result": disp_res,
        "permdisp_distances": disp_df,
        "permanova_full": perma_full,
        "permanova_loo": perma_loo
    }


#%%

diag = outlier_diagnostics(counts_group, L=lev, value_col="density_per_mu",
                           normalize="hellinger", metric="euclidean",
                           permutations=999, seed=7)

#%%
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(diag["permanova_full"])          # your overall pseudo-F and p
    print(diag["medoid_outliers"].head(20))  # mice far from their group medoid (flagged)
    print(diag["permdisp_result"])         # test if groups differ in dispersion
    print(diag["permanova_loo"].head(20))

#%%

import numpy as np, pandas as pd
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

def build_wide(counts, L=5, value_col="density_per_mm3"):
    X = (counts.pivot_table(index="mouse",
                            columns=f"level{L}_id",
                            values=value_col,
                            aggfunc="sum",
                            fill_value=0)
                .sort_index())
    # drop regions absent everywhere
    return X.loc[:, (X > 0).any(axis=0)]

def clr_transform(X, eps=1e-6):
    P = X.div(X.sum(axis=1), axis=0)
    Z = np.log(P + eps)
    return Z.sub(Z.mean(axis=1), axis=0)


def per_region_log(counts, L):
    
    d = counts.copy()
    d['log_density'] = np.log(d["density_per_mu"])
    
    out = []
    for roi, sub in d.groupby(f'level{L}_id'):
        gA = sub.loc[sub.group=='landmark','log_density'].to_numpy()
        gB = sub.loc[sub.group=='ctx','log_density'].to_numpy()
        if len(gA)>=2 and len(gB)>=2:
            t, p = ttest_ind(gA, gB, equal_var=False)  # Welch
            # Hedges' g
            na, nb = len(gA), len(gB)
            sa2, sb2 = gA.var(ddof=1), gB.var(ddof=1)
            sp = np.sqrt(((na-1)*sa2 + (nb-1)*sb2) / (na+nb-2))
            d_unb = (gA.mean() - gB.mean()) / (sp + 1e-12)
            J = 1 - 3/(4*(na+nb)-9)  # small-sample correction
            g_hedges = J * d_unb
            out.append((roi, t, p, g_hedges, gA.mean()-gB.mean()))
    out = pd.DataFrame(out, columns=[f'level{L}_id','t','p','hedges_g','diff_logdens'])
    
    out['p_adj'] = multipletests(out['p'], method='fdr_bh')[1]
    
    out['region'] = out[f'level{L}_id'].astype('Int64')

    labels = atlas[['id','name','acronym']].rename(columns={'id':'region'})
    log_labeled = out.merge(labels, on='region', how='left')
    return log_labeled[["region","acronym","name"] + [c for c in log_labeled.columns if c!="region"]].sort_values("p")


def per_region_welch_t(counts, L=5, value_col="density_per_mu"):
    X = build_wide(counts, L=L, value_col=value_col)
    X_clr = clr_transform(X)

    groups = (counts[["mouse","group"]].drop_duplicates()
                .set_index("mouse").reindex(X_clr.index)["group"])

    A = X_clr.loc[groups.eq("ctx")]
    B = X_clr.loc[groups.eq("landmark")]

    rows = []
    for reg in X_clr.columns:
        a, b = A[reg].dropna(), B[reg].dropna()
        t, p = ttest_ind(a, b, equal_var=False)
        # simple effect size on CLR scale (Hedges g approx; small-n correction optional)
        sd_pooled = np.sqrt((a.var(ddof=1)+b.var(ddof=1))/2)
        g = (a.mean() - b.mean()) / (sd_pooled if sd_pooled>0 else np.nan)
        rows.append({"region": reg, "t": t, "p": p, "g_clr": g,
                     "mean_ctx": a.mean(), "mean_landmark": b.mean()})
    out = pd.DataFrame(rows)
    out["q"] = multipletests(out["p"], method="fdr_bh")[1]
    return out.sort_values("p")
#%% ttests
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    tt = per_region_welch_t(counts_group, L=lev,value_col="count")
    tt_labeled = (tt.assign(region=tt["region"].astype("Int64"))
                .merge(atlas.rename(columns={"id":"region"}), on="region", how="left"))
    print(tt_labeled[["region","acronym","name"] + [c for c in tt.columns if c!="region"]].head(20))
    
    ll = per_region_log(counts_group, L=lev)
    print(ll.head(20))

#%%

from scipy.stats import spearmanr

def spearman_regionwise(counts, behav_df, y_col, L=5, value_col="density_per_mu",
                        mode="rel", control_group=True):
    # matrix
    X = build_wide(counts, L=L, value_col=value_col)
    if mode == "rel":
        X = X.div(X.sum(axis=1).replace(0,np.nan), axis=0).fillna(0)
    elif mode == "hellinger":
        X = np.sqrt(X.div(X.sum(axis=1).replace(0,np.nan), axis=0).fillna(0))
    elif mode == "clr":
        X = clr_transform(X)

    # behavior aligned
    y = behav_df.set_index("mouse")[y_col].reindex(X.index)

    # optional: partial Spearman by residualizing both X and y on group
    if control_group:
        g = (counts[["mouse","group"]].drop_duplicates()
               .set_index("mouse").reindex(X.index)["group"].astype("category"))
        G = pd.get_dummies(g, drop_first=True)  # one column: landmark vs ctx
        # residualize each region on G
        G_ = np.c_[np.ones(len(G)), G.values]
        P = G_ @ np.linalg.pinv(G_.T @ G_) @ G_.T
        X = X - (P @ X.values)
        # residualize y too
        y = y - pd.Series(P @ y.to_numpy(), index=y.index)

    rows = []
    for reg in X.columns:
        rho, p = spearmanr(X[reg], y, nan_policy="omit")
        rows.append({"region": reg, "rho": rho, "p": p})
    out = pd.DataFrame(rows)
    out["q"] = multipletests(out["p"], method="fdr_bh")[1]
    return out.sort_values("q")

#%%
mode = "hellinger"#"rel"#"clr", 
res_test = spearman_regionwise(counts_group, p_df, "test_score", L=lev, mode=mode)
res_gap  = spearman_regionwise(counts_group, p_df, "gap_lmk_ctx", L=lev, mode=mode)
res_training_len  = spearman_regionwise(counts_group, p_df, "training_len", L=lev, mode=mode)

#%%
def add_acronym_by_id(df, atlas, id_col="region", prefix=""):
    lab = atlas[["id","acronym","name","depth"]].rename(columns={"id": id_col})
    out = df.merge(lab, on=id_col, how="left")
    if prefix:
        out = out.rename(columns={
            "acronym": f"{prefix}_acr",
            "name":    f"{prefix}_name",
            "depth":   f"{prefix}_depth"
        })
    return out
res_test_labeled = add_acronym_by_id(res_test, atlas, id_col="region")
res_gap_labeled = add_acronym_by_id(res_gap, atlas, id_col="region")
res_training_len_labeled = add_acronym_by_id(res_training_len, atlas, id_col="region")
#%% spearman results


with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(res_test_labeled.head(15))
    print(res_gap_labeled.head(15))
    print(res_training_len_labeled.head(15))


#%%

# import numpy as np
# import pandas as pd

# # Suppose cells has xt, yt, zt in µm in atlas space
# coords = cells[["xt","yt","zt"]].to_numpy()

# # Define voxel grid
# voxel_size = 100.0  # µm
# x_bins = np.arange(coords[:,0].min(), coords[:,0].max()+voxel_size, voxel_size)
# y_bins = np.arange(coords[:,1].min(), coords[:,1].max()+voxel_size, voxel_size)
# z_bins = np.arange(coords[:,2].min(), coords[:,2].max()+voxel_size, voxel_size)

# # Histogram
# hist, edges = np.histogramdd(coords, bins=(x_bins, y_bins, z_bins))

# # Density per voxel (cells/mm³)
# voxel_mm3 = (voxel_size**3) * 1e-9
# densities = hist / voxel_mm3

# print("Voxel grid shape:", densities.shape)
