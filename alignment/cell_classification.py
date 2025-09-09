import utils
import pandas as pd
import numpy as np

landmark_sessions = {"landmark1", "landmark2"}
ctx_sessions = {"ctx1", "ctx2"}
all_sessions = {"s0", "landmark1", "landmark2", "ctx1", "ctx2"}

def is_test_specific(sessions):
    return not "s0" in sessions

def is_mixed(sessions):
    return not ctx_sessions.isdisjoint(sessions) and not(landmark_sessions.isdisjoint(sessions))


def is_specific_to_type(sessions, target_group, opposing_group):
    return target_group.issubset(sessions) and opposing_group.isdisjoint(sessions)

def appeared_in_sessions(row_sessions, sessions_to_check, exclusive=False):
    if exclusive:
        return (set(sessions_to_check) == row_sessions)# or (set(sessions_to_check+["s0"]) == row_sessions)
    for session in sessions_to_check:
        if session in row_sessions:
            return True
    return False

def get_cell_status(row, id_pair):
    if id_pair[0] in row["detected_in_sessions"] and (not row[f'is_dim_by_bg_{id_pair[0]}']):
        if id_pair[1] in row["detected_in_sessions"] and not(row[f'is_dim_by_bg_{id_pair[1]}']):
            return "const"
        else:
            return "off"
    elif id_pair[1] in row["detected_in_sessions"] and not (row[f'is_dim_by_bg_{id_pair[1]}']):
        return "on"
    return "_"


    
def on_off_cells(df, id_pairs):
    for id_pair in id_pairs:
        if (f'int_optimized_{id_pair[0]}' in df.columns) and (f'int_optimized_{id_pair[1]}' in df.columns):
            df[f"{id_pair[0]}_to_{id_pair[1]}"] = df.apply(get_cell_status, 
                                                           id_pair = id_pair, 
                                                           axis=1)

    return df
    
def up_down_cells(df, id_pairs, threshold=0.2):
    for id_pair in id_pairs:
        id1, id2 = id_pair
        df[f'int_optimized_{id1}'] = pd.to_numeric(df[f'int_optimized_{id1}'])
        df["lower_lim"] = df[f'int_optimized_{id1}'] * (1-threshold)
        df["upper_lim"] = df[f'int_optimized_{id1}'] * (1+threshold)
        col_name = f"{id1}_to_{id2}"
        df[col_name] = "stable"
        df.loc[df[f'int_optimized_{id2}']>df["upper_lim"],col_name] = "up"
        df.loc[df[f'int_optimized_{id2}']<df["lower_lim"],col_name] = "down"
        
    return df


def cell_count_for_sessions(df, sessions_to_check, exclusive=False):
    df["is_in_sessions"] = df["detected_in_sessions"].apply(appeared_in_sessions,
                                                            sessions_to_check = sessions_to_check, exclusive=exclusive)
    return df.loc[df["is_in_sessions"]].shape[0]

def count_cells_per_tendency_group(df, id_pairs, group, normalize=True):
    """
    Parameters
    ----------
    df : df with cell pooled from all sessions
    id_pairs : (a,b) pairs of sessions
    group : which group out of "up", "down" and "stable" to quantify

    Returns
    -------
    group_occurences_per_pair : for each pair from id_pairs returns the number
    of cells that belong to the specified group based on transition from a to b

    """
    group_occurences_per_pair = []
    for id_pair in id_pairs:
        if (f'int_optimized_{id_pair[0]}' in df.columns) and (f'int_optimized_{id_pair[1]}' in df.columns):
            df_filtered = df.loc[df[f"{id_pair[0]}_to_{id_pair[1]}"] == group]
            number_in_pair = df_filtered.shape[0]
            if normalize:
                number_in_pair /= cell_count_for_sessions(df, set(id_pair))
            group_occurences_per_pair.extend([number_in_pair])
        
    print(group, group_occurences_per_pair)
    return group_occurences_per_pair

def gather_group_percentages_across_mice(regions, id_pairs, config, groups=["on", "off", "const"], ttype = "presence",normalize=True, dfs = None):
    data = []
    
    if dfs is None:
        dfs = []
        for mouse, region in regions:
            dfs += [utils.read_pooled_with_background(mouse, region, config)]
        
    for i, df in enumerate(dfs):
        df = df.copy()
        mouse, region = regions[i]
        if ttype == 'presence':
            df = on_off_cells(df, id_pairs)
        else:
            df = up_down_cells(df, id_pairs)
        for group in groups:
            percentages = count_cells_per_tendency_group(df, 
                                                         id_pairs, 
                                                         group=group, 
                                                         normalize=normalize)
            for (id_pair, pct) in zip(id_pairs, percentages):
                if (f'int_optimized_{id_pair[0]}' in df.columns) and (f'int_optimized_{id_pair[1]}' in df.columns):
                    data.append({
                        "mouse_id": f"m{mouse}r{region}",
                        "session_from": id_pair[0],
                        "session_to": id_pair[1],
                        "session_pair": f"{id_pair[0]}_{id_pair[1]}",
                        "group": group,
                        "percentage": pct
                    })
    return pd.DataFrame(data)

def mark_cells_specificity_class(df):
    df["test_specific"] = df["detected_in_sessions"].apply(is_test_specific)
    df["ctx_specific"] = df["detected_in_sessions"].apply(is_specific_to_type,
                                                          target_group = ctx_sessions,
                                                          opposing_group = landmark_sessions)
    df["landmark_specific"] = df["detected_in_sessions"].apply(is_specific_to_type,
                                                               target_group = landmark_sessions,
                                                               opposing_group = ctx_sessions)
    df["is_mixed"] = df["detected_in_sessions"].apply(is_mixed)
    df["is_transient"] = df["n_sessions"] == 1
    df["is_intermediate"] = (df["n_sessions"] < 4) & ~df["is_transient"]
    df["is_persistent"] = df["n_sessions"] >3
    return df

    

def count_cells_in_spec_class(df, cl, normalize=True):
    if cl in df.columns:
        cl_count = df[df[cl]].shape[0]
        if normalize:
            cl_count /= df.shape[0]
        return cl_count
    

def gather_cells_specificity_percentages_across_mice(regions, config, classes, normalize=True, filterby=None, dfs=None):
    data = []
    if dfs is None:
        dfs = []
        for mouse, region in regions:
            dfs += [utils.read_pooled_with_background(mouse, region, config)]
        
    for i, df in enumerate(dfs):
        mouse, region = regions[i]
        df = utils.read_pooled_with_background(mouse, region, config)
        df = mark_cells_specificity_class(df)
        
        if filterby is not None:
            df = df[df[filterby]]
        
        for cl in classes:
            percentage = count_cells_in_spec_class(df, cl, normalize)
            print(mouse, cl, percentage)
            data.append({
                "mouse_id": f"m{mouse}r{region}",
                "spec_class": cl,
                "percentage": percentage
            })
    return pd.DataFrame(data)



# Fallback parser if you don't want to depend on your utils.parse_setlike
def _parse_setlike(x):
    if isinstance(x, (set, list, tuple)): return set(map(str, x))
    if isinstance(x, str):
        # tolerant for "{a,b}" or "['a', 'b']" or "a,b"
        x = x.strip().strip("{}[]()")
        if not x: return set()
        parts = [p.strip().strip("'\"") for p in x.split(",")]
        return set(p for p in parts if p)
    return set()

def _denom_by_mouse(df_mouse: pd.DataFrame, a: str, b: str, how="union") -> int:
    """
    Count denominator for a single mouse: |A∪B| or |A∩B|.
    Uses 'detected_in_sessions' if present; otherwise falls back to
    presence in intensity columns int_optimized_{session}.
    """
    df_mouse = df_mouse.loc[~(df_mouse[f"is_dim_by_bg_{a}"]&df_mouse[f"is_dim_by_bg_{b}"])]
    if "detected_in_sessions" in df_mouse.columns:
        ssets = df_mouse["detected_in_sessions"].map(_parse_setlike)
        if how == "union":
            return int((ssets.apply(lambda s: a in s or b in s)).sum())
        else:  # intersection
            return int((ssets.apply(lambda s: a in s and b in s)).sum())
    # fallback: look for non-missing intensities
    a_col = f"int_optimized_{a}"
    b_col = f"int_optimized_{b}"
    has_a = df_mouse[a_col].notna() if a_col in df_mouse.columns else pd.Series(False, index=df_mouse.index)
    has_b = df_mouse[b_col].notna() if b_col in df_mouse.columns else pd.Series(False, index=df_mouse.index)
    if how == "union":
        return int((has_a | has_b).sum())
    else:
        return int((has_a & has_b).sum())




def gather_group_counts_across_mice(
    all_mice,
    id_pairs,
    groups=None,                 # default depends on ttype
    ttype="presence",            # "presence" -> on/off/const; "intensity" -> up/down/stable
    normalize=True,
    mouse_col = "mouse",
    denominator="union",         # "union" (|A∪B|) or "intersection" (|A∩B|) if you ever want that
    return_counts=False,          # also return a wide counts table ready for GLM
    canonical: bool = False
):
    """
    Build a long table with counts and (optionally) percentages per Mouse×Pair×Group.
    Optionally return a wide counts table for GLMs (turnover/direction).
    """

    if groups is None:
        groups = ["on","off","const"] if ttype == "presence" else ["up","down","stable"]

    df = all_mice.copy()

    df = df.loc[df["n_sessions"]<3]


    # 2a) status columns (once on the whole table)
    if ttype == "presence":
        # expects to create columns like f"{a}_to_{b}" with {"on","off","const"}
        df = on_off_cells(df, id_pairs)
    else:
        # expects {"up","down","stable"} in f"{a}_to_{b}"
        df = up_down_cells(df, id_pairs)

    # 2b) aggregate per mouse (and region)
    rows = []
    grp_keys = [mouse_col]

    for (a, b) in id_pairs:
        pair_col = f"{a}_to_{b}"
        if pair_col not in df.columns:
            # skip silently if a/b columns missing in this dataset slice
            continue

        # background summaries per mouse (useful covariate context)
        # do it pairwise so A/B map properly
        bgA = f"background_{a}"
        bgB = f"background_{b}"
        have_bgA = bgA in df.columns
        have_bgB = bgB in df.columns

        for gvals, sub in df.groupby(grp_keys, observed=True, sort=False):
            # gvals is a tuple if region is included; make accessors
            
            #sub = sub.loc[~(sub[f'is_dim_by_bg_{a}']|sub[f'is_dim_by_bg_{b}'])]
            mouse_val = gvals if isinstance(gvals, (str, int)) else gvals[0]
            region_val = None
            if len(grp_keys) == 2:
                region_val = gvals[1]

            # counts of statuses within this mouse (sub-table)
            vc = sub[pair_col].astype(str).str.lower().value_counts()
            counts = {g: int(vc.get(g, 0)) for g in groups}

            # denominator for this mouse (union/intersection)
            n = _denom_by_mouse(sub, a, b, how=denominator)
            print(np.array(list(counts.values()))/n)
            print(n)
            # background summaries (mouse-level, not pair-filtered)
            bg_A_mean = float(sub[bgA].mean()) if have_bgA else np.nan
            bg_B_mean = float(sub[bgB].mean()) if have_bgB else np.nan
            bg_A_std  = float(sub[bgA].std())  if have_bgA else np.nan
            bg_B_std  = float(sub[bgB].std())  if have_bgB else np.nan

            # label to write
            pair_label_out = utils.canonical_pair(a,b) if canonical else f"{a}_to_{b}"

            for grp in groups:
                cnt = counts[grp]
                pct = (cnt / n) if (normalize and n > 0) else (np.nan if normalize else None)
                row = {
                    "Mouse": mouse_val,
                    "mouse_id": str(mouse_val),
                    "Pair": pair_label_out,
                    "session_from": a,
                    "session_to": b,
                    "group": grp,
                    "count": cnt,
                    "n": int(n),
                    "percentage": pct,
                    "bg_A": bg_A_mean,
                    "bg_B": bg_B_mean,
                    "bg_std_A": bg_A_std,
                    "bg_std_B": bg_B_std,
                }
                if region_val is not None:
                    row["Region"] = region_val
                    row["mouse_id"] = f"m{mouse_val}r{region_val}"
                rows.append(row)

    df_long = pd.DataFrame(rows)

    if not return_counts:
        return df_long

    # wide counts per Mouse×Pair
    index_cols = ["Mouse","mouse_id","Pair","bg_A","bg_B","bg_std_A","bg_std_B"]
    if "Region" in df_long.columns:
        index_cols.insert(1, "Region")

    df_counts = (
        df_long
        .pivot_table(index=index_cols, columns="group", values="count",
                     aggfunc="sum", fill_value=0)
        .reset_index()
    )

    # ensure expected columns exist
    for col in ["on","off","const","up","down","stable"]:
        if col not in df_counts.columns:
            df_counts[col] = 0

    # denominator per Mouse×Pair
    n_per = df_long.groupby(index_cols)["n"].max().reset_index(name="n")
    df_counts = df_counts.merge(n_per, on=index_cols, how="left")

    # derived proportions
    if ttype == "presence":
        df_counts["changed"] = df_counts["on"] + df_counts["off"]
        df_counts["prop_changed"] = np.where(df_counts["n"] > 0,
                                             df_counts["changed"] / df_counts["n"], np.nan)
        df_counts["prop_on"] = np.where(df_counts["changed"] > 0,
                                        df_counts["on"] / df_counts["changed"], np.nan)
    else:
        df_counts["changed"] = df_counts["up"] + df_counts["down"]
        df_counts["prop_changed"] = np.where(df_counts["n"] > 0,
                                             df_counts["changed"] / df_counts["n"], np.nan)
        df_counts["prop_up"] = np.where(df_counts["changed"] > 0,
                                        df_counts["up"] / df_counts["changed"], np.nan)

    return df_long, df_counts
