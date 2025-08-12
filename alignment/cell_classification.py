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
    if id_pair[0] in row["detected_in_sessions"]:
        if id_pair[1] in row["detected_in_sessions"]:
            return "const"
        else:
            return "off"
    elif id_pair[1] in row["detected_in_sessions"]:
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



def gather_group_counts_across_mice(
    regions,
    id_pairs,
    config,
    groups=None,                 # default depends on ttype
    ttype="presence",            # "presence" -> on/off/const; "intensity" -> up/down/stable
    normalize=True,
    dfs=None,
    denominator="union",         # "union" (|A∪B|) or "intersection" (|A∩B|) if you ever want that
    return_counts=False          # also return a wide counts table ready for GLM
):
    """
    Build a long table with counts and (optionally) percentages per Mouse×Pair×Group.
    Optionally return a wide counts table for GLMs (turnover/direction).
    """

    # Choose default groups by type
    if groups is None:
        groups = ["on", "off", "const"] if ttype == "presence" else ["up", "down", "stable"]

    rows = []

    # 1) Load dataframes if not provided
    if dfs is None:
        dfs = []
        for mouse, region in regions:
            dfs.append(utils.read_pooled_with_background(mouse, region, config))

    # 2) Loop mice
    for (i, df_raw) in enumerate(dfs):
        df = df_raw.copy()
        mouse, region = regions[i]

        # 2a) Create per-pair status columns
        if ttype == "presence":
            df = on_off_cells(df, id_pairs)      # expects values in {"on","off","const"}
        else:
            df = up_down_cells(df, id_pairs)     # expects values in {"up","down","stable"}

        # 2b) For each pair, get group counts and denominator
        for (a, b) in id_pairs:
            if (f'int_optimized_{a}' in df.columns) and (f'int_optimized_{b}' in df.columns):
                pair_col = f"{a}_to_{b}"
    
                # counts for this pair
                vc = df[pair_col].astype(str).str.lower().value_counts()
                counts = {g: int(vc.get(g, 0)) for g in groups}
    
                # denominator
                if denominator == "union":
                    n = cell_count_for_sessions(df, {a, b}, exclusive=False)  # |A∪B|
                elif denominator == "intersection":
                    n = cell_count_for_sessions(df, {a, b}, exclusive=True)   # |A∩B|
                else:
                    raise ValueError('denominator must be "union" or "intersection"')
    
                # write one row per group
                for g in groups:
                    cnt = counts[g]
                    pct = (cnt / n) if (normalize and n > 0) else (np.nan if normalize else None)
                    rows.append({
                        "Mouse": mouse,
                        "Region": region,
                        "mouse_id": f"m{mouse}r{region}",
                        "session_from": a,
                        "session_to": b,
                        "Pair": f"{a}_to_{b}",
                        "bg_A" : df[f"background_{a}"].mean(),
                        "bg_B" : df[f"background_{b}"].mean(),
                        "bg_std_A" : df[f"background_{a}"].std(),
                        "bg_std_B" : df[f"background_{b}"].std(),
                        "group": g,
                        "count": cnt,
                        "n": n,
                        "percentage": pct
                    })

    df_long = pd.DataFrame(rows)

    if not return_counts:
        return df_long

    # 3) Build a wide counts table per Mouse×Pair (ready for GLMs)
    # Pivot group counts
    df_counts = (
        df_long
        .pivot_table(index=["Mouse","Region","mouse_id","Pair", "bg_A", "bg_B", "bg_std_A", "bg_std_B"],
                     columns="group", values="count", aggfunc="sum", fill_value=0)
        .reset_index()
    )

    # Make sure expected columns exist
    for col in ["on","off","const","up","down","stable"]:
        if col not in df_counts.columns:
            df_counts[col] = 0

    # Denominator per Pair (union)
    n_per_pair = df_long.groupby(["Mouse","Region","mouse_id","Pair"])["n"].max().reset_index(name="n")
    df_counts = df_counts.merge(n_per_pair, on=["Mouse","Region","mouse_id","Pair"], how="left")

    # Derived columns depending on ttype
    if ttype == "presence":
        df_counts["changed"] = df_counts["on"] + df_counts["off"]
        df_counts["prop_changed"] = np.where(df_counts["n"] > 0,
                                             df_counts["changed"] / df_counts["n"], np.nan)
        df_counts["prop_on"] = np.where(df_counts["changed"] > 0,
                                        df_counts["on"] / df_counts["changed"], np.nan)
    else:
        # intensity classification: "up/down/stable"
        df_counts["changed"] = df_counts["up"] + df_counts["down"]
        df_counts["prop_changed"] = np.where(df_counts["n"] > 0,
                                             df_counts["changed"] / df_counts["n"], np.nan)
        df_counts["prop_up"] = np.where(df_counts["changed"] > 0,
                                        df_counts["up"] / df_counts["changed"], np.nan)

    return df_long, df_counts
