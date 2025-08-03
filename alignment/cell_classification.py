import utils
import pandas as pd

landmark_sessions = {"landmark1", "landmark2"}
ctx_sessions = {"ctx1", "ctx2"}
all_sessions = {"s0", "landmark1", "landmark2", "ctx1", "ctx2"}

def is_test_specific(sessions):
    return not "s0" in sessions

def is_mixed(sessions):
    return not ctx_sessions.isdisjoint(sessions) and not(landmark_sessions.isdisjoint(sessions))


def is_specific_to_type(sessions, target_group, opposing_group):
    return target_group.issubset(sessions) and opposing_group.isdisjoint(sessions)

def appeared_in_sessions(row_sessions, sessions_to_check):
    for session in sessions_to_check:
        if session in row_sessions:
            return True
    return False

def get_cell_status(row, id_pair):
    if id_pair[0] in row["detected_in_sessions"]:
        if id_pair[1] in row["detected_in_sessions"]:
            return "stable"
        else:
            return "down"
    elif id_pair[1] in row["detected_in_sessions"]:
        return "up"
    return "_"
    
def up_down_cells(df, id_pairs):
    for id_pair in id_pairs:
        df[f"{id_pair[0]}_to_{id_pair[1]}"] = df.apply(get_cell_status, 
                                                       id_pair = id_pair, 
                                                       axis=1)
    return df


def cell_count_for_sessions(df, sessions_to_check):
    df["is_in_sessions"] = df["detected_in_sessions"].apply(appeared_in_sessions,
                                                            sessions_to_check = sessions_to_check)
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
        df_filtered = df.loc[df[f"{id_pair[0]}_to_{id_pair[1]}"] == group]
        number_in_pair = df_filtered.shape[0]
        if normalize:
            number_in_pair /= cell_count_for_sessions(df, set(id_pair))
        group_occurences_per_pair.extend([number_in_pair])
    print(group, group_occurences_per_pair)
    return group_occurences_per_pair

def gather_group_percentages_across_mice(regions, id_pairs, config, groups=("up", "down", "stable"), normalize=True):
    data = []
    for mouse, region in regions:
        df = utils.read_pooled_cells(mouse, region, config)
        for group in groups:
            df = up_down_cells(df, id_pairs)
            percentages = count_cells_per_tendency_group(df, 
                                                         id_pairs, 
                                                         group=group, 
                                                         normalize=True)
            for (id_pair, pct) in zip(id_pairs, percentages):
                data.append({
                    "mouse_id": f"m{mouse}r{region}",
                    "session_from": id_pair[0],
                    "session_to": id_pair[1],
                    "group": group,
                    "percentage": pct
                })
    return pd.DataFrame(data)

def mark_cells_specificity_class(df):
    df["test_specific"] = df["detected_in_sessions"].apply(is_test_specific, axis=1)
    df["ctx_specific"] = df["detected_in_sessions"].apply(is_specific_to_type,
                                                          target_group = ctx_sessions,
                                                          opposing_group = landmark_sessions,
                                                          axis=1)
    df["landmark_specific"] = df["detected_in_sessions"].apply(is_specific_to_type,
                                                               target_group = landmark_sessions,
                                                               opposing_group = ctx_sessions,
                                                               axis=1)
    df["is_mixed"] = df["detected_in_sessions"].apply(is_mixed, axis=1)
    return df

def count_cells_in_spec_class(df, cl, normalize=True):
    if cl in df.columns:
        cl_count = df[df[cl]].shape[0]
        if normalize:
            cl_count /= df.shape[0]
        return cl_count
    

def gather_cells_specificity_percentages_across_mice(regions, config, classes, normalize=True):
    data = []
    for mouse, region in regions:
        df = utils.read_pooled_cells(mouse, region, config)
        df = mark_cells_specificity_class(df)
        for cl in classes:
            percentage = count_cells_in_spec_class(df, cl, normalize)
            data.append({
                "mouse_id": f"m{mouse}r{region}",
                "spec_class": cl,
                "percentage": percentage
            })
    return pd.DataFrame(data)
