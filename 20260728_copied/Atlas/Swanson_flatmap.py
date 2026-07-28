import numpy as np
from iblatlas.plots import plot_swanson_vector
from iblatlas.atlas import BrainRegions
from iblatlas.atlas import AllenAtlas
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, to_hex
from matplotlib.colors import Normalize
import os

# br = BrainRegions()
# plot_swanson_vector(br=br, annotate=True)
# plt.savefig(r"P:\Histological_analysis\_Braion_regions.pdf", format="pdf", bbox_inches="tight")


# indices = br.acronym2index('VISa')
# swanson_indices = np.unique(br.mappings['Swanson'])
# for index in swanson_indices:
#     print("################")
#     # print(br.id[index])
#     print(br.acronym[index])
#     print(br.level[index])
#     # print(br.parent[index])
#     # pi = br.parent[index]
#     # parent_index = int(pi) if not pd.isna(pi) else None
#     parent_id = br.parent[index]
#     print(parent_id)
#     if not pd.isna(parent_id):
#         print(br.id2acronym(parent_id))
#         # parent_index = br.id2index(parent_id)
#         # print(parent_index)
#         # print(br.level[parent_index])
#         # print(br.acronym[parent_index])


def swanson_mapping(csv_path):
    dir = os.path.dirname(csv_path)
    df = pd.read_csv(csv_path)

    acronyms =  df.columns.tolist()[3:]
    rename_map = {
        "MOant": "MOs",
        "MOmid": "MOp",
        "MOpost": "SSp-tr",
        "SSant": "SSp-m",
        "SSmid":"SSp-n",
        "SSpost":"SSs"
    }
    acronyms_for_temp_plot = [rename_map.get(name, name) for name in acronyms]

    # br = BrainRegions()
    # ontology_acronyms = br.acronym  # Swanson atlas の登録済み acronym 一覧
    # # 存在しない acronym を探す
    # invalid = [ac for ac in acronyms_for_temp_plot if ac not in ontology_acronyms]
    # print("存在しないacronyms:", invalid)

    # cmap = LinearSegmentedColormap.from_list("silver_red", ["silver", "#fa5f2f"], N=256) #"#E9967A"
    cmap = LinearSegmentedColormap.from_list("silver_red", ["white", "#f50707"], N=256)  # "#E9967A"

    for group_name, group_df in df.groupby("Group"):
        numeric_df = group_df[acronyms].apply(pd.to_numeric, errors='coerce')  # 数値以外はNaNに
        values = numeric_df.mean().to_numpy()
        # values = group_df[acronyms].mean().to_numpy()
        print(group_name)
        print(acronyms_for_temp_plot)
        print(values)
        normalized_values = np.clip(values, 0, 100) / 100  # 0〜1のfloat値に変換

        print(normalized_values)
        fig, ax = plt.subplots(figsize=(6, 6))
        plot_swanson_vector(acronyms_for_temp_plot, normalized_values,
                            annotate=False,
                            # annotate_list=['MO', 'SS', 'ACA', 'ORB'],
                            cmap=cmap,
                            vmin=0.0, vmax=1,
                            ax =ax,
                            empty_color='silver')

        sm = cm.ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap=cmap)
        sm.set_array([])  # ダミー
        cbar = fig.colorbar(sm, ax=ax, shrink=0.6)  # ← どの Axes から余白をもらうか指定
        cbar.set_label("Expression (%)")

        plt.savefig(os.path.join(dir,group_name+"_swanson.pdf"), format="pdf", bbox_inches="tight")

def main():
    csv_path = r"P:\Histological_analysis\Pup_expression_summary.csv"
    swanson_mapping(csv_path)

if __name__ == "__main__":
    main()