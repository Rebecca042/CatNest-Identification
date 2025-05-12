import os
import re
import tempfile

import pandas as pd
import datetime
import time
import multiprocessing

from PIL import Image
from meteostat import Hourly, Point
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.discovery.inductive.variants import imf
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness_algorithm
from pm4py.algo.discovery.footprints import algorithm as footprints_discovery
from pm4py.objects.conversion.process_tree import converter as tree_converter
from pm4py.objects.log.util import dataframe_utils
from joblib import Parallel, delayed

from GenerateProcessDataWeather import generate_log_weather_dependent
from GenerateProcessData import generate_log

results_summary = []

def get_resized_image_path(original_path, width_px):
    img = Image.open(original_path)
    aspect_ratio = img.height / img.width
    height_px = int(width_px * aspect_ratio)

    temp_dir = tempfile.gettempdir()
    filename = os.path.basename(original_path)
    resized_path = os.path.join(temp_dir, f"resized_{width_px}px_{filename}")

    if not os.path.exists(resized_path):  # Avoid regenerating
        resized_img = img.resize((width_px, height_px), Image.LANCZOS)
        resized_img.save(resized_path)

    return resized_path

def visualize_petri_net(net, im, fm, label="visualization"):
    gviz = pn_visualizer.apply(net, im, fm, parameters={"format": "png", "engine": "dot"})

    activity_to_icon = {
        'ComingHome': '../Processes/Activities/ComingHome.png',
        'Cooking': "../Processes/Activities/CookingPerson.png",
        'Eating': "../Processes/Activities/Eating.png",
        'ComputerGames': "../Processes/Activities/GamingPerson.png",
        'WatchingTV': "../Processes/Activities/WatchingTV.png",
        'Reading(Book)': "../Processes/Activities/ReadingPerson.png",
        'Cat-cuddles': "../Processes/Activities/Cat-cuddelingPerson.png",
        'Sleeping': "../Processes/Activities/SleepingPerson.png"
    }

    nodes_to_update = {}
    for line in gviz.body:
        node_id_match = re.match(r'^\s*(\S+)\s*\[', line)

        # Try the regex for quoted labels first
        label_search = re.search(r'label=(.*?)[\s\]]', line)
        label_search_quoted = re.search(r'label="([^"]*)"', line)
        if label_search_quoted:
            label_search = label_search_quoted
        else:
            label_search_unquoted = re.search(r'label\s*=\s*"([^"]*)"', line)
            if label_search_unquoted:
                label_search = label_search_quoted
        if node_id_match and label_search:
            node_id = node_id_match.group(1)
            label_text = label_search.group(1).strip()
            if 'shape=circle' not in line and label_text in activity_to_icon:
                path = get_resized_image_path(activity_to_icon[label_text], width_px=80)
                nodes_to_update[node_id] = (label_text, path)

    updated_node_ids = set()
    for i, statement in enumerate(gviz.body):
        match_node = re.match(r'^\s*(\S+)\s*\[.*label="?([^"]*)"?(.*)border=', statement)
        if match_node:
            node_id = match_node.group(1)
            if node_id in nodes_to_update and node_id not in updated_node_ids:
                label_text, path = nodes_to_update[node_id]
                path = path.replace("\\", "/")
                html_label = f"<table><tr><td>{label_text}</td></tr><tr><td><img src='{path}' /></td></tr></table>"
                gviz.body[i] = re.sub(r'label=.+? border=', f'label=<{html_label}> border=', gviz.body[i])
                updated_node_ids.add(node_id)

    output_dir = os.path.join("visualizations", label)
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f'petri_net_with_images_{label}')
    gviz.render(filename, format='png', view=False, cleanup=True)
    print(f"Generated {filename}.png")


def mine_setting(df_setting):
    log = log_converter.apply(df_setting)
    return inductive_miner.apply(log, parameters={"noiseThreshold": 0.0})


def conduct_comparison(df, label):
    df_pm = df.rename(columns={
        'Case ID': 'case:concept:name',
        'Activity': 'concept:name',
        'Timestamp': 'time:timestamp'
    })
    df_pm['time:timestamp'] = pd.to_datetime(df_pm['time:timestamp'])
    df_pm = dataframe_utils.convert_timestamp_columns_in_df(df_pm)

    start_global = time.time()
    log_global = log_converter.apply(df_pm)
    process_tree_global = inductive_miner.apply(log_global, parameters={"noiseThreshold": 0.0})
    process_tree_global0 = inductive_miner.apply(log_global, variant=imf, parameters={"noiseThreshold": 0.1})
    end_global = time.time()
    global_time = end_global - start_global
    print(f"Global mining time: {global_time:.2f} seconds")

    parallel_time = 0

    if 'Setting' in df_pm.columns:
        setting_ids = df_pm['Setting'].unique().tolist()
        setting_dfs = [group for _, group in df_pm.groupby('Setting')]

        for sid, sdf in zip(setting_ids, setting_dfs):
            print(f"Setting {sid} → {len(sdf)} rows, {sdf['case:concept:name'].nunique()} unique cases")

        start_parallel = time.time()
        results = Parallel(n_jobs=multiprocessing.cpu_count(), prefer="processes")(
            delayed(mine_setting)(df_setting) for df_setting in setting_dfs
        )
        end_parallel = time.time()
        parallel_time = end_parallel - start_parallel
        print(f"Parallel mining time: {parallel_time:.2f} seconds")
    else:
        print("Column 'Setting' not found in the data.")
        return

    net_global, im_global, fm_global = tree_converter.apply(process_tree_global)
    visualize_petri_net(net_global, im_global, fm_global, label=label)

    for sid, df_setting in zip(setting_ids, setting_dfs):
        log_setting = log_converter.apply(df_setting)
        fitness = replay_fitness_algorithm.apply(log_setting, net_global, im_global, fm_global)
        avg_fitness = fitness['averageFitness']
        print(f"Setting {sid} — Fitness against global model: {avg_fitness:.3f}")

    fp_global = footprints_discovery.apply(process_tree_global)
    fp_setting = footprints_discovery.apply(results[0])

    common = fp_global['sequence'] & fp_setting['sequence']
    unique_global = fp_global['sequence'] - fp_setting['sequence']
    unique_setting = fp_setting['sequence'] - fp_global['sequence']

    results_summary.append({
        'Label': label,
        'GlobalTime': global_time,
        'ParallelTime': parallel_time,
        'CommonRelations': len(common),
        'OnlyInGlobal': len(unique_global),
        'OnlyInSetting': len(unique_setting)
    })


if __name__ == "__main__":
    multiprocessing.freeze_support()

    log_sizes = [5000, 10000, 20000, 50000, 100000]
    munich = Point(48.1351, 11.5820)

    for size in log_sizes:
        print(f"\n=== Generating standard log with {size} events ===")
        df_std = generate_log(size)
        df_std = df_std.sort_values(by=['Case ID', 'Timestamp']).reset_index(drop=True)
        conduct_comparison(df_std, label=f"Standard_{size}")


        print(f"\n=== Generating weather-dependent log with {size} cases ===")
        if size in [50000, 100000]:
            print(f"Skipping weather-dependent generation for size {size} due to performance constraints.")
        else:
            df_weather = generate_log_weather_dependent(num_cases=size, location=munich)
            df_weather = df_weather.sort_values(by=['Case ID', 'Timestamp']).reset_index(drop=True)
            conduct_comparison(df_weather, label=f"Weather_{size}")

    print("\n=== Summary of All Runs ===")
    df_summary = pd.DataFrame(results_summary)
    print(df_summary)
    df_summary.to_csv("comparison_summary.csv", index=False)
