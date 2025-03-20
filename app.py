import pickle
import os
import re
import json
import logging

import gradio as gr
import pandas as pd

from trustfuse.conflicting_dataset.dataset import (DynamicDataset,
                                                   StaticDataset)
from trustfuse.evaluation import evaluation
from trustfuse.visualization import visualization
import settings
import utils


THEME = gr.themes.Default(primary_hue="orange",
                          secondary_hue="blue")

# Absolute paths to load available datasets
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIRECTORY = 'data/input_trustfuse/wikiconflict/'

# Existing datasets in the app
DYNAMIC_DATASETS_AVAILABLE = {
    "Paris monuments (WikiConflict)": {},
    "Machine Learning (WikiConflict)": {},
    "Football World Cup 2022 (WikiConflict)": {},
}
STATIC_DATASETS_AVAILABLE = {
    "Flights": {
        "attr_type_path": os.path.join(BASE_DIR, "data",
                                       "configurations",
                                       "truthfinder", "flights",
                                       "types.json"),
        "params_path": os.path.join(BASE_DIR, "data",
                                    "input_trustfuse",
                                    "flights", "dataset_parameters.json"),
        "data_folder": os.path.join(BASE_DIR, "data",
                                    "input_trustfuse",
                                    "flights", "conflicting_data"),
        "gt_folder": os.path.join(BASE_DIR, "data",
                                  "input_trustfuse",
                                  "flights", "ground_truth"),
        "preprocess": os.path.join(BASE_DIR, "data",
                                   "configurations",
                                   "truthfinder", "flights",
                                   "preprocess_configuration.json")
    },
    "Stocks": {
        "attr_type_path": os.path.join(BASE_DIR, "data",
                                       "configurations",
                                       "truthfinder", "stocks",
                                       "types.json"),
        "params_path": os.path.join(BASE_DIR, "data",
                                    "input_trustfuse",
                                    "stocks", "dataset_parameters.json"),
        "data_folder": os.path.join(BASE_DIR, "data",
                                    "input_trustfuse",
                                    "stocks", "conflicting_data"),
        "gt_folder": os.path.join(BASE_DIR, "data",
                                  "input_trustfuse",
                                  "stocks", "ground_truth"),
        "preprocess": os.path.join(BASE_DIR, "data",
                                   "configurations",
                                   "truthfinder", "stocks",
                                   "preprocess_configuration.json")
    },
    "Weather": {
        "attr_type_path": os.path.join(BASE_DIR, "data",
                                       "configurations",
                                       "truthfinder", "weather",
                                       "types.json"),
        "params_path": os.path.join(BASE_DIR, "data",
                                    "input_trustfuse",
                                    "weather", "dataset_parameters.json"),
        "data_folder": os.path.join(BASE_DIR, "data",
                                    "input_trustfuse",
                                    "weather", "conflicting_data"),
        "gt_folder": os.path.join(BASE_DIR, "data",
                                  "input_trustfuse",
                                  "weather", "ground_truth"),
        "preprocess": os.path.join(BASE_DIR, "data",
                                   "configurations",
                                   "truthfinder", "weather",
                                   "preprocess_configuration.json")
    },
    "Book": {
        "attr_type_path": os.path.join(BASE_DIR, "data",
                                       "configurations",
                                       "truthfinder", "book",
                                       "types.json"),
        "params_path": os.path.join(BASE_DIR, "data",
                                    "input_trustfuse",
                                    "book", "dataset_parameters.json"),
        "data_folder": os.path.join(BASE_DIR, "data",
                                    "input_trustfuse",
                                    "book", "conflicting_data"),
        "gt_folder": os.path.join(BASE_DIR, "data",
                                  "input_trustfuse",
                                  "book", "ground_truth"),
        "preprocess": os.path.join(BASE_DIR, "data",
                                   "configurations",
                                   "truthfinder", "book",
                                   "preprocess_configuration.json")
    },
}

# global variables
dataset_global = None
colors_map = None
metrics_global = None

### Dataset loading functions
def load_dataset(dataset, mode):
    """Load dataset in pickle format as described in the doc"""
    global dataset_global
    if dataset is None:
        return None, "📂 No dataset selected.", "", ""
    dataset_name = dataset.name

    with open(os.path.join(BASE_DIR, "data", "configurations", "property_types.pkl"), 'rb') as f:
        attr_types = pickle.load(f)
        # Define types not in configuration file
        attr_types["label_en"] = "string"
        attr_types["label_fr"] = "string"
        attr_types["description_en"] = "string"
        attr_types["description_fr"] = "string"

    try:
        if dataset_name.endswith('.csv'):
            df = pd.read_csv(dataset_name)

        elif dataset_name.endswith('.pkl'):
            dataset_global = DynamicDataset(dataset_name,
                                            attribute_types=attr_types,
                                            **settings.DATASET_PARAMETERS["WikiConflict"],
                                            entity_as="string") # entity_as="qid" / "string"
            dataset_global.make_post_preprocess_copy()
            preprocess_file_path = os.path.join(BASE_DIR, "data", "configurations",
                                                "truthfinder", "wikiconflict", 
                                                "preprocess_configuration.json")
            with open(preprocess_file_path , encoding="utf-8") as preprocessing_config_file:
                preprocess_config = json.load(preprocessing_config_file)
                logging.info("DATA PREPROCESSING")
                dataset_global.apply_data_preprocessing(preprocess_config)
                logging.info("METADATA PREPROCESSING")
                dataset_global.apply_metadata_preprocessing(preprocess_config)

            df = dataset_global.data_pp[0]

        else:
            return (None,
                    "⚠️ Format not supported.",
                    "<p style='color:red'>⚠️ No dataset selected, the graph cannot be generated.</p>",
                    "",
                    "?",
                    "")
        return (df,
                "✅ Dataset successfully loaded.",
                generate_graph(0, mode),
                generate_partial_order_graph(0),
                gr.update(value="Bucket #0",
                          choices=[f"Bucket #{key}" for key in dataset_global.data]))
    except Exception as e:
        print(e)
        return (None,
                f"❌ Error : {str(e)}",
                "<p style='color:red'>⚠️ No dataset selected, the graph cannot be generated.</p>",
                "",
                "?",
                "")


### Dataset loading functions
def load_available_dataset(dataset_name, mode, progress=gr.Progress()):
    """Load dataset in pickle format as described in the doc"""
    global dataset_global

    if dataset_name in DYNAMIC_DATASETS_AVAILABLE:

        with open(os.path.join(BASE_DIR, "data", "configurations", "property_types.pkl"), 'rb') as f:
            attr_types = pickle.load(f)
            # Define types not in configuration file
            attr_types["label_en"] = "string"
            attr_types["label_fr"] = "string"
            attr_types["description_en"] = "string"
            attr_types["description_fr"] = "string"

        try:
            if dataset_name.endswith('.csv'):
                df = pd.read_csv(dataset_name)

            elif dataset_name.endswith('.pkl'):
                dataset_global = DynamicDataset(dataset_name,
                                                attribute_types=attr_types,
                                                **settings.DATASET_PARAMETERS["WikiConflict"],
                                                entity_as="string") # entity_as="qid" / "string"
                dataset_global.make_post_preprocess_copy()
                preprocess_file_path = os.path.join(BASE_DIR, "data", "configurations",
                                                    "truthfinder", "wikiconflict", 
                                                    "preprocess_configuration.json")
                with open(preprocess_file_path , encoding="utf-8") as preprocessing_config_file:
                    preprocess_config = json.load(preprocessing_config_file)
                    logging.info("DATA PREPROCESSING")
                    dataset_global.apply_data_preprocessing(preprocess_config, progress=progress)
                    logging.info("METADATA PREPROCESSING")
                    dataset_global.apply_metadata_preprocessing(preprocess_config, progress=progress)

                df = dataset_global.data_pp[0]

            else:
                return "⚠️ Format not supported."
            return "✅ Dataset successfully loaded."
        except Exception as e:
            print(e)
            return f"❌ Error : {str(e)}"

    if dataset_name in STATIC_DATASETS_AVAILABLE:
        file_path = STATIC_DATASETS_AVAILABLE[dataset_name]["attr_type_path"]
        with open(file_path, encoding="utf-8") as f:
            attr_types = json.load(f)
        file_path = STATIC_DATASETS_AVAILABLE[dataset_name]["params_path"]
        with open(file_path, encoding="utf-8") as f:
            parameters = json.load(f)

        try:
            data_folder_path = STATIC_DATASETS_AVAILABLE[dataset_name]["data_folder"]
            gt_folder_path = STATIC_DATASETS_AVAILABLE[dataset_name]["gt_folder"]
            data_folder = [os.path.join(data_folder_path, bucket)
                           for bucket in os.listdir(data_folder_path)]
            gt_folder = [os.path.join(gt_folder_path, bucket)
                         for bucket in os.listdir(gt_folder_path)]

            dataset_global = StaticDataset([data_folder, gt_folder],
                                attribute_types=attr_types,
                                gradio=True,
                                **parameters)
            dataset_global.make_post_preprocess_copy()
            preprocess_file_path = STATIC_DATASETS_AVAILABLE[dataset_name]["preprocess"]

            with open(preprocess_file_path , encoding="utf-8") as preprocessing_config_file:
                preprocess_config = json.load(preprocessing_config_file)
                logging.info("DATA PREPROCESSING")
                dataset_global.apply_data_preprocessing(preprocess_config, progress=progress)
                logging.info("METADATA PREPROCESSING")
                dataset_global.apply_metadata_preprocessing(preprocess_config, progress=progress)

            df = dataset_global.data_pp[0]

            return "✅ Dataset successfully loaded."
        except Exception as e:
            print(e)
            return f"❌ Error : {str(e)}"
        

def display_dataset(file_output, mode):
    df = dataset_global.data_pp[0]
    return (df,
            generate_graph(0, mode),
            generate_partial_order_graph(0),
            gr.update(value="Bucket #0",
                      choices=[f"Bucket #{key}" for key in dataset_global.data]))


def load_dataset_from_folders(data_folder, gt_folder, type_mapping, dataset_parameters, mode):
    """Load datasets from two folders containing a list of buckets"""
    global dataset_global
    if data_folder is None or gt_folder is None:
        return None, "📂 No dataset selected.", "", ""

    with open(type_mapping, encoding="utf-8") as f:
        attr_types = json.load(f)

    with open(dataset_parameters, encoding="utf-8") as f:
        parameters = json.load(f)

    try:
        dataset_global = StaticDataset([data_folder, gt_folder],
                            attribute_types=attr_types,
                            gradio=True,
                            **parameters)
        dataset_global.make_post_preprocess_copy()
        preprocess_file_path = os.path.join("data", "configurations",
                                                "truthfinder", "flights",
                                                "preprocess_configuration.json")

        with open(preprocess_file_path , encoding="utf-8") as preprocessing_config_file:
            preprocess_config = json.load(preprocessing_config_file)
            logging.info("DATA PREPROCESSING")
            dataset_global.apply_data_preprocessing(preprocess_config)
            logging.info("METADATA PREPROCESSING")
            dataset_global.apply_metadata_preprocessing(preprocess_config)

        df = dataset_global.data_pp[0]

        return (df,
                "✅ Dataset successfully loaded.",
                generate_graph(0, mode),
                generate_partial_order_graph(0),
                gr.update(value="Bucket #0",
                          choices=[f"Bucket #{key}" for key in dataset_global.data]))
    except Exception as e:
        print(e)
        return (None,
                f"❌ Error : {str(e)}",
                "<p style='color:red'>⚠️ No dataset selected, the graph cannot be generated.</p>",
                "?",
                "")


def generate_graph(bid, mode):
    """Generate the graph visualization for the bucket bid"""
    global colors_map
    if colors_map is None:
        colors_map = visualization.set_colors_map(dataset_global.attributes)
    net = visualization.visualize(dataset_global,
                            dataset_global.attributes,
                            colors_map,
                            graph=mode,
                            gradio=True,
                            bid = bid)

    graph_html = net.generate_html()
    graph_html = utils.fix_html(graph_html)
    with open(f"graph_{bid}.html", "w", encoding="utf-8") as file:
        file.write(graph_html)

    return (""" <iframe style="width: 100%; height: 600px; """
            """ margin:0 auto" frameborder="0" """
            f""" srcdoc='{graph_html}'></iframe> """)


def generate_partial_order_graph(bid):
    """Generate the partial order graph of the bucket bid"""
    net = visualization.visualize_partial_orders(dataset_global, bid)
    if net is not None:
        graph_html = net.generate_html()
        fixed_html = graph_html.replace("'", "\"")
        return (""" <iframe style="width: 100%; """
                """ height: 300px; margin:0 auto" """
                f"""frameborder="0" srcdoc='{fixed_html}'></iframe>""")
    return ("<p style='color:#FF7900'>"
            "⚠️ No partial orders for the current bucket.</p>")


def update_dataframe(index, mode, n_sources):
    """Update the bucket"""
    return (f"Bucket #{index}", dataset_global.data_pp[index],
            generate_graph(index, mode), generate_partial_order_graph(index),
            top_n(n_sources, index))


def update_dataframe_from_dropdown(index, mode, n_sources):
    """Load the bucket from the selected one and update the visualization"""
    index = int(re.search(r"Bucket #(\d+)", index).group(1))
    return (index, dataset_global.data_pp[index],
            generate_graph(index, mode),
            generate_partial_order_graph(index),
            top_n(n_sources, index))


def prev_dataframe(index, mode, n_sources):
    """Load the previous bucket and update the visualization"""
    index = (index - 1) % len(dataset_global.data_pp)
    return index, *update_dataframe(index, mode, n_sources)


def next_dataframe(index, mode, n_sources):
    """Load the next bucket and update the visualization"""
    index = (index + 1) % len(dataset_global.data_pp)
    return index, *update_dataframe(index, mode, n_sources)


def toggle_display(choice, bid):
    """Change the the visualization choice among Input/Output"""
    dataset_mapping = {
        "Input": dataset_global.data_pp[bid],
        "Output": dataset_global.fmt_fused_data[bid]
    }
    return dataset_mapping[choice], generate_graph(bid, choice)


def apply_model(df, model_name, bucket_state, progress=gr.Progress()):
    """Run the fusion model on the loaded dataset"""
    if df is None:
        return "⚠️ No dataset selected."
    global metrics_global
    logging.info("Apply %s", model_name)
    model = settings.MODEL_MAP[model_name](dataset_global,
                                           progress=progress,
                                           **settings.MODEL_PARAMETERS[model_name])
    for bid, inputs in progress.tqdm(model.model_input.items(),
                                     desc="Fusion"):
        results = model.fuse(dataset_global, bid, inputs, progress=progress)
        logging.info("Performing reverse mapping")
        dataset_global.reverse_mapping(results, bid, progress)
        logging.info("Metrics computation")
        _, metrics = evaluation.get_metrics(dataset_global,
                                            dataset_global.attributes,
                                            mode="positive",
                                            progress=progress)
        logging.info("Metrics computed")
        metrics = metrics.style.applymap(utils.color_gradient, subset=metrics.columns[1:])
        metrics_html = utils.display_metrics(metrics, model_name)
        metrics_global = metrics_html
        yield "BID", [f"Bucket #{key}" for key in dataset_global.fmt_fused_data]


def update_bucket_id(available_buckets):
    return gr.update(choices=[f"Bucket #{key}"
                               for key in dataset_global.fmt_fused_data])


def display_model_output(model_information, bucket_state):
    return ("Output",
            gr.update(visible=True),
            gr.update(visible=True, value=top_n(10, bucket_state)),
            metrics_global)


def update_visibility(choice):
    """Updates the component display according to the choice of the user."""
    if choice == "Single file (Pickle)":
        return (gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False))
    else:
        return (gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False))


def update_selector_visibility():
    """Change the visibility of File selector component"""
    return gr.update(visible=True)


def top_n(n, bucket_state):
    sorted_scores = sorted(dataset_global.weights_dict[bucket_state].items(),
                           key=lambda x: x[1],
                           reverse=True)

    max_n = min(10, len(sorted_scores))

    try:
        n = int(n) if n else max_n
        n = max(1, min(n, len(sorted_scores)))
    except ValueError:
        return "⚠️ Enter a valid number"

    return sorted_scores[:n]


with gr.Blocks(theme=THEME, fill_height=True) as trustfuse_demo:
    gr.Markdown("## 🌋 TrustFuse", elem_id="title")

    with gr.Row():
        with gr.Column(scale=1):
            # Dataset selectors
            with gr.Column(scale=1):
                with gr.Column():
                    choice_mode = gr.Radio(
                        ["Single file (Pickle)", "Two separate folders (Data + GT)"],
                        label="Select the dataset format to be loaded",
                        value="Single file (Pickle)"
                    )
                    available_datasets = gr.Dropdown(
                        choices=(list(STATIC_DATASETS_AVAILABLE)
                                 + list(DYNAMIC_DATASETS_AVAILABLE)),
                        label="or select an available dataset",
                        value="",
                        interactive=True
                    )
                dataset_selector = gr.File(label="Select the pickle file",
                                           interactive=True,
                                           file_count="single",
                                           file_types=[".pkl"],
                                           height=150)
                with gr.Row():
                    data_selector = gr.File(label="📂 Select conflicting data folder",
                                          file_count="directory",
                                          interactive=True, height=150, visible=False)
                    gt_selector = gr.File(label="📂 Select GT data folder",
                                        file_count="directory",
                                        interactive=True, height=150, visible=False)

                parameters_selector = gr.File(label="Select the dataset parameters",
                                        file_count="single",
                                        interactive=True, height=140, visible=False)

                type_selector = gr.File(label="Select attribute/datatatype mapping file",
                                        file_count="single",
                                        interactive=True, height=140, visible=False)

            bucket_state = gr.State(0)
            available_buckets = gr.State([0])
            file_output = gr.Textbox(label="📄 Dataset",
                                     interactive=False)
            bucket_id = gr.Dropdown(label="Current bucket",
                                    choices=["Bucket #0"],
                                    interactive=True, value="")
            with gr.Row():
                prev_button = gr.Button("⬅ Previous bucket")
                next_button = gr.Button("Next bucket ➡")

            model_selector = gr.Dropdown(["CRH", "CATD", "KDEm", "GTM",
                                 "TruthFinder", "TKGC", "SLIMFAST", 
                                 "LTM", "ACCU"], label="🌋 Choose a fusion model")
            model_information = gr.Textbox(label="Model state", interactive=False)
            run_button = gr.Button("🚀 Run", interactive=True)
            toggle = gr.Radio(["Input", "Output", "Ground Truth"],
                              label="Display", value="Input")
            with gr.Column():
                partial = gr.Markdown("#### 🎯 Partial orders")
                partial_order = gr.HTML()
            dataset_state = gr.State(None)

        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("Graph"):
                    graph_output = gr.HTML()
                with gr.Tab("Table"):
                    table_output = gr.Dataframe(label="📋 Data loaded")
            # with gr.Row(scale=1):
            #     gr.Checkbox(label="Save metrics", visible=True,)
            metric_output = gr.HTML()
            n_sources = gr.Textbox(value="",
                                   label="Enter the Top N of desired sources",
                                   visible=False)
            top_n_sources = gr.Dataframe(headers=["Source", "Score"], visible=False)

        n_sources.change(fn=top_n, inputs=[n_sources, bucket_state], outputs=[top_n_sources])
        prev_button.click(prev_dataframe,
                          inputs=[bucket_state,
                                  toggle,
                                  n_sources],
                          outputs=[bucket_state,
                                   bucket_id,
                                   table_output,
                                   graph_output,
                                   partial_order,
                                   top_n_sources])

        next_button.click(next_dataframe,
                          inputs=[bucket_state,
                                  toggle,
                                  n_sources],
                          outputs=[bucket_state,
                                   bucket_id,
                                   table_output,
                                   graph_output,
                                   partial_order,
                                   top_n_sources])

        toggle.change(toggle_display,
                      inputs=[toggle,
                              bucket_state],
                      outputs=[table_output,
                               graph_output])

        available_datasets.change(load_available_dataset,
                                  inputs=[available_datasets, toggle],
                                  outputs=[file_output])

        file_output.change(display_dataset,
                           inputs=[file_output, toggle],
                           outputs=[table_output,
                                   graph_output,
                                   partial_order,
                                   bucket_id])

        dataset_selector.change(load_dataset,
                                inputs=[dataset_selector,
                                        toggle],
                                outputs=[table_output,
                                         file_output,
                                         graph_output,
                                         partial_order,
                                         bucket_id])
        
        model_information.change(display_model_output,
                                 inputs=[model_information, bucket_state],
                                 outputs=[toggle,
                                  n_sources,
                                  top_n_sources,
                                  metric_output])
# Model events
        run_button.click(apply_model,
                         inputs=[table_output,
                                 model_selector,
                                 bucket_state],
                         outputs=[model_information,
                                  available_buckets])
        available_buckets.change(update_bucket_id,
                                 inputs=[available_buckets],
                                 outputs=[bucket_id])

        bucket_id.input(update_dataframe_from_dropdown,
                        inputs=[bucket_id,
                                toggle,
                                n_sources],
                        outputs=[bucket_state,
                                 table_output,
                                 graph_output,
                                 partial_order,
                                 top_n_sources])

        choice_mode.change(update_visibility,
                           choice_mode,
                           [dataset_selector,
                            data_selector,
                            gt_selector])
        
        data_selector.change(update_selector_visibility,
                             outputs=[gt_selector])
        gt_selector.change(update_selector_visibility,
                           outputs=[parameters_selector])
        gt_selector.change(update_selector_visibility,
                           outputs=[type_selector])

        type_selector.change(load_dataset_from_folders,
                             inputs=[data_selector,
                                     gt_selector,
                                     type_selector,
                                     parameters_selector,
                                     toggle],
                            outputs=[table_output,
                                    file_output,
                                    graph_output,
                                    partial_order,
                                    bucket_id])

trustfuse_demo.queue().launch()
