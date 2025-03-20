import pandas as pd
import os
import tqdm
from typing import Dict, Tuple, Mapping
import pickle
import copy
import tabulate
import networkx
from trustfuse.conflicting_dataset.preprocessing import (DATA_PREPROCESSING_FUNCTIONS,
                                                         METADATA_PREPROCESSING_FUNCTIONS,
                                                         data_preprocessing)


def rename_data_header(df_dict, header):
	for k in df_dict:
		df_dict[k].rename(
			columns={i: header[i] for i in range(len(header))},
			inplace=True
			)
        

class Dataset:
    def __init__(self, entity_col_name: str, attribute_types):

        # Column of the dataset that represents the entities
        self.entity_col_name = entity_col_name
        # Ground Truth Data
        self.gt_data = {}
        # Conflicting Data
        self.data = {}
        # List of attributes involved in each bucket
        self.attributes = {}
        # Dict mapping between attribute/datatype
        self.attribute_types = attribute_types
        # Make copies for reverse mapping after preprocessing
        self.seed_data = {}
        self.seed_gt_data = {}
        # Init formatted result variables
        self.fmt_fused_data = {}

        self.weights_dict = {}
        self.fused_data = {}
        # postpreprocess
        self.data_pp = {}

    def __str__(self):
        """Print the attributes of the instance"""
        return ("\n".join(f"{cle}: {valeur}"
                          for cle, valeur in self.__dict__.items()))


    def set_attributes(self, attributes):
        """Attributes setter"""
        self.attributes = attributes


    def apply_preprocessors(self, preprocessors, modify_structure=False, **kwargs):
        """Apply preprocessors"""
        for preprocessor in preprocessors:
            if preprocessor in DATA_PREPROCESSING_FUNCTIONS:
                data_preprocessing(
                    self,
                    DATA_PREPROCESSING_FUNCTIONS[preprocessor],
                    modify_structure=modify_structure,
                    **preprocessors[preprocessor],
                    **kwargs
                    )


    def apply_data_preprocessing(self, preprocessors, **kwargs):
        """Apply preprocessing on data

        Args:
            preprocessors (Dict): Dict with preprocessors and its parameter
        """
        if "modify_structure" in preprocessors:
            self.apply_preprocessors(preprocessors["modify_structure"],
                                     modify_structure=True,
                                     **kwargs)
        # Copy data before the preprocess step in order to save data
        # before indexing it for Reverse Mapping to display result in
        # the original format. This copy will support metrics computation
        self.make_post_preprocess_copy()
        if "modify_data" in preprocessors:
            self.apply_preprocessors(preprocessors["modify_data"],
                                     **kwargs)


    def apply_metadata_preprocessors(self, preprocessors, modify_structure=False, **kwargs):
        """Apply metadata preprocessors"""
        for preprocessor in preprocessors:
            if preprocessor in METADATA_PREPROCESSING_FUNCTIONS:
                METADATA_PREPROCESSING_FUNCTIONS[preprocessor](self,
                                                               **preprocessors[preprocessor],
                                                               **kwargs)


    def apply_metadata_preprocessing(self, preprocessors, **kwargs):
        """Apply preprocessing on data

        Args:
            preprocessors (Dict): Dict with preprocessors and its parameter
        """
        if "modify_structure" in preprocessors:
            self.apply_metadata_preprocessors(preprocessors["modify_structure"],
                                              modify_structure=True,
                                              **kwargs)
        if "modify_data" in preprocessors:
            self.apply_metadata_preprocessors(preprocessors["modify_data"],
                                              **kwargs)


    def serialize(self, path):
        """Serialize the dataset object and its state to save experiments"""
        with open(path, 'wb') as file:
            pickle.dump(self, file)


    def make_post_preprocess_copy(self):
        """Make a copy of conflicting data after the preprocessing
        step used to compute metrics and allow the reverse mapping after 
        the fusion step
        """
        for bid, _ in self.data.items():
            self.data_pp[bid] = self.data[bid].copy(deep=True)
            # Make copies of the data + GT data before preprocessing
            # for reverse mapping aftre the fusion stage
            self.seed_data[bid] = self.data[bid].copy(deep=True)
            self.seed_gt_data[bid] = self.gt_data[bid].copy(deep=True)


    def reverse_mapping(self, unified_result: Mapping[str, Tuple], bid, progress=tqdm) \
        -> Tuple[Mapping[str, pd.DataFrame], Mapping[str, Dict]]:
        """Apply a reverse mapping to display fusion result
        as the same format of the input data"""
        # fusion_results: Tuple[np.array, np.array]
        # Iterate over the buckets
        print(f"Bucket ID = {bid}")
        # Creation of an "index" column as primary key
        # for mapping between seed and transformed data
        # Create an index column with the number
        # of each line in the Dataframe
        self.data[bid]["index"] = self.data[bid].index
        # Create the indexes for preprocessed
        # data on all attributes & entity column
        self.data[bid].set_index(
            self.attributes[bid] + [self.entity_col_name],
            inplace=True
            )

        self.seed_data[bid]["index"] = self.seed_data[bid].index
        self.seed_data[bid].set_index(
            [self.entity_col_name] + ["index"],
            inplace=True
            )

        # We retrieve the confidence scores of the sources
        fusion_result = unified_result[bid]["truth"]
        self.weights_dict[bid] = unified_result[bid]["weights"]
        # Create a dict to construct a Dataframe of the results
        self.fused_data[bid] = {a: [] for a in self.attributes[bid]}
        self.fused_data[bid][self.entity_col_name] = []
        # Make a copy of the dictionary
        self.fmt_fused_data[bid] = copy.deepcopy(self.fused_data[bid])

        for ent in progress.tqdm(fusion_result, desc="Reverse Mapping"):
            # Add the entity in the Entity column of the dict
            self.fused_data[bid][self.entity_col_name].append(ent)
            self.fmt_fused_data[bid][self.entity_col_name].append(ent)
            # Iterate over each attribute a of the entity e
            for a in fusion_result[ent]:
                # formatted list to fill in
                fmt_list = []
                # Iterate over the values of each property of the entity
                for v in fusion_result[ent][a]:
                    if v is not None:
                        # Firstly find the index where attribute == value from transformed dataset
                        subset = self.data[bid].xs(ent, level=self.entity_col_name)
                        corresponding_index = subset[
                            (subset.index.get_level_values(a) == v)
                        ]["index"].iloc[0]
                        # Then find the seed value with the corresponding index
                        seed_subset = self.seed_data[bid].xs(
                            ent,
                            level=self.entity_col_name
                        )
                        seed_value = seed_subset[
                            (seed_subset.index.get_level_values("index") == corresponding_index)
                        ][a].iloc[0]
                        fmt_list.append(seed_value)
                    else:
                        fmt_list.append(None)
                self.fused_data[bid][a].append(fusion_result[ent][a])
                self.fmt_fused_data[bid][a].append(fmt_list)
        self.fmt_fused_data[bid] = pd.DataFrame(data=self.fmt_fused_data[bid])
        self.fused_data[bid] = pd.DataFrame(data=self.fused_data[bid])

        return self.fused_data, self.weights_dict


    def print_table(self, metrics):
        """_summary_

        Args:
            metrics (_type_): _description_
        """
        for bid, result in self.fmt_fused_data.items():
            print(f"Bucket level: precision={round(metrics['buckets'][bid]['b_p'], 2)}  "
                f"recall={round(metrics['buckets'][bid]['b_r'], 2)}  "
                f"accuracy={round(metrics['buckets'][bid]['b_acc'], 2)}  "
                f"f1-score={round(metrics['buckets'][bid]['b_f1_score'], 2)} "
                f"completion rate={round(metrics['buckets'][bid]['c_rate'], 2)}")
            print(tabulate.tabulate(result.iloc[:100], headers='keys', tablefmt='grid'))


class StaticDataset(Dataset):
    """To handle literature datasets"""
    def __init__(self,
                 data_folder: str,
                 headers=None,
                 sep="\t",
                 gradio=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.headers = headers
        self.sep = sep
        if gradio:
            bucket_id = 0
            for file_name in tqdm.tqdm(data_folder[0]):
                self.data[bucket_id] = pd.read_csv(file_name,
                                                header=None,
                                                sep=sep,
                                                encoding='ISO-8859-1')
                self.data[bucket_id] = self.data[bucket_id] \
                    .drop(columns=[18], errors='ignore')
                bucket_id += 1
            bucket_id = 0
            for file_name in tqdm.tqdm(data_folder[1]):
                self.gt_data[bucket_id] = pd.read_csv(file_name,
                                                    header=None,
                                                    sep=sep,
                                                    encoding='ISO-8859-1')
                self.gt_data[bucket_id] = self.gt_data[bucket_id] \
                    .drop(columns=[17], errors='ignore')
                bucket_id += 1
        else:
            gt_data_path = os.path.join(data_folder, 'ground_truth')
            conflicting_data_path = os.path.join(data_folder, 'conflicting_data')
            bucket_id = 0
            for file_name in tqdm.tqdm(os.listdir(conflicting_data_path)):
                bucket_file = os.path.join(conflicting_data_path, file_name)
                self.data[bucket_id] = pd.read_csv(bucket_file,
                                                header=None,
                                                sep=sep,
                                                encoding='ISO-8859-1')
                self.data[bucket_id] = self.data[bucket_id] \
                    .drop(columns=[18], errors='ignore')
                bucket_id += 1
            bucket_id = 0
            for file_name in tqdm.tqdm(os.listdir(gt_data_path)):
                bucket_file = os.path.join(gt_data_path, file_name)
                self.gt_data[bucket_id] = pd.read_csv(bucket_file,
                                                    header=None,
                                                    sep=sep,
                                                    encoding='ISO-8859-1')
                self.gt_data[bucket_id] = self.gt_data[bucket_id] \
                    .drop(columns=[17], errors='ignore')
                bucket_id += 1

        # Define headers if they are not specified in the dataset
        if headers is not None:
            rename_data_header(self.data, headers[0])
            rename_data_header(self.gt_data, headers[1])

        # Define the attributes for each bucket
        for bid, _ in self.data.items():
            self.attributes[bid] = []
            self.attributes[bid].extend(list(self.data[bid].columns))
            self.attributes[bid].remove(self.entity_col_name)
            self.attributes[bid].remove('Source')


class DynamicDataset(Dataset):
    """Class to handle dynamic dataset (WikiConflict)"""
    def __init__(self, buckets_file, entity_as, **kwargs):
        super().__init__(**kwargs)

        with open(buckets_file, "rb") as f:
            buckets_by_qid = pickle.load(f)

        self.partial_orders = {}

        for bid in buckets_by_qid:
            # Load the right buckets
            if entity_as == "string":
                self.gt_data[bid] = buckets_by_qid[bid]["GT"]["value"]
                self.data[bid] = buckets_by_qid[bid]["data"]["value"]
            else:
                self.gt_data[bid] = buckets_by_qid[bid]["GT"]["qid"]
                self.data[bid] = buckets_by_qid[bid]["data"]["qid"]

            # Each bucket must include the previous bucket
            if bid - 1 in self.data:
                # Conflicting data
                self.data[bid] = pd.concat([self.data[bid-1], self.data[bid]],
                                           ignore_index=True, sort=False)
                # Ground Truth data
                previous_gt_props = self.gt_data[bid-1] \
                    .columns.difference(self.gt_data[bid].columns)
                self.gt_data[bid] = pd.concat([self.gt_data[bid],
                                               self.gt_data[bid-1][previous_gt_props]],
                                               axis=1)
                for attr, _ in buckets_by_qid[bid-1]["GT"]["value_order"].items():
                    if attr not in buckets_by_qid[bid]["GT"]["value_order"]:
                        buckets_by_qid[bid]["GT"]["value_order"][attr] = \
                            buckets_by_qid[bid-1]["GT"]["value_order"][attr]

            # Define the attributes for each bucket
            self.attributes[bid] = list(self.data[bid].columns)
            self.attributes[bid].remove(self.entity_col_name)
            self.attributes[bid].remove('Source')

            self.partial_orders[bid] = self.create_partial_order_graphs(
                buckets_by_qid[bid]["GT"]["value_order"])


    def create_partial_order_graphs(self, partial_orders):
        """Generate the taxonomy in a directed graph 
        structure of partial orders.

        Args:
            partial_orders (dict): partials orders as nested lists
        """
        graphs = {}
        for attr, content in partial_orders.items():
            # Check if there is at least one partial order
            if len(content) > 0:
                graph = networkx.DiGraph()
                edges = []
                for partial_order in partial_orders[attr]:
                    max_depth = len(partial_order) - 1
                    partial_order_reversed = partial_order.copy()
                    partial_order_reversed.reverse()
                    roots = []
                    for root in partial_order_reversed[0]:
                        roots.append(root)
                        graph.add_node(root, depth=0, max_depth=max_depth,
                                       coeff=0/max_depth, leaf=False)
                    new_roots = []
                    leaf = False
                    for depth, more_specific_values in enumerate(partial_order_reversed[1:]):
                        if depth == len(partial_order_reversed[1:]) - 1:
                            leaf = True
                        for value in more_specific_values:
                            graph.add_node(value, depth=depth+1, max_depth=max_depth,
                                           coeff=(depth+1)/max_depth, leaf=leaf)
                            edges.extend([(value, root) for root in roots])
                            new_roots.append(value)
                        roots = new_roots.copy()

                graph.add_edges_from(edges, label="more_specific")
                graphs[attr] = graph
        return graphs
