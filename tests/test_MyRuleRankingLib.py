import tensorflow as tf
import numpy as np
import copy
import tensorflow_ranking as tfr
import tensorflow_recommenders as tfrs
import pandas as pd
from sklearn.model_selection import train_test_split
import math
import os
import google.protobuf.text_format as pbtext
import tqdm

def test_data_loader(directory=None):
    if directory == None:
        directory = os.getcwd()
    else:
        directory = directory # pylint: disable=self-assigning-variable;
    path_to_labeled_dir = "Labeled_Data/ExcelParser_Outputs"
    directory_path = os.path.join(directory, path_to_labeled_dir)
    file_list = os.listdir(directory_path)
    df_dict = {}
    rule_dict = {}
    package = {}
    rule_names = []
    for i in range(len(file_list)):
        if "RuleText" in file_list[i]:
            df = pd.read_csv(os.path.join(directory_path, file_list[i]))
            ruletext = encode_checker(df["Text"].iloc[0])
            rulename = file_list[i].replace("RuleText.csv", "")
            rule_dict[rulename] = ruletext
        if ("RuleText" not in file_list[i]) and (file_list[i][-3:] == ".h5"):
            df = pd.read_hdf(os.path.join(directory_path, file_list[i]))
            rulename = file_list[i][:-3]
            df_dict[rulename] = df
            rule_names.append(rulename)
        elif ("RuleText" not in file_list[i]) and (file_list[i][-3:] == "csv"):
            df = pd.read_csv(os.path.join(directory_path, file_list[i]))
            rulename = file_list[i][:-4]
            df_dict[rulename] = df
            rule_names.append(rulename)

    package["Rules"] = rule_dict
    package["Dataframes"] = df_dict

    # package structure
    # package = { 'Rules': { 'RuleName': str },
    #             'Dataframes': { 'RuleName': pandas df }}

    assert isinstance(package, dict) and isinstance(rule_names, list)

# some words are represented as byte strings that cannot be encoded as utf-8
# this function removes those words
def encode_checker(txt):
    all_words = txt.split()
    can_be_encoded = []
    cannot_be_encoded = {}
    for i in range(len(all_words)):
        word = all_words[i]
        try:
            word.encode("utf-8")
            can_be_encoded.append(word)
        except UnicodeEncodeError:
            cannot_be_encoded[i] = word
    if len(can_be_encoded) == 0:
        return cannot_be_encoded
    elif len(can_be_encoded) > 0 and len(cannot_be_encoded) > 0:
        new_text = " ".join(can_be_encoded)
        return new_text, cannot_be_encoded
    else:
        new_text = " ".join(can_be_encoded)
        return new_text

# Helper function used to print datasets
def print_dictionary_dataset(dataset):
    for i, element in enumerate(dataset):
        print("Element {}:".format(i))
        for feature_name, feature_value in element.items():
            print("{:>14} = {}".format(feature_name, feature_value))

def mask_rows(nrules, rule_rl, com_rl, max_com_rl):
    # rule_emb_repeat is a padded tensor so create a boolean mask to convert it to ragged
    # find amount of rules
    # boolean mask input

    msk_rows = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
    # use row lengths to create input for boolean mask
    for i in range(nrules):
        #         row = tf.repeat(rule_rl[i], repeats=com_rl.__getitem__(i), axis=0)
        #         if com_rl.__getitem__(i) < max_com_rl:
        row = tf.repeat(rule_rl[i], repeats=com_rl[i], axis=0)
        if com_rl[i] < max_com_rl:
            # num of True values
            len_row = tf.cast(tf.shape(row)[0], tf.dtypes.int32)
            # num of False values
            pads = tf.repeat([0], repeats=(max_com_rl - len_row), axis=0)
            # concat together to ensure len(row) is 'max_com_rl' long
            new_row = tf.concat([row, pads], 0)
            msk_rows = msk_rows.write(i, new_row)
        else:
            msk_rows = msk_rows.write(i, row)

    msk_rows = msk_rows.stack()
    msk_3d = tf.sequence_mask(tf.cast(msk_rows, tf.dtypes.int32))

    return msk_3d

def remove_empty_lists(rt):
    nrl = rt.nested_row_lengths()
    empties = tf.squeeze(tf.where(nrl[1] == 0), axis=1)
    diff = tf.expand_dims(rt.nested_row_splits[0][1:], axis=0) - tf.expand_dims(
        empties, axis=1
    )
    diff_absolute = tf.where(diff <= 0, diff.dtype.limits[1], diff)
    diff_min = tf.argmin(diff_absolute, axis=1)
    counts = tf.unique_with_counts(diff_min)
    to_subtract = tf.scatter_nd(
        tf.expand_dims(counts.y, 1),
        counts.count,
        tf.cast(tf.shape(nrl[0]), tf.dtypes.int64),
    )
    non_empties = tf.squeeze(tf.where(nrl[1] != 0), axis=1)
    nrl_updated = tf.gather(nrl[1], non_empties, axis=None)
    result = tf.RaggedTensor.from_nested_row_lengths(
        rt.flat_values, (nrl[0] - tf.cast(to_subtract, tf.int64), nrl_updated)
    )
    return result

def inspect_ragged_results(tfds, y_pred, rule_names):
    num_rules = len(rule_names)
    for x in tfds.take(num_rules):
        inputs = dict(
            labeledscore=x["score"],
            rule_embs=x["rule_embs"],
            comment_embs=x["comment_embs"],
            rule=x["rule"],
            comment=x["comments"],
            name=x["names"],
            hyperlink=x["hyperlinks"],
        )

    rule_list = inputs["rule"].numpy()
    name_lists = inputs["name"].numpy()
    hyperlink_lists = inputs["hyperlink"].numpy()
    # ragged tensor y_pred to list
    score_lists = y_pred.to_list()
    dataframes = {}
    for i in range(len(rule_list)):
        data = {}
        # flatten nested score_list
        data["PredScore"] = [item for sublist in score_lists[i] for item in sublist]
        data["LabeledScore"] = inputs["labeledscore"][i].numpy().tolist()
        data["Name"] = [name.decode("utf-8") for name in name_lists[i]]
        data["Hyperlink"] = [link.decode("utf-8") for link in hyperlink_lists[i]]
        df = pd.DataFrame.from_dict(data)
        sorted_df = df.sort_values(by="PredScore", ascending=False)
        # remove rows that were artificially created to make a rectangular dataset
        filtered_df = sorted_df[sorted_df["Name"] != "NA"] # pylint: disable=unsubscriptable-object
        dataframes[rule_names[i]] = filtered_df

    # dict structure
    
    # {"rule_name": dataframe}
    return dataframes

# for non-ragged results
def inspect_results_v1(tfds, y_pred, rule_names):
    num_rules = len(rule_names)
    for x in tfds.take(num_rules):
        y = x["score"]
        inputs = dict(
            rule_embs=x["rule_embs"],
            comment_embs=x["comment_embs"],
            rule=x["rule"],
            comment=x["comment"],
            name=x["name"],
            hyperlink=x["hyperlink"],
        )

    rule_list = inputs["rule"].numpy()
    name_lists = inputs["name"].numpy()
    hyperlink_lists = inputs["hyperlink"].numpy()
    score_lists = [score_list for score_list in y_pred]
    dataframes = {}
    for i in range(len(rule_list)):
        data = {}
        data["PredScore"] = score_lists[i].ravel()
        data["LabeledScore"] = np.round(y.numpy()[i], 4)
        data["Name"] = [name.decode("utf-8") for name in name_lists[i]]
        data["Hyperlink"] = [link.decode("utf-8") for link in hyperlink_lists[i]]
        df = pd.DataFrame.from_dict(data)
        sorted_df = df.sort_values(by="PredScore", ascending=False)
        # remove rows that were artificially created to make a rectangular dataset
        filtered_df = sorted_df[sorted_df["LabeledScore"] > 0] # pylint: disable=unsubscriptable-object
        dataframes[rule_names[i]] = filtered_df

    # dict structure
    # {"rule_name": dataframe}
    return dataframes

# function to take package from data_loader and create a train/val/test dataset
def train_val_test_package(package, train_size, val_size):
    val_size = val_size / (1 - train_size)
    train_dfs = {}
    val_dfs = {}
    test_dfs = {}

    train_package = {}
    val_package = {}
    test_package = {}

    X_cols = ["Name", "Hyperlink", "Comment"]
    for df_name in package["Dataframes"].keys():
        df = package["Dataframes"][df_name]
        X_train, X_, y_train, y_ = train_test_split(
            df[X_cols],
            df["Score"],
            train_size=train_size,
            random_state=42,
            stratify=df["Score"],
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_, y_, train_size=val_size, random_state=42, stratify=y_
        )

        train_dfs[df_name] = pd.concat([X_train, y_train], axis=1)
        val_dfs[df_name] = pd.concat([X_val, y_val], axis=1)
        test_dfs[df_name] = pd.concat([X_test, y_test], axis=1)

    train_package["Rules"] = package["Rules"]
    val_package["Rules"] = package["Rules"]
    test_package["Rules"] = package["Rules"]

    train_package["Dataframes"] = train_dfs
    val_package["Dataframes"] = val_dfs
    test_package["Dataframes"] = test_dfs

    return train_package, val_package, test_package

def text_chunker(package):
    # max number of tokens bert can use
    chunk_size = 512
    rules = [rule_text for rule_text in package["Rules"].values()]
    dfs = list(package["Dataframes"].values())
    split_texts = {"rule": [], "comment": [], "score": [], "name": [], "hyperlink": []}
    rule_most_chunks = 0
    comments_most_chunks = 0
    rule_most_comments = 0
    for i in range(len(rules)):
        rule = rules[i]
        # splits the text document by whitespace
        all_words = rule.split()
        chunks = math.ceil(len(all_words) / chunk_size)
        if chunks > rule_most_chunks:
            rule_most_chunks = chunks
        chunk_list = []
        # for every 512 word chunk of text
        for j in range(chunks):
            if j < chunks - 1:
                # obtain the j-th chunk of 512 words
                chunk_j = all_words[j * chunk_size : (j + 1) * chunk_size]
            else:
                chunk_j = all_words[j * chunk_size :]
            # join those 512 words into one chunk
            chunk_txt = encode_checker(" ".join(chunk_j))
            # append to chunk list
            chunk_list.append(chunk_txt)
        # chunk_list append to rule list
        split_texts["rule"].append(chunk_list)
        rule_df = dfs[i]
        rule_comments = []
        rule_scores = []
        rule_names = []
        rule_links = []
        for j in range(len(rule_df["Comment"])):
            comment = rule_df["Comment"].iloc[j]
            score = rule_df["Score"].iloc[j]
            name = rule_df["Name"].iloc[j]
            hyperlink = rule_df["Hyperlink"].iloc[j]
            all_words = comment.split()
            chunks = math.ceil(len(all_words) / chunk_size)
            if chunks > comments_most_chunks:
                comments_most_chunks = chunks
            chunk_list = []
            for k in range(chunks):
                if k < chunks - 1:
                    chunk_k = all_words[k * chunk_size : (k + 1) * chunk_size]
                else:
                    chunk_k = all_words[k * chunk_size :]
                chunk_txt = encode_checker(" ".join(chunk_k))
                chunk_list.append(chunk_txt)

            rule_comments.append(chunk_list)
            rule_scores.append(score)
            rule_names.append(name)
            rule_links.append(hyperlink)
        # finding which rule has the most comments
        if len(rule_comments) > rule_most_comments:
            rule_most_comments = len(rule_comments)

        split_texts["comment"].append(rule_comments)
        split_texts["score"].append(rule_scores)
        split_texts["name"].append(rule_names)
        split_texts["hyperlink"].append(rule_links)

    return split_texts, rule_most_chunks, comments_most_chunks, rule_most_comments

def text_ragged_embeddings(
    split_texts, model
):
    # cwd = os.getcwd()
    cwd = "/home/herrerast@AD.SEC.GOV/Desktop/Persistent_Data/personal/RaggedTensorBert"
    models_dir = os.path.join(cwd, "models")
    smallbert_filepath = os.path.join(models_dir, "small_bert")
    bert_filepath = os.path.join(models_dir, "bert_uncased_L12-H768-A12_v4")
    roberta_filepath = os.path.join(models_dir, "roberta_en_cased_L-24_H-1024_A-16_1")
    preprocessor_filepath = os.path.join(models_dir, "bert_preprocessor")

    smallbert = tf.saved_model.load(smallbert_filepath)
    bert = tf.saved_model.load(bert_filepath)
    roberta = tf.saved_model.load(roberta_filepath)

    bert_preprocessor = tf.saved_model.load(preprocessor_filepath)

    new_dict = copy.deepcopy(split_texts)
    new_dict["rule_embs"] = []
    new_dict["comment_embs"] = []

    # the output embeddings are different for various versions of bert
    emb_mods = {"small_bert": smallbert, "bert": bert, "roberta": roberta}

    bert_model = emb_mods[model]
    # embed every comment
    for i in range(len(new_dict["rule"])):
        rule_tensor = tf.ragged.constant(new_dict["rule"][i], dtype=tf.string)
        rule_inp = bert_preprocessor(rule_tensor)
        rule_outputs = bert_model(rule_inp)["pooled_output"]
        rule_reshape = tf.reshape(rule_outputs, (1, -1)).numpy().tolist()
        # new_dict['rule_embs'].append(rule_reshape[0])
        new_dict["rule_embs"].append(rule_reshape[0])

        tensor_comment_list = []
        for j in tqdm.tqdm(range(len(new_dict["comment"][i]))):
            comment_tensor = tf.ragged.constant(
                new_dict["comment"][i][j], dtype=tf.string
            )
            comment_inp = bert_preprocessor(comment_tensor)
            # output's embedding are of shape(num_chunks, embedding_dims)
            comment_outputs = bert_model(comment_inp)["pooled_output"]
            comm_reshape = tf.reshape(comment_outputs, (1, -1)).numpy().tolist()
            tensor_comment_list.append(comm_reshape[0])
            # rejoin all the split strings back into one string for easier inspection later
            new_comment_tensor = tf.strings.join(comment_tensor)
            # new_dict['comment'][i][j] = [new_comment_tensor.numpy()]
            new_dict["comment"][i][j] = new_comment_tensor.numpy().decode("utf-8")

        # rejoin split strings
        new_rule_tensor = tf.strings.join(rule_tensor, separator=" ")
        # new_dict['rule'][i] = new_rule_tensor.numpy()
        new_dict["rule"][i] = new_rule_tensor.numpy().decode("utf-8")

        # shape of ragged tensor: (1 , num_comments (ragged), embedding_size (ragged))
        # does the fact that .shape results in [1,None,None] mean you can have differing number of comments?
        new_dict["comment_embs"].append(tensor_comment_list)

    return new_dict

# converts split texts dict to appropriate tfds dimensions
def tfds_creator(split_texts):
    num_rules = len(split_texts["rule"])
    # keys: rule ,comment, score, hyperlink, name, comment_embs, rule_embs
    for key in split_texts.keys():
        try:
            if (key != "comment_embs") and (key != "score") and (key != "comment"):
                split_texts[key] = tf.ragged.constant(split_texts[key])
            elif key == "score":
                split_texts[key] = tf.ragged.constant(
                    split_texts[key], dtype=tf.float32
                )
            else:
                pass
                # split_texts[key] = tf.squeeze(tf.stack(split_texts[key], axis=0))
        except ValueError:
            print(key, "Scalar tensor has no len()")
        except TypeError:
            print(key, "pylist may not be a RaggedTensor or RaggedTensorValue.")

    # split_texts['rule_embs'] = tf.squeeze(split_texts['rule_embs'])

    tfds = tf.data.Dataset.from_tensor_slices(split_texts)
    tfds = tfds.batch(num_rules)
    return tfds

# a function to flatten the emb lists in comment_embs list
def flatten_com_embs(embs):
    new_dict = copy.deepcopy(embs)
    lengths_list = []
    flattened_embs = []
    for i in range(len(new_dict["comment_embs"])):
        emb_lengths = []
        flat_embs = [
            item for sublist in new_dict["comment_embs"][i] for item in sublist
        ]
        flattened_embs.append(flat_embs)
        for j in range(len(new_dict["comment_embs"][i])):
            emb = new_dict["comment_embs"][i][j]
            emb_length = len(emb)
            emb_lengths.append(emb_length)
        lengths_list.append(emb_lengths)

    new_dict["flattened_embs"] = flattened_embs
    new_dict["emb_lengths"] = lengths_list

    return new_dict

# a function to create a ragged dataset
# https://www.tensorflow.org/guide/ragged_tensor
# https://www.tensorflow.org/api_docs/python/tf/io/RaggedFeature
def ragged_tfds_creator(flat_embs):
    ragged_batch = []
    new_dict = copy.deepcopy(flat_embs)
    batch_num = len(new_dict["rule"])
    # convert each list in the dict to a string
    for i in range(len(new_dict["rule"])):
        # necessary to put string in list
        rule_str = str([new_dict["rule"][i]])
        com_str = str(new_dict["comment"][i])
        score_str = str(new_dict["score"][i])
        name_str = str(new_dict["name"][i])
        link_str = str(new_dict["hyperlink"][i])
        rule_emb_str = str(new_dict["rule_embs"][i])
        com_embs = str(new_dict["flattened_embs"][i])
        com_emb_len_str = str(new_dict["emb_lengths"][i])
        # necessary to put int in list
        rule_length_str = str([len(new_dict["rule_embs"][i])])

        ragged_batch.append(
            pbtext.Merge(
                r"""
        features {
            feature {key: "rule" value {bytes_list {value: """
                + rule_str
                + """} } }
            feature {key: "comments" value {bytes_list {value: """
                + com_str
                + """} } }
            feature {key: "scores" value {float_list {value: """
                + score_str
                + """} } }
            feature {key: "names" value {bytes_list {value: """
                + name_str
                + """} } } 
            feature {key: "hyperlinks" value {bytes_list {value: """
                + link_str
                + """} } }
            feature {key: "rule_embs" value {float_list {value: """
                + rule_emb_str
                + """} } } 
            feature {key: "comment_embs" value {float_list {value: """
                + com_embs
                + """} } } 
            feature {key: "com_emb_lengths" value {int64_list {value: """
                + com_emb_len_str
                + """} } }
            feature {key: "rule_lengths" value {int64_list {value: """
                + rule_length_str
                + """} } } 
            }""",
                tf.train.Example(),
            ).SerializeToString()
        )

    features = {
        # Zero partitions: returns 1D tf.Tensor for each Example.
        "rule": tf.io.RaggedFeature(value_key="rule", dtype=tf.string),
        "comments": tf.io.RaggedFeature(value_key="comments", dtype=tf.string),
        "score": tf.io.RaggedFeature(value_key="scores", dtype=tf.float32),
        "hyperlinks": tf.io.RaggedFeature(value_key="hyperlinks", dtype=tf.string),
        "names": tf.io.RaggedFeature(value_key="names", dtype=tf.string),
        # One partition: returns 2D tf.RaggedTensor for each Example.
        "rule_embs": tf.io.RaggedFeature(value_key="rule_embs", dtype=tf.float32),
        # Two partitions: returns 3D tf.RaggedTensor for each Example.
        "comment_embs": tf.io.RaggedFeature(
            value_key="comment_embs",
            dtype=tf.float32,
            partitions=[tf.io.RaggedFeature.RowLengths("com_emb_lengths")],
        ),
        "com_emb_lengths": tf.io.RaggedFeature(
            value_key="com_emb_lengths", dtype=tf.int64
        ),
        "rule_lengths": tf.io.RaggedFeature(value_key="rule_lengths", dtype=tf.int64),
    }

    feature_dict = tf.io.parse_example(ragged_batch, features) # pylint: disable=no-value-for-parameter;
    ragged_dataset = tf.data.Dataset.from_tensor_slices(feature_dict)
    return ragged_dataset.batch(batch_num), feature_dict

class RankingModel(tfrs.Model):
    def __init__(self, loss):
        super().__init__()
        # tf.keras.layers.attention

        # Compute predictions.
        self.score_model = tf.keras.Sequential(
            [
                # Learn multiple dense layers.
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(8, activation="relu"),
                tf.keras.layers.Dropout(0.5),
                # Make rating predictions in the final layer.
                tf.keras.layers.Dense(1),
            ]
        )

        self.task = tfrs.tasks.Ranking(
            loss=loss,
            metrics=[
                # Normalized Discounted Cumulative Gain
                tfr.keras.metrics.NDCGMetric(name="ndcg_metric", ragged=True)
            ],
        )

    def call(self, features=None): # pylint: disable=arguments-differ;
        # batch_size = num rules to feed at one time
        # (batch_size, ragged)
        rule_embeddings = features["rule_embs"]

        # (batch_size, list_size, ragged)
        comment_embeddings = features["comment_embs"]

        # find the row lengths of the embeddings
        com_rl = comment_embeddings.row_lengths()
        rule_rl = rule_embeddings.row_lengths()

        # find largest row lengths
        max_com_rl = tf.reduce_max(com_rl)

        # find num of rules
        #         nrules = tf.get_static_value(tf.shape(features['comment_embs']).num_row_partitions)
        nrules = tf.shape(features["rule"])[0]

        # repeat the augmented data `max_com_rl` number of time on a padded tensor
        rule_embeddings_repeated = tf.repeat(
            tf.expand_dims(rule_embeddings.to_tensor(), 1), repeats=max_com_rl, axis=1
        )

        # boolean mask for creating a ragged tensor
        msk_3d = mask_rows(nrules, rule_rl, com_rl, max_com_rl)

        # shape is (2, max_com_rl, ragged)
        # this is bc there are empty elements e.g. [[0.9, 0.1], [], [], []]
        ragged_rule_embs_repeated = tf.ragged.boolean_mask(
            rule_embeddings_repeated, msk_3d
        )

        # empty elements removed so shape is (2, (ragged_1, ... , ragged_N), (ragged))
        ragged_rule_embs_repeated_ = remove_empty_lists(ragged_rule_embs_repeated)

        # concatenate
        concatenated_embeddings = tf.concat(
            [
                ragged_rule_embs_repeated_.with_row_splits_dtype("int32"),
                comment_embeddings.with_row_splits_dtype("int32"),
            ],
            2,
        )
        # determine the max emb size
        max_concat = 50_000
        # set max emb size for each rule
        concat_rl = tf.repeat(max_concat, repeats=nrules, axis=0)
        # create boolean mask
        mask = mask_rows(nrules, concat_rl, com_rl, max_com_rl)
        # pad concat embs to max emb size
        concat_emb_tensor = concatenated_embeddings.to_tensor(
            default_value=0.0, shape=[nrules, max_com_rl, max_concat]
        )

        rag_concat_rep = tf.ragged.boolean_mask(concat_emb_tensor, mask)
        rag_concat_rep_ = remove_empty_lists(rag_concat_rep)
        # merge dimensions
        merge_concat = rag_concat_rep_.merge_dims(0, 1)
        # find row lengths for creating ragged tensor
        row_splits = tf.concat([tf.constant([0]), tf.cumsum(com_rl, axis=0)], 0)
        # ragged tensor is shape (nrules, (ragged_1, ... , ragged_nrules), custom_emb_size)
        ragged_tensor = tf.RaggedTensor.from_row_splits(
            merge_concat.to_tensor(), row_splits
        )

        return self.score_model(ragged_tensor)

    def compute_loss(self, features): # pylint: disable=arguments-differ;
        labels = features.pop("score")

        scores = self(features)

        return self.task(
            labels=labels,
            predictions=tf.squeeze(scores, axis=-1),
        )