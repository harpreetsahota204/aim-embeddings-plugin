## Apple AIMv2 Embeddings Plugins

### Plugin Overview

This plugin allows you to compute embeddings using Apple AIMv2 one your FiftyOne datasets.

**Note*:* This plugin only computing supports image embeddings. The only AIMv2 model to support CLIP like text-image similarity is [`aimv2-large-patch14-224-lit`](https://huggingface.co/apple/aimv2-large-patch14-224-lit). Zero-shot prediction with this model is implemented as part of the [Zero-Shot Prediction plugin](https://github.com/jacobmarks/zero-shot-prediction-plugin).

#### Supported Models

This plugin supports all currently released versions of the [AIMv2 collection](https://huggingface.co/collections/apple/aimv2-6720fe1558d94c7805f7688c):

- `apple/aimv2-large-patch14-224`
- `apple/aimv2-huge-patch14-224`
- `apple/aimv2-1B-patch14-224`
- `apple/aimv2-3B-patch14-224`
- `apple/aimv2-large-patch14-336`
- `apple/aimv2-huge-patch14-336`
- `apple/aimv2-1B-patch14-336`
- `apple/aimv2-3B-patch14-336`
- `apple/aimv2-large-patch14-448`
- `apple/aimv2-huge-patch14-448`
- `apple/aimv2-1B-patch14-448`
- `apple/aimv2-3B-patch14-448`
- `apple/aimv2-large-patch14-224-distilled`
- `apple/aimv2-large-patch14-336-distilled`
- `apple/aimv2-large-patch14-native`


## Installation

If you haven't already, install FiftyOne:

```shell
pip install -U fiftyone transformers
```

Then, install the plugin:

```shell
fiftyone plugins download https://github.com/harpreetsahota204/aim-embeddings-plugin
```

### Embedding Types

The plugin supports two types of embeddings:

- **Class Token Embedding (`cls`)**: A single embedding vector derived from the special classification token. This represents the global semantic context of an image.
  
- **Mean Pooling Embedding (`mean`)**: An embedding vector computed by averaging the representations of all image patches. This captures distributed contextual information across the entire input.

## Usage in FiftyOne App

You can compute AIMv2 embeddings directly through the FiftyOne App:

1. Launch the FiftyOne App with your dataset
2. Open the "Operators Browser" by clicking on the Operator Browser icon above the sample grid or by typing backtick (`)
3. Type "compute_aimv2_embeddings"
4. Configure the following parameters:
   - **Model**: Select one of the supported AIMv2 models
   - **Embedding Type**: Choose between:
     - `cls` - Class token embedding for global semantic context
     - `mean` - Mean pooling embedding for distributed contextual information
   - **Field Name**: Enter the name for the embeddings field (e.g., "aimv2_embeddings")
5. Click "Execute" to compute embeddings for your dataset

The embeddings will be stored in the specified field name and can be used for similarity searches, visualization, or other downstream tasks. 

**Note:** text-image similarity search is not currently supported.

## Operators

### `compute_aimv2_embeddings`

This operator computes image embeddings using an AIMv2 model.

## Operator usage via SDK

Once the plugin has been installed, you can instantiate the operator as follows:

```python
import fiftyone.operators as foo

embedding_operator = foo.get_operator("@harpreetsahota/aimv2_embeddings/compute_aimv2_embeddings")
```

You can then compute embeddings on your dataset by running the operator with your desired parameters:

```python
# Run the operator on your dataset
embedding_operator(
    dataset,
    model_name="apple/aimv2-large-patch14-224",  # Choose any supported model
    embedding_type="cls",  # Either "cls" or "mean"
    field_name="aimv2_embeddings",  # Name for the embeddings field
)
```

If you're running in a notebook, it's recommended to launch a [Delegated operation](https://docs.voxel51.com/plugins/using_plugins.html#delegated-operations) by running `fiftyone delegated launch` in terminal, then run as follows:

```python
await embedding_operator(
    dataset,
    model_name="apple/aimv2-large-patch14-224",  # Choose any supported model
    embedding_type="cls",  # Either "cls" or "mean"
    field_name="aimv2_embeddings",  # Name for the embeddings field
)
```

# Citation

You can read the paper [here](https://arxiv.org/abs/2411.14402).

```bibtex
@misc{fini2024multimodal,
    title={Multimodal Autoregressive Pre-training of Large Vision Encoders},
    author={Enrico Fini and Mustafa Shukor and Xiujun Li and Philipp Dufter and Michal Klein and David Haldimann and Sai Aitharaju and Victor Guilherme Turrisi da Costa and Louis Béthune and Zhe Gan and Alexander T Toshev and Marcin Eichner and Moin Nabi and Yinfei Yang and Joshua M. Susskind and Alaaeldin El-Nouby},
    year={2024},
    eprint={2411.14402},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```