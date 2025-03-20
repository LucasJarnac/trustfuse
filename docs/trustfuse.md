# Using TrustFuse

## Run a fusion model

#### ``fusion_pipeline.py``

Runs a basic data fusion pipeline.

```
python fusion_pipeline.py --dataset-path DATASET_PATH --attr-types ATTR_TYPES --model CRH --dataset-name DATASET_NAME --preprocess-config .\data\configurations\crh\book\preprocess_configuration.json --dynamic
```

with
* ``DATASET_PATH``: (e.g., data/input_trustfuse/&lt;dataset_name&gt;)
* ``ATTR_TYPES``: file that maps attributes to a data type (e.g., data/configurations/&lt;model_name&gt;/&lt;dataset_name&gt;/types.json)
* ``MODEL``: name of the fusion model
* ``DATASET_NAME``: name of the dataset
* ``PREPROCESS_CONFIG``: preprocessing file that contains functions to be applied on the dataset before data fusion (e.g., data/configurations/&lt;model_name&gt;/&lt;dataset_name&gt;/preprocess_configuration.json)
* ``DYNAMIC (optional)``: add ``--dynamic`` to the command only if you use WikiConflict dataset

## Models 

### Models implemented in TrustFuse (more to come)

* [CRH](https://dl.acm.org/doi/pdf/10.1145/2588555.2610509)
* [ACCU](https://dl.acm.org/doi/pdf/10.14778/1687627.1687690)
* [CATD](https://dl.acm.org/doi/pdf/10.14778/2735496.2735505)
* [GTM](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=2ffe1157df90ce94cb91f28074b43b58135cedac)
* [KDEm](https://dl.acm.org/doi/pdf/10.1145/2939672.2939837)
* [LTM](https://dl.acm.org/doi/pdf/10.14778/2168651.2168656)
* [SLiMFast](https://dl.acm.org/doi/pdf/10.1145/3035918.3035951)
* [TruthFinder](https://dl.acm.org/doi/pdf/10.1145/1281192.1281309)

### Data

- [Existing fusion datasets also integrated into my repository](http://lunadong.com/fusionDataSets.htm)


#### <i class="fab fa-github"></i> Links to GitHub repositories that inspired me and from which some models have been reused or modified:

- [TDH](https://github.com/woohwanjung/truthdiscovery)

- [LTM](https://github.com/yishangru/TruthDiscovery/tree/master)

- [Multiple models](https://github.com/MengtingWan/KDEm)

- [SlimFast, ACCU](https://github.com/HoloClean/RecordFusion/)