## Requirements

### Python requirements

See [requirements.txt](requirements.txt)

* Python 3.11
* lmdb==1.4.0
* pandas==1.5.3
* tqdm==4.66.2
* typing==3.7.4.3
* scipy==1.10.1
* numpy==1.24.2
* tabulate==0.9.0
* requests==2.31.0
* gradio==5.16.0
* pyvis==0.3.2
* networkx==3.1
* scikit-learn==1.2.1
* Wikipedia-API==0.7.3

### Data requirements

To test fusion models or to reproduce the paper experiments you will need the WikiConflict dataset and other datasets from the literature.

* The datasets from the literature used in our experiments (Flights, Stocks, Weather, Restaurant, Book) have been copied into this repository but are also available [here](http://lunadong.com/fusionDataSets.htm). A description of these datasets is provided in the link. These datasets are often used to evaluate fusion models.

* WikiConflict is a dataset built from the history of Wikidata revisions since its creation in 2012. WikiConflict contains several sub-datasets that are each associated with a Wikipedia category. For example, the ``Monuments in Paris`` sub-dataset contains all revisions of Wikidata entities associated with Wikipedia pages in the ``Monuments in Paris`` category or sub-categories.


WikiConflict consists of a pickle file that stores conflicting data under the structure of a dictionary of buckets assigned to an index in the chronological order of bucket closure (explained in the paper): 


```python
{
  BID: {
    "data": {
        "value": pandas.DataFrame({...}),
        "qid": pandas.DataFrame({...})
    },
    "GT": {
        "value": pandas.DataFrame({...}),
        "qid": pandas.DataFrame({...}),
        "value_order": list(...),
        "qid_order": list(...)
    }
  }
}
```

For example, for the ``located in the administrative territorial entity`` property of entity ``Eiffel Tower (Q243)``, the pandas DataFrame looks like this:

| Source                               | Entity   | located in the administrative territorial entity   |    
|:-------------------------------------|:---------|:---------------------------------------------------|
| Danrok                               | Q243     | 7th arrondissement of Paris                        |
| Dega180                              | Q243     | Paris                                              |
| Docu                                 | Q243     | Île-de-France                                      |
| Haplology                            | Q243     | Île-de-France                                      |
| Haplology                            | Q243     | Paris                                              |
| Haplology                            | Q243     | 7th arrondissement of Paris                        |
| 190.162.120.233                      | Q243     | 7th arrondissement of Paris                        |
| 190.162.120.233                      | Q243     | Île-de-France                                      |
| 190.162.120.233                      | Q243     | Santiago                                           |
| 2A01:E34:EC09:3820:9DF:101:F75A:6862 | Q243     | Mars                                               |
| 80.15.152.116                        | Q243     | 18th arrondissement of Paris                       |
| Ayack                                | Q243     | 7th arrondissement of Paris                        |
| 90.88.107.246                        | Q243     | 5th Parliament of the United Kingdom               |

Or for the ``height`` property:

| Source         | Entity   | height        |
|:---------------|:---------|:--------------|
| Ayack          | Q243     | +324metre     |
| Jklamo         | Q243     | +324metre     |
| Sjoerddebruin  | Q243     | +324metre     |
| YMS            | Q243     | +324metre     |
| 188.130.107.65 | Q243     | +125metre     |
| 196.217.134.32 | Q243     | +300metre     |
| 193.49.248.104 | Q243     | +325metre     |
| 86.197.111.59  | Q243     | +10centimetre |
| NicoScribe     | Q243     | +324metre     |
| 176.151.127.31 | Q243     | +324.16metre  |
| HaeB           | Q243     | +324metre     |


For ground truth (GT) dataframes look like the same as above except that only one line is present and each cell is a list of one or more correct values (depending on the property).

Partial orders correspond to nested lists according to the number of different specificity levels, for example:

```python
partial_order = [['7th arrondissement of Paris'], ['Paris'], ['Île-de-France']]
```
is a partial order that indicates ``7th arrondissement of Paris < Paris < Île-de-France`` with ``x < y`` meaning that x is more specific than y.
