# GraphHAM
This repository contains the source code and dataset for our paper **"An Efficient Automatic Meta-Path Selection for Social Event Detection via Hyperbolic Space"**.

# Requirements
* Python>=3.7
* torch>=1.4.0
* scipy>=1.2.1
* networkx>=2.4
* scikit-learn>=0.20.3

# To run GraphHAM
```python run.py```

# Datasets
The datasets used in GraphHAM are [Kawarith](https://github.com/alaa-a-a/kawarith)[1], [CrisisLexT6](https://crisislex.org/data-collections.html#CrisisLexT26)[2], and [Twitter2012](https://github.com/RingBDStack/KPGNN/tree/main/datasets/Twitter)[3]. To comply with [Twitter’s policies](https://developer.twitter.com/en/developer-terms/agreement-and-policy), we only uploaded the processed Kawarith dataset for demonstration. You can retrieve the complete tweet object via the dataset's link and a valid Twitter API.


# Ablation Studies
All variables can be modified in ```config.py```

# Citation
If you find this repository helpful, please consider citing the following paper.
```
@inproceedings{10.1145/3589334.3645526,
author = {Qiu, Zitai and Ma, Congbo and Wu, Jia and Yang, Jian},
title = {An Efficient Automatic Meta-Path Selection for Social Event Detection via Hyperbolic Space},
year = {2024},
isbn = {9798400701719},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3589334.3645526},
doi = {10.1145/3589334.3645526},
pages = {2519–2529},
numpages = {11},
location = {<conf-loc>, <city>Singapore</city>, <country>Singapore</country>, </conf-loc>},
series = {WWW '24}
}
```

# Reference
[1] Alaa Alharbi and Mark Lee. 2021. Kawarith: an Arabic Twitter corpus for crisis events. In Proceedings of the Sixth Arabic Natural Language Processing Workshop. 42–52.

[2] Alexandra Olteanu, Carlos Castillo, Fernando Diaz, and Sarah Vieweg. 2014. Crisislex: A lexicon for collecting and filtering microblogged communications in crises. In Proceedings of the international AAAI conference on web and social media, Vol. 8. 376–385.

[3] Andrew J McMinn, Yashar Moshfeghi, and Joemon M Jose. 2013. Building a large-scale corpus for evaluating event detection on twitter. In Proceedings of the 22nd ACM international conference on Information & Knowledge Management. 409–418.
