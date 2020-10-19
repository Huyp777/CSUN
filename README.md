# CLEAR

Cross-Modal Semantic Alignment for Moment Localization via Nature Language
============================================================================
We propose an end-to-end Coarse-to-fine cross-modaL sEmantic Alignment netwoRk, dubbed ClEAR, to efﬁciently localization target moments within the given video via diverse natural language queries. <br>
Concretely, we first design a dual-path neural network, comprising two independent modules: the video encoding network (VEN) and the query encoding network (QEN). <br>
Thereinto, the VEN applies our proposed hierarchical semantic strategy to the input video for generating the corresponding moment candidates and modeling their semantic relevance. The QEN adopts the word embedding based bi-directional LSTM network (Bi-LSTM) to understand the corresponding semantics of the diverse given queries.<br>
Afterwards, we develop a multi-granularity interaction network (MIN) to achieve high-quality moment localization in an effective coarse-to-fine manner. To be more specific, it first utilizes efficient coarse-grained semantic pruning to filter out corresponding semantic ranges and ignore irrelevant parts, and then performs fine-grained semantic fusing for accurate moments localization.<br>
We conduct extensive experiments on two benchmark datasets ActivityNet Captions and TACoS. The experimental results show that our proposed model is more effective, efﬁcient than the state-of-the-art models.<br>
The introduction of CLEAR in details will be given in the form of an authorized patent and a published paper later.<br>
An illustration of the framework of CLEAR is shown in the following figure.
![CLEAR](initmodel.png)<br>

## Dateset

- TACoS: [http://www.coli.uni-saarland.de/projects/smile/page.php?id=tacos](http://www.coli.uni-saarland.de/projects/smile/page.php?id=tacos)
- ActivityNet: [http://activity-net.org/challenges/2016/download.html#imshuffle](http://activity-net.org/challenges/2016/download.html#imshuffle)
- ActivityNet Captions: [https://cs.stanford.edu/people/ranjaykrishna/densevid/](https://cs.stanford.edu/people/ranjaykrishna/densevid/)



## How to run

Please place the data files to the appropriate path and set it in tacos.py and activitynet_captions.py.
```
python tacos.py
```
or
```
python activitynet_captions.py
```


