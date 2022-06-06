### Exploitation

##### Data Exploration

In this project, we used the Amazon dataset to classify Amazon products to their categories. Given this dataset, a question that might come to mind is whether the frequency of two products being bought together is related with their category.

In the follow in Histogram we will figure out whether two items that are more frequently bought together, are also more likely to belong in the same class. We end up with the following graph, where the x axis is the frequency of two products being bought together (divided into 10 bars) and the y axis is the percentage of items that belong in the same category.

![1](C:\Users\Mao Zhipeng\Documents\2022Spring\Network machine learning\project\1.png)

As we see, the items that are frequently bought together, are less likely to belong in the same category. Therefore the edges attribute is not sufficient in order to classify the products, and as a result we certainly need to take advantage of the nodes features.

##### Evaluation Metrics

In the dataset, each product can have multiple categories. Therefore, the classification problem is a multi-label problem. F1-micro score is a typical metrics to assess the quality of such problems. 

$\text{Micro F1-score}=\frac{2*\text{Micro-precision} * \text{Micro-recall}}{\text{Micro-precision} + \text{Micro-recall}}$

where $\text{Micro-precision}$ and $\text{Micro-recall}$ are calculated across all the classes.

##### Baseline

According to the visualization results, products of the same category are close to each other in the Euclidean space of the node feature. Therefore, we use KNN (K=5) as the baseline, i.e. assign one item with the category that the K nearest items vote. Before doing KNN, we use PCA to reduce the dimensionality of the node features. However, the results show that retaining the original node feature get the best results.

| Dimensionality | Test set F1-micro Score |
| -------------- | ----------------------- |
| 200            | 0.3131                  |
| 100            | 0.2867                  |
| 50             | 0.2146                  |
| 10             | 0.1324                  |

We also tried Ridge Classifier and obtained F1-micro score 0.0833 on the test set, which is far worse than KNN.

##### Graph Sampling

The Amazon dataset is too large to do a full-batch training of GNN because GPU memory is limited. Also, in standard SGD, if we sample nodes independently in a mini-batch, sampled nodes tend to be isolated from each other and GNN cannot access to neighboring nodes to aggregate their node features. Therefore, we consider using graph samplers to enable message-passing over small subgraphs in each mini-batch and only the subgraphs need to be loaded on a GPU at a time. 

Existent methods include neighbor sampling [Hamilton et al. NeurIPS 2017] (https://arxiv.org/abs/1706.02216), Cluster-GCN [Chiang et al. KDD 2019] (https://arxiv.org/abs/1905.07953) and GraphSAINT sampler (https://arxiv.org/abs/1907.04931), because in  the GraphSAINT paper the GraphSAINT sampler outperforms the other two. Specifically, we use the random walk based sampler with walk length 4, batch size 1500, num_steps (The number of iterations per epoch) 5, sample coverage (How many samples per node should be used to compute normalization statistics) 100. 



##### Network Architecture

The architecture we first consider is 3 ClusterGCNConv layers, which is the ClusterGCN graph convolutional operator from [Chiang et al. KDD 2019] the paper ($\mathbf{X}^{\prime} = \left( \mathbf{\hat{A}} + \lambda \cdot
\textrm{diag}(\mathbf{\hat{A}}) \right) \mathbf{X} \mathbf{W}_1 +
\mathbf{X} \mathbf{W}_2$

where $\mathbf{\hat{A}} = {(\mathbf{D} + \mathbf{I})}^{-1}(\mathbf{A}+\mathbf{I})$), with elu (Exponential Linear Unit) as the activation function. Using hidden dimension of 512, learning rate 0.001 and Adam optimizer, we achieve best validation set F1-micro score 0.7510 (the corresponding F1-score on the test set is 0.6918), and best test set F1-micro score 0.7163. 

Considering the previous knowledge that the category of one item may not have much to do with its neighbors in the graph. Instead, the node feature itself may contain more information about its category. Therefore,  we add residual connection between the convolution layers.



After adding the residual connection, we achieve a significantly better result, with best validation set F1-micro score 0.7774 (the corresponding F1-score on the test set is 0.7394) and best test set F1-micro score 0.7588. 

We tried several other Convolution layers and we keep the number of the layers 3, hidden dimension 512, learning rate 0.001 across all methods, and select the best results during 500 epochs of training. The results show than ClusterGCN and SAGEConv outperform the other two.

| Convolution   layer | Best validation set F1-micro score (corresponding test set score) | Best test set F1-micro score |
| ------------------- | ------------------------------------------------------------ | ---------------------------- |
| SAGEConv            | 0.7907 (0.7080)                                              | 0.7541                       |
| GATConv             | 0.7172 (0.6613)                                              | 0.7017                       |
| TransformerConv     | 0.7900 (0.6879)                                              | 0.7496                       |
| ClusterGCN          | 0.7774 (0.7394)                                              | 0.7588                       |

We also tried different layer numbers to investigate if deeper models can lead to better results. According to the results, model with 4 and 5 layers outperform those with 2 and 3 significantly.

| Layer number | Best validation set F1-micro score (corresponding test set score) | Best test set F1-micro score |
| ------------ | ------------------------------------------------------------ | ---------------------------- |
| 2            | 0.7152 (0.6863)                                              | 0.7065                       |
| 3            | 0.7774 (0.7394)                                              | 0.7588                       |
| 4            | 0.7924 (0.7649)                                              | 0.7784                       |
| 5            | 0.7917 (0.7618)                                              | 0.7808                       |



##### 

