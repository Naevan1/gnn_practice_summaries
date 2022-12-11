# gnn_practice_summaries


Size and shapes change in graphs, which is a problem when we know that in general NN expect a specific input( e.g. with images of different size we resize.) So we need a method that can handle arbitrary inptut shapes.

Graphs have isomoprhism -> 2 graphs that look different can be structurally identical( while images if u flip one they are not the same, in graphs only order of nodes change.) So the algorithm that handles the graph data must be permutation invariant

-Graph structure is non-eucliean, distance between nodes etc doesnt make sense.

Fundamental Idea of a GNN : Learn a , for neural networks, SUITABLE REPRESENTATION of graph data . Thats called representation learning.

So taking in a Graph with nodes,edges,node features, edge features and adjacency matrix, pass it through the GNN with message passing layers, and output new representations which are called embeddings, for each of the nodes. These node embeddings contain BOTH the STRUCTURAL AND THE FEATURE information of the other nodes in the graph. This means that each node, knows something about the other nodes, the connection to these nodes, and it's context in the graph. The embeddings can finally be used to perform predictions. e.g. for node level predictions, you would use the node embedding of a specific unlabeled node to obtain a prediction. ( periexei oles tis plirofories pou thes apo trigiro auto to embedding! gia auto ginete!) If u wanted graph level predicitons you would take all of the node embeddings, combine them in a certain way,and get a representation of the whole graph.

OR , you could do the following : include pooling operations iteratevly to compress the graph into a fixed size vector representation. This representation can then be used to do predictions.

Similar nodes ( nodes with similar feature or similar context) will lead to similar node embeddings, the same way as similar graphs lead to similar graph embeddings using a gnn.

*Size of node embeddings is a hyper parameter. It starts with the amount of features, but after the GNN operations it changes, usually increasing, and they no longer " make sense" in a typical way. (they cannot DIRECTLY be interpreted, as they are an artifiical compound of the node and edge information on the whole graph.)*

*Lastly, edge features can also be processed in the gnn, and will be combined into these node embeddings.*

Message passing Layers are the most imporant thing in a GNN : They are responsible for converting the node and edge features into the NODE EMBEDDINGS.

*ALL IN ALL : The NODE EMBEDDINGS hold information on the whole graph - node properties,adjacencies, overall graph structure and connectivity - including the context of the NODE ITSELF in the graph.*





This takes information from our current node state(feature vector) and the infromaation about our direct neighbhours node states.

So for example for h1^(k) takes info from h2^(k),h3^(k) and h4^(k),aggregates them and after combining them and updates it into h1^(k+1).

States( feature vectors) are denoted by h, and we are in time step k.

This means that h1^(k+1) contains information from itseld and the blue nodes ( the direct 1 step neighbhours)

This procedure is done for every node.

Notice how the green node, which is 2 steps away from yellow, does not have the information from yellow yet, but it will change on k+2.

MP can be viwed as number of hops.

MP is always done with neighbours just as with each layer each node has more information. Every message passing step corresponds to a layer in the GNN.

Every feature vector contains info from the other nodes and the node itself.

The # of MP layers is a hyperparameter that depends on graph data we use.

When you have many more layers than needed in a GNN you have a problem called oversmoothing, which makes all the nodes too similar and indistinquisable.

Every GNN architecture follows the same concept of MP with the UPDATE and AGGREGATE , just using different things.


Now to implement a GNN. It works the same way as any other pytorch module. We just use different layers, which come from torch_geometric.nn

Here we use GCNConv (graph convolution layer which is MESSAGE PASSSING layer.) Others exist like graph attention layer etc.

So this works by defining the 2 fucntions init and forward. In init we define layers , here we have 4 message passing layers. The first layer is the transformaton layer which transforms our data.num_features(9 features) into embeddings of size 64. Then we have 3 more message passing steps and a Linear output layer, which has output shape of 1 (data.num_classes) which means we want to perform regression.

In our forward function , we pass in the node features(x), the edge informatio(edge_index) and the batch_index.

By "hidden" he means pretty much the updated node states. So the first time you pass in x and edge_index, activate it, and then you use the updated node feature vector(hidden) to update the next one in the next message passing layer and so on.

We want graph level predictions so we want to take all the node states of our graph into a graph representation. The first idea would be to append the feature vectors, but the problem is that for one graph we might have 20 nodes but for another we would have 50. That would be a problem since it would not be a global solution. So we need a global pooling operation to do it . One way is using mean&max pooling. You first take the mean of all the feature vectors and the max of them and append them to 1 feature vector so one representation Smarter ways exist than this but its good enough. Another way would be to use graph pooling and that would be decreasing graph nodes over time , like for example starting with 5 then 4 etc... until you end up with a representation that contains all information over the graph.

in any case we use Global pooling with mean and max.

So thats why on self.out we say embedding times 2. On Global pooling, we do global max pool and global mean pool adn then we concatenate them. so we have 2 vectors appended ,thats why we need twice the size. batch_index is also important.

In our example, we start with 9 node features GCNConv(9,64) , then project them to 64 (64,64) 3 times, then on out Linear(in_featres=128,out=1) which is the stacked compressed representation (regression)

For node level predicitons , in case you work on that( or edge level) you encounter binary masks. Those indicate which nodes have(or dont have) a class labeled.


DataLoader work a bit differently in pytorch geom. in deep learning its used for training speed up or convergance. Here its more useful. We have graphs of different shapes and sizes of nodes. This works by concatenating all node features in a large matrix and combining the individual adjacency information in a huge adjacency information matrix. The individual graphs are disconnected in this adjacency matrix, so no information will be passed between the graphs, only in the graphs. We can simply input a large graph, pass it through the MP layers, and get an updated node embedding for each node for each graph.

Next we define the train function

I think batches is what it means. It batches for better computation ,here 64 nodes, and every time we iterate over those batches (the train ones are included in loader, the test ones in test_loader) and predict the prediction and embeddings, calculate loss function, optimize and return the end results , which are prediction for each graph(molecule) and embedding
