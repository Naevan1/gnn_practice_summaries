# gnn_practice_summaries
Uploading my practice summaries on GNN's in order to have them available for my work laptop
Here I will include notes on GNN from different videos and stuff - I just use this instead fo textbook to save some time for me.

Size and shapes change in graphs, which is a problem when we know that in general NN expect a specific input( e.g. with images of different size we resize.) So we need a method that can handle arbitrary inptut shapes.

    Graphs have isomoprhism -> 2 graphs that look different can be structurally identical( while images if u flip one they are not the same, in graphs only order of nodes change.) So the algorithm that handles the graph data must be permutation invariant

-Graph structure is non-eucliean, distance between nodes etc doesnt make sense.

Fundamental Idea of a GNN : Learn a , for neural networks, SUITABLE REPRESENTATION of graph data . Thats called representation learning.

So taking in a Graph with nodes,edges,node features, edge features and adjacency matrix, pass it through the GNN with message passing layers, and output new representations which are called embeddings, for each of the nodes. These node embeddings contain BOTH the STRUCTURAL AND THE FEATURE information of the other nodes in the graph. This means that each node, knows something about the other nodes, the connection to these nodes, and it's context in the graph. The embeddings can finally be used to perform predictions. e.g. for node level predictions, you would use the node embedding of a specific unlabeled node to obtain a prediction. ( periexei oles tis plirofories pou thes apo trigiro auto to embedding! gia auto ginete!) If u wanted graph level predicitons you would take all of the node embeddings, combine them in a certain way,and get a representation of the whole graph.

OR , you could do the following : include pooling operations iteratevly to compress the graph into a fixed size vector representation. This representation can then be used to do predictions.

Similar nodes ( nodes with similar feature or similar context) will lead to similar node embeddings, the same way as similar graphs lead to similar graph embeddings using a gnn.

Size of node embeddings is a hyper parameter. It starts with the amount of features, but after the GNN operations it changes, usually increasing, and they no longer " make sense" in a typical way. (they cannot DIRECTLY be interpreted, as they are an artifiical compound of the node and edge information on the whole graph.)

Lastly, edge features can also be processed in the gnn, and will be combined into these node embeddings.

Message passing Layers are the most imporant thing in a GNN : They are responsible for converting the node and edge features into the NODE EMBEDDINGS.

ALL IN ALL : The NODE EMBEDDINGS hold information on the whole graph - node properties,adjacencies, overall graph structure and connectivity - including the context of the NODE ITSELF in the graph.
