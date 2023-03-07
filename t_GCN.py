from torch_geometric.datasets import KarateClub
import matplotlib.pyplot as plt
import networkx as nx
import torch
#print(torch.cuda.is_available())
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils import to_networkx
# Import dataset from PyTorch Geometric
dataset = KarateClub()

# Print information
print(dataset)
print('------------')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

data = dataset[0]

print(f'x = {data.x.shape}')
print(data.x)


A = to_dense_adj(data.edge_index)[0].numpy().astype(int)
print(f'A = {A.shape}')
print(A)


G = to_networkx(data, to_undirected=True)
plt.figure(figsize=(12,12))
plt.axis('off')
nx.draw_networkx(G,
                pos=nx.spring_layout(G, seed=0),
                with_labels=True,
                node_size=800,
                node_color=data.y,
                cmap="hsv",
                vmin=-2,
                vmax=3,
                width=0.8,
                edge_color="grey",
                font_size=14
                )
plt.show()

from torch.nn import Linear
from torch_geometric.nn import GCNConv, RGCNConv


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #self.gcn = GCNConv(dataset.num_features, 3)
        self.gcn = RGCNConv(dataset.num_features, 3, num_relations=1)
        self.out = Linear(3, dataset.num_classes)

    def forward(self, x, edge_index, edge_attr):
        h = self.gcn(x, edge_index, edge_attr).relu()
        z = self.out(h)
        return h, z

model = GCN()
print(model)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

# Calculate accuracy
def accuracy(pred_y, y):
    return (pred_y == y).sum() / len(y)

# Data for animations
embeddings = []
losses = []
accuracies = []
outputs = []

# Training loop
for epoch in range(201):
    # Clear gradients
    optimizer.zero_grad()

    # Forward pass
    data.edge_attr = torch.zeros(data.edge_index.shape[1], dtype=torch.long)
    h, z = model(data.x, data.edge_index, data.edge_attr)

    # Calculate loss function
    loss = criterion(z, data.y)

    # Calculate accuracy
    acc = accuracy(z.argmax(dim=1), data.y)

    # Compute gradients
    loss.backward()

    # Tune parameters
    optimizer.step()

    # Store data for animations
    embeddings.append(h)
    losses.append(loss)
    accuracies.append(acc)
    outputs.append(z.argmax(dim=1))

    # Print metrics every 10 epochs
    if epoch % 10 == 0:
        print(f'Epoch {epoch:>3} | Loss: {loss:.2f} | Acc: {acc*100:.2f}%')

embed = embeddings[200].detach().cpu().numpy()

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')
ax.patch.set_alpha(0)
plt.tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)
ax.scatter(embed[:, 0], embed[:, 1], embed[:, 2],
           s=200, cmap="hsv", vmin=-2, vmax=3)

plt.show()