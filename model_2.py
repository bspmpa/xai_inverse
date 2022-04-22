import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.config=config
        self.bn1 = nn.BatchNorm1d(14, affine=True)
        if self.config.name == 'adult':
            dim_list = [9, 16, 7, 15, 6, 5, 2, 42]
        else:
            dim_list = [13, 4, 5, 3, 3, 3, 4, 13, 5]
        self.embedding_list =nn.ModuleList([nn.Embedding(dim, 1) for dim in dim_list])
        self.linear1 = nn.Linear(14, config.hidden_dim)
        self.linear2 = nn.Linear(config.hidden_dim, 16)
        self.linear3 = nn.Linear(config.hidden_dim2, 16)
        self.linear4 = nn.Linear(16, 2)
        # self.linear_s = nn.Linear(config.input_dim, config.hidden_dim)

        # self.num_labels = config.num_labels
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)
        # self.dropout_c = nn.Dropout(config.dropout)
        # self.dropout_s = nn.Dropout(config.dropout)
        self.softmax = nn.Softmax(dim=1)



    def forward(self, x, y=None):
        x_emb=[]
        if self.config.name=='adult':
            emb_index = [0, 1, 2, 3, 4, 5, 6, 7]
        else:
            emb_index = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        for index, emb in zip(emb_index, self.embedding_list):
            x_emb.append(emb(x[:,index]))
        embed_result = torch.cat(x_emb, 1)
        cat = []
        cat.append(embed_result.float())
        cat.append(x[:, 8:].float())
        cat = torch.cat(cat, 1)
        x = self.bn1(cat)


        logits = self.linear1(x)
        # logits = self.dropout1(logits)
        logits = torch.relu(logits)
        logits = self.linear2(logits)
        # logits = self.dropout2(logits)
        # logits = torch.relu(logits)
        # logits = self.linear3(logits)
        logits = self.dropout3(logits)
        logits = torch.relu(logits)
        logits = self.linear4(logits)
        # logits = self.softmax(logits)

        if y is not None:
            r = torch.argmax(logits, dim=1)
            loss_function = nn.CrossEntropyLoss()
            loss = loss_function(logits, y)

            return loss, logits[:,1],r

        else:
            logits = self.softmax(logits)
            r = torch.argmax(logits, dim=1)
            return logits,r