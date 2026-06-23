import torch
import torch.nn as nn

class EmbeddingToDETRInput(nn.Module):
    def __init__(self, input_dim, num_tokens, embed_dim):
        super(EmbeddingToDETRInput, self).__init__()
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim
        self.linear = nn.Linear(input_dim, num_tokens * embed_dim)
        self.positional_encoding = nn.Parameter(self._generate_positional_encoding(num_tokens, embed_dim), requires_grad=False)

    def _generate_positional_encoding(self, num_tokens, embed_dim):
        pe = torch.zeros(num_tokens, embed_dim)
        position = torch.arange(0, num_tokens, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe

    def forward(self, x):
        bs = x.size(0)
        x = self.linear(x)
        x = x.view(bs, self.num_tokens, self.embed_dim)
        x = x + self.positional_encoding
        return x

# # Example usage
# bs = 8  # batch size
# input_dim = 1024  # input embedding dimension
# num_tokens = 100  # number of tokens expected by DETR
# embed_dim = 256  # embedding dimension expected by DETR
#
# embeddings = torch.randn(bs, input_dim)  # your (bs, 1024) embeddings
# transformer_input = EmbeddingToDETRInput(input_dim, num_tokens, embed_dim)
# detr_input = transformer_input(embeddings)  # shape will be (bs, num_tokens, embed_dim)
# print(detr_input.shape)
# # Now, detr_input can be fed into the DETR model
# # detr_model = DETRModel()  # your DETR model initialization
# # outputs = detr_model(detr_input)  # forward pass through DETR
