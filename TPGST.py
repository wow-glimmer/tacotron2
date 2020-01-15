import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.utils import weight_norm as norm
import numpy as np
import module as mm


class ReferenceEncoder(nn.Module):
    """
    Reference Encoder.
    6 convs + GRU + FC
    :param in_channels: Scalar.
    :param embed_size: Scalar.
    :param activation_fn: activation function
    """
    def __init__(self, in_channels=1, embed_size=256, activation_fn=None):
        super(ReferenceEncoder, self).__init__()
        self.embed_size = embed_size
        channels = [in_channels, 32, 32, 64, 64, 128, embed_size]
        self.convs = nn.ModuleList([
            mm.Conv2d(channels[c], channels[c+1], 3, stride=2, bn=True, bias=False, activation_fn=torch.relu)
            for c in range(len(channels)-1)
        ]) # (Batch, Time_domain/r, 128)
        self.gru = nn.GRU(self.embed_size*2, self.embed_size, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(embed_size, embed_size),
        )
        self.activation_fn = activation_fn

    def forward(self, x, hidden=None):
        """
        :param x: (Batch, 1, Time_domain, n_mel_channels) Tensor. Mel Spectrogram
        :param hidden: Tensor. initial hidden state for gru
        Returns:
            y_: (Batch, 1, Embedding) Reference Embedding
        
        :
            Time_domain = Time_domain
            n_mel_channels = n_mel_channels
            Batch = batch
        """
        #print(x.shape)
        #y_ = x.unsqueeze(3).transpose(2,3) # (Batch, n_mel_channels, time_domain) -> [Batch, n_mel_channels, time_domain, 1] -> [Batch, n_mel_channels, 1, time_domain]
        #y_ = x.view(x.size(0), -1, 80).unsqueeze(1)
        y_ = x.transpose(1, 2).unsqueeze(1) # (Batch, n_mel_channels, time_domain) -> (Batch, time_domain, n_mel_channels) -> [Batch, 1, time_domain, n_mel_channels]
        #print("0: "+str(y_.shape))
        
        for i in range(len(self.convs)):
            y_ = self.convs[i](y_)
        
        #print("1: "+str(y_.shape))
        # (Batch, C, Time_domain//64, n_mel_channels//64)
        y_ = y_.transpose(1, 2) # (Batch, Time_domain//64, C, n_mel_channels//64)
        #print("2: "+str(y_.shape))
        shape = y_.shape
        y_ = y_.contiguous().view(shape[0], shape[1], shape[2]*shape[3]) # merge last 2 dimensions
        #print("3: "+str(y_.shape))
        # y_ as input, hidden as inital state, normally none
        y_, out = self.gru(y_, hidden) # out = (1, Batch, Embedding)
        #print("4: "+str(out.shape))
        
                            # hidden_state -> y_
        y_ = out.squeeze(0) # (1, Batch, Embedding) -> (Batch, Embedding)
        y_ = self.fc(y_) # (Batch, Embedding)
        
        y_ = self.activation_fn(y_) if self.activation_fn is not None else y_
        return y_.unsqueeze(1) # (Batch, 1, Embedding)


class MultiHeadAttention(nn.Module):
    """
    Multi-head Attention
    :param n_units: Scalars.
    :param embed_size : Scalars.
    """
    def __init__(self, n_units=128, embed_size=256, num_heads=8, token_num=10):
        super(MultiHeadAttention, self).__init__()
        self.split_size = n_units // num_heads
        self.conv_Q = mm.Conv1d(embed_size, n_units, 1) # in_channels, out_channels, kernel_size
        self.conv_K = mm.Conv1d(embed_size, n_units, 1) # in_channels, out_channels, kernel_size
        self.fc_Q = nn.Sequential(
            nn.Linear(n_units, n_units),
            nn.Tanh(),
        )
        self.fc_K = nn.Sequential(
            nn.Linear(n_units, n_units),
            nn.Tanh(),
        )
        self.fc_V = nn.Sequential(
            nn.Linear(embed_size, self.split_size),
            nn.Tanh(),
        )
        self.fc_A = nn.Sequential(
            nn.Linear(n_units, token_num),
            nn.Tanh(),
        )
        

    def forward(self, ref_embedding, token_embedding):
        """
        :param ref_embedding: (Batch, 1, Embedding) Reference embedding
        :param token_embedding: (Batch, token_num, embed_size) Token Embedding
        Returns:
            y_: (Batch, token_num) Tensor. STime_domainle attention weight
        """
        # (Batch, 1, n_units)
        Q = self.fc_Q(self.conv_Q(ref_embedding.transpose(1,2)).transpose(1,2))  # (Batch, 1, Embedding) -> (Batch, Embedding, 1) -> (Batch, Embedding, 1) ->
        K = self.fc_K(self.conv_K(token_embedding.transpose(1,2)).transpose(1,2))  # (Batch, token_num, n_units)
        V = self.fc_V(token_embedding)  # (Batch, token_num, n_units)
        
        Q = torch.stack(Q.split(self.split_size, dim=-1), dim=0) # (n_heads, Batch, 1, n_units//n_heads)
        K = torch.stack(K.split(self.split_size, dim=-1), dim=0) # (n_heads, Batch, token_num, n_units//n_heads)
        V = torch.stack(V.split(self.split_size, dim=-1), dim=0) # (n_heads, Batch, token_num, n_units//n_heads)
        
        inner_A = torch.softmax(
            torch.matmul(Q, K.transpose(-2, -1)) / self.split_size**0.5,
            dim=-1
        ) # (n_heads, Batch, 1, token_num)
        
        y_ = torch.matmul(inner_A, V)  # (n_heads, Batch, 1, n_units//n_heads)
        
        y_ = torch.cat(y_.split(1, dim=0), dim=-1).squeeze() # (Batch, n_units)
        
        y_ = self.fc_A(y_) # (Batch, token_num)
        return y_

class GST(nn.Module):
    """
    STime_domainle Token Layer
    Reference Encoder + Multi-head Attention, token embeddings
    :param embed_size: Scalar.
    :param n_units: Scalar. for multihead attention ***
    """
    def __init__(self, hparams, embed_size=256, n_units=128, token_num=10, in_channels=1):
        super(GST, self).__init__()
        self.token_embedding = nn.Parameter(torch.zeros([token_num, embed_size])) # (token_num, Embedding)
        self.ref_encoder = ReferenceEncoder(in_channels=in_channels, embed_size=embed_size, activation_fn=torch.tanh)
        self.att = MultiHeadAttention(n_units, embed_size)

        init.normal_(self.token_embedding, mean=0., std=0.5)
        # init.orthogonal_(self.token_embedding)
 
    def forward(self, ref, ref_mode=True):
        """
        :param ref: (Batch, Time_domain, n_mel_channels) Tensor containing reference audio or (Batch, token_num) if not ref_mode
        :param ref_mode: Boolean. whether it is reference mode
        Returns:
            :y_: (Batch, 1, Embedding) STime_domainle embedding
            :A: (Batch, token_num) Tensor. Combination weight.
        """
        token_embedding = self.token_embedding.unsqueeze(0).expand(ref.size(0), -1, -1) # (Batch, token_num, Embedding)
        if ref_mode:
            ref = self.ref_encoder(ref) # (Batch, 1, Embedding)
            A = torch.softmax(self.att(ref, token_embedding), dim=-1) # (Batch, token_num)
            # A = torch.softmax(self.att(ref, token_embedding)) # (Batch, token_num)
        else:
            A = torch.softmax(ref, dim=-1)
        y_ = torch.sum(A.unsqueeze(-1) * token_embedding, dim=1, keepdim=True) # (Batch, 1, Embedding)
        y_ = torch.tanh(y_)
        return y_ #, A

