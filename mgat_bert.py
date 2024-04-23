# coding:utf-8
import torch
import numpy as np
import copy
import math
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable



class MGATBert(nn.Module):
    def __init__(self, bert, args):
        super().__init__()
        hidden_dim = args.bert_dim // 2
        self.args = args
        self.enc = ContextEncoder(bert, args)
        self.classifier = nn.Linear(hidden_dim*2, args.polarities_dim)

    def forward(self, inputs):
        hiddens = self.enc(inputs)
        logits = self.classifier(hiddens)
        return logits, hiddens


class ContextEncoder(nn.Module):
    def __init__(self, bert, args):
        super().__init__()
        self.bert = bert
        self.args = args
        self.opt = args
        self.bert_dim = args.bert_dim
        self.bert_layernorm = LayerNorm(self.bert_dim)
        self.bert_drop= nn.Dropout(args.bert_dropout)
        self.hidden_dim = self.bert_dim // 2
        self.dense = nn.Linear(args.bert_dim , args.polarities_dim)
        self.saattn = SelfAttention(args.bert_dim)
        self.mgat = MGAT(args.bert_dim, args.bert_dim, 2)
    def reset_parameters(self):
        torch.nn.init.eye_(self.inp_map.weight)
        torch.nn.init.zeros_(self.inp_map.bias)

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids, \
        attention_mask,text, aspect_mask =inputs

        maxlen = self.args.max_length
        text = text[:, :maxlen]
        bert_spc_out = self.bert(text_bert_indices, token_type_ids=bert_segments_ids, attention_mask=attention_mask,
                                  )
        bert_spc_out = self.bert_layernorm(bert_spc_out)
        bert_spc_out = self.bert_drop(bert_spc_out)

        sa_attn, sa_prob, sa_mean = self.saattn(bert_spc_out)


        context = self.mgat(bert_spc_out,sa_prob)
        mean_context = torch.mean(context, dim=1)





        return mean_context




class SelfAttention(torch.nn.Module):
    def __init__(self, input_dim, num_heads=1):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.query = torch.nn.Linear(input_dim, input_dim)
        self.key = torch.nn.Linear(input_dim, input_dim)
        self.value = torch.nn.Linear(input_dim, input_dim)

    def forward(self, inputs, aspect_mask=None):
        batch_size, seq_len, _ = inputs.size()
        # 计算 query、key 和 value
        q = self.query(inputs)
        k = self.key(inputs)
        v = self.value(inputs)

        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1) ** 0.5)  # (batch_size, seq_len, seq_len)
        if aspect_mask is not None:
            attention_scores = attention_scores.masked_fill(aspect_mask.unsqueeze(1).bool(), float('-inf')) 
        attention_probs = F.softmax(attention_scores, dim=-1) 
       
        context = torch.matmul(attention_probs, v)  # (batch_size, seq_len, dim)
        mean_context = torch.sum(context, dim=1, keepdim=True) / (seq_len ** 0.5)  # (batch_size, 1, dim)
        return context, attention_probs, mean_context



class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads

        self.W = nn.ModuleList([nn.Linear(input_dim, 100) for _ in range(num_heads)])
        self.fc = nn.Linear(output_dim, output_dim)

    def forward(self, h, adj):
        # Compute attention coefficients for each head
        alpha = []
        for k in range(self.num_heads):
            W_h = self.W[k](h)  # Linear transformation
            alpha_k = torch.matmul(torch.matmul(W_h, adj.transpose(1, 2)),W_h.transpose(1, 2))  # Compute attention scores
            alpha.append(alpha_k)
        alpha = torch.stack(alpha, dim=1)  # Combine attention scores from all heads

        # Normalize attention coefficients
        alpha = F.softmax(alpha, dim=-1)

        # Compute output for each head and concatenate
        outputs = []
        for k in range(self.num_heads):
            output_k = torch.matmul(alpha[:, k], h)  # Weighted sum of input features
            outputs.append(output_k)
        outputs = torch.stack(outputs, dim=1)  # Combine outputs from all heads
        output = torch.mean(outputs, dim=1)  # Average over all heads

        # Linear transformation and non-linearity
        output = self.fc(output)
        output = F.relu(output)

        return output


class MGAT(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads):
        super(MGAT, self).__init__()
        self.attention = MultiHeadAttention(input_dim, output_dim, num_heads)

    def forward(self, h, adj):
        return self.attention(h, adj)
