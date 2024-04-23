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
        bert_spc_out, _ = self.bert(text_bert_indices, token_type_ids=bert_segments_ids, attention_mask=attention_mask,
                                    return_dict=False)
        bert_spc_out = self.bert_layernorm(bert_spc_out)
        bert_spc_out = self.bert_drop(bert_spc_out)
        ##attention
        sa_attn, sa_prob, sa_mean = self.saattn(bert_spc_out)
        # sent_out= asp_attn + sa_attn

        context = self.mgat(bert_spc_out,sa_prob)
        mean_context = torch.mean(context, dim=1)

        # 通过全连接层进行分类
     #   logits = self.dense(mean_context)
       # logits = F.softmax(logits,dim=-1)


        return mean_context




def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

#
# class MultiHeadAttention(nn.Module):
#
#     def __init__(self, h, d_model, aspect_aware=False, relation_aware=False, dropout=0.1):
#         super(MultiHeadAttention, self).__init__()
#         assert d_model % h == 0
#         self.d_k = d_model // h
#         self.h = h
#         self.aspect_aware = aspect_aware
#         self.relation_aware = relation_aware
#         self.linears = clones(nn.Linear(d_model, d_model), 2)
#         self.dropout = nn.Dropout(p=dropout)
#         if aspect_aware:
#             self.weight_m = nn.Parameter(torch.Tensor(self.h, self.d_k, self.d_k))
#             self.bias_m = nn.Parameter(torch.Tensor(1))
#             self.dense = nn.Linear(d_model, self.d_k)
#         if relation_aware:
#             self.linear_query = nn.Linear(d_model, d_model)
#             self.linear_key = nn.Linear(d_model, self.d_k)
#
#     def forward(self, query, key, aspect=None, mask=None):
#         if mask is not None:
#             mask = mask[:, :, :query.size(1)]
#             mask = mask.unsqueeze(1)  # (B, 1, 1, seq)
#
#         nbatches = query.size(0)
#         if self.relation_aware:
#             query = self.linear_query(query).view(nbatches, -1, self.h, self.d_k)  # (B, H, seq, dim)
#             key = self.linear_key(key)  # key: relation adj(B, seq, seq, dim)
#         else:
#             query, key = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
#                           for l, x in zip(self.linears, (query, key))]  # multiple heads dim: d/z
#         # aspect aware attention
#         if self.aspect_aware:
#             batch, a_dim = aspect.size()[0], aspect.size()[1]
#             aspect = aspect.unsqueeze(1).expand(batch, self.h, a_dim)  # (batch, heads, dim)
#             aspect = self.dense(aspect)  # (batch, heads, dim/heads)
#             aspect = aspect.unsqueeze(2).expand(batch, self.h, query.size()[2],
#                                                 self.d_k)  # (batch_size, heads, seq, dim)
#             attn = self.attention(query, key, aspect=aspect, mask=mask, dropout=self.dropout)
#         else:
#             attn = self.attention(query, key, mask=mask, dropout=self.dropout)  # self-att score
#         return attn
#
#     def attention(self, query, key, aspect=None, mask=None, dropout=None):
#         d_k = query.size(-1)
#         scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
#         # aspect-aware attention
#         if self.aspect_aware:
#             batch = len(scores)
#             p = self.weight_m.size(0)
#             max = self.weight_m.size(1)
#             weight_m = self.weight_m.unsqueeze(0).expand(batch, p, max, max)  # (B, heads, dim, dim)
#             # attention scores
#             aspect_scores = torch.tanh(torch.add(torch.matmul(torch.matmul(aspect, weight_m), key.transpose(-2, -1)),
#                                                  self.bias_m))  # [16,5,41,41]
#             scores = torch.add(scores, aspect_scores)  # self-attn + aspect-aware attn
#
#         if mask is not None:
#             scores = scores.masked_fill(mask == 0, -1e18)
#         if not self.relation_aware:  # using relation do not choose the softmax
#             scores = F.softmax(scores, dim=-1)
#             if dropout is not None:
#                 scores = dropout(scores)
#         return scores


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


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
        # 计算注意力分数
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1) ** 0.5)  # (batch_size, seq_len, seq_len)
        if aspect_mask is not None:
            attention_scores = attention_scores.masked_fill(aspect_mask.unsqueeze(1).bool(), float('-inf'))  # 掩码
        attention_probs = F.softmax(attention_scores, dim=-1)  # 归一化得到注意力权重
        # 计算加权平均的上下文表示
        context = torch.matmul(attention_probs, v)  # (batch_size, seq_len, dim)
        mean_context = torch.sum(context, dim=1, keepdim=True) / (seq_len ** 0.5)  # (batch_size, 1, dim)
        return context, attention_probs, mean_context


class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(SiameseNetwork, self).__init__()

        # 孪生网络共享的编码器部分
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # 最后一层用于计算文本相似性分数的全连接层
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, text1, text2):
        # 编码文本序列
   #     encoded_text1 = self.encoder(text1)
      #  encoded_text2 = self.encoder(text2)

        # 计算两个文本序列的相似性分数
        similarity_score = torch.sigmoid(self.fc(torch.abs(text1 - text2)))
        return similarity_score


class AMRAttention(nn.Module):
    def __init__(self, amr_dim, text_dim, hidden_dim):
        super(AMRAttention, self).__init__()
        self.amr_linear = nn.Linear(amr_dim, hidden_dim)
        self.text_linear = nn.Linear(text_dim, hidden_dim)
        self.output_linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, amr_matrix, text_hidden, aspect_mask):
        # 将 AMR 矩阵和文本隐藏表示分别进行线性变换
        amr_transformed = self.amr_linear(amr_matrix)
        text_transformed = self.text_linear(text_hidden)
        # 使用点积注意力计算注意力权重
        attention_scores = torch.matmul(amr_transformed, text_transformed.transpose(1, 2))
        attention_weights = F.softmax(attention_scores, dim=0)

        mask_asp = attention_weights.masked_fill(aspect_mask.unsqueeze(1).bool(), float('-inf'))

        # 将注意力权重应用到文本隐藏表示上
        attended_text = torch.matmul(attention_weights.transpose(1, 2), text_hidden)

        # 将注意力加权的文本表示进行线性变换
        output = self.output_linear(attended_text)

        return output, attention_weights, mask_asp


class BiAttention(nn.Module):
    def __init__(self, amr_dim, text_dim, hidden_dim):
        super(BiAttention, self).__init__()
        self.amr_linear = nn.Linear(amr_dim, hidden_dim)
        self.text_linear = nn.Linear(text_dim, hidden_dim)
        self.output_linear = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(self, amr_matrix, text_hidden):
        # 将 AMR 矩阵和文本隐藏表示分别进行线性变换
        amr_transformed = self.amr_linear(amr_matrix)
        text_transformed = self.text_linear(text_hidden)

        # 使用点积注意力计算 AMR 对文本的注意力权重
        amr_attention_scores = torch.matmul(amr_transformed, text_transformed.transpose(1, 2))
        amr_attention_weights = F.softmax(amr_attention_scores, dim=0)

        # 使用点积注意力计算文本对 AMR 的注意力权重
        text_attention_scores = torch.matmul(text_transformed, amr_transformed.transpose(1, 2))
        text_attention_weights = F.softmax(text_attention_scores, dim=0)

        # 将注意力权重应用到对应的表示上
        attended_amr = torch.matmul(amr_attention_weights.transpose(1, 2), amr_transformed)
        attended_text = torch.matmul(text_attention_weights.transpose(1, 2), text_hidden)

        # 将注意力加权的 AMR 表示和文本表示进行合并
        combined_representation = torch.cat((attended_amr, attended_text), dim=-1)

        # 将合并后的表示进行线性变换
        output = self.output_linear(combined_representation)

        return output, amr_attention_weights, text_attention_weights


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
