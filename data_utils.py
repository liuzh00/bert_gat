import os
import sys

sys.path.append(r'./Parser/src_joint')
import re
import torch
import json
import pickle
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer
from torch.utils.data import Dataset
from collections import defaultdict


def ParseData(data_path):
    """Parse row data"""
    with open(data_path) as infile:
        all_data = []
        data = json.load(infile)
        for d in data:
            for aspect in d['aspects']:
                text_list = list(d['token'])
                tok = list(d['token'])  # word token
                length = len(tok)  # real length
                # if args.lower == True:
                tok = [t.lower() for t in tok]
                tok = ' '.join(tok)
                asp = list(aspect['term'])  # aspect
                asp = [a.lower() for a in asp]
                asp = ' '.join(asp)
                label = aspect['polarity']  # label
                pos = list(d['pos'])  # pos_tag
                head = list(d['head'])  # head
                deprel = list(d['deprel'])  # deprel

                # position
                aspect_post = [aspect['from'], aspect['to']]
                post = [i - aspect['from'] for i in range(aspect['from'])] \
                       + [0 for _ in range(aspect['from'], aspect['to'])] \
                       + [i - aspect['to'] + 1 for i in range(aspect['to'], length)]
                # aspect mask
                if len(asp) == 0:
                    mask = [1 for _ in range(length)]  # for rest16
                else:
                    mask = [0 for _ in range(aspect['from'])] \
                           + [1 for _ in range(aspect['from'], aspect['to'])] \
                           + [0 for _ in range(aspect['to'], length)]

                sample = {'text': tok, 'aspect': asp, 'pos': pos, 'post': post, 'head': head, \
                          'deprel': deprel, 'length': length, 'label': label, 'mask': mask, \
                          'aspect_post': aspect_post, 'text_list': text_list}
                all_data.append(sample)

    return all_data


def build_tokenizer(fnames, max_length, data_file):
    parse = ParseData
    if os.path.exists(data_file):
        print('loading tokenizer:', data_file)
        tokenizer = pickle.load(open(data_file, 'rb'))
    else:
        tokenizer = Tokenizer.from_files(fnames=fnames, max_length=max_length, parse=parse)
        pickle.dump(tokenizer, open(data_file, 'wb'))
    return tokenizer


class Vocab(object):
    ''' vocabulary of dataset '''

    def __init__(self, vocab_list, add_pad, add_unk):
        self._vocab_dict = dict()
        self._reverse_vocab_dict = dict()
        self._length = 0
        if add_pad:
            self.pad_word = '<pad>'
            self.pad_id = self._length
            self._length += 1
            self._vocab_dict[self.pad_word] = self.pad_id
        if add_unk:
            self.unk_word = '<unk>'
            self.unk_id = self._length
            self._length += 1
            self._vocab_dict[self.unk_word] = self.unk_id
        for w in vocab_list:
            self._vocab_dict[w] = self._length
            self._length += 1
        for w, i in self._vocab_dict.items():
            self._reverse_vocab_dict[i] = w

    def word_to_id(self, word):
        if hasattr(self, 'unk_id'):
            return self._vocab_dict.get(word, self.unk_id)
        return self._vocab_dict[word]

    def id_to_word(self, id_):
        if hasattr(self, 'unk_word'):
            return self._reverse_vocab_dict.get(id_, self.unk_word)
        return self._reverse_vocab_dict[id_]

    def has_word(self, word):
        return word in self._vocab_dict

    def __len__(self):
        return self._length

    @staticmethod
    def load_vocab(vocab_path: str):
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

    def save_vocab(self, vocab_path):
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)


class Tokenizer(object):
    ''' transform text to indices '''

    def __init__(self, vocab, max_length, lower, pos_char_to_int, pos_int_to_char):
        self.vocab = vocab
        self.max_length = max_length
        self.lower = lower

        self.pos_char_to_int = pos_char_to_int
        self.pos_int_to_char = pos_int_to_char

    @classmethod
    def from_files(cls, fnames, max_length, parse, lower=True):
        corpus = set()
        pos_char_to_int, pos_int_to_char = {}, {}
        for fname in fnames:
            for obj in parse(fname):
                text_raw = obj['text']
                if lower:
                    text_raw = text_raw.lower()
                corpus.update(Tokenizer.split_text(text_raw))
        return cls(vocab=Vocab(corpus, add_pad=True, add_unk=True), max_length=max_length, lower=lower,
                   pos_char_to_int=pos_char_to_int, pos_int_to_char=pos_int_to_char)

    @staticmethod
    def pad_sequence(sequence, pad_id, maxlen, dtype='int64', padding='post', truncating='post'):
        x = (np.zeros(maxlen) + pad_id).astype(dtype)
        if truncating == 'pre':
            trunc = sequence[-maxlen:]
        else:
            trunc = sequence[:maxlen]
        trunc = np.asarray(trunc, dtype=dtype)
        if padding == 'post':
            x[:len(trunc)] = trunc
        else:
            x[-len(trunc):] = trunc
        return x

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        if not isinstance(text, list):
            text = Tokenizer.split_text(text)
        if self.lower:
            text = [w.lower() for w in text]
        sequence = [self.vocab.word_to_id(w) for w in text]
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence.reverse()
        return Tokenizer.pad_sequence(sequence, pad_id=self.vocab.pad_id, maxlen=self.max_length,
                                      padding=padding, truncating=truncating)

    @staticmethod
    def split_text(text):
        # for ch in ["\'s", "\'ve", "n\'t", "\'re", "\'m", "\'d", "\'ll", ",", ".", "!", "*", "/", "?", "(", ")", "\"", "-", ":"]:
        #     text = text.replace(ch, " "+ch+" ")
        return text.strip().split()


class ABSAData(Dataset):
    ''' PyTorch standard dataset class '''

    def __init__(self, fname, tokenizer, opt, vocab_help):

        parse = ParseData
        post_vocab, deppost_vocab, pos_vocab, dep_vocab, pol_vocab = vocab_help
        data = list()
        polarity_dict = {'positive': 0, 'negative': 1, 'neutral': 2}
        for obj in parse(fname):
            text = tokenizer.text_to_sequence(obj['text_list'])
            aspect = tokenizer.text_to_sequence(obj['aspect'])  # max_length=10
            post = [post_vocab.stoi.get(t, post_vocab.unk_index) for t in obj['post']]
            post = tokenizer.pad_sequence(post, pad_id=opt.pad_id, maxlen=opt.max_length, dtype='int64', padding='post',
                                          truncating='post')
            dep_post = self.build_dependent_position_adj(tokenizer, obj['text_list'], obj['head'], opt)
            dep_post = []
            pos = [pos_vocab.stoi.get(t, pos_vocab.unk_index) for t in obj['pos']]
            pos = tokenizer.pad_sequence(pos, pad_id=opt.pad_id, maxlen=opt.max_length, dtype='int64', padding='post',
                                         truncating='post')
            deprel = [dep_vocab.stoi.get(t, dep_vocab.unk_index) for t in obj['deprel']]
            deprel = tokenizer.pad_sequence(deprel, pad_id=opt.pad_id, maxlen=opt.max_length, dtype='int64',
                                            padding='post', truncating='post')
            mask = tokenizer.pad_sequence(obj['mask'], pad_id=opt.pad_id, maxlen=opt.max_length, dtype='int64',
                                          padding='post', truncating='post')

            adj = np.ones(opt.max_length) * opt.pad_id
            if opt.parseadj:
                from  LALParser.src_joint.absa_parser  import headparser
                # * adj
                headp, syntree = headparser.parse_heads(obj['text'])
                adj = softmax(headp[0])
                adj = np.delete(adj, 0, axis=0)
                adj = np.delete(adj, 0, axis=1)
                adj -= np.diag(np.diag(adj))
                if not opt.direct:
                    adj = adj + adj.T
                adj = adj + np.eye(adj.shape[0])
                adj = np.pad(adj, (0, opt.max_length - adj.shape[0]), 'constant')

            if opt.parsehead:
                from LALParser.src_joint.absa_parser import headparser
                headp, syntree = headparser.parse_heads(obj['text'])
                syntree2head = [[leaf.father for leaf in tree.leaves()] for tree in syntree]
                head = tokenizer.pad_sequence(syntree2head[0], pad_id=opt.pad_id, maxlen=opt.max_length, dtype='int64',
                                              padding='post', truncating='post')
            else:
                head = tokenizer.pad_sequence(obj['head'], pad_id=opt.pad_id, maxlen=opt.max_length, dtype='int64',
                                              padding='post', truncating='post')
            length = obj['length']
            polarity = polarity_dict[obj['label']]
            data.append({
                'text': text,
                'aspect': aspect,
                'post': post,
                'dep_post': dep_post,
                'pos': pos,
                'deprel': deprel,
                'head': head,
                'adj': adj,
                'mask': mask,
                'length': length,
                'polarity': polarity
            })

        self._data = data

    def __getitem__(self, index):
        return self._data[index]

    def __len__(self):
        return len(self._data)

    def build_dependent_position_adj(self, tokenizer, tokens, head, opt):
        """
        构建句法依赖距离邻接矩阵
        """
        token_range = []
        token_start = 0
        for i, w, in enumerate(tokens):
            token_end = token_start + len([tokenizer.vocab.word_to_id(w)])
            token_range.append([token_start, token_end-1])
            token_start = token_end

        word_pair_deppost = torch.zeros(opt.max_length, opt.max_length).long()
        tmp = [[0] * len(tokens) for _ in range(len(tokens))]
        for i in range(len(tokens)):
            j = head[i]
            if j == 0:
                continue
            tmp[i][j - 1] = 1
            tmp[j - 1][i] = 1

        tmp_dict = defaultdict(list)
        for i in range(len(tokens)):
            for j in range(len(tokens)):
                if tmp[i][j] == 1:
                    tmp_dict[i].append(j)

        word_level_degree = [[4] * len(tokens) for _ in range(len(tokens))]

        for i in range(len(tokens)):
            node_set = set()
            word_level_degree[i][i] = 0
            node_set.add(i)
            for j in tmp_dict[i]:
                if j not in node_set:
                    word_level_degree[i][j] = 1
                    node_set.add(j)
                for k in tmp_dict[j]:
                    if k not in node_set:
                        word_level_degree[i][k] = 2
                        node_set.add(k)
                        for g in tmp_dict[k]:
                            if g not in node_set:
                                word_level_degree[i][g] = 3
                                node_set.add(g)

        for i in range(len(tokens)):
            start, end = token_range[i][0], token_range[i][1]
            for j in range(len(tokens)):
                s, e = token_range[j][0], token_range[j][1]
                for row in range(start, end + 1):
                    for col in range(s, e + 1):
                        word_pair_deppost[row][col] = word_level_degree[i][j]

        return word_pair_deppost

    def dep_position_weight(self, input, adj, dep_post_adj, aspect_mask, maxlen):
        """
        句法依存距离权重
        """
        weight = torch.zeros_like(adj)
        for i in range(input):  # 构建句法依存距离权重矩阵
            max_distance = dep_post_adj[i].max().item()  # 样本最大句法依赖距离
            for j in range(maxlen):
                row_aspect = (aspect_mask[i][j] == 1).item()  # 判断该行词是否方面词
                for t in range(maxlen):
                    col_aspect = (aspect_mask[i][t] == 1).item()  # 判断该列词是否方面词
                    # 行是方面词且和列有依赖
                    if row_aspect and adj[i][j][t] == 1:  # 若行是方面词且有依赖关系
                        weight[i][j][t] = 1 - dep_post_adj[i][j][t] / (max_distance + 1)
                    # 列是方面词且与行有依赖
                    if col_aspect and adj[i][j][t] == 1:  # 若列是方面词且有依赖关系
                        weight[i][j][t] = 1 - dep_post_adj[i][j][t] / (max_distance + 1)
        adj = adj + weight  # A = A * (L + 1)
        padding = -9e15 * torch.ones_like(adj)
        adj = torch.where(adj > 0, adj, padding)
        return adj  # srd = A * (L + 1)

class ABSABertData(Dataset):
    def __init__(self, fname, tokenizer, opt, vocab_help=None):
        self.data = []

        parse = ParseData
        polarity_dict = {'positive': 0, 'negative': 1, 'neutral': 2}


        for obj in parse(fname):
            polarity = polarity_dict[obj['label']]
            text = obj['text']
            term = obj['aspect']
            term_start = obj['aspect_post'][0]
            term_end = obj['aspect_post'][1]
            text_list = obj['text_list']
            head = list(obj["head"])
            head = [int(x) for x in head]
            length = obj['length']

         #   pos = tokenizer.pad_sequence(pos, opt.pad_id, opt.max_length)
            left, term, right = text_list[: term_start], text_list[term_start: term_end], text_list[term_end:]



            left_tokens, term_tokens, right_tokens = [], [], []
            left_tok2ori_map, term_tok2ori_map, right_tok2ori_map = [], [], []

            for ori_i, w in enumerate(left):
                for t in tokenizer.tokenize(w):
                    left_tokens.append(t)  # * ['expand', '##able', 'highly', 'like', '##ing']
                    left_tok2ori_map.append(ori_i)  # * [0, 0, 1, 2, 2]
            asp_start = len(left_tokens)
            offset = len(left)
            for ori_i, w in enumerate(term):
                for t in tokenizer.tokenize(w):
                    term_tokens.append(t)
                    # term_tok2ori_map.append(ori_i)
                    term_tok2ori_map.append(ori_i + offset)
            asp_end = asp_start + len(term_tokens)
            offset += len(term)
            for ori_i, w in enumerate(right):
                for t in tokenizer.tokenize(w):
                    right_tokens.append(t)
                    right_tok2ori_map.append(ori_i + offset)

            while len(left_tokens) + len(right_tokens) > tokenizer.max_seq_len - 2 * len(term_tokens) - 3:
                if len(left_tokens) > len(right_tokens):
                    left_tokens.pop(0)
                    left_tok2ori_map.pop(0)
                else:
                    right_tokens.pop()
                    right_tok2ori_map.pop()

            bert_tokens = left_tokens + term_tokens + right_tokens
            tok2ori_map = left_tok2ori_map + term_tok2ori_map + right_tok2ori_map
            truncate_tok_len = len(bert_tokens)
            tok_adj = np.zeros(
                (truncate_tok_len, truncate_tok_len), dtype='float32')


            bert_head = [head[idx] for idx in tok2ori_map]  # head

            tokens = tokenizer.convert_tokens_to_ids(bert_tokens)

            aspect_tokens = tokenizer.convert_tokens_to_ids(term_tokens)
            context_asp_ids = [tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(
                bert_tokens) + [tokenizer.sep_token_id] + tokenizer.convert_tokens_to_ids(term_tokens) + [
                                  tokenizer.sep_token_id]
            context_asp_len = len(context_asp_ids)
            paddings = [0] * (tokenizer.max_seq_len - context_asp_len)
            context_len = len(bert_tokens)
            context_asp_seg_ids = [0] * (1 + context_len + 1) + [1] * (len(term_tokens) + 1) + paddings
            src_mask = [0] + [1] * context_len + [0] * (opt.max_length - context_len - 1)
            aspect_mask = [0] + [0] * asp_start + [1] * (asp_end - asp_start)
            aspect_mask = aspect_mask + (opt.max_length - len(aspect_mask)) * [0]
            context_asp_attention_mask = [1] * context_asp_len + paddings
            context_asp_ids += paddings
            context_asp_ids = np.asarray(context_asp_ids, dtype='int64')
            context_asp_seg_ids = np.asarray(context_asp_seg_ids, dtype='int64')
            context_asp_attention_mask = np.asarray(context_asp_attention_mask, dtype='int64')
            src_mask = np.asarray(src_mask, dtype='int64')
            aspect_mask = np.asarray(aspect_mask, dtype='int64')

            tokens = tokens + [0] * (opt.max_length - truncate_tok_len)
            tokens = np.asarray(tokens, dtype='int64')
            text_bert_indices = tokens
            bert_head = bert_head + [0] * (opt.max_length - len(bert_head))
            bert_head = np.asarray(bert_head, dtype='int64')


            src_mask = [0] + [1] * context_len + [0] * (opt.max_length - context_len - 1)
            src_mask = np.asarray(src_mask, dtype='int64')

            data = {
                'concat_bert_indices': context_asp_ids,#concat_bert_indices
                'bert_segments_ids': context_asp_seg_ids, #concat_segments_indices
             #   'text_bert_indices': text_bert_indices,
                'attention_mask': context_asp_attention_mask,
                'asp_start': asp_start,
                'asp_end': asp_end,
                'src_mask': src_mask,
                'aspect_mask': aspect_mask,
                'text': tokens,
                'head': bert_head,
                'length': length,
                'polarity': polarity,
           #     'pos':bert_pos,

            }
            self.data.append(data)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    def build_dependent_position_adj(self,tokenizer, tokens, head,opt):
        """
        构建句法依赖距离邻接矩阵
        """
        token_range = []
        token_start = 0
        for i, w, in enumerate(tokens):

            token_end = token_start + len([tokenizer.convert_tokens_to_ids(w)])
            token_range.append([token_start, token_end-1])
            token_start = token_end

        word_pair_deppost = torch.zeros(opt.max_length, opt.max_length).long()
        tmp = [[0] * len(tokens) for _ in range(len(tokens))]
        for i in range(len(tokens)):

            j = head[i]
            if j == 0:
                continue
            tmp[i][j - 1] = 1
            tmp[j - 1][i] = 1

        tmp_dict = defaultdict(list)
        for i in range(len(tokens)):
            for j in range(len(tokens)):
                if tmp[i][j] == 1:
                    tmp_dict[i].append(j)

        word_level_degree = [[4] * len(tokens) for _ in range(len(tokens))]

        for i in range(len(tokens)):
            node_set = set()
            word_level_degree[i][i] = 0
            node_set.add(i)
            for j in tmp_dict[i]:
                if j not in node_set:
                    word_level_degree[i][j] = 1
                    node_set.add(j)
                for k in tmp_dict[j]:
                    if k not in node_set:
                        word_level_degree[i][k] = 2
                        node_set.add(k)
                        for g in tmp_dict[k]:
                            if g not in node_set:
                                word_level_degree[i][g] = 3
                                node_set.add(g)

        for i in range(len(tokens)):
            start, end = token_range[i][0], token_range[i][1]
            for j in range(len(tokens)):
                s, e = token_range[j][0], token_range[j][1]
                for row in range(start, end ):
                    for col in range(s, e ):

                        word_pair_deppost[row][col] = word_level_degree[i][j]

        return word_pair_deppost

def _load_wordvec(data_path, embed_dim, vocab=None):
    with open(data_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        word_vec = dict()
        if embed_dim == 200:
            for line in f:
                tokens = line.rstrip().split()
                if tokens[0] == '<pad>' or tokens[0] == '<unk>':  # avoid them
                    continue
                if vocab is None or vocab.has_word(tokens[0]):
                    word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
        elif embed_dim == 300:
            for line in f:
                tokens = line.rstrip().split()
                if tokens[0] == '<pad>':  # avoid them
                    continue
                elif tokens[0] == '<unk>':
                    word_vec['<unk>'] = np.random.uniform(-0.25, 0.25, 300)
                word = ''.join((tokens[:-300]))
                if vocab is None or vocab.has_word(tokens[0]):
                    word_vec[word] = np.asarray(tokens[-300:], dtype='float32')
        else:
            print("embed_dim error!!!")
            exit()

        return word_vec


def build_embedding_matrix(vocab, embed_dim, data_file, glove_dir):
    if os.path.exists(data_file):
        print('loading embedding matrix:', data_file)
        embedding_matrix = pickle.load(open(data_file, 'rb'))
    else:
        print('loading word vectors...')
        embedding_matrix = np.zeros((len(vocab), embed_dim))
        word_vec = _load_wordvec(glove_dir, embed_dim, vocab)
        for i in range(len(vocab)):
            vec = word_vec.get(vocab.id_to_word(i))
            if vec is not None:
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(data_file, 'wb'))
    return embedding_matrix


def softmax(x):
    if len(x.shape) > 1:
        # matrix
        tmp = np.max(x, axis=1)
        x -= tmp.reshape((x.shape[0], 1))
        x = np.exp(x)
        tmp = np.sum(x, axis=1)
        x /= tmp.reshape((x.shape[0], 1))
    else:
        # vector
        tmp = np.max(x)
        x -= tmp
        x = np.exp(x)
        tmp = np.sum(x)
        x /= tmp
    return x


class Tokenizer4BertGCN:
    def __init__(self, max_seq_len, pretrained_bert_name):
        self.max_seq_len = max_seq_len
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id

    def tokenize(self, s):
        return self.tokenizer.tokenize(s)

    def convert_tokens_to_ids(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)


    def pad_sequence(self,sequence, pad_id, maxlen, dtype='int64', padding='post', truncating='post'):
        x = (np.zeros(maxlen) + pad_id).astype(dtype)
        if truncating == 'pre':
            trunc = sequence[-maxlen:]
        else:
            trunc = sequence[:maxlen]
        trunc = np.asarray(trunc, dtype=dtype)
        if padding == 'post':
            x[:len(trunc)] = trunc
        else:
            x[-len(trunc):] = trunc
        return x
