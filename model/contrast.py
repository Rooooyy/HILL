import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss

from transformers import AutoTokenizer
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertEncoder
from transformers.file_utils import ModelOutput
from transformers import BertModel

from torch_geometric.utils import to_dense_adj
from torch_geometric.loader import DataLoader

from model.graph import GraphEncoder
from model.hill import HRLEncoder, GTData, GNNEncoder
from model.coding_tree import get_tree_data


class BertPoolingLayer(nn.Module):
    def __init__(self, avg='cls'):
        super(BertPoolingLayer, self).__init__()
        self.avg = avg

    def forward(self, x):
        if self.avg == 'cls':
            x = x[:, 0, :]
        else:
            x = x.mean(dim=1)
        return x


class BertOutputLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x):
        x = self.dense(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class NTXent(nn.Module):

    def __init__(self, config, tau=1., transform=True):
        super(NTXent, self).__init__()
        self.tau = tau
        self.transform = transform
        self.norm = 1.
        if transform:
            self.transform = nn.Sequential(
                nn.Dropout(config.hidden_dropout_prob),
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.ReLU(),
                nn.Dropout(config.hidden_dropout_prob),
                nn.Linear(config.hidden_size, config.hidden_size),
            )
        else:
            self.transform = None

    def forward(self, x):
        if self.transform:
            x = self.transform(x)  # original hgclr
        n = x.shape[0]
        x = F.normalize(x, p=2, dim=1) / np.sqrt(self.tau)
        # 2B * 2B
        sim = x @ x.t()
        sim[np.arange(n), np.arange(n)] = -1e9

        logprob = F.log_softmax(sim, dim=1)

        m = 2

        labels = (np.repeat(np.arange(n), m) + np.tile(np.arange(m) * n // m, n)) % n
        # remove labels pointet to itself, i.e. (i, i)
        labels = labels.reshape(n, m)[:, 1:].reshape(-1)
        loss = -logprob[np.repeat(np.arange(n), m - 1), labels].sum() / n / (m - 1) / self.norm

        return loss


class BaseModelOutputWithPoolingAndCrossAttentions(ModelOutput):
    last_hidden_state = None
    pooler_output = None
    hidden_states = None
    past_key_values = None
    attentions = None
    cross_attentions = None
    input_embeds = None


class ContrastEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

    def forward(
            self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0,
            embedding_weight=None,
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length: seq_length + past_key_values_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        if embedding_weight is not None:
            if len(embedding_weight.size()) == 2:
                embedding_weight = embedding_weight.unsqueeze(-1)
            inputs_embeds = inputs_embeds * embedding_weight
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings, inputs_embeds


class ContrastBertModel(BertPreTrainedModel):

    _keys_to_ignore_on_load_missing = [r"position_ids"]

    # Copied from transformers.models.bert.modeling_bert.BertModel.__init__
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = ContrastEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = None
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # Copied from transformers.models.bert.modeling_bert.ContrastBertModel.forward
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            embedding_weight=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``: ``1`` for
            tokens that are NOT MASKED, ``0`` for MASKED tokens.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if not self.config.is_decoder:
            use_cache = False

        # input_ids = input_ids[0]
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output, inputs_embeds = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
            embedding_weight=embedding_weight,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output, inputs_embeds) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
            inputs_embeds=inputs_embeds,
        )


class BertAndGraphModel(BertPreTrainedModel):
    def __init__(self, config, local_config):
        super(BertAndGraphModel, self).__init__(config)
        self.config = config
        self.local_config = local_config
        self.num_labels = config.num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(config.name_or_path)
        self.text_drop = nn.Dropout(config.hidden_dropout_prob)
        self.bert = BertModel(config)
        self.bert_pool = BertPoolingLayer('cls')
        self.structure_encoder = None

        # Parse edge list of label hierarchy
        label_hier = torch.load(os.path.join(self.local_config.data_dir, self.local_config.dataset, 'slot.pt'))
        path_dict = {}
        for s in label_hier:
            for v in label_hier[s]:
                path_dict[v] = s
        self.edge_list = [[v, i] for v, i in path_dict.items()]
        self.edge_list += [[i, v] for v, i in path_dict.items()]
        self.edge_list = torch.tensor(self.edge_list).transpose(0, 1)
        # Graph Data
        self.graph = GTData(x=None, edge_index=self.edge_list)

        self.trans_dup = nn.Sequential(nn.Linear(config.num_labels, config.num_labels),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(config.num_labels, config.num_labels),
                                        nn.Dropout(p=local_config.structure_encoder.trans_dp)
                                     )
        # For label attention
        if local_config.label:
            self.label_type = local_config.label_type
            self.label_dict = torch.load(os.path.join(local_config.data_dir, local_config.dataset, 'bert_value_dict.pt'))
            self.label_dict = {i: self.tokenizer.decode(v) for i, v in self.label_dict.items()}
            self.label_name = []
            for i in range(len(self.label_dict)):
                self.label_name.append(self.label_dict[i])
            self.label_name = self.tokenizer(self.label_name, padding='longest')['input_ids']
            self.label_name = nn.Parameter(torch.tensor(self.label_name, dtype=torch.long), requires_grad=False)

            self.attn = FusionLayer1(config, local_config)
            self.label_embeddings = nn.Embedding(config.num_labels, config.hidden_size)

            self.trans_proj = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(config.hidden_size, config.hidden_size),
                                            nn.Dropout(p=local_config.structure_encoder.trans_dp)
                                            )
        else:
            self.trans_proj = nn.Sequential(nn.Linear(config.hidden_size, local_config.hidden_dim),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(local_config.hidden_dim, local_config.hidden_dim),
                                       nn.Dropout(p=local_config.structure_encoder.trans_dp)
                                       )
        self.init_weights()

    def batch_duplicate(self, text_embeds, repeats=None):
        if repeats is None:
            rep = self.num_labels
        text_embeds = text_embeds.unsqueeze(1)
        text_embeds = torch.repeat_interleave(text_embeds, repeats=rep, dim=1)
        text_embeds = self.trans_proj(text_embeds)
        text_embeds = torch.transpose(text_embeds, -1, -2)
        text_embeds = self.trans_dup(text_embeds)
        text_embeds = torch.transpose(text_embeds, -1, -2)
        return text_embeds

    def align_graph(self, embeds, batch_size):
        batch_graph = [copy.deepcopy(self.graph) for _ in range(batch_size)]
        for i in range(batch_size):
            batch_graph[i].x = embeds[i]
        return batch_graph

    @staticmethod
    def extract_local_hierarchy(node_embeds, labels, node_mask):
        return torch.where(labels.unsqueeze(-1) == 0, node_mask.expand_as(node_embeds), node_embeds)

    # -----------------Freezing--------------------------
    @staticmethod
    def __children(module):
        return module if isinstance(module, (list, tuple)) else list(module.children())

    def __apply_leaf(self, module, func):
        c = self.__children(module)
        if isinstance(module, nn.Module):
            func(module)
        if len(c) > 0:
            for leaf in c:
                self.__apply_leaf(leaf, func)

    def __set_trainable(self, module, flag):
        self.__apply_leaf(module, lambda m: self.__set_trainable_attr(m, flag))

    @staticmethod
    def __set_trainable_attr(module, flag):
        module.trainable = flag
        for p in module.parameters():
            p.requires_grad = flag

    def freeze(self):
        self.__set_trainable(self.bert, False)  # freeze all params in bert

    def unfreeze_all(self):
        self.__set_trainable(self.bert, True)

    def unfreeze(self, start_layer, end_layer, pooler=True):
        """
        # Unfreeze the params in bert.encoder ranged from $[start_layer, end_layer]$
        while keeping other parameters in bert freeze.
        # You can unfreeze the last layer of bert by calling set_trainable(model.bert.encoder.layer[23], True)
        """
        self.__set_trainable(self.bert, False)
        for i in range(start_layer, end_layer+1):
            self.__set_trainable(self.bert.encoder.layer[i], True)
        if self.bert.pooler is not None:
            self.__set_trainable(self.bert.pooler, pooler)


class BertAndCodingTreeModel(BertAndGraphModel):
    def __init__(self, config, local_config):
        super(BertAndCodingTreeModel, self).__init__(config, local_config)
        # Coding tree
        self.tree, self.fb_keys = self.construct_coding_tree()
        self.init_weights()

    def construct_coding_tree(self):
        tree = GTData(x=None, edge_index=self.edge_list)
        adj = to_dense_adj(self.edge_list, max_num_nodes=self.num_labels).squeeze(0)
        nodeSize, edgeSize, edgeMat = get_tree_data(adj, self.local_config.tree_depth)
        tree.treeNodeSize = torch.LongTensor(nodeSize).view(1, -1)
        for layer in range(1, self.local_config.tree_depth+1):
            tree['treePHLayer%s' % layer] = torch.ones([nodeSize[layer], 1])  # place holder
            tree['treeEdgeMatLayer%s' % layer] = torch.LongTensor(edgeMat[layer]).T
        fb_keys = [key for key in tree.keys if key.find('treePHLayer') >= 0]
        return tree, fb_keys

    def align_tree(self, embeds, batch_size):
        batch_tree = [copy.deepcopy(self.tree) for _ in range(batch_size)]
        for i in range(batch_size):
            batch_tree[i].x = embeds[i]
        return batch_tree


class StructureContrast(BertAndCodingTreeModel):
    def __init__(self, config, local_config):
        """
        HILL: Hierarchy-aware Information Lossless contrastive Learning
        """
        super(StructureContrast, self).__init__(config, local_config)
        self.contrastive_lossfct = NTXent(config, tau=local_config.contrast.tau, transform=False)
        self.cls_loss = local_config.cls_loss  # Whether to use classification loss
        self.contrast_loss = local_config.contrast_loss  # Whether to use contrastive loss
        self.multi_label = local_config.multi_label
        self.lamda = local_config.lamda  # weight of contrastive loss

        self.structure_encoder = HRLEncoder(local_config)
        self.output_type = local_config.hrl_output
        hidden_size = config.hidden_size if not self.output_type == 'concat' else config.hidden_size * 2
        self.classifier = nn.Linear(hidden_size, config.num_labels)  # structure_encoder.output_dim := hidden_size

        self.text_proj = nn.Sequential(nn.Linear(config.hidden_size, local_config.contrast.proj_dim),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(local_config.contrast.proj_dim, local_config.contrast.proj_dim)
                                       )
        self.tree_proj = nn.Sequential(nn.Linear(local_config.structure_encoder.output_dim, local_config.contrast.proj_dim),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(local_config.contrast.proj_dim, local_config.contrast.proj_dim)
                                       )
        self.node_mask = nn.Parameter(torch.randn(local_config.hidden_dim))
        self.init_weights()  # Warning! This is NOT training BERT from scratch

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        pooled_cls_embed = self.text_drop(self.bert_pool(hidden_states))

        batch_size = hidden_states.shape[0]

        loss = 0
        contrast_logits = None

        text_embeds = self.batch_duplicate(pooled_cls_embed)
        batch_tree = self.align_tree(text_embeds, batch_size)
        tree_loader = DataLoader(batch_tree, batch_size=batch_size, follow_batch=self.fb_keys)
        contrast_output = self.structure_encoder(next(iter(tree_loader)))

        if self.output_type == 'tree':
            logits = self.classifier(contrast_output)  # hill
        elif self.output_type == 'residual':
            logits = self.classifier(pooled_cls_embed + contrast_output)  # hill + bert
        elif self.output_type == 'concat':
            logits = self.classifier(torch.cat([pooled_cls_embed, contrast_output], dim=1))  # [bert, hill]
        else:
            logits = self.classifier(pooled_cls_embed)  # bert

        if labels is not None:
            if self.training:
                if not self.multi_label:
                    loss_fct = CrossEntropyLoss()
                    target = labels.view(-1)
                else:
                    loss_fct = nn.BCEWithLogitsLoss()
                    target = labels.to(torch.float32)
                if self.cls_loss:
                    loss += loss_fct(logits.view(-1, self.num_labels), target)
                if self.contrast_loss:
                    contrastive_loss = self.contrastive_lossfct(
                        torch.cat([self.text_proj(pooled_cls_embed), self.tree_proj(contrast_output)], dim=0),)
                    loss += contrastive_loss * self.lamda

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions,
            'contrast_logits': contrast_logits,
        }

class GraphContrast(BertAndGraphModel):
    def __init__(self, config, local_config):
        """
        Ablation model for r.p. HRLEncoder with GIN/GAT/GCN
        """
        super(GraphContrast, self).__init__(config, local_config)
        self.contrastive_lossfct = NTXent(config, tau=local_config.contrast.tau, transform=False)
        self.cls_loss = local_config.cls_loss  # Whether to use classification loss
        self.contrast_loss = local_config.contrast_loss  # Whether to use contrastive loss
        self.multi_label = local_config.multi_label
        self.lamda = local_config.lamda  # weight of contrastive loss

        self.structure_encoder = GNNEncoder(local_config)
        self.output_type = local_config.hrl_output
        hidden_size = config.hidden_size if not self.output_type == 'concat' else config.hidden_size * 2
        self.classifier = nn.Linear(hidden_size, config.num_labels)  # structure_encoder.output_dim := hidden_size

        self.text_proj = nn.Sequential(nn.Linear(config.hidden_size, local_config.contrast.proj_dim),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(local_config.contrast.proj_dim, local_config.contrast.proj_dim)
                                       )
        self.graph_proj = nn.Sequential(nn.Linear(local_config.structure_encoder.output_dim, local_config.contrast.proj_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(local_config.contrast.proj_dim, local_config.contrast.proj_dim)
                                        )

        self.init_weights()  # Warning! This is NOT training BERT from scratch

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        pooled_cls_embed = self.text_drop(self.bert_pool(hidden_states))

        batch_size = hidden_states.shape[0]

        loss = 0
        contrast_logits = None

        text_embeds = self.batch_duplicate(pooled_cls_embed)
        batch_graph = self.align_graph(text_embeds, batch_size)
        graph_loader = DataLoader(batch_graph, batch_size=batch_size)
        contrast_output = self.structure_encoder(next(iter(graph_loader)))

        if self.output_type == 'tree':
            logits = self.classifier(contrast_output)  # gnn
        elif self.output_type == 'residual':
            logits = self.classifier(pooled_cls_embed + contrast_output)  # gnn + bert
        elif self.output_type == 'concat':
            logits = self.classifier(torch.cat([pooled_cls_embed, contrast_output], dim=1))  # [bert, gnn]
        else:
            logits = self.classifier(pooled_cls_embed)  # bert

        if labels is not None:
            if self.training:
                if not self.multi_label:
                    loss_fct = CrossEntropyLoss()
                    target = labels.view(-1)
                else:
                    loss_fct = nn.BCEWithLogitsLoss()
                    target = labels.to(torch.float32)

                if self.cls_loss:
                    loss += loss_fct(logits.view(-1, self.num_labels), target)
                if self.contrast_loss:
                    contrastive_loss = self.contrastive_lossfct(
                        torch.cat([self.text_proj(pooled_cls_embed), self.graph_proj(contrast_output)], dim=0), )
                    loss += contrastive_loss * self.lamda

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions,
            'contrast_logits': contrast_logits,
        }

class ContrastModel(BertPreTrainedModel):
    def __init__(self, config, local_config):
        """
        Vanilla HGCLR
        """
        super(ContrastModel, self).__init__(config)
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.bert = ContrastBertModel(config)
        self.pooler = BertPoolingLayer('cls')
        self.contrastive_lossfct = NTXent(config, tau=local_config.structure_encoder.tau)
        self.cls_loss = local_config.cls_loss
        self.contrast_loss = local_config.contrast_loss
        self.token_classifier = BertOutputLayer(config)

        self.graph_encoder = GraphEncoder(config, graph=local_config.graph,
                                          layer=local_config.structure_encoder.layer,
                                          data_path=os.path.join(local_config.data_dir, local_config.dataset),
                                          threshold=local_config.structure_encoder.thre,
                                          tau=local_config.structure_encoder.tau)
        self.lamb = local_config.lamda
        self.init_weights()
        self.multi_label = local_config.multi_label

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        contrast_mask = None

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            embedding_weight=contrast_mask,

        )
        pooled_output = outputs[0]
        pooled_output = self.dropout(self.pooler(pooled_output))

        loss = 0
        contrastive_loss = None
        contrast_logits = None

        logits = self.classifier(pooled_output)
        # logits = self.token_classifier(pooled_output)

        if labels is not None:
            if not self.multi_label:
                loss_fct = CrossEntropyLoss()
                target = labels.view(-1)
            else:
                loss_fct = nn.BCEWithLogitsLoss()
                target = labels.to(torch.float32)

            if self.cls_loss:
                if self.num_labels == 1:
                    #  We are doing regression
                    loss_fct = MSELoss()
                    loss += loss_fct(logits.view(-1), labels.view(-1))
                else:
                    loss += loss_fct(logits.view(-1, self.num_labels), target)

            if self.training:
                contrast_mask = self.graph_encoder(outputs['inputs_embeds'],
                                                   attention_mask, labels, lambda x: self.bert.embeddings(x))

                contrast_output = self.bert(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=None,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    embedding_weight=contrast_mask,
                )
                contrast_sequence_output = self.dropout(self.pooler(contrast_output[0]))
                contrast_logits = self.classifier(contrast_sequence_output)
                # contrast_logits = self.token_classifier(contrast_sequence_output)
                contrastive_loss = self.contrastive_lossfct(
                    torch.cat([pooled_output, contrast_sequence_output], dim=0), )

                loss += loss_fct(contrast_logits.view(-1, self.num_labels), target) \

            if contrastive_loss is not None and self.contrast_loss:
                loss += contrastive_loss * self.lamb

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions,
            'contrast_logits': contrast_logits,
        }