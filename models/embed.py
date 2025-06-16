import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

import math
import pandas as pd
import ast


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[: token_embedding.size(0), :])


class TemporalEmbedding(nn.Module):
    def __init__(self, d_input, emb_info="all"):
        super(TemporalEmbedding, self).__init__()

        self.emb_info = emb_info
        self.minute_size = 4
        hour_size = 24
        weekday = 7

        if self.emb_info == "all":
            self.minute_embed = nn.Embedding(self.minute_size, d_input)
            self.hour_embed = nn.Embedding(hour_size, d_input)
            self.weekday_embed = nn.Embedding(weekday, d_input)
        elif self.emb_info == "time":
            self.time_embed = nn.Embedding(self.minute_size * hour_size, d_input)
        elif self.emb_info == "weekday":
            self.weekday_embed = nn.Embedding(weekday, d_input)

    def forward(self, time, weekday):
        if self.emb_info == "all":
            hour = torch.div(time, self.minute_size, rounding_mode="floor")
            minutes = time % 4

            minute_x = self.minute_embed(minutes)
            hour_x = self.hour_embed(hour)
            weekday_x = self.weekday_embed(weekday)

            return hour_x + minute_x + weekday_x
        elif self.emb_info == "time":
            return self.time_embed(time)
        elif self.emb_info == "weekday":
            return self.weekday_embed(weekday)


class POINet(nn.Module):
    def __init__(self, poi_vector_size, out):
        super(POINet, self).__init__()

        self.buffer_num = 11

        # 11 -> poi_vector_size*2 -> 11
        if self.buffer_num == 11:
            self.linear1 = torch.nn.Linear(self.buffer_num, poi_vector_size * 2)
            self.linear2 = torch.nn.Linear(poi_vector_size * 2, self.buffer_num)
            self.dropout2 = nn.Dropout(p=0.1)
            self.norm1 = nn.LayerNorm(self.buffer_num)

            # 11*poi_vector_size -> poi_vector_size
            self.dense = torch.nn.Linear(self.buffer_num * poi_vector_size, poi_vector_size)
            self.dropout_dense = nn.Dropout(p=0.1)

        # poi_vector_size -> poi_vector_size*4 -> poi_vector_size
        self.linear3 = torch.nn.Linear(poi_vector_size, poi_vector_size * 4)
        self.linear4 = torch.nn.Linear(poi_vector_size * 4, poi_vector_size)
        self.dropout3 = nn.Dropout(p=0.1)
        self.dropout4 = nn.Dropout(p=0.1)
        self.norm2 = nn.LayerNorm(poi_vector_size)

        # poi_vector_size -> out
        self.fc = nn.Linear(poi_vector_size, out)

    def forward(self, x):
        # first
        if self.buffer_num == 11:
            x = self.norm1(x + self._ff_block(x))
        # flat
        x = x.view([x.shape[0], x.shape[1], x.shape[2] * x.shape[3]])
        if self.buffer_num == 11:
            x = self.dropout_dense(F.relu(self.dense(x)))
        # second
        x = self.norm2(x + self._dense_block(x))
        return self.fc(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(F.relu(self.linear1(x)))
        return self.dropout2(x)

    def _dense_block(self, x: Tensor) -> Tensor:
        x = self.linear4(self.dropout3(F.relu(self.linear3(x))))
        return self.dropout4(x)


class CoordEmbedding(nn.Module):
    def __init__(self, spa_embed_dim, coord_dim=2, frequency_num=16,
        max_radius=10000, min_radius=1000, freq_init="geometric") -> None:
        super().__init__()
        # self.embed_dim = embed_dim
        self.frequency_num = frequency_num
        self.coord_dim = coord_dim 
        self.max_radius = max_radius
        self.min_radius = min_radius
        self.spa_embed_dim = spa_embed_dim
        self.freq_init = freq_init

        # the frequence we use for each block, alpha in ICLR paper
        freq_list = _cal_freq_list(self.freq_init, self.frequency_num, self.max_radius, self.min_radius)
        freq_list = torch.tensor(freq_list)
        self.register_buffer("freq_list", freq_list)
        freq_mat = torch.unsqueeze(freq_list, axis=1)
        freq_mat = freq_mat.repeat(1, 6)
        self.register_buffer("freq_mat", freq_mat)
        # self.freq_mat shape: (frequency_num, 6)
        

        # there unit vectors which is 120 degree apart from each other
        unit_vec1 = torch.tensor([1.0, 0.0])                        # 0
        unit_vec2 = torch.tensor([-1.0/2.0, math.sqrt(3)/2.0])      # 120 degree
        unit_vec3 = torch.tensor([-1.0/2.0, -math.sqrt(3)/2.0])     # 240 degree
        self.register_buffer("unit_vec1", unit_vec1)
        self.register_buffer("unit_vec2", unit_vec2)
        self.register_buffer("unit_vec3", unit_vec3)

        self.input_embed_dim = self.cal_input_dim()
        self.ffn = nn.Linear(self.input_embed_dim, self.spa_embed_dim)

    # def cal_freq_list(self):
    # def cal_freq_mat(self):
    #     # freq_mat shape: (frequency_num, 1)
    #     freq_mat = torch.unsqueeze(self.freq_list, axis=1)
    #     # self.freq_mat shape: (frequency_num, 6)
    #     self.freq_mat = freq_mat.repeat(1, 6)
    #     # self.freq_mat = np.repeat(freq_mat, 6, axis = 1)

    def cal_input_dim(self):
        # compute the dimention of the encoded spatial relation embedding
        return int(6 * self.frequency_num)

    def make_input_embeds(self, coords):
        """
        Args:
            coords: a tensor with shape (seq_len, batch_size, coord_dim)
        """
        # if type(coords) == np.ndarray:
        #     assert self.coord_dim == np.shape(coords)[2]
        #     coords = list(coords)
        # elif type(coords) == list:
        #     assert self.coord_dim == len(coords[0][0])
        # else:
        #     raise Exception("Unknown coords data type for GridCellSpatialRelationEncoder")
        
        # # (batch_size, num_context_pt, coord_dim) -> (seq_len, batch_size, coord_dim)
        # coords_mat = np.asarray(coords).astype(float)
        # batch_size = coords_mat.shape[0]
        # num_context_pt = coords_mat.shape[1]

        seq_len, batch_size = coords.shape[0], coords.shape[1]

        # compute the dot product between [deltaX, deltaY] and each unit_vec
        # (batch_size, num_context_pt, 1)
        angle_mat1 = torch.unsqueeze(torch.matmul(coords, self.unit_vec1), axis=-1)
        # (batch_size, num_context_pt, 1)
        angle_mat2 = torch.unsqueeze(torch.matmul(coords, self.unit_vec2), axis=-1)
        # (batch_size, num_context_pt, 1)
        angle_mat3 = torch.unsqueeze(torch.matmul(coords, self.unit_vec3), axis=-1)

        # (batch_size, num_context_pt, 6)
        # angle_mat = np.concatenate([angle_mat1, angle_mat1, angle_mat2, angle_mat2, angle_mat3, angle_mat3], axis = -1)
        angle_mat = torch.stack([angle_mat1, angle_mat1, angle_mat2, angle_mat2, angle_mat3, angle_mat3], dim=-1)
        # (batch_size, num_context_pt, 1, 6)
        angle_mat = torch.unsqueeze(angle_mat, axis=-2)
        # (batch_size, num_context_pt, frequency_num, 6)
        angle_mat = torch.repeat_interleave(angle_mat, self.frequency_num, axis=-2)
        # (batch_size, num_context_pt, frequency_num, 6)
        angle_mat = angle_mat * self.freq_mat
        # (batch_size, num_context_pt, frequency_num*6)
        spr_embeds = torch.reshape(angle_mat, (seq_len, batch_size, -1))

        # make sinuniod function
        # sin for 2i, cos for 2i+1
        # spr_embeds: (batch_size, num_context_pt, frequency_num*6=input_embed_dim)
        spr_embeds[:, :, 0::2] = torch.sin(spr_embeds[:, :, 0::2])  # dim 2i
        spr_embeds[:, :, 1::2] = torch.cos(spr_embeds[:, :, 1::2])  # dim 2i+1
        
        return spr_embeds
    
    def forward(self, coords):
        """
        Given a list of coords (deltaX, deltaY), give their spatial relation embedding
        Args:
            coords: a tensor with shape (seq_len, batch_size, coord_dim)
        Return:
            sprenc: Tensor shape (seq_len, batch_size, spa_embed_dim)
        """
        spr_embeds = self.make_input_embeds(coords)

        # spr_embeds: (batch_size, num_context_pt, input_embed_dim)
        # spr_embeds = torch.FloatTensor(spr_embeds)
        if self.ffn is not None:
            return self.ffn(spr_embeds)
        else:
            return spr_embeds


def _cal_freq_list(freq_init, frequency_num, max_radius, min_radius):
    if freq_init == "random":
        # the frequence we use for each block, alpha in ICLR paper
        # freq_list shape: (frequency_num)
        freq_list = np.random.random(size=[frequency_num]) * max_radius
    elif freq_init == "geometric":
        # freq_list = []
        # for cur_freq in range(frequency_num):
        #     base = 1.0/(np.power(max_radius, cur_freq*1.0/(frequency_num-1)))
        #     freq_list.append(base)

        # freq_list = np.asarray(freq_list)

        log_timescale_increment = (math.log(float(max_radius) / float(min_radius)) /
        (frequency_num*1.0 - 1))

        timescales = min_radius * np.exp(
            np.arange(frequency_num).astype(np.float32) * log_timescale_increment)

        freq_list = 1.0/timescales

    return freq_list


class RoadNetEmbedding(nn.Module):
    def __init__(self, d_input, config):
        super(RoadNetEmbedding, self).__init__()

        # Get the pre-trained road (node) embeddings
        weight = self._get_weight("data/2024/locations_fsq_nyc_with_embeddings.csv")
        weight = torch.FloatTensor(weight)
        self.road_embed = nn.Embedding.from_pretrained(weight)
        self.road_embed.weight.requires_grad = False

        self.project = nn.Linear(weight.shape[1], d_input)

        # print(self.road_embed(torch.LongTensor([0, 1, 2])))
        
    def _get_weight(self, weight_file):
        loc_df = pd.read_csv(weight_file)
        node_embeddings = loc_df["node_embedding"].apply(lambda x: ast.literal_eval(x))
        node_embeddings = np.stack(node_embeddings.values)

        # Add two embeddings at 0 and 1th position to account for the padding and unknown token
        node_embeddings = np.vstack([np.zeros(node_embeddings.shape[1]), np.zeros(node_embeddings.shape[1]), node_embeddings])

        return node_embeddings

    def forward(self, x):
        # print(self.road_embed(torch.LongTensor([0, 1, 2]).to(x.device)))
        road_embeddings = self.road_embed(x)
        road_embeddings = self.project(road_embeddings)
        return road_embeddings


class AllEmbedding(nn.Module):
    def __init__(self, d_input, config, total_loc_num, loc_embbedder, if_pos_encoder=True, emb_info="all", emb_type="add") -> None:
        super(AllEmbedding, self).__init__()
        # embedding layers
        self.d_input = d_input
        self.emb_type = emb_type

        # Add road network embedding
        self.if_include_road = config.if_embed_roadnet
        if self.if_include_road:
            self.emb_road = RoadNetEmbedding(d_input, config)

        # location embedding
        self.loc_embed_method = config.loc_embed_method

        if self.loc_embed_method == "vanilla":
            self.emb_loc = nn.Embedding(total_loc_num, d_input)
        else:
            self.emb_loc = loc_embbedder
            # for param in self.emb_loc.parameters():
            #     param.requires_grad = False
        
        # location type
        self.loc_type = config.loc_type

        # if self.emb_type == "add":
        #     self.emb_loc = nn.Embedding(total_loc_num, d_input)
        # else:
        #     self.emb_loc = nn.Embedding(total_loc_num, d_input - config.time_emb_size)

        # spatial encoding
        self.if_include_spatial = config.if_embed_spatial
        if self.if_include_spatial:
            if self.emb_type == "add":
                self.spatial_embedding = CoordEmbedding(d_input)
            else:
                self.spatial_embedding = CoordEmbedding(config.spatial_emb_size)

        # time is in minutes, possible time for each day is 60 * 24 // 30
        self.if_include_time = config.if_embed_time
        if self.if_include_time:
            if self.emb_type == "add":
                self.temporal_embedding = TemporalEmbedding(d_input, emb_info)
            else:
                self.temporal_embedding = TemporalEmbedding(config.time_emb_size, emb_info)

        # duration is in minutes, possible duration for two days is 60 * 24 * 2// 30
        self.if_include_duration = config.if_embed_duration
        if self.if_include_duration:
            self.emb_duration = nn.Embedding(60 * 24 * 2 // 30, d_input)

        # poi
        self.if_include_poi = config.if_embed_poi
        if self.if_include_poi:
            self.poi_net = POINet(config.poi_original_size, d_input)

        # position encoder for transformer
        self.if_pos_encoder = if_pos_encoder
        if self.if_pos_encoder:
            self.pos_encoder = PositionalEncoding(d_input, dropout=0.1)
        else:
            self.dropout = nn.Dropout(0.1)

    def forward(self, src, context_dict) -> Tensor:
        N, B = src.shape[0], src.shape[1]
        if self.loc_embed_method == "calliper":
            if self.loc_type == "x-y":
                coords = torch.stack([context_dict["x_X"], context_dict["y_X"]], dim=-1)  # (seq_len, batch_size, 2)
            elif self.loc_type == "lon-lat":
                coords = torch.stack([context_dict["lon_X"], context_dict["lat_X"]], dim=-1)  # (seq_len, batch_size, 2)
            else:
                raise ValueError("Unknown location type")
            # reshape coords from (seq_len, batch_size, 2) to (batch_size * seq_len, 2)
            coords = coords.reshape(-1, 2)
            emb = self.emb_loc(coords).float()
            # reshape back to (seq_len, batch_size, 2, d_input)
            emb = emb.reshape(N, B, self.d_input)
        elif self.loc_embed_method == "ctle":
            src_batch_first = src.transpose(0, 1)
            emb = self.emb_loc(src_batch_first)
            emb = emb.transpose(0, 1)
        else:
            emb = self.emb_loc(src)  # (126, 256, 128) (L, B, N)
        

        if self.if_include_road:
            emb = emb + self.emb_road(src)

        if self.if_include_spatial:
            coords = torch.stack([context_dict["lat"], context_dict["long"]], dim=-1)  # (seq_len, batch_size, 2)
            # coords = torch.transpose(coords, 0, -1)
            emb = emb + self.spatial_embedding(coords)

        if self.if_include_time:
            if self.emb_type == "add":
                emb = emb + self.temporal_embedding(context_dict["time"], context_dict["weekday_X"])
            else:
                emb = torch.cat([emb, self.temporal_embedding(context_dict["time"], context_dict["weekday_X"])], dim=-1)

        if self.if_include_duration:
            emb = emb + self.emb_duration(context_dict["dur_X"])

        if self.if_include_poi:
            emb = emb + self.poi_net(context_dict["poi"])

        if self.if_pos_encoder:
            return self.pos_encoder(emb * math.sqrt(self.d_input))
        else:
            return self.dropout(emb)
