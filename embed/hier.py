from itertools import zip_longest

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from utils.earlystopping import EarlyStopping
from transformers import get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import StepLR


class HierEmbedding(nn.Module):
    def __init__(self, token_embed_size, num_vocab, week_embed_size, hour_embed_size, duration_embed_size):
        super().__init__()
        self.num_vocab = num_vocab
        self.token_embed_size = token_embed_size
        self.embed_size = token_embed_size + week_embed_size + hour_embed_size + duration_embed_size

        self.token_embed = nn.Embedding(num_vocab, token_embed_size)
        self.token_embed.weight.data.uniform_(-0.5/token_embed_size, 0.5/token_embed_size)
        self.week_embed = nn.Embedding(7, week_embed_size)
        self.hour_embed = nn.Embedding(24, hour_embed_size)
        if duration_embed_size > 0:
            self.duration_embed = nn.Embedding(24, duration_embed_size)
        else:
            self.duration_embed = None

        self.dropout = nn.Dropout(0.1)

    def forward(self, token, week, hour, duration):
        token = self.token_embed(token)
        week = self.week_embed(week)
        hour = self.hour_embed(hour)
        if self.duration_embed is not None:
            duration = self.duration_embed(duration)
            out_embed = torch.cat([token, week, hour, duration], dim=-1)
        else:
            out_embed = torch.cat([token, week, hour], dim=-1)

        return out_embed


class Hier(nn.Module):
    def __init__(self, embed: HierEmbedding, hidden_size, num_layers, share=True, dropout=0.1):
        super().__init__()
        self.embed = embed
        self.add_module('embed', self.embed)
        self.encoder = nn.LSTM(self.embed.embed_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        if share:
            self.out_linear = nn.Sequential(nn.Linear(hidden_size, self.embed.token_embed_size), nn.LeakyReLU())
        else:
            self.out_linear = nn.Sequential(nn.Linear(hidden_size, self.embed.token_embed_size),
                                            nn.LeakyReLU(),
                                            nn.Linear(self.embed.token_embed_size, self.embed.num_vocab))
        self.share = share

    def forward(self, token, week, hour, duration, valid_len, **kwargs):
        """
        :param token: sequences of tokens, shape (batch, seq_len)
        :param week: sequences of week indices, shape (batch, seq_len)
        :param hour: sequences of visit time slot indices, shape (batch, seq_len)
        :param duration: sequences of duration slot indices, shape (batch, seq_len)
        :return: the output prediction of next vocab, shape (batch, seq_len, num_vocab)
        """
        embed = self.embed(token, week, hour, duration)  # (batch, seq_len, embed_size)
        packed_embed = pack_padded_sequence(embed, valid_len, batch_first=True, enforce_sorted=False)
        encoder_out, (hidden, cell) = self.encoder(packed_embed)  # (batch, seq_len, hidden_size)
        # unpack the packed sequence
        encoder_out, _ = pad_packed_sequence(encoder_out, batch_first=True)
        out = self.out_linear(encoder_out)  # (batch, seq_len, token_embed_size)

        if self.share:
            out = torch.matmul(out, self.embed.token_embed.weight.transpose(0, 1))  # (total_valid_len, num_vocab)
        return out

    def static_embed(self):
        return self.embed.token_embed.weight[:self.embed.num_vocab].detach().cpu().numpy()


class HierDataset(Dataset):
    def __init__(self, sentences, weekdays, start_mins):
        self.data = sentences
        self.weekdays = weekdays
        self.start_mins = start_mins
        self.lens = [len(sent) for sent in sentences]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        loc_ids = torch.tensor(self.data[idx], dtype=torch.long)
        weekday = torch.tensor(self.weekdays[idx], dtype=torch.long)
        start_min = torch.tensor(self.start_mins[idx], dtype=torch.long)
        length = torch.tensor(self.lens[idx], dtype=torch.long)

        return loc_ids, weekday, start_min, length

def collate_fn(batch):
    loc_ids, weekdays, start_mins, lengths = zip(*batch)
    loc_ids = pad_sequence(loc_ids, batch_first=True, padding_value=0)
    weekdays = pad_sequence(weekdays, batch_first=True, padding_value=0)
    start_mins = pad_sequence(start_mins, batch_first=True, padding_value=0)
    lengths = torch.stack(lengths)
    return loc_ids, weekdays, start_mins, lengths


def get_hier_dataloader(dataset, batch_size, data_type="train"):
    if data_type == "train":
        shuffle = True
    else:
        shuffle = False
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


def train_hier(train_loader, val_loader, hier_model, device, logger, log_dir, config):
    # user_ids, src_tokens, src_weekdays, src_ts, src_lens = zip(*dataset.gen_sequence(select_days=0))
    # Log the model structure
    logger.info(hier_model)
    hier_model = hier_model.to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(hier_model.parameters(), lr=0.001, weight_decay=0.000001)
    n_batches = len(train_loader)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=len(train_loader) * config.num_warmup_epochs,
        num_training_steps=len(train_loader) * config.num_training_epochs,
    )
    scheduler_ES = StepLR(optimizer, step_size=config.lr_step_size, gamma=config.lr_gamma)
    logger.info(f"Current learning rate: {scheduler.get_last_lr()[0]:.6f}")

    scheduler_count = 0
    # initialize the early_stopping object
    early_stopping = EarlyStopping(logger,
        log_dir, patience=config["patience"], verbose=config.verbose, delta=0.0001
    )

    for epoch in range(config["num_epoch"]):
        for i, inputs in enumerate(train_loader):
            # src_token, src_weekday, src_t, src_len = zip(*batch)

            src_token, weekdays, start_mins, lengths = inputs  
            src_token = src_token.to(device)
            src_weekday = weekdays.to(device)
            src_hour = (start_mins // 60).long().to(device)
            src_len = lengths  # (B,)

            # src_t = torch.from_numpy(np.transpose(np.array(list(zip_longest(*src_t, fillvalue=0))))).float().to(device)
            # src_len = torch.tensor(src_len).long().to(device)

            # src_hour = (src_t % (24 * 60 * 60) / 60 / 60).long()
            # src_duration = ((src_t[:, 1:] - src_t[:, :-1]) % (24 * 60 * 60) / 60 / 60).long()
            # src_duration = torch.clamp(src_duration, 0, 23)
            if config["duration_embed_size"] > 0:
                pass
            else:
                src_duration = None

            hier_out = hier_model(token=src_token[:, :-1], week=src_weekday[:, :-1], hour=src_hour[:, :-1],
                                   duration=src_duration, valid_len=src_len-1)  # (batch, seq_len, num_vocab)
            hier_out = hier_out.view(-1, hier_out.size(-1))  # (batch*seq_len, num_vocab)
            trg_token = src_token[:, 1:].flatten()  # (batch*seq_len,)
            # trg_token = pack_padded_sequence(src_token[:, 1:], src_len-1, batch_first=True, enforce_sorted=False).data
            loss = loss_func(hier_out, trg_token)

            if (i + 1) % config["print_step"] == 0:
                logger.info(f"Epoch {epoch+1} {100*(i+1)/n_batches:.1f}% batch {i}, loss: {loss.item():.5f}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler_count == 0:
                scheduler.step()

        # At the end of the epoch, do a pass on the validation set
        total_val_loss = 0
        with torch.no_grad():
            for inputs in val_loader:
                src_token, weekdays, start_mins, lengths = inputs  
                src_token = src_token.to(device)
                src_weekday = weekdays.to(device)
                src_hour = (start_mins // 60).long().to(device)
                src_len = lengths  # (B,)

                if config["duration_embed_size"] > 0:
                    pass
                else:
                    src_duration = None
                hier_out = hier_model(token=src_token[:, :-1], week=src_weekday[:, :-1], hour=src_hour[:, :-1],
                                   duration=src_duration, valid_len=src_len-1)  # (batch, seq_len, num_vocab)
                hier_out = hier_out.view(-1, hier_out.size(-1))  # (batch*seq_len, num_vocab)
                trg_token = src_token[:, 1:].flatten()
                total_val_loss += loss_func(hier_out, trg_token).item()

        # val_loss
        val_loss = total_val_loss / len(val_loader)
        logger.info(f"Epoch {epoch+1}, val loss: {val_loss}")
        val_loss_dict = {"val_loss": val_loss}
        early_stopping(val_loss_dict, hier_model)

        if early_stopping.early_stop:
            logger.info("=" * 50)
            logger.info("Early stopping")
            if scheduler_count == 2:
                break
            scheduler_count += 1
            hier_model.load_state_dict(torch.load(log_dir + "/checkpoint.pt"))
            early_stopping.early_stop = False
            early_stopping.counter = 0
            scheduler_ES.step()
            print("Learing rate decayed to 1/10")

        if config.verbose:
            # print("Current learning rate: {:.5f}".format(scheduler.get_last_lr()[0]))
            # print("Current learning rate: {:.5f}".format(scheduler_ES.get_last_lr()[0]))
            logger.info("Current learning rate: {:.6f}".format(optimizer.param_groups[0]["lr"]))
            logger.info("=" * 50)

        # if early_stopping.early_stop:
        #     logger.info("=" * 50)
        #     logger.info("Early stopping")
        #     break
    return hier_model.static_embed()
