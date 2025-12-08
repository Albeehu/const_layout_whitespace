import torch
import torch.nn as nn

from model.util import TransformerWithToken


class Generator(nn.Module):
    def __init__(self, dim_latent, num_label,
                 d_model=512, nhead=8, num_layers=4):
        super().__init__()

        self.fc_z = nn.Linear(dim_latent, d_model // 2)
        self.emb_label = nn.Embedding(num_label, d_model // 2)
        self.fc_in = nn.Linear(d_model, d_model)

        te = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                        dim_feedforward=d_model // 2)
        self.transformer = nn.TransformerEncoder(te, num_layers=num_layers)

        self.fc_out = nn.Linear(d_model, 4)

    def forward(self, z, label, padding_mask):
        z = self.fc_z(z)
        l = self.emb_label(label)
        x = torch.cat([z, l], dim=-1)
        x = torch.relu(self.fc_in(x)).permute(1, 0, 2)

        x = self.transformer(x, src_key_padding_mask=padding_mask)

        x = self.fc_out(x.permute(1, 0, 2))
        x = torch.sigmoid(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, num_label, d_model=512,
                 nhead=8, num_layers=4, max_bbox=66):
        super().__init__()

        # encoder
        self.emb_label = nn.Embedding(num_label, d_model)
        self.fc_bbox = nn.Linear(4, d_model)
        self.enc_fc_in = nn.Linear(d_model * 2, d_model)

        self.enc_transformer = TransformerWithToken(d_model=d_model,
                                                    dim_feedforward=d_model // 2,
                                                    nhead=nhead, num_layers=num_layers)

        self.fc_out_disc = nn.Linear(d_model, 1)

        # decoder
        self.pos_token = nn.Parameter(torch.rand(max_bbox, 1, d_model))
        self.dec_fc_in = nn.Linear(d_model * 2, d_model)

        te = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                        dim_feedforward=d_model // 2)
        self.dec_transformer = nn.TransformerEncoder(te,
                                                     num_layers=num_layers)

        self.fc_out_cls = nn.Linear(d_model, num_label)
        self.fc_out_bbox = nn.Linear(d_model, 4)

    def forward(self, bbox, label, padding_mask, reconst=False):
        """
        bbox: [B, N, 4]
        label: [B, N]
        padding_mask: [B, N]  True 表示 padding
        """
        B, N, _ = bbox.size()

        # ----- Encoder -----
        b = self.fc_bbox(bbox)                 # [B, N, Cb]
        l = self.emb_label(label)             # [B, N, Cl]
        x = self.enc_fc_in(torch.cat([b, l], dim=-1))  # [B, N, d_model]
        x = torch.relu(x).permute(1, 0, 2)    # [N, B, d_model]

        x = self.enc_transformer(x, src_key_padding_mask=padding_mask)  # [N, B, d_model]

        # 拿第 0 個 token 當 global feature
        x_cls = x[0]                          # [B, d_model]

        # ----- adversarial score -----
        logit_disc = self.fc_out_disc(x_cls).squeeze(-1)   # [B]

        # 只要判別，不做重建
        if not reconst:
            return logit_disc

        # ----- reconstruction branch -----
        # 確保 pos_token 長度足夠
        max_pos = self.pos_token.size(0)
        if N > max_pos:
            raise ValueError(
                f"N = {N} exceeds number of pos_token = {max_pos}. "
                f"Increase max_num_elements when creating the model."
            )

        # 把 global feature 複製成 N 個位置
        x_dec = x_cls.unsqueeze(0).expand(N, -1, -1)       # [N, B, d_model]

        # 加上 position token
        t = self.pos_token[:N].expand(-1, B, -1)           # [N, B, d_model]

        # concat global + position 當 decoder 輸入
        x_dec = torch.cat([x_dec, t], dim=-1)              # [N, B, 2*d_model]
        x_dec = torch.relu(self.dec_fc_in(x_dec))          # [N, B, d_model]

        x_dec = self.dec_transformer(x_dec, src_key_padding_mask=padding_mask)  # [N, B, d_model]

        # 展平成只剩有效 token
        x_dec = x_dec.permute(1, 0, 2)[~padding_mask]      # [M, d_model] (M = 非 padding 的 token 數)

        # ----- cls + bbox head -----
        logit_cls = self.fc_out_cls(x_dec)                 # [M, num_classes]
        bbox_pred = torch.sigmoid(self.fc_out_bbox(x_dec)) # [M, 4]

        # **一定要回傳三個值**
        return logit_disc, logit_cls, bbox_pred

    # def forward(self, bbox, label, padding_mask, reconst=False):
    #     B, N, _ = bbox.size()
    #     b = self.fc_bbox(bbox)
    #     l = self.emb_label(label)
    #     x = self.enc_fc_in(torch.cat([b, l], dim=-1))
    #     x = torch.relu(x).permute(1, 0, 2)

    #     x = self.enc_transformer(x, src_key_padding_mask=padding_mask)
    #     x = x[0]

    #     # logit_disc: [B,]
    #     logit_disc = self.fc_out_disc(x).squeeze(-1)

    #     if not reconst:
    #         return logit_disc

    #     else:
    #         max_pos = self.pos_token.size(0)
    #     if N > max_pos:
    #         raise ValueError(
    #             f"N = {N} exceeds number of pos_token = {max_pos}. "
    #             f"Increase max_num_elements when creating the model."
    #         )
    #         x = x.unsqueeze(0).expand(N, -1, -1)
    #         t = self.pos_token[:N].expand(-1, B, -1)
    #         print("x shape:", x.shape)
    #         print("t shape:", t.shape)
    #         x = torch.cat([x, t], dim=-1)
    #         x = torch.relu(self.dec_fc_in(x))

    #         x = self.dec_transformer(x, src_key_padding_mask=padding_mask)
    #         x = x.permute(1, 0, 2)[~padding_mask]

    #         # logit_cls: [M, L]    bbox_pred: [M, 4]
    #         logit_cls = self.fc_out_cls(x)
    #         bbox_pred = torch.sigmoid(self.fc_out_bbox(x))

            # return logit_disc, logit_cls, bbox_pred
