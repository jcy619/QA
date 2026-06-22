import torch
import torch.nn as nn

class DecoderLayer(nn.Module):
    def __init__(self,dim=384,num_heads=6,mlp_ratio=4.,dropout=0.1,depth_i = 0):
        super().__init__()
        self.depth_i = depth_i
        self.decoder_dim = dim
        self.num_heads = num_heads
        self.self_attn = nn.MultiheadAttention(self.decoder_dim,self.num_heads,dropout)

        self.gate_proj = nn.Linear(dim, dim)
        self.sigmoid = nn.Sigmoid()
        self.multi_attn = nn.MultiheadAttention(self.decoder_dim,self.num_heads,dropout)

        dim_feedforward = int(mlp_ratio*self.decoder_dim)
        self.linear1 = nn.Linear(self.decoder_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, self.decoder_dim)

        self.norm1 = nn.LayerNorm(self.decoder_dim)
        self.norm2 = nn.LayerNorm(self.decoder_dim)
        self.norm3 = nn.LayerNorm(self.decoder_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.GELU()

    def forward(self,x):
        tgt = x['tgt']
        memory = x['x']
        query_pos = x['query_pos']
    
    
        q = k = query_pos
        
        attn_out, weight = self.self_attn(q, k, value=tgt)

        gate = self.sigmoid(self.gate_proj(attn_out))
        tgt = tgt + self.dropout1(attn_out * (1 + gate))
        tgt = self.norm1(tgt)
        
        query = tgt
        
        key = value = memory
        tgt2 = self.multi_attn(query=query, key=key, value=value,attn_mask=None, key_padding_mask=None)[0]  # num_queries,bs,c
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        output = {'tgt':tgt,'x':memory,'query_pos':query_pos}
        return output

class QueryHead(nn.Module):
    def __init__(self,num_classes=100,num_queries=20,embed_dim=192,num_patches=65,dec_layers=6,num_heads=12,mlp_ratio=2.):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.dec_layers = dec_layers
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.query_embed = nn.Embedding(self.num_queries,self.embed_dim)
        self.tgt_proj = nn.Linear(self.embed_dim, self.embed_dim)
        
        self.decoder_blocks = nn.Sequential(*[
            DecoderLayer(
                dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                dropout=0.1,
                depth_i=i,
            )
            for i in range(self.dec_layers)])
        self.head = nn.Linear(self.embed_dim,self.num_classes) if num_classes > 0 else nn.Identity()
        #self.avgpool = nn.AdaptiveAvgPool1d(1)
        #self.head = nn.Linear(self.embed_dim,1)
        

    def forward(self,x):
        bs,N,c = x.shape
        
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1).to(x.device)

        # ===== mean + max  =====
        mean_feat = x.mean(dim=1)          # (bs, c)
        max_feat = x.max(dim=1).values     # (bs, c)
        global_feat = mean_feat + max_feat # (bs, c)

        tgt = self.tgt_proj(global_feat).unsqueeze(0)   # (1, bs, c)
        tgt = tgt.repeat(self.num_queries, 1, 1)        # (num_queries, bs, c)

        
        x = x.permute(1, 0, 2)
        
        input = {"tgt": tgt, 'x': x, "query_pos": query_embed}
        
    
        x = self.decoder_blocks(input)
        
        tgt = x['tgt'].permute(1, 0, 2)  # (bs,nq,embed_dim)
        #print(tgt.shape)
        x = self.head(tgt) #(bs,nq,1)
        #print(x.shape)
        x =x.flatten(1,2) #(bs,1)
        #print(x.shape)
        return x


        q = k = query_pos
        
        attn_out, weight = self.self_attn(q, k, value=tgt)

        gate = self.sigmoid(self.gate_proj(attn_out))
        tgt = tgt + self.dropout1(attn_out * (1 + gate))
        tgt = self.norm1(tgt)
        
        query = tgt
        
        key = value = memory
        tgt2 = self.multi_attn(query=query, key=key, value=value,attn_mask=None, key_padding_mask=None)[0]  # num_queries,bs,c
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        output = {'tgt':tgt,'x':memory,'query_pos':query_pos}
        return output

class QueryHead(nn.Module):
    def __init__(self,num_classes=100,num_queries=20,embed_dim=192,num_patches=65,dec_layers=6,num_heads=12,mlp_ratio=2.):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.dec_layers = dec_layers
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.query_embed = nn.Embedding(self.num_queries,self.embed_dim)
        self.tgt_proj = nn.Linear(self.embed_dim, self.embed_dim)
        
        self.decoder_blocks = nn.Sequential(*[
            DecoderLayer(
                dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                dropout=0.1,
                depth_i=i,
            )
            for i in range(self.dec_layers)])
        self.head = nn.Linear(self.embed_dim,self.num_classes) if num_classes > 0 else nn.Identity()
        #self.avgpool = nn.AdaptiveAvgPool1d(1)
        #self.head = nn.Linear(self.embed_dim,1)
        

    def forward(self,x):
        bs,N,c = x.shape
        
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1).to(x.device)

        # ===== mean + max  =====
        mean_feat = x.mean(dim=1)          # (bs, c)
        max_feat = x.max(dim=1).values     # (bs, c)
        global_feat = mean_feat + max_feat # (bs, c)

        tgt = self.tgt_proj(global_feat).unsqueeze(0)   # (1, bs, c)
        tgt = tgt.repeat(self.num_queries, 1, 1)        # (num_queries, bs, c)

        
        x = x.permute(1, 0, 2)
        
        input = {"tgt": tgt, 'x': x, "query_pos": query_embed}
        
    
        x = self.decoder_blocks(input)
        
        tgt = x['tgt'].permute(1, 0, 2)  # (bs,nq,embed_dim)
        #print(tgt.shape)
        x = self.head(tgt) #(bs,nq,1)
        #print(x.shape)
        x =x.flatten(1,2) #(bs,1)
        #print(x.shape)
        return x


