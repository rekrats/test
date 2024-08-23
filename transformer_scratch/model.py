import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self, Embedding_d: int, Vocab_size: int):
        super().__init__()
        self.Embedding_d = Embedding_d
        self.Vocab_size = Vocab_size
        self.Embeddings = nn.Embedding(Vocab_size, Embedding_d)
        # nn.embedding 接受两个参数，第一个 num 即所有 embeddings 的总数也就是词汇数
        # 第二个参数是目标嵌入的维度
    def forward(self, x):
        return self.Embeddings(x) * math.sqrt(self.Embedding_d)
    
class PositionEncoding(nn.Module):
    def __init__(self, Embedding_d: int, Seq_len: int, dropout: float)->None:
        super().__init__()
        self.Embedding_d = Embedding_d
        self.Seq_len = Seq_len
        self.dropout = nn.Dropout(dropout)
        # 如果不注册 PE 的话，当模型移动到 GPU 上的时候这个矩阵还会在 CPU 上
        # Which causes Runtime Error
        PE = torch.zeros(self.Seq_len, self.Embedding_d)
        Positions = torch.arange(0, self.Seq_len, dtype = float).unsqueeze(1)
        div_mode = math.exp(torch.arange(0, self.Embedding_d, 2) * math.log(-10000) / self.Embedding_d)
        PE[: , 0::2] = torch.sin(Positions * div_mode)
        PE[: , 1::2] = torch.cos(Positions * div_mode)
        PE.unsqueeze(0) # (1, seqlen, embed_d)

        self.register_buffer("PE", PE)

    def forward(self, x): # 注意：x 的 size 为 (n, x_len, Eembedding)
        # PE 为 (1(broadcast), maxseq_len(partially used), embedding)
        # 所以这个 PE 是设计好的一个编码矩阵，直接用来加就行
        x = x + (self.PE[: , :x.shape[1], :]).requires_grad(False)
        return self.dropout(x)
    
class LayerNomalization(nn.Module):
    # 正则化层，两个可学习参数 alpha and bias
    def __init__(self, eps : float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.parameter(torch.ones(1))
        self.bias = nn.parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True) # why dim = -1?
        std = x.std(dim = -1, keepdim = True) 
        return self.alpha * ((x - mean)/(std + self.eps)) + self.bias
    
class FeedingForward(nn.Module):
    # attention 只是一个机制
    # 主要的对映射的模拟（或者说推理？）还是要由 MLP 完成
    def __init__(self, Embedding_d: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.L1 = nn.Linear(Embedding_d, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.L2 = nn.Linear(d_ff, Embedding_d)

    def forward(self, x):
        return self.L2(self.dropout(nn.ReLU(self.L1(x))))
    
class MultiHeadAttentionBlock(nn.Module):
    # 这一层应当接受三个输入 qkv 输出whatever
    def __init__(self, Embedding_d: int, h: int, Dropout: float) -> None:
        # seqlen 后面再用，从输入 x 中获取
        super().__init__()
        self.Embedding_d = Embedding_d
        self.h = h
        self.dropout = nn.Dropout(Dropout)

        assert Embedding_d % h == 0
        self.d_k = Embedding_d // h

        self.Wq = nn.Linear(self.Embedding_d, self.Embedding_d)
        self.Wk = nn.Linear(self.Embedding_d, self.Embedding_d)
        self.Wv = nn.Linear(self.Embedding_d, self.Embedding_d)
        self.Wo = nn.Linear(self.Embedding_d, self.Embedding_d)
    @staticmethod
    def attention(self, query, key, value, mask, Dropout: nn.Dropout):
        # query, key, value 的规格均为(batch, h, len, dk)
        dk = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(dk)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim = -1)
        if Dropout is not None:
            attention_scores = Dropout(attention_scores)
        # attention_scores 的规格为(batch, h, len, len)
        # Xout 的规格为(batch, h, len, dk)
        return attention_scores , (attention_scores @ value)

    def forward(self, q, k, v, mask):
        # x 的规格是(batch, xlen, embed_d)
        # 第一步是获取 query key value 
        query = self.Wq(q) # 维度均为(batch, xlen, embed_d)
        key = self.Wk(k) # 这里的中间量不需要注册，reason skip
        value = self.Wv(v)
        # cut into h pieces
        #(batch, len, embed_d) --> (batch, h, len, dk)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)
        # Xout 的规格为(batch, h, len, dk), 也就是 h 个 head，拼接成 H 之后和 Wo 做运算
        attention_scores, Xout = self.attention(query, key, value, mask, self.dropout)
        Xout = Xout.transpose(1, 2).contiguous().view(Xout.shape[0], -1, self.Embedding_d)

        return self.Wo(Xout)
    
class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNomalization()
    def forward(self, x, sublayer):
        # 接受一个层和 x，返回这个层的正则化加上残差的结果
        # residual connection 的中文翻译是残差连接
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    def __init__(self, atten: MultiHeadAttentionBlock, feed: FeedingForward, dropout: float) -> None:
        super().__init__()
        self.attenblock = atten # MultiHeadAttentionBlock
        self.feedblock = feed # FeedingForward
        self.residuals = nn.ModuleList( ResidualConnection(dropout) for _ in range(2) )
    def forward(self, x, mask):
        # 需要一个 mask 作为每一个 encoder block 的输入
        x = self.residuals[0](x, lambda x: self.attenblock(x, x, x, mask))
        x = self.residuals[1](x, self.feedblock)
        return x
    
class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNomalization()
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
class DecoderBlock(nn.Module):
    def __init__(self, self_ATT: MultiHeadAttentionBlock, cross_ATT: MultiHeadAttentionBlock, feed: FeedingForward, dropout: float) -> None:
        super().__init__()
        # 解码器需要与编码器交互，需要区分自注意力和交叉注意力
        # 同样地，需要来自编码器的 ENCODER MASK 和解码器的 DECODER MASK
        self.self_ATT = self_ATT
        self.cross_ATT = cross_ATT
        self.feed = feed
        self.residuals = nn.ModuleList( ResidualConnection(dropout) for _ in range(3) )
    def forward(self, x, encoder_output, en_mask, de_mask):
        x = self.residuals[0](x, lambda x: self.self_ATT(x, x, x, de_mask))
        x = self.residuals[1](x, lambda x: self.cross_ATT(x, encoder_output, encoder_output, en_mask))
        x = self.residuals[2](x, self.feed)
        return x
    
class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNomalization()
    def forward(self, x, encoder_output, en_mask, de_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, en_mask, de_mask)
        return self.norm(x)
    
class ProjectLayer(nn.Module):
    def __init__(self, Embedding_d: int, Vocab_size: int) -> None:
        super().__init__()
        self.W = nn.Linear(Embedding_d, Vocab_size)
    def forward(self, x):
        x = self.W(x)
        return torch.log_softmax(x, dim = -1)
    
class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, srembd: InputEmbedding, tgembd: InputEmbedding, srpos: PositionEncoding, tgpos: PositionEncoding, projector: ProjectLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.srembd = srembd
        self.tgembd = tgembd
        self.srpos = srpos
        self.tgpos = tgpos 
        self.projector = projector
    def ENCODE(self, x, src_mask):
        x = self.srembd(x)
        x = self.srpos(x)
        return self.encoder(x, src_mask)
    def DECODE(self, x, encoder_output, src_mask, tgt_mask):
        x = self.tgembd(x)
        x = self.tgpos(x)
        return self.decoder(x, encoder_output, src_mask, tgt_mask)
    def project(self, x):
        return self.projector(x)
    
def BuildTransformer(sr_vocab: int, tg_vocab: int, sr_seqlen: int, tr_seqlen: int, d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048)->Transformer:
    # build embedding and position encoding
    srembd = InputEmbedding(d_model, sr_vocab)
    tgembd = InputEmbedding(d_model, tg_vocab)
    srpos = PositionEncoding(d_model, sr_seqlen, dropout)
    tgpos = PositionEncoding(d_model, tr_seqlen, dropout)

    # build encoder and decoder

    Encoder_blocks = []
    for _ in range(N):
        encoder_selfatt = MultiHeadAttentionBlock(d_model, h, dropout)
        encoder_feed = FeedingForward(d_model, d_ff, dropout)
        Encoder_blocks.append(EncoderBlock(encoder_selfatt, encoder_feed, dropout))
    
    Decoder_blocks = []
    for _ in range(N):
        decoder_selfatt = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_crossatt = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_feed = FeedingForward(d_model, d_ff, dropout)
        Decoder_blocks.append(DecoderBlock(decoder_selfatt, decoder_crossatt, decoder_feed, dropout))
    encoder = Encoder(nn.ModuleList(Encoder_blocks))
    decoder = Decoder(nn.ModuleList(Decoder_blocks))

    # build projector
    projector = ProjectLayer(d_model, tg_vocab)

    transformer = Transformer(encoder, decoder, srembd, tgembd, srpos, tgpos, projector)
    # initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer

    
