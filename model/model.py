import math
import torch
import torch.nn as nn 
from torch.autograd import Function
from torch.nn.parameter import Parameter 
from torch.nn import functional as F
from torch.nn._functions.thnn import rnnFusedPointwise as fusedBackend 

def encoder_lstm(input, hidden, weight, feedback, mask_u, mask_w):
    
    hx_origin, cx = hidden 
    W, U, G = weight 
    
    if mask_u is not None:
        hx_origin = hx_origin*mask_u  

    if G is not None: 
        gh = F.sigmoid(hx_origin.bmm(G)) 
        gh = gh.expand_as(hx_origin) 
        hx_origin = gh * hx_origin 

    if feedback is True: 
        hx = [] 
        for i in range(len(hx_origin)):
            hx.append(hx_origin[i])
    
        hx = torch.cat(hx, 1) 
        hx = hx.repeat(len(hx_origin), 1, 1) 

    else: 
        hx = hx_origin 

    hx_next = []
    cx_next = [] 
    
    for i in range(hx.size(0)): 
        
        if mask_w is not None: 
        	input = input*mask_w[i]
        
        igates = F.linear(input, W[i]) 
        hgates = F.linear(hx[i], U[i]) 
        
        state = fusedBackend.LSTMFused.apply
        
        input, cy = state(igates, hgates, cx[i]) 
        
        hx_next.append(input)
        cx_next.append(cy) 
    
    hx_next = torch.stack(hx_next, 0) 
    cx_next = torch.stack(cx_next, 0) 
    
    return hx_next, cx_next

def decoder_lstm(output, hidden, weight, feedback, mask_u, mask_w):

    hx_origin, cx = hidden 
    W, U, G, L = weight 
    
    if mask_u is not None:
        hx_origin = hx_origin*mask_u

    if G is not None: 
        gh = F.sigmoid(hx_origin.bmm(G)) 
        gh = gh.expand_as(hx_origin) 
        hx_origin = gh * hx_origin 
 
    if feedback is True:
        hx = [] 
        for i in range(len(hx_origin)):
            hx.append(hx_origin[i])
        
        hx = torch.cat(hx, 1) 
        hx = hx.repeat(len(hx_origin), 1, 1) 
       
    else: 
        hx = hx_origin 
    
    hx_next = []
    cx_next = [] 
    
#    input = Variable(output.data.new(hx_origin[0].size(0), hx_origin[0].size(1)*4).zero_())

    input = output.new(hx_origin[0].size(0), hx_origin[0].size(1)*4).zero_().requires_grad_()
    
    for i in range(hx.size(0)): 
        
        hgates = F.linear(hx[i], U[i]) 
        if i==0: 
        	igates = input
        else:
        	if mask_w is not None: 
        		input = input*mask_w[i-1] 
        
        	igates = F.linear(input, W[i-1]) 
        
        state = fusedBackend.LSTMFused.apply 
        input, cy = state(igates, hgates, cx[i])
        
        hx_next.append(input) 
        cx_next.append(cy) 
        
    hx_next = torch.stack(hx_next, 0) 
    cx_next = torch.stack(cx_next, 0) 
    
    return input, (hx_next, cx_next)
        

def Recurrent(feedback, hidden_tied):

    inner = encoder_lstm 
    
    def forward(input, hidden, weight, mask_u, mask_w): 

        steps = input.size(0) 
        output = [] 
        
        enc_hidden = [] 
        enc_context = [] 
        
        for t in range(steps): 
            hidden = inner(input[t], hidden, weight, feedback, mask_u, mask_w) 
            if (hidden_tied) and (mask_w) is not None and (t!=steps-1): 
                enc_hidden.append(hidden[0]) 
                enc_context.append(hidden[1]) 

        if (hidden_tied) and (mask_w is not None): 
            enc_hidden = torch.stack(enc_hidden, 0)
            enc_context = torch.stack(enc_context, 0) 
            enc_hiddens = (enc_hidden, enc_context) 
        else:
            enc_hiddens = None
        
        return hidden, enc_hiddens

    return forward


def Recurrent_Decoder(feedback, hidden_tied):
    
    inner = decoder_lstm 
    
    def forward(output, hidden, steps, linear, weight, mask_u, mask_w): 
        
        outputs = [] 
        _, _, _, L = weight

        dec_hidden = []
        dec_context = [] 
        
        for t in range(steps-1): 
            output = linear(output) 
            output, hidden = inner(output, hidden, weight, feedback, mask_u, mask_w) 

            if (hidden_tied) and (mask_w) is not None and (t!=steps-1): 
                dec_hidden.append(hidden[0])
                dec_context.append(hidden[1])

            if mask_w is not None: 
                output = output*mask_w[-1] 
            
            output = F.linear(output, L) 
            outputs.append(output) 
 
        outputs = torch.stack(outputs, 0) 

        if (hidden_tied) and (mask_w is not None): 
            # reverse
            dec_hidden = torch.stack(dec_hidden[::-1], 0) 
            dec_context = torch.stack(dec_context[::-1], 0) 
            dec_hiddens = (dec_hidden, dec_context) 

        else:
            dec_hiddens = None

        return outputs, dec_hiddens
    
    return forward

		
class Encoder(nn.Module): 
    def __init__(self, ninp, nhid, nlayers, dropout=0.2, h_dropout=0.0, feedback=False, gated=False, hidden_tied=False): 
        super(Encoder, self).__init__() 
        
        self.ninp = ninp 
        self.nhid = nhid
        self.nlayers = nlayers 
        self.v_dropout = dropout
        self.h_dropout = h_dropout 
        
        self.feedback = feedback
        self.gated = gated
        self.hidden_tied = hidden_tied
        
        self.linear = nn.Linear(ninp, nhid) 
        
        self.w_weight = Parameter(torch.empty(nlayers, 4*nhid, nhid)) 
        if feedback: 
        	self.u_weight = Parameter(torch.empty(nlayers, 4*nhid, nlayers*nhid)) 
        else:
        	self.u_weight = Parameter(torch.empty(nlayers, 4*nhid, nhid)) 
        
        if gated:
            self.g_weight = Parameter(torch.empty(nlayers, nhid, 1)) 
        else:
            self.g_weight = None 
        
        self._all_weights = ['w_weight', 'u_weight', 'g_weight'] 
        
        self.reset_parameters() 

    def reset_parameters(self):
        stdv = 1.0/math.sqrt(self.nhid) 
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv) 

    @property
    def all_weights(self): 
        return [getattr(self, weight) for weight in self._all_weights] 

    def init_hidden(self, bsz): 
        weight = next(self.parameters()).data
	    
        return (weight.new(self.nlayers, bsz, self.nhid).zero_().requires_grad_(), 
        weight.new(self.nlayers, bsz, self.nhid).zero_().requires_grad_()) 


    def forward(self, input, hidden=None): 
        
        # input: seqlen by batch by dim 
        bsz = input.size(1) 
        if hidden is None:
            hidden = self.init_hidden(bsz)
        
        input = self.linear(input) 
        
        if self.v_dropout>0 and self.training: 
            self.mask_w = input.new(self.nlayers, bsz, self.nhid).bernoulli_(1-self.v_dropout).div(1-self.v_dropout).requires_grad_() 
        else:
            self.mask_w = None 
        
        if self.h_dropout>0 and self.training: 
            self.mask_u = input.new(self.nlayers, bsz, self.nhid).bernoulli_(1-self.h_dropout).div(1-self.h_dropout).requires_grad_() 
        else:
            self.mask_u = None 
        
        hidden, enc_hiddens = Recurrent(self.feedback, self.hidden_tied)(input, hidden, self.all_weights, self.mask_u, self.mask_w)
        return hidden, enc_hiddens

class Decoder(nn.Module): 
    def __init__(self, nout, nhid, nlayers, dropout=0.2, h_dropout=0.0, feedback=False, gated=False, hidden_tied=False):

        super(Decoder, self).__init__()
        self.nout = nout
        self.nhid = nhid
        self.nlayers = nlayers 
        self.v_dropout = dropout
        self.h_dropout = h_dropout 

        self.feedback = feedback 
        self.gated = gated 
        self.hidden_tied = hidden_tied

        self.linear = nn.Linear(nout, nhid) 
        
        self.w_weight = Parameter(torch.empty(nlayers-1, 4*nhid, nhid)) 
        if feedback: 
        	self.u_weight = Parameter(torch.empty(nlayers, 4*nhid, nlayers*nhid))
        else:
        	self.u_weight = Parameter(torch.empty(nlayers, 4*nhid, nhid)) 

        if gated:
            self.g_weight = Parameter(torch.empty(nlayers, nhid, 1)) 
        else:
            self.g_weight = None 

        self.l_weight = Parameter(torch.empty(nout, nhid)) 

        self._all_weights = ['w_weight', 'u_weight', 'g_weight', 'l_weight']
        self.reset_parameters() 
    
    def reset_parameters(self):
        stdv = 1.0/math.sqrt(self.nhid) 
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv) 
    
    @property
    def all_weights(self): 
    	return [getattr(self, weight) for weight in self._all_weights] 
    
    def forward(self, hidden, steps):
        
        bsz = hidden[0].size(1) 
        output = F.linear(hidden[0][-1], self.l_weight) 
        
        if self.v_dropout>0 and self.training: 
            self.mask_w = output.new(self.nlayers, bsz, self.nhid).bernoulli_(1-self.v_dropout).div(1-self.v_dropout).requires_grad_() 
        else:
            self.mask_w = None 
        
        if self.h_dropout>0 and self.training: 
            self.mask_u = output.new(self.nlayers, bsz, self.nhid).bernoulli_(1-self.h_dropout).div(1-self.h_dropout).requires_grad_() 
        else:
            self.mask_u = None 
        
        if steps>1: 
            outputs, dec_hiddens = Recurrent_Decoder(self.feedback, self.hidden_tied)(output, hidden, steps, self.linear, self.all_weights, self.mask_u, self.mask_w) 
            outputs = torch.cat([output.unsqueeze(0), outputs], 0)
        else: 
            outputs = output.unsqueeze(0) 
            dec_hiddens = None 
        
        return outputs, dec_hiddens

class EncDecAD(nn.Module):
    def __init__(self, ninp, nhid, nout, nlayers, dropout=0.2, h_dropout=0.0, feedback=False, gated=False, hidden_tied=False):
        super(EncDecAD, self).__init__() 
        
        self.ninp = ninp
        self.nhid = nhid
        self.nout = nout 
        self.nlayers = nlayers 
        self.dropout = dropout
        self.h_dropout = h_dropout 

        self.feedback = feedback 
        self.gated = gated
        self.hidden_tied = hidden_tied
        
        self.linear = nn.Linear(ninp, nhid) 
        self.encoder = Encoder(nhid, nhid, nlayers, dropout=dropout, h_dropout=h_dropout, feedback=feedback, gated=gated, hidden_tied=hidden_tied)
        self.decoder = Decoder(nout, nhid, nlayers, dropout=dropout, h_dropout=h_dropout, feedback=feedback, gated=gated, hidden_tied=hidden_tied) 

    def forward(self, input, hidden=None):
        
        bsz = input.size(1) 
        emb = self.linear(input.view(-1, input.size(-1))) 
        emb = emb.view(-1, bsz, emb.size(-1)) 
        hidden, enc_hiddens = self.encoder(emb, hidden)
        output, dec_hiddens = self.decoder(hidden, input.size(0)) 
        
        return output, hidden, enc_hiddens, dec_hiddens

    def init_hidden(self, bsz): 
        weight = next(self.parameters()).data 
        
        return (weight.new(self.nlayers, bsz, self.nhid).zero_().required_grad_(), # hidden
                weight.new(self.nlayers, bsz, self.nhid).zero_().required_grad_()) # context
