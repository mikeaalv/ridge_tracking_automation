# RNN and modified structure for ridge tracking automation
#Yue Wu
#UGA 08/25/2021
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all__=['gru_mlp_rnn' 'gru_rnn']

def line1d(in_features,out_features):
    return nn.Linear(in_features,out_features,bias=False)

def line1dbias(in_features,out_features):
    return nn.Linear(in_features,out_features,bias=True)

## a gru network with controled inside MLP
class gru_mlp_cell(nn.Module):
    def __init__(self,input_size,hidden_size,numlayer=0,bias=True,p=0.0):
        super(gru_mlp_cell,self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.x2h=line1dbias(input_size,3*hidden_size)
        self.h2h=line1dbias(hidden_size,3*hidden_size)
        self.layer1=self._make_layer(hidden_size,numlayer=numlayer,p=p)
        # self.reset_parameters()
    
    def reset_parameters(self):
        std=1.0/math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std,std)
    
    def forward(self,input,hidden):
        ## a condensed reimplementation(approximation) for effieciency (as used in pytroch c++ and other python implementaion)
        gate_input=self.x2h(input)#input transformation
        gate_hidden=self.h2h(hidden)#hidden state transformation
        # print('gate_input{}'.format(gate_input.shape))
        # gate_input=gate_input.squeeze()
        # gate_hidden=gate_hidden.squeeze()
        i_r,i_i,i_n=gate_input.chunk(3,1)#To: reset gate,update gate, update h
        h_r,h_i,h_n=gate_hidden.chunk(3,1)#To: reset gate,update gate,update h
        resetgate=torch.sigmoid(i_r+h_r)
        updategate=torch.sigmoid(i_i+h_i)
        h_n_new=self.layer1(h_n)
        newh=torch.tanh(i_n+resetgate*h_n_new)
        hy=newh+updategate*(hidden-newh)
        # print('resetgate{} updategate{} hidden{}'.format(resetgate.shape,updategate.shape,hidden.shape))
        return hy
        
    def _make_layer(self,hidden_size,numlayer=0,p=0.0):
        layers=[]
        ##block will pass the arguments to the two block types
        for _ in range(0,numlayer):
            layers.append(line1dbias(hidden_size,hidden_size))
            layers.append(nn.Dropout(p=p))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

## the wrapper for rnn model
class RNN_Model(nn.Module):
    def __init__(self,input_dim,output_dim,hidden_dim,input_dim_0,numlayer,type='gru',bias=True,p=0.0):
        ##initialvec initial condition and t0
        super(RNN_Model,self).__init__()
        self.hidden_dim=hidden_dim
        # print('{}\n'.format(type))
        if type=='gru':
            self.rnncell=nn.GRUCell(input_dim,hidden_dim,bias=True)
        elif type=='gru_mlp':
            self.rnncell=gru_mlp_cell(input_dim,hidden_dim,numlayer,p=p)
        
        self.inputlay=line1dbias(input_dim_0,hidden_dim)
        self.outputlay=line1dbias(hidden_dim,output_dim)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight,mode='fan_out')
        
    def forward(self,x,initialvec):
        ##initialvec: input1 [Y(t_0) t_0], initial condition
        ##x: input2  [theta, t_k, delta t_k], theta and time
        hiddeninput0=initialvec
        self.hiddeninput0=hiddeninput0
        h0=self.inputlay(self.hiddeninput0)
        outs=[]
        hn=h0
        for seq in range(x.size(1)):#time direction
        
            # a=x[:,seq,:]
            # print('{} {}'.format(a.shape,hn.shape))
            
            hn=self.rnncell(x[:,seq,:],hn)
            outs.append(self.outputlay(hn))
        # print('outs{}'.format(outs))
        # outtensor=torch.cat(outs)
        outtensor=torch.stack(outs)
        outtensor=torch.transpose(outtensor,0,1).contiguous()
        size3d=outtensor.shape
        outtensor=outtensor.view(size3d[0]*size3d[1],-1)
        return outtensor

def _rnnnet(ninput,noutput,num_layer,ncellscale,type,**kwargs):
    # ninput: #input,
    # num_response: #response,
    # block: block structure,
    # layers: #layers,
    # pretrained: pretrained or not(not currently used)
    # progress: progress(not currently used),
    # ncellscale: scale factor for hidden layer size
    # **kwargs: to add other parameters
    input_dim=ninput
    output_dim=noutput
    input_dim_0=nspec+1
    hidden_dim=int(input_dim_0*(ncellscale+1))
    # print('input_dim {} output_dim {} input_dim_0 {} hidden_dim {}'.format(input_dim,output_dim,input_dim_0,hidden_dim))
    model=RNN_Model(input_dim,output_dim,hidden_dim,input_dim_0,num_layer,type,**kwargs)
    return model

def gru_mlp_rnn(ninput,noutput,num_layer,p=0.0,ncellscale=1.0,**kwargs):
    r"""rnn model adapted from
    a controled number of layer is added within gru
    """
    kwargs['p']=p
    type='gru_mlp'
    return _rnnnet(ninput,noutput,num_layer,ncellscale,type,**kwargs)

def gru_rnn(ninput,noutput,num_layer,p=0.0,ncellscale=1.0,**kwargs):
    r"""the original gru in pytorch
    """
    kwargs['p']=p
    type='gru'
    return _rnnnet(ninput,noutput,num_layer,ncellscale,type,**kwargs)
