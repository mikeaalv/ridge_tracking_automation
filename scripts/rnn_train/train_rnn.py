# training script for RNN model
import argparse
import os
import random
import shutil
import time
import warnings
import pickle
import numpy as np
import math
import sys
import copy
import h5py
from nltk import flatten
import re
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from itertools import permutations

import torch
import torch.nn.parallel
import torch.utils.data as utils
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.utils.data.sampler import Sampler

# sys.path.insert(1,'PATH')
import nnt_struc as models

model_names=sorted(name for name in models.__dict__
    if (name.endswith("_mlp") or name.endswith("_rnn")) and callable(models.__dict__[name]))
##default parameters
args_internal_dict={
    "batch_size": (50000,int),
    "test_validate_ratio": (0.2,float), # ratio of sample for test&validation in each epoch (they will be devided by half)
    "test_batch_size": (50000,int),
    "epochs": (10,int),
    "learning_rate": (0.01,float),
    "momentum": (0.5,float),
    "no_cuda": (False,bool),
    "seed": (1,int),
    "log_interval": (10,int),
    "net_struct": ("resnet18_mlp",str),
    "layersize_ratio": (1.0,float),#use the input vector size to calcualte hidden layer size
    "optimizer": ("adam",str),##adam
    "normalize_flag": ("Y",str),#whether the input data in X are normalized (Y) or not (N)
    "batchnorm_flag": ("Y",str),# whether batch normalization is used (Y) or not (N). Not working for resnet
    "num_layer": (0,int),#number of layer, not work for resnet
    "timetrainlen": (101,int), #the length of time-series to use in training
    "inputfile": ("sparselinearode_new.small.stepwiseadd.mat",str),## the file name of input data
     "p": (0.0,float),#probability used in dropout
     "gpu_use": (1,int),# whehter use gpu (1) or not (0)
     "scheduler": ("",str),# the lr decay scheduler choices: step, plateau, cyclelr
     "lr_print": (0,int),
     "rnn_struct": (0,int),#whether use rnn structure
     "sampler": ("block",str),##sampler to use. "block" sampler or "individual" sampler
     "timeshift_transformp": (0.0,float),##transformation input data by shift initial condition and time. This is the probability that such transform is performed
     "linearcomb_transformp": (0.0,float)##transform input data by random combine two samples. This is the probability that such transform is performed
}
###fixed parameters: for communication related parameter within one node
fix_para_dict={#"world_size": (1,int),
               # "rank": (0,int),
               # "dist_url": ("env://",str),#"tcp://127.0.0.1:FREEPORT"
               "gpu": (None,int),
               # "multiprocessing_distributed": (False,bool),
               # "dist_backend": ("nccl",str), ##the preferred way approach of parallel gpu
               "workers": (1,int)
}
inputdir="../data/"
def train(args,model,train_loader,optimizer,epoch,device,ntime,scheduler):
    model.train()
    trainloss=[]
    for batch_idx, (data, target) in enumerate(train_loader):
        # print("checkerstart")
        # if args.gpu is not None:
        #     data=data.cuda(args.gpu,non_blocking=True)
        #
        # target=target.cuda(args.gpu,non_blocking=True)
        
        # Transform ******
        if args.timeshift_transformp>0.0 and args.sampler=="block": # linear combination transform
            data, target=trans_time_shift(data,target,args,ntime)
        
        if args.linearcomb_transformp>0: # time shift transform
            data, target=trans_lin_comb(data,target,args,ntime)
                
        if args.rnn_struct==0:
            data,target=data.to(device),target.to(device)
            output=model(data)
        else:
            #reshape data for rnn intput
            sizes=data.shape
            ntheta_real=args.ntheta-1-args.nspec
            nsample_loc=int(sizes[0]/ntime)
            timeseq=range(0,sizes[0]-ntime+1,ntime)
            fixinput_ind=range(ntheta_real,ntheta_real+args.nspec)
            allind=set(range(0,args.ntheta))
            time_var_ind=list(allind.difference(set(fixinput_ind)))
            initialvec_theta=(data[timeseq,:])[:,fixinput_ind]
            #nsample*(nspec+1)
            zerotime=np.repeat(0.0,nsample_loc).reshape(nsample_loc,-1)
            initialvec=torch.tensor(np.concatenate((initialvec_theta,zerotime),axis=1)).float()
            timevarinput=data[:,time_var_ind]
            timevec=timevarinput[:,-1]
            deltimevec=timevec[1::]-timevec[0:-1]
            deltimevec=torch.cat((torch.tensor([0.0]),deltimevec),0)
            deltimevec[deltimevec<0]=0
            timevarinput=torch.cat((timevarinput,deltimevec.view(sizes[0],-1)),1)
            #nsample*ntime*(ntheta-1-nspec)
            timevarinput=timevarinput.view(nsample_loc,ntime,len(time_var_ind)+1)
            # print('timevarinput{} initialvec{}'.format(timevarinput.shape,initialvec.shape))
            timevarinput,initialvec,target=timevarinput.to(device),initialvec.to(device),target.to(device)
            output=model(timevarinput,initialvec)
            
        # loss=F.nll_loss(output,target)
        # print('output{} target{}'.format(output.shape,target.shape))
        loss=F.mse_loss(output,target,reduction='mean')
        optimizer.zero_grad()
        loss.backward()
        # plot_grad_flow(model.named_parameters())
        optimizer.step()
        
        if batch_idx % args.log_interval==0:
            if args.lr_print==1:
                lrstr=' lr: '+str(get_lr(optimizer))
            else:
                lrstr=''
                
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss(per sample): {:.6f}{}'.format(
                epoch,batch_idx*len(data),len(train_loader.dataset),
                100. * batch_idx*len(data)/len(train_loader.dataset),loss.item()*ntime,lrstr))
        
        if args.scheduler=='cyclelr':#clclicLR need to make steps for each mini-batch
            scheduler.step()
        
        trainloss.append(loss.item())
    
    return (sum(trainloss)/len(trainloss))*ntime

def test(args,model,test_loader,device,ntime):
    model.eval()
    test_loss=[]
    with torch.no_grad():
        for data, target in test_loader:
            # if args.gpu is not None:
            #     data=data.cuda(args.gpu,non_blocking=True)
            # target=target.cuda(args.gpu,non_blocking=True)
            if args.rnn_struct==0:
                data,target=data.to(device),target.to(device)
                output=model(data)
            else:
                #reshape data for rnn intput
                sizes=data.shape
                ntheta_real=args.ntheta-1-args.nspec
                nsample_loc=int(sizes[0]/ntime)
                timeseq=range(0,sizes[0]-ntime+1,ntime)
                fixinput_ind=range(ntheta_real,ntheta_real+args.nspec)
                allind=set(range(0,args.ntheta))
                time_var_ind=list(allind.difference(set(fixinput_ind)))
                initialvec_theta=(data[timeseq,:])[:,fixinput_ind]
                #nsample*(nspec+1)
                zerotime=np.repeat(0.0,nsample_loc).reshape(nsample_loc,-1)
                initialvec=torch.tensor(np.concatenate((initialvec_theta,zerotime),axis=1)).float()
                timevarinput=data[:,time_var_ind]
                timevec=timevarinput[:,-1]
                deltimevec=timevec[1::]-timevec[0:-1]
                deltimevec=torch.cat((torch.tensor([0.0]),deltimevec),0)
                deltimevec[deltimevec<0]=0
                timevarinput=torch.cat((timevarinput,deltimevec.view(sizes[0],-1)),1)
                #nsample*ntime*(ntheta-1-nspec)
                timevarinput=timevarinput.view(nsample_loc,ntime,len(time_var_ind)+1)
                # print('timevarinput{} initialvec{}'.format(timevarinput.shape,initialvec.shape))
                timevarinput,initialvec,target=timevarinput.to(device),initialvec.to(device),target.to(device)
                output=model(timevarinput,initialvec)

            # test_loss += F.nll_loss(output,target,reduction='sum').item() # sum up batch loss
            test_loss.append(F.mse_loss(output,target,reduction='mean').item()) # sum
    test_loss_mean=sum(test_loss)/len(test_loss)
    print('\nTest set: Average loss (per sample): {:.4f}\n'.format(test_loss_mean*ntime))
    return test_loss_mean*ntime

def parse_func_wrap(parser,termname,args_internal_dict):
    commandstring='--'+termname.replace("_","-")
    defaulval=args_internal_dict[termname][0]
    typedef=args_internal_dict[termname][1]
    parser.add_argument(commandstring,type=typedef,default=defaulval,
                        help='input '+str(termname)+' for training (default: '+str(defaulval)+')')
    
    return(parser)

def save_checkpoint(state,is_best,is_best_train,filename='checkpoint.resnetode.tar'):
    torch.save(state,filename)
    if is_best:
        shutil.copyfile(filename,'model_best.resnetode.tar')
    
    if is_best_train:
        shutil.copyfile(filename,'model_best_train.resnetode.tar')

class batch_sampler_block(Sampler):
    """
    user defined sampler to make sure random sampling blocks
    default replacement=FALSE
    default drop_last=FALSE
    """
    def __init__(self,datasource,blocks,nblock=1):
        """
            datasource: data set
            blocks: block list
            nblocks: number of block for each batch

         EX code:
         datasource=np.array([0,1,2,3,4,5,6,7,8,9])
         blocks=np.array([0,0,1,1,2,2,3,3,4,4])
         nblocks=2
         list(batch_sampler_block(datasource,blocks,nblock=nblocks))
        """
        self.datasource=datasource
        self.blocks=blocks
        self.nblock=nblock
        self.uniblocks=np.unique(self.blocks)
        self.blocksize=int(len(self.datasource)/len(self.uniblocks))
    
    def __iter__(self):
        n=len(self.uniblocks)
        indreorder=torch.randperm(n).tolist()
        batch=[]
        for idx in indreorder:
            indnew=list(range(idx*self.blocksize,(idx+1)*self.blocksize))
            batch.append(indnew)
            if len(batch)==self.nblock: ##number of block in each minibatch
                yield flatten(batch)
                batch=[]
        
        if len(batch) > 0:
            yield flatten(batch)
    
    def __len__(self):
        return self.nblock*self.blocksize

def get_lr(optimizer):#output the lr as scheduler is used
    for param_group in optimizer.param_groups:
        return param_group['lr']

def plot_grad_flow(named_parameters):
    '''From https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/8
    
    Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow
    
    used for gradient checking
    '''
    ave_grads=[]
    max_grads=[]
    layers=[]
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            # print('p {} mean {}'.format(p.shape,p.grad.abs().mean()))
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)),max_grads,alpha=0.1,lw=1,color="c")
    # print('max_grads {}'.format(max_grads.__len__()))
    plt.bar(np.arange(len(max_grads)),ave_grads,alpha=0.1,lw=1,color="b")
    plt.hlines(0,0,len(ave_grads)+1,lw=2,color="k")
    plt.xticks(range(0,len(ave_grads),1),layers,rotation="vertical",fontsize=3)
    plt.xlim(left=0,right=len(ave_grads))
    plt.ylim(bottom= -0.001,top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0],[0],color="c",lw=4),
                Line2D([0],[0],color="b",lw=4),
                Line2D([0],[0],color="k",lw=4)],['max-gradient','mean-gradient','zero-gradient'])
    plt.savefig("test.pdf")
    sys.exit('plotting')

def trans_lin_comb(data,target,args,ntime):
    """
    Data augmentation: produce new training sample
    doesn't follow the format of transformer as the function takes two sample and produce one sample
    """
    ntheta_real=args.ntheta-1-args.nspec
    theta_ind=list(range(0,ntheta_real))
    timevec=data[:,-1]
    unique_time,counts=np.unique(timevec,return_counts=True)
    time_duplicated=unique_time[counts>1]
    # nsamp=math.floor(time_duplicated.__len__()*args.linearcomb_transformp)
    # seletimes=random.sample(range(0,len(time_duplicated)),nsamp)
    for seletime_ind in range(0,len(time_duplicated)):
        seletime=time_duplicated[seletime_ind]
        dupmask=np.equal(np.array(timevec),np.array(seletime))
        duploc=list(np.squeeze(np.where(dupmask)))
        perm=list(permutations(set(range(0,len(duploc))),2))##pair permutations
        nsamp_eachtime=math.floor(len(duploc)*args.linearcomb_transformp)##p percent of the number of time duplicated samples
        seletperms=random.sample(range(0,len(perm)),nsamp_eachtime)
        for permind in seletperms:
            permpair=perm[permind]
            duplocpair=[duploc[x] for x in list(permpair)]
            a_ratio=random.uniform(0,1)
            data[duplocpair[1],theta_ind]=data[duplocpair[0],theta_ind]*a_ratio+data[duplocpair[1],theta_ind]*(1.0-a_ratio)
            target[duplocpair[1],:]=target[duplocpair[0],:]*a_ratio+target[duplocpair[1],:]*(1.0-a_ratio)
    
    return data, target

def trans_time_shift(data,target,args,ntime):
    """
    Data augmentation: produce time shift sample
    doesn't follow the format of transformer
    """
    inputsize=data.shape
    outputsize=target.shape
    nsample=inputsize[0]
    niteration=math.floor(nsample/ntime)
    nblocksamp=math.floor(ntime*args.timeshift_transformp)##for each block
    ini_ind=list(range(inputsize[1]-1-outputsize[1],inputsize[1]-1))
    for iteration in range(0,niteration):
        blockinds=list(range(iteration*ntime,(iteration+1)*ntime))
        perm=list(permutations(set(range(0,ntime)),2))##pair permutations
        seleseq=random.sample(range(0,len(perm)),nblocksamp)
        for seleele in seleseq:
            currpair=perm[seleele]
            currpairind=[blockinds[x] for x in currpair]
            timevec=[data[x,-1] for x in currpairind]
            ind_order=[0, 1]##[small big]
            if timevec[0]>timevec[1]:
                ind_order=[1, 0]
            data[currpairind[ind_order[1]],-1]=timevec[ind_order[1]]-timevec[ind_order[0]]+args.mintime
            data[currpairind[ind_order[1]],ini_ind]=target[currpairind[ind_order[0]],:]
    return data, target

def main():
    # Training settings load-in through command line
    parser=argparse.ArgumentParser(description='PyTorch Example')
    for key in args_internal_dict.keys():
        parser=parse_func_wrap(parser,key,args_internal_dict)
    
    for key in fix_para_dict.keys():
        parser=parse_func_wrap(parser,key,fix_para_dict)
    
    args=parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic=True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    
    # args.distributed=args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node=torch.cuda.device_count()
    # ngpus_per_node=1
    # Since we have ngpus_per_node processes per node, the total world_size
    # needs to be adjusted accordingly
    ## args.world_size=ngpus_per_node*args.world_size
    # Use torch.multiprocessing.spawn to launch distributed processes: the
    # main_worker process function
    ## arg_transf=copy.deepcopy(args)
    ## arg_transf.ngpus_per_node=ngpus_per_node
    ## mp.spawn(main_worker,nprocs=ngpus_per_node,args=(ngpus_per_node,args))
    main_worker(args.gpu,ngpus_per_node,args)

def main_worker(gpu,ngpus_per_node,args):
    global best_acc1
    # args.gpu=gpu
    # if args.gpu is not None:
    #     print("Use GPU: {} for training".format(args.gpu))
    
    # For multiprocessing distributed training, rank needs to be the
    # global rank among all the processes
    ## args.rank=args.rank*ngpus_per_node+gpu
    ## dist.init_process_group(backend=args.dist_backend,init_method="env://",#args.dist_url,
    ## world_size=args.world_size,rank=args.rank)
    
    ##read the matlab matrix as Xvar and ResponseVar
    inputfile=args.inputfile
    f=h5py.File(inputdir+inputfile,'r')
    data=f.get('inputstore')
    Xvar=np.array(data).transpose()
    data=f.get('outputstore')
    ResponseVar=np.array(data).transpose()
    data=f.get('samplevec')
    samplevec=np.array(data)
    samplevec=np.squeeze(samplevec.astype(int)-1)##block index
    data=f.get('parastore')
    parastore=np.array(data)##omega normalizer
    data=f.get('nthetaset')
    nthetaset=int(np.array(data)[0][0])##block number
    data=f.get('ntime')
    ntimetotal=int(np.array(data)[0][0])##time seq including [training part, extrapolation part]
    f.close()
    ntime=args.timetrainlen
    # ResponseVarnorm=(ResponseVar-ResponseVar.mean(axis=0))/ResponseVar.std(axis=0)
    ResponseVarnorm=ResponseVar## the response variable was originally scale by omega {scaling} but not centered. and no more normalization will be done
    ##separation of train and test set
    nsample=(Xvar.shape)[0]
    ntheta=(Xvar.shape)[1]
    nspec=(ResponseVarnorm.shape)[1]
    simusamplevec=np.unique(samplevec)
    separation=['train','validate','test']
    numsamptest_validate=math.floor((simusamplevec.__len__())*args.test_validate_ratio/2)
    sampleind=set(range(0,nsample))
    simusampeind=set(range(0,nthetaset))
    ## a preset whole time range for test, validation (groups)
    simusamplevec_test=random.sample(simusampeind,numsamptest_validate)
    simusamplevec_validate=random.sample(simusampeind.difference(set(simusamplevec_test)),numsamptest_validate)
    ##index of training, testing, and validation
    testind=np.sort(np.where(np.isin(samplevec,simusamplevec_test)))[0]
    validateind=np.sort(np.where(np.isin(samplevec,simusamplevec_validate)))[0]
    testvalidte_ind_union=set(testind)
    testvalidte_ind_union=testvalidte_ind_union.union(set(validateind))
    trainind=np.sort(np.array(list(sampleind.difference(testvalidte_ind_union))))#index for training set
    ntrainset=nthetaset-numsamptest_validate*2
    sizeset={"train": (ntrainset), "validate": (numsamptest_validate), "test": (numsamptest_validate)}
    ind_separa={"train": (trainind), "validate": (validateind), "test": (testind)}
    ##training block index (time range) keep in the training time block
    timeind={x: np.tile(np.concatenate((np.repeat(1,ntime),np.repeat(0,ntimetotal-ntime))),sizeset[x]) for x in separation}
    time_in_ind={}
    time_extr_ind={}
    for x in separation:
        tempind=ind_separa[x]
        time_in_ind[x]=tempind[timeind[x]==1]
        time_extr_ind[x]=tempind[timeind[x]==0]

    ##train validate test "block" ind
    samplevec_separa={x: samplevec[time_in_ind[x]] for x in separation}
    Xvar_separa={x: Xvar[list(ind_separa[x]),:] for x in separation}
    Xvarnorm=np.empty_like(Xvar)
    # Xvar_norm_separa={}
    if args.normalize_flag is 'Y':
        ##the normalization if exist should be after separation of training and testing data to prevent leaking
        ##normalization (X-mean)/sd
        ##normalization include time. Train and test model need to have at least same range or same mean&sd for time
        del(Xvar)
        for x in separation:
            Xvartemp=Xvar_separa[x]
            meanvec=Xvartemp.mean(axis=0)
            stdvec=Xvartemp.std(axis=0)
            for coli in range(0,len(meanvec)):
                Xvartemp[:,coli]=(Xvartemp[:,coli]-meanvec[coli])/stdvec[coli]
            
            # Xvar_norm_separa[x]=copy.deepcopy(temp_norm_mat)
            Xvarnorm[list(ind_separa[x]),:]=copy.deepcopy(Xvartemp)
        
    else:
        # Xvar_norm_separa={x: Xvar_separa[x] for x in separation}
        Xvarnorm=np.copy(Xvar)
        del(Xvar)
    
    #samplevecXX repeat id vector, XXind index vector
    inputwrap={"Xvarnorm": (Xvarnorm),
        "ResponseVar": (ResponseVar),
        "trainind": (trainind),
        "testind": (testind),
        "validateind": (validateind),
        "ind_separa": (ind_separa),
        "time_in_ind": (time_in_ind),
        "time_extr_ind": (time_extr_ind),
        "samplevec": (samplevec),
        # "samplewholeselec": (samplewholeselec),
        "samplevec_separa": (samplevec_separa),
        # "Xvarmean": (Xvarmean),## these two value: Xvarmean, Xvarstd can be used for "new" test data not used in the original normalization
        # "Xvarstd": (Xvarstd),
        "inputfile": (inputfile),
        "ngpus_per_node": (ngpus_per_node),## number of gpus
        "numsamptest_validate": (numsamptest_validate),#number of testing samples
        "timeind": (timeind)
    }
    with open("pickle_inputwrap.dat","wb") as f1:
        pickle.dump(inputwrap,f1,protocol=4)##protocol=4 if there is error: cannot serialize a bytes object larger than 4 GiB
    
    del(inputwrap)
    
    Xtensor={x: torch.Tensor(Xvarnorm[list(time_in_ind[x]),:]) for x in separation}
    Resptensor={x: torch.Tensor(ResponseVar[list(time_in_ind[x]),:]) for x in separation}
    Dataset={x: utils.TensorDataset(Xtensor[x],Resptensor[x]) for x in separation}
    # train_sampler=torch.utils.data.distributed.DistributedSampler(traindataset)
    nblock=int(args.batch_size/ntime)
    # nblocktest=int(args.test_batch_size/ntime)
    # traindataloader=utils.DataLoader(traindataset,batch_size=args.batch_size,
    #     shuffle=(train_sampler is None),num_workers=args.workers,pin_memory=True,sampler=train_sampler)
    #
    # testdataloader=utils.DataLoader(testdataset,batch_size=args.test_batch_size,
    #     shuffle=False,num_workers=args.workers,pin_memory=True,sampler=test_sampler)
    if args.sampler=="block": # block sampler
        sampler={x: batch_sampler_block(Dataset[x],samplevec_separa[x],nblock=nblock) for x in separation}
        dataloader={x: utils.DataLoader(Dataset[x],num_workers=args.workers,pin_memory=True,batch_sampler=sampler[x]) for x in separation}
    elif args.sampler=="individual": #individual random sampler
        dataloader={x: utils.DataLoader(Dataset[x],batch_size=args.batch_size,shuffle=True,num_workers=args.workers,pin_memory=True) for x in separation}

    args.mintime=np.min(Xvarnorm[:,-1])
    ninnersize=int(args.layersize_ratio*ntheta)
    ##store data
    with open("pickle_dataloader.dat","wb") as f1:
        pickle.dump(dataloader,f1,protocol=4)
    
    dimdict={
        "nsample": (nsample,int),
        "ntheta": (ntheta,int),
        "nspec": (nspec,int),
        "ninnersize": (ninnersize,int)
    }
    args.nsample=nsample
    args.ntheta=ntheta
    args.nspec=nspec
    with open("pickle_dimdata.dat","wb") as f3:
        pickle.dump(dimdict,f3,protocol=4)
    
    ##free up some space (not currently set)
    ##create model
    if bool(re.search("[rR]es[Nn]et",args.net_struct)):
        model=models.__dict__[args.net_struct](ninput=ntheta,num_response=nspec,p=args.p,ncellscale=args.layersize_ratio)
    elif args.rnn_struct==1:
        model=models.__dict__[args.net_struct](ntheta=ntheta,nspec=nspec,num_layer=args.num_layer,ncellscale=args.layersize_ratio,p=args.p)
    else:
        model=models.__dict__[args.net_struct](ninput=ntheta,num_response=nspec,nlayer=args.num_layer,p=args.p,ncellscale=args.layersize_ratio,batchnorm_flag=(args.batchnorm_flag is 'Y'))
    
    # model.eval()
    # if args.gpu is not None:
    #     torch.cuda.set_device(args.gpu)
    #     model.cuda(args.gpu)
    #     # When using a single GPU per process and per
    #     # DistributedDataParallel, we need to divide the batch size
    #     # ourselves based on the total number of GPUs we have
    #     args.batch_size=int(args.batch_size/ngpus_per_node)
    #     args.workers=int((args.workers+ngpus_per_node-1)/ngpus_per_node)
    #     model=torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    # else:
    # DistributedDataParallel will divide and allocate batch_size to all
    # available GPUs if device_ids are not set
    
    # model=torch.nn.DataParallel(model).cuda()
    model=torch.nn.DataParallel(model)
    if args.gpu_use==1:
        device=torch.device("cuda:0")#cpu
    else:
        device=torch.device("cpu")
    
    model.to(device)
    if args.optimizer=="sgd":
        optimizer=optim.SGD(model.parameters(),lr=args.learning_rate,momentum=args.momentum)
    elif args.optimizer=="adam":
        optimizer=optim.Adam(model.parameters(),lr=args.learning_rate)
    elif args.optimizer=="nesterov_momentum":
        optimizer=optim.SGD(model.parameters(),lr=args.learning_rate,momentum=args.momentum,nesterov=True)
    
    if args.scheduler=='step':
        scheduler=lr_scheduler.StepLR(optimizer,step_size=200,gamma=0.5)
    elif args.scheduler=='plateau':
        scheduler=lr_scheduler.ReduceLROnPlateau(optimizer,'min',factor=0.5)
    elif args.scheduler=='cyclelr':
        scheduler=lr_scheduler.CyclicLR(optimizer,args.learning_rate/100,args.learning_rate,step_size_up=1000,cycle_momentum=False,mode="triangular2")
    else:
        scheduler=None
    
    cudnn.benchmark=True
    ##model training
    for epoch in range(1,args.epochs+1):
        msetr=train(args,model,dataloader["train"],optimizer,epoch,device,ntime,scheduler)
        msevalidate=test(args,model,dataloader["validate"],device,ntime)
        if scheduler is not None:
            if args.scheduler=='step':
                scheduler.step()
            elif args.scheduler=='plateau':
                scheduler.step(msevalidate)##based on validation set. This is fine as we use train|validate|test separation
        if epoch==1:
            best_msevalidate=msevalidate
            best_train_mse=msetr
        
        # is_best=acc1>best_acc1
        is_best=msevalidate<best_msevalidate
        is_best_train=msetr<best_train_mse
        best_msevalidate=min(msevalidate,best_msevalidate)
        best_train_mse=min(msetr,best_train_mse)
        save_checkpoint({
            'epoch': epoch,
            'arch': args.net_struct,
            'state_dict': model.state_dict(),
            'best_acc1': best_msevalidate,
            'best_acctr': best_train_mse,
            'optimizer': optimizer.state_dict(),
            'args_input': args,
        },is_best,is_best_train)
    
    print('\nFinal test MSE\n')
    acctest=test(args,model,dataloader["test"],device,ntime)

if __name__ == '__main__':
    main()
