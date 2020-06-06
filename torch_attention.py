from __future__ import print_function
import input
import time
from config import *
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerEncoderModel(nn.Module):
 
    def __init__(self, ninp, nhead, nhid, nlayers , dropout=dropout_rate):
        super(TransformerEncoderModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'TransformerEncoder'
        model_size = 2
        self.mut1_encoder = nn.Linear(35,35*model_size)
        torch.nn.init.xavier_uniform_(self.mut1_encoder.weight)
        torch.nn.init.normal_(self.mut1_encoder.bias)
        self.mut2_encoder = nn.Linear(35,35*model_size)
        torch.nn.init.xavier_uniform_(self.mut2_encoder.weight)
        torch.nn.init.normal_(self.mut2_encoder.bias)
        self.mut3_encoder = nn.Linear(35,35*model_size)
        torch.nn.init.xavier_uniform_(self.mut3_encoder.weight)
        torch.nn.init.normal_(self.mut3_encoder.bias)
        self.mut4_encoder = nn.Linear(35,35*model_size)
        torch.nn.init.xavier_uniform_(self.mut4_encoder.weight)
        torch.nn.init.normal_(self.mut4_encoder.bias)

        self.spec_encoder = nn.Linear(34,34*model_size)
        torch.nn.init.xavier_uniform_(self.spec_encoder.weight)
        torch.nn.init.normal_(self.spec_encoder.bias)
        self.sim_encoder = nn.Linear(15,15*model_size)
        torch.nn.init.xavier_uniform_(self.sim_encoder.weight)
        torch.nn.init.normal_(self.sim_encoder.bias)
        self.comp_encoder = nn.Linear(37,37*model_size)
        torch.nn.init.xavier_uniform_(self.comp_encoder.weight)
        torch.nn.init.normal_(self.comp_encoder.bias)

        # self.cat_encoder = nn.Linear(9,ninp,bias= False)
        # self.small_encoder = nn.Linear(121,ninp,bias= False)
        # self.big_encoder = nn.Linear(418,ninp,bias= False)
        # self.final_linear = nn.Linear(ninp,2,bias=False)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        self.mut_linear = nn.Linear(4*35*model_size,35*model_size)
        torch.nn.init.xavier_uniform_(self.mut_linear.weight)
        torch.nn.init.normal_(self.mut_linear.bias)

        self.spec_mut_linear = nn.Linear(69*model_size,32*model_size)
        torch.nn.init.xavier_uniform_(self.spec_mut_linear.weight)
        torch.nn.init.normal_(self.spec_mut_linear.bias)

        self.total_linear = nn.Linear(84*model_size,128)
        torch.nn.init.xavier_uniform_(self.total_linear.weight)
        torch.nn.init.normal_(self.total_linear.bias)
        self.final_linear = nn.Linear(128,2)
        torch.nn.init.xavier_uniform_(self.final_linear.weight)
        torch.nn.init.zeros_(self.final_linear.bias)
        # encoder_mut_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        # encoder_fb_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        # encoder_total_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        # self.transformer_mut_encoder = TransformerEncoder(encoder_mut_layers, nlayers)
        # self.transformer_fb_encoder = TransformerEncoder(encoder_fb_layers, nlayers)
        # self.transformer_total_encoder = TransformerEncoder(encoder_total_layers, nlayers)

    def post_fc(self,src):
        src = self.dropout(src)
        src = self.relu(src)
        return src

    def forward(self, src):
        mut1 = src[:,34:69]
        mut1 = self.mut1_encoder(mut1)
        mut1 = self.post_fc(mut1)

        mut2 = src[:,69:104]
        mut2 = self.mut2_encoder(mut2)
        mut2 = self.post_fc(mut2)

        mut3 = src[:,104:139]
        mut3 = self.mut3_encoder(mut3)
        mut3 = self.post_fc(mut3)

        mut4 = src[:,139:174]
        mut4 = self.mut4_encoder(mut4)
        mut4 = self.post_fc(mut4)
        # mut = self.transformer_mut_encoder(torch.stack((mut1, mut2, mut3, mut4),1))
        # mut = torch.sum(mut,dim = 1)
        mut = self.mut_linear(torch.cat([mut1, mut2, mut3, mut4],1))
        mut = self.post_fc(mut)
        # fb_cat = src[:,226:235]
        # fb_cat = self.cat_encoder(fb_cat)
        # fb_small = src[:,235:356]
        # fb_small = self.small_encoder(fb_small)
        # fb_big = src[:,-418:]
        # fb_big = self.big_encoder(fb_big)
        # fb = self.transformer_fb_encoder(torch.stack((fb_cat,fb_small,fb_big),1))
        # fb = torch.sum(fb,dim=1)

        comp = src[:,174:211]
        comp = self.comp_encoder(comp)
        # print(comp[0,:])
        comp = self.post_fc(comp)
        # print(comp[0,:])
        sim = src[:,211:226]
        sim = self.sim_encoder(sim)
        # print(sim[0,:])
        sim = self.post_fc(sim)
        # print(sim[0,:])
        spec = src[:,:34]
        spec = self.spec_encoder(spec)
        spec = self.post_fc(spec)

        spec_mut = self.spec_mut_linear(torch.cat([mut,spec],1))
        spec_mut = self.post_fc(spec_mut)

        # output = self.transformer_total_encoder(torch.stack((mut,fb,comp,sim,spec),1))
        output = self.total_linear(torch.cat([spec_mut,sim,comp],1))
        output = self.post_fc(output)

        output = self.final_linear(output)
        # size = output.size()
        # output = self.final_linear(output.view(size[0],size[1]*size[2]))
        # output = output.softmax(1)
        return output

'''
Main function for executing the model
@param trainFile: .csv filename of training features
@param trainLabelFile: .csv filename of training labels
@param testFile: .csv filename of test features
@param testLabelFile: .csv filename of test labels
@param groupFile: group filename
@param suspFile: output file name storing the prediction results of model, typically the results name will be suspFile+epoch_num
@param loss: the loss function configurations controlled in command
@param featureNum: number of input features
@param nodeNum: hidden node number per layer 
@return N/A
'''  
def run(trainFile, trainLabelFile, testFile,testLabelFile, groupFile, suspFile,loss, featureNum, nodeNum):
    # Construct model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerEncoderModel(32, 2, 128, 1)
    datasets = input.read_data_sets(trainFile, trainLabelFile, testFile, testLabelFile, groupFile)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    model.to(device)
    if(loss == 1):
        criterion = nn.CrossEntropyLoss()
    else:
        print("ERROR: other loss funcs except softmax are not supported!")
        exit(-1)
    for epoch in range(training_epochs):
        model.train()
        total_loss = 0
        start_time = time.time()
        total_batch = int(datasets.train.num_instances/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y ,batch_g= datasets.train.next_batch(batch_size)
            batch_x = torch.Tensor(batch_x).to(device)
            # print(batch_y)
            batch_y = torch.Tensor(batch_y).argmax(1).to(device)
            # Run optimization op (backprop) and cost op (to get loss value)
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            # print(output)
            # Compute average loss
            total_loss += loss.item()
            log_interval = 1
            # if i % log_interval == 0 and i > 0:
            if False:
                cur_loss = total_loss / i
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | '
                    'lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                        epoch, i, total_batch, scheduler.get_lr()[0],
                        elapsed * 1000 / log_interval,
                        cur_loss, math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()
        # Display logs per epoch step
        
        
        if epoch % dump_step ==(dump_step-1):
            #Write Result
            model.eval() # Turn on the evaluation mode
            total_loss = 0.
            with torch.no_grad():
                data = torch.Tensor(datasets.test.instances).to(device)
                label =  torch.Tensor(datasets.test.labels).argmax(1).to(device)
                result = model(data)
                # print(result.size())
                result = nn.Softmax(1)(result)
                total_loss += criterion(result, label).item()
            with open(suspFile+'-'+str(epoch+1),'w') as f:
                for susp in result[:,0]:
                    # print(susp.item())
                    f.write(str(susp.item())+'\n')
        # scheduler.step()

        #print(" Optimization Finished!")