import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from IPython.core.debugger import Tracer

    
class ContrastiveLossWithAttention(nn.Module):
    r"""
    """
    def __init__(self):
        super(ContrastiveLossWithAttention, self).__init__()
    
    
       
        

    def forward(self, pred_dsmat: Tensor, gt_perm: Tensor,pred_perm:Tensor,src_ns: Tensor, tgt_ns: Tensor, beta_value) -> Tensor:
        
        batch_num = pred_dsmat.shape[0]
      
        pred_dsmat = torch.clamp(pred_dsmat, min=0.0, max=1.0)

        
        all_ones_tensor= torch.ones_like(gt_perm)
        all_zeros_tensor = torch.zeros_like(gt_perm)
        gt_predicted_values = torch.mul(pred_dsmat,gt_perm)
        
        
        beta = torch.full_like(pred_dsmat,beta_value)
        
        column_gt_values = torch.matmul(torch.ones_like(gt_predicted_values),gt_predicted_values)
        column_gt_values_minus_beta=column_gt_values-beta
        gt_available_columns= torch.matmul(torch.ones_like(gt_predicted_values),gt_perm)
        
        
        attention_tgt = torch.mul(torch.ge(pred_dsmat,column_gt_values_minus_beta).float(),gt_available_columns)
        attention_tgt_without_gt = attention_tgt - gt_perm
        attention_tgt_predicted_values_without_gt = torch.mul(attention_tgt_without_gt,pred_dsmat)  
        attention_tgt_negatives_selected = attention_tgt_predicted_values_without_gt
        
        
        row_gt_values = torch.matmul(gt_predicted_values,torch.ones_like(gt_predicted_values))
        row_gt_values_minus_beta = row_gt_values-beta
        gt_available_rows = torch.matmul(gt_perm,torch.ones_like(gt_predicted_values))
       
        attention_src = torch.mul(torch.ge(pred_dsmat,row_gt_values_minus_beta).float(),gt_available_rows)
        attention_src_without_gt = attention_src - gt_perm
        attention_src_predicted_values_without_gt = torch.mul(attention_src_without_gt,pred_dsmat)
        attention_src_negatives_selected = attention_src_predicted_values_without_gt
      
        
        def calculateLoss(pred_dsmat,gt_perm,gt_predicted_values, row_gt_values, column_gt_values, attention_src_negatives_selected, attention_tgt_negatives_selected,all_zeros_tensor,b):
           
         
            corresponding_target_indices = (gt_perm == 1).nonzero(as_tuple=True)[1]
            
            attention_src_negatives_selected_squared = torch.square(attention_src_negatives_selected)
            attention_tgt_negatives_selected_squared = torch.square(attention_tgt_negatives_selected)
            gt_predicted_values_squared = torch.square(gt_predicted_values)
            
            
            src_negative_sum = torch.sum(attention_src_negatives_selected_squared,1)
            src_positive_sum = torch.sum(gt_predicted_values_squared,1)
            tgt_negative_sum = torch.sum(attention_tgt_negatives_selected_squared,0)
            
            corresponding_tgt_negative_sum = torch.index_select(tgt_negative_sum, 0, corresponding_target_indices)
            overall_negative_sum = src_negative_sum + corresponding_tgt_negative_sum 
            denominator = 1+overall_negative_sum
            probability = src_positive_sum/denominator
            elementwise_loss = -0.5 * torch.log(probability)
            
            loss = torch.sum(elementwise_loss)
            return loss
            
        
    
        
       
        loss = torch.tensor(0.).to(pred_dsmat.device)
        n_sum = torch.zeros_like(loss)
        for b in range(batch_num):
            
           
            loss += calculateLoss(pred_dsmat[b, :src_ns[b], :tgt_ns[b]],gt_perm[b, :src_ns[b], :tgt_ns[b]], gt_predicted_values[b, :src_ns[b], :tgt_ns[b]], row_gt_values[b, :src_ns[b], :tgt_ns[b]], column_gt_values[b, :src_ns[b], :tgt_ns[b]], attention_src_negatives_selected[b, :src_ns[b], :tgt_ns[b]], attention_tgt_negatives_selected[b, :src_ns[b], :tgt_ns[b]], all_zeros_tensor[b, :src_ns[b], :tgt_ns[b]],b)
            
            n_sum += src_ns[b].to(n_sum.dtype).to(pred_dsmat.device)
         
            
        return loss/n_sum