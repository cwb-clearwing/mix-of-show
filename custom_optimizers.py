from torch.optim.optimizer import Optimizer, required
import torch
import math
import copy
import time
from torch import nn
import numpy as np

class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.0, correct_bias=False):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def reset_state(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state["exp_avg"] = torch.zeros_like(p.data)
                state["exp_avg_sq"] = torch.zeros_like(p.data)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if 'correct_bias' in group and group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)
                if group["weight_decay"] > 0.0:
                    p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])

        return loss
    
class SGD(Optimizer):
    def __init__(self, params, lr, weight_decay, betas=(0.9, 0.98), eps=1e-6, correct_bias=True):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super().__init__(params, defaults)
    def step(self,closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                if group["weight_decay"] > 0.0:
                    p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])
                p.data.add_(p.grad.data, alpha=-group['lr'])
            
class RGD_Opt(Optimizer):

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.999), momentum=0, weight_decay=0.01            
                 ,epsilon=1e-6,theta=0.1,eps=1e-6,reg=0,correct_bias=True,opt_type="RGD"):
        """

        :param params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        :param lr (float, optional): learning rate (default: 1e-1)
        :param momentum: (float, optional): momentum factor (default: 0)
        :param weight_decay: (float, optional): weigt decay factor (default: 0)
        :param epsilon: (float, optional): momentum factor (default: 1e-4)
        :param update_freq: (int, optional): update frequency to compute inverse (default: 1)
        """
        defaults = dict(lr=lr, betas=betas, eps=eps, reg=reg, momentum=momentum, weight_decay=weight_decay, correct_bias=correct_bias, epsilon=epsilon, timer1=0, timer2=0, theta=theta,opt_type=opt_type)
        super(RGD_Opt, self).__init__(params, defaults)                                                                                                                                                                                                                                                     
    
    def reset_state(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                temp = p.grad.detach().clone()
                temp = temp.view(temp.size(0), -1)
                n,m = temp.size()
                state['step'] = 0
                state["Lt"] = (group["reg"]) * torch.eye(n=n, device=p.device)
                state["Rt"] = (group["reg"]) * torch.eye(n=m, device=p.device)
                state["pos_l"] = 0 * torch.eye(n=n, device=p.device)
                state["pos_r"] = 0 * torch.eye(n=m, device=p.device)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            opt_type = group['opt_type']
            for p in group["params"]:
                if p.grad is None:
                    continue
                if hasattr(p,'is_matrix') and p.is_matrix:
                    state = self.state[p]
                    temp = p.grad.detach().clone()
                    original_weight_shape = temp.shape
                    temp = temp.view(temp.size(0), -1)
                    n,m = temp.size()
                    if len(state) == 0:
                        state["step"] = 0
                        state["L0"] = (group["reg"]) * torch.eye(n=n, device=p.device)
                        state["R0"] = (group["reg"]) * torch.eye(n=m, device=p.device)
                        state["Lt"] = (group["reg"]) * torch.eye(n=n, device=p.device)
                        state["Rt"] = (group["reg"]) * torch.eye(n=m, device=p.device)
                        state["pos_l"] = torch.zeros(n,n, device=p.device)
                        state["pos_r"] = torch.zeros(m,m, device=p.device)
                        state["neg_l"] = torch.zeros(n,n, device=p.device)
                        state["neg_r"] = torch.zeros(m,m, device=p.device)
                        state["exp_avg"] = torch.zeros_like(temp.data)
                        state["step-size"] = 1
                    if hasattr(p,'is_matrix') and p.is_matrix:
                        if opt_type=="PRGD":
                            state["pos_l"] = ((group["epsilon"]) * torch.eye(n=n, device=p.device) + torch.diag(torch.diag(temp.matmul(temp.t()), 0))) ** (+1/4)
                            state["neg_l"] = torch.diag(torch.diag(state["pos_l"], 0) ** (-1))
                            state["pos_r"] = ((group["epsilon"]) * torch.eye(n=m, device=p.device) + torch.diag(torch.diag(temp.t().matmul(temp), 0))) ** (+1/4)
                            state["neg_r"] = torch.diag(torch.diag(state["pos_r"], 0) ** (-1))
                            p.grad = (state["neg_l"].matmul(temp.matmul(state["neg_r"]))).view(original_weight_shape)
                        elif opt_type=="RAdaGrad":
                            beta2 = group["betas"][1]
                            beta1 = group["betas"][0]
                            skip = 1
                            Lt = state["Lt"]
                            Rt = state["Rt"]
                            # bias correction for RAdaGrad
                            if 'correct_bias' in group and group["correct_bias"]:  # No bias correction for Bert
                                bias_correction2 = 1.0 - beta2 ** (state["step"] + 1)
                            else:
                                bias_correction2 = 1.0

                            if state["step"] % skip == 0:
                                Lt.mul_(beta2).add_(torch.diag(torch.diag(p.grad.matmul(p.grad.t()), 0)), alpha=(1-beta2))
                                Rt.mul_(beta2).add_(torch.diag(torch.diag(p.grad.t().matmul(p.grad), 0)), alpha=(1-beta2))

                            # Revise the estimate 2nd momentum to L0 + E[GG*]
                            compute_Lt = Lt + state["L0"] * (bias_correction2 - beta2 ** (state["step"] + 1))
                            compute_Lt.div_(bias_correction2)
                            compute_Rt = Rt + state["R0"] * (bias_correction2 - beta2 ** (state["step"] + 1))
                            compute_Rt.div_(bias_correction2)
                            
                            state["pos_l"] = torch.diag((torch.diag(compute_Lt, 0) ** (+1/4)))
                            state["neg_l"] = torch.diag(torch.diag(state["pos_l"], 0) ** (-1))
                            state["pos_r"] = torch.diag((torch.diag(compute_Rt, 0) ** (+1/4)))
                            state["neg_r"] = torch.diag(torch.diag(state["pos_r"], 0) ** (-1))
                            
                            # Pre Grad : L grad R
                            pre_grad = state["neg_l"]@p.grad@state["neg_r"]
                            p.grad = pre_grad.view(original_weight_shape)
                            p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])
                        elif opt_type=="RAdam":
                            ### 超参数部分
                            # 防止溢出而设置的decay系数beta
                            beta2 = group["betas"][1]
                            beta1 = group["betas"][0]
                            # 跳过步数
                            skip = 1
                            ### 超参数结束
                            # bias correction for RAdam
                            if 'correct_bias' in group and group["correct_bias"]:  # No bias correction for Bert
                                bias_correction1 = 1.0 - beta1 ** (state["step"] + 1)
                                bias_correction2 = 1.0 - beta2 ** (state["step"] + 1)
                            else:
                                bias_correction1 = 1.0
                                bias_correction2 = 1.0
                            
                            # 梯度累加量
                            exp_avg = state["exp_avg"]
                            exp_avg.mul_(beta1).add_(temp, alpha=1.0 - beta1)
                            #exp_avg.div_(bias_correction1)
                            
                            # 对Lt和Rt进行累加，并且保持量级维持在AdamW相同的水平
                            Lt = state["Lt"]
                            Rt = state["Rt"]

                            if state["step"] % skip == 0:
                                Lt.mul_(beta2).add_(torch.diag(torch.diag(temp.matmul(temp.t()), 0)), alpha=(1-beta2))
                                Rt.mul_(beta2).add_(torch.diag(torch.diag(temp.t().matmul(temp), 0)), alpha=(1-beta2))
                                
                            # 实际计算中需要把累加量修正到 L0 + GG* 的期望量
                            compute_Lt = Lt + state["L0"] * (bias_correction2 - beta2 ** (state["step"] + 1))
                            compute_Lt.div_(bias_correction2)
                            compute_Rt = Rt + state["R0"] * (bias_correction2 - beta2 ** (state["step"] + 1))
                            compute_Rt.div_(bias_correction2)
                            compute_exp = exp_avg / bias_correction1
                            
                            state["pos_l"] = torch.diag((torch.diag(compute_Lt, 0) ** (+1/4)))
                            state["neg_l"] = torch.diag(torch.diag(state["pos_l"], 0) ** (-1))
                            state["pos_r"] = torch.diag((torch.diag(compute_Rt, 0) ** (+1/4)))
                            state["neg_r"] = torch.diag(torch.diag(state["pos_r"], 0) ** (-1))
                            
                            pre_grad = state["neg_l"]@compute_exp@state["neg_r"]
                            p.grad = pre_grad.view(original_weight_shape)
                            p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])
                        else:
                            state["pos_l"] = torch.eye(n=n, device=p.device)
                            state["neg_l"] = state["pos_l"]
                            state["neg_r"] = torch.eye(n=m, device=p.device)
                            state["pos_r"] = state["neg_r"]
                        grad = p.grad.data
                        if grad.is_sparse:
                            raise RuntimeError("Rgd does not support sparse gradients, please consider SparseAdam instead")
                        state["step"] += 1
                        step_size = group["lr"]
                        p.data.add_(-grad, alpha=step_size)
                        # Post-integration step
                        original_shape = p.grad.shape
                        Dim2_shape = p.grad.view(p.grad.size(0), -1).size()
                        n,m = Dim2_shape
                        if not hasattr(p,'is_matrix') or not p.is_matrix:
                            continue
                        grad = p.grad.clone()

                        if p.s == None:
                            state["step"] = 0
                            if hasattr(p,'is_matrix') and p.is_matrix:
                                state["svd"] = torch.linalg.svd(p.data.view(Dim2_shape))
                                u, s ,vh = state["svd"]
                                p.s = s
                                p.u = u
                                p.vh = vh
                                p.data = (u[:,0:p.r].matmul(torch.diag(s[0:p.r]).matmul(vh[0:p.r,:]))).view(original_shape)

                        if hasattr(p,'is_matrix') and p.is_matrix:
                            grad1 = p.data.view(Dim2_shape).clone()
                            M_1 = p.u[:,0:p.r].t()@state["pos_l"]@p.u[:,0:p.r]
                            M_1_inv = torch.inverse(M_1)
                            y1h_part = M_1_inv@p.u[:,0:p.r].t()@state["pos_l"]@grad1
                            y1h = y1h_part@(torch.eye(n=m, device=p.device) - p.vh[0:p.r,:].t()@p.vh[0:p.r,:])
                            M_2 = p.vh[0:p.r,:]@state["pos_r"]@p.vh[0:p.r,:].t()
                            M_2_inv = torch.inverse(M_2)
                            y2_part = grad1@state["pos_r"]@p.vh[0:p.r,:].t()@M_2_inv
                            y2 = (torch.eye(n=n, device=p.device) - p.u[:,0:p.r]@p.u[:,0:p.r].t())@y2_part
                            k0 = y1h_part@p.vh[0:p.r,:].t() + p.u[:,0:p.r].t()@y2_part - y1h_part@state["pos_r"]@p.vh[0:p.r,:].t()@M_2_inv
                            # Use 2*qr to replace svd
                            q1,k1 = torch.linalg.qr(y1h.t())
                            q2,k2 = torch.linalg.qr(y2)
                            M = torch.cat((torch.cat((k0,k2),0),torch.cat((k1.t(),torch.zeros(k0.size()[0],k0.size()[0],device=p.device)),0)),1)
                            Small = torch.clone(M)
                            u_m, s, vh_m = torch.linalg.svd(Small)
                            u = torch.cat((p.u[:,0:p.r], q2),1)@u_m
                            vh = vh_m@torch.cat((p.vh[0:p.r,:], q1.t()),0)
                            rmax = p.r
                            tmp = 0.0

                            compression = False
                            if compression and rmax >= p.minimum_rank: 
                                tol = group["theta"] * torch.linalg.norm(s)
                                rmax = int(np.floor(s.shape[0] / 2))
                                for j in range(0, 2 * rmax - 1):
                                    tmp = torch.linalg.norm(s[j:2 * rmax - 1])
                                    if tmp < tol:
                                        rmax = j
                                        break
                                rmax = min([rmax, p.r])
                                rmax = max([rmax, 2])
                                p.s[:rmax] = s[:rmax]
                                p.u[:,:rmax] = u[:,:rmax]
                                p.vh[:rmax,:] = vh[:rmax, :]
                                p.data = (p.u[:,:rmax]@torch.diag(s[:rmax])@p.vh[:rmax,:]).view(original_shape)
                                p.r = rmax
                            else:
                                p.s[:rmax] = s[:rmax]
                                p.u[:,:rmax] = u[:,:rmax]
                                p.vh[:rmax,:] = vh[:rmax, :]
                                p.data = (p.u[:,:rmax]@torch.diag(s[:rmax])@p.vh[:rmax,:]).view(original_shape)    
        return loss