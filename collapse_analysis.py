import torch
import tqdm
import numpy as np
import torch.nn.functional as F
from scipy.sparse.linalg import svds
import torch.nn as nn
import math

class graphs():
    def __init__(self):
        self.accuracy     = []
        self.loss         = []
        self.reg_loss     = []

        # NC1: Sw -> within-class covariance, Sb -> between-class covariance
        self.Sw_invSb     = []
        self.avg_dis      = []
        self.tr_Sw      = []

        # NC2 : M -> class means; W: classifiers
        self.norm_M_CoV   = []
        self.norm_W_CoV   = []
        self.cos_M        = []
        self.cos_W        = []
        self.degree_M     = []
        self.degree_W     = []
        self.norm_M_avg   = []
        self.norm_W_avg   = []
        self.norm_W_std   = []

        # NC3
        self.W_M_dist     = []
        
        # NC4: Nearest Class Center
        self.NCC_mismatch = []

        # Decomposition
        self.MSE_wd_features = []
        self.LNC1 = []
        self.LNC23 = []
        self.Lperp = []
        
def graphNC(h, target, output, C, classifier, graphs):
    N             = [0 for _ in range(C)]
    mean          = [0 for _ in range(C)]
    dis           = [0 for _ in range(C)]
    Sw            = 0
    loss          = 0
    net_correct   = 0
    NCC_match_net = 0

    for computation in ['Mean','Cov']:
        for c in range(C):
            # features belonging to class c
            idxs = (target == c).nonzero(as_tuple=True)[0]
            if len(idxs) == 0: # If no class-c in this batch
                continue
            h_c = h[idxs,:] # B CHW

            if computation == 'Mean':
                # update class means
                mean[c] += torch.sum(h_c, dim=0) # CHW
                N[c] += h_c.shape[0]
                
            elif computation == 'Cov':
                # update within-class cov
                z = h_c - mean[c].unsqueeze(0) # B CHW
                dis[c] += torch.norm(h_c - mean[c], dim=1).sum()
                cov = torch.matmul(z.unsqueeze(-1), # B CHW 1
                                z.unsqueeze(1))  # B 1 CHW
                Sw += torch.sum(cov, dim=0)
                # during calculation of within-class covariance, calculate:
                # 1) network's accuracy
                net_pred = torch.argmax(output[idxs,:], dim=1)
                net_correct += sum(net_pred==target[idxs]).item()
                # 2) agreement between prediction and nearest class center
                NCC_scores = torch.stack([torch.norm(h_c[i,:] - M.T,dim=1) \
                                        for i in range(h_c.shape[0])])
                NCC_pred = torch.argmin(NCC_scores, dim=1)
                NCC_match_net += sum(NCC_pred==net_pred).item()
            
        if computation == 'Mean':
            for c in range(C):
                mean[c] /= N[c]
                M = torch.stack(mean).T
            loss /= sum(N)
        elif computation == 'Cov':
            Sw /= sum(N)
            for c in range(C):   
                dis[c] /= N[c]
                dis[c] = round(dis[c].item(), 4)
    # print(dis)
    graphs.tr_Sw.append(torch.trace(Sw).item())
    avg_dis = np.mean(dis)
    graphs.avg_dis.append(avg_dis)
    graphs.accuracy.append(net_correct/sum(N))
    graphs.NCC_mismatch.append(1-NCC_match_net/sum(N))

    # global mean
    muG = torch.mean(M, dim=1, keepdim=True) # CHW 1

    # between-class covariance
    M_ = M - muG
    Sb = torch.matmul(M_, M_.T) / C

    # avg norm
    W  = classifier.weight
    M_norms = torch.norm(M_,  dim=0)
    W_norms = torch.norm(W.T, dim=0)
    graphs.norm_M_avg.append((torch.mean(M_norms)).item())
    graphs.norm_W_avg.append((torch.mean(W_norms)).item())
    graphs.norm_W_std.append((torch.std(W_norms)).item())
    graphs.norm_M_CoV.append((torch.std(M_norms)/torch.mean(M_norms)).item())
    graphs.norm_W_CoV.append((torch.std(W_norms)/torch.mean(W_norms)).item())

    # tr{Sw Sb^-1}
    Sw = Sw.cpu().numpy()
    Sb = Sb.cpu().numpy()
    eigvec, eigval, _ = svds(Sb, k=C-1)
    inv_Sb = eigvec @ np.diag(eigval**(-1)) @ eigvec.T 
    graphs.Sw_invSb.append(np.trace(Sw @ inv_Sb))
    
    # inv_Sb = torch.pinverse(Sb)
    # Sw_invSb = torch.trace(Sw @ inv_Sb).item()
    # graphs.Sw_invSb.append(Sw_invSb)

    # ||W^T - M_||
    normalized_M = M_ / torch.norm(M_,'fro')
    normalized_W = W.T / torch.norm(W.T,'fro')
    graphs.W_M_dist.append((torch.norm(normalized_W - normalized_M)**2).item())

    # mutual coherence
    def coherence(V): 
        G = V.T @ V
        G += torch.ones((C,C)).cuda() / (C-1)
        G -= torch.diag(torch.diag(G))
        return torch.norm(G,1).item() / (C*(C-1))
    def degree(V): 
        G = V.T @ V
        G -= torch.diag(torch.diag(G))
        avg_cos = torch.norm(G,1).item() / (C*(C-1))
        degree = math.degrees(math.acos(avg_cos))
        return degree
    graphs.cos_M.append(coherence(M_/M_norms))
    graphs.cos_W.append(coherence(W.T/W_norms))
    graphs.degree_M.append(degree(M_/M_norms))
    graphs.degree_W.append(degree(W.T/W_norms))

def MSE_decom(labels, H, W, b, C, N, M):
    W = W.T # p * K
    eW = torch.cat((W, b.unsqueeze(0))) # extended W, (p+1) * K
    Y = F.one_hot(labels, num_classes=C).float() # n * K
    eH = torch.cat((H, torch.ones((H.size(0), 1)).cuda()), dim=1) # extended H, n * (p+1)
    L2loss = (Y-eH@eW).norm(p=2) ** 2
    St = eH.T @ eH
    inv_St = torch.inverse(St)
    eW_LS = inv_St @ eH.T @ Y
    # W_LS = torch.pinverse(H) @ Y
    eH_hat = []
    eM = torch.cat((M, torch.ones((M.size(0), 1)).cuda()), dim=1) # extended M, K * (p+1)
    for i in range(labels.size(0)):
        eH_hat.append(eM[labels[i]])
    eH_hat = torch.stack(eH_hat)
    print(eH_hat.size())
    LNC1 = ((eH - eH_hat) @ eW_LS).norm(p=2) ** 2
    LNC23 = (Y - eH_hat@eW_LS).norm(p=2) ** 2
    Lperp = (eH @ (eW-eW_LS)).norm(p=2) ** 2
    print('||W_LS-W||', (eW-eW_LS).norm())
    return L2loss.item(), LNC1.item(), LNC23.item(), Lperp.item()

def myNC1(labels, H, C, M):
    H = H.cpu()
    n = H.size(0)
    one_n = torch.ones((n, 1))
    mu_G = H.mean(dim=0).unsqueeze(0).detach().cpu() # 1 * P
    # print('H:', H.device,'one_n:', one_n.device, 'muG:', mu_G.device)
    H_hat = H - one_n @ mu_G
    Y = F.one_hot(labels, num_classes=C).float().cpu() # n * K
    Y_hat = Y - 1/n * one_n @ one_n.T @ Y
    St = H_hat.T @ H_hat
    Sb = H_hat.T @ Y_hat @ Y_hat.T @ H_hat
    H_bar = []
    for i in range(n):
        H_bar.append(M[labels[i]])
    H_bar = torch.stack(H_bar)
    H_bar = H_bar.cpu()
    Sw = (H - H_bar).T @ (H - H_bar)
    # Sw = Sw.detach().cpu().numpy()
    # Sb = Sb.cpu().numpy()
    # eigvec, eigval, _ = svds(Sb, k=C-1)
    # # print('eigen vector:', eigvec, 'eigen value:', eigval)
    # inv_Sb = eigvec @ np.diag(eigval**(-1)) @ eigvec.T 
    inv_Sb = torch.pinverse(Sb)
    Sw_invSb = torch.trace(Sw @ inv_Sb).item()
    return Sw_invSb
    
class graphs():
    def __init__(self):
        self.accuracy     = []
        self.loss         = []
        self.reg_loss     = []

        # NC1
        self.Sw_invSb     = []
        self.avg_dis      = []
        self.Sw_norm      = []

        # NC2
        self.norm_M_CoV   = []
        self.norm_W_CoV   = []
        self.cos_M        = []
        self.cos_W        = []
        self.degree_M     = []
        self.degree_W     = []
        self.norm_M_avg   = []
        self.norm_W_avg   = []
        self.norm_W_std   = []

        # NC3
        self.W_M_dist     = []
        
        # NC4
        self.NCC_mismatch = []

        # Decomposition
        self.MSE_wd_features = []
        self.LNC1 = []
        self.LNC23 = []
        self.Lperp = []