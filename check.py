#trash file, ignore
import torch, numpy as np, json
from model.gnn_uc import GNN_UC
from model.cnn_bilstm_baseline import CNN_BiLSTM_Baseline
from dataset.graph_dataset import UCGraphDataset
from torch_geometric.loader import DataLoader
from utils.dc_powerflow import build_ptdf_matrix
from data.ieee30_system import GENERATORS, LINES, N_BUS, N_GEN, T_HORIZON
import gurobipy as gp
from gurobipy import GRB
import argparse

def lp_cost(z, net_load, contingency, costs):
    removed = None if contingency == -1 else contingency
    gen_bus = [g['bus']-1 for g in GENERATORS]
    line_rates = [ln[5] for ln in LINES]
    try:
        PTDF = build_ptdf_matrix(N_BUS, LINES, ref_bus_idx=0, removed_line_idx=removed)
    except ValueError:
        return None
    env = gp.Env(empty=True); env.setParam('OutputFlag',0); env.start()
    m = gp.Model(env=env)
    p = m.addVars(N_GEN, T_HORIZON, lb=0.0)
    obj = gp.LinExpr()
    for g, gen in enumerate(GENERATORS):
        for t in range(T_HORIZON):
            if z[g,t] == 1:
                m.addConstr(p[g,t] >= gen['Pmin'])
                m.addConstr(p[g,t] <= gen['Pmax'])
                obj += costs[g,0] * p[g,t]
                obj += gen['c_noload']
            else:
                m.addConstr(p[g,t] == 0.0)
    for t in range(T_HORIZON):
        m.addConstr(gp.quicksum(p[g,t] for g in range(N_GEN))
                    == float(net_load[:,t].sum()))
    for l in range(len(LINES)):
        if l == removed: continue
        rate = line_rates[l]
        for t in range(T_HORIZON):
            flow = (gp.quicksum(PTDF[l,gen_bus[g]]*p[g,t]
                                for g in range(N_GEN)
                                if abs(PTDF[l,gen_bus[g]])>1e-8)
                    - float(sum(PTDF[l,n]*net_load[n,t] for n in range(N_BUS))))
            m.addConstr(flow <= rate); m.addConstr(flow >= -rate)
    m.setObjective(obj, GRB.MINIMIZE); m.optimize()
    if m.Status == GRB.OPTIMAL:
        cost = m.ObjVal; m.dispose(); env.dispose(); return cost
    m.dispose(); env.dispose(); return None

device = torch.device('cpu')
fs_path = 'dataset_output/feat_stats.json'
test_ds = UCGraphDataset('dataset_output', split='test', feat_stats=fs_path)
raw = dict(np.load('dataset_output/test.npz', allow_pickle=True))
with open(fs_path) as f: fs = json.load(f)
mu0=float(fs['0']['mean']); sig0=float(fs['0']['std'])

nominal_costs = np.array([[g['c_p'][0],g['c_p'][1]] for g in GENERATORS], dtype=np.float32)

gnn_ckpt = torch.load('model_output/best_model.pt', map_location=device, weights_only=False)
ga = argparse.Namespace(**gnn_ckpt['args'])
gnn = GNN_UC(in_features=4, d_h=ga.d_h, n_heads=ga.n_heads,
             gat_layers=ga.gat_layers, lstm_hidden=ga.lstm_hidden, n_gen=6, dropout=0.0)
gnn.load_state_dict(gnn_ckpt['model_state']); gnn.eval()

bl_ckpt = torch.load('baseline_output/best_baseline.pt', map_location=device, weights_only=False)
ba = argparse.Namespace(**bl_ckpt['args'])
bl = CNN_BiLSTM_Baseline(n_bus=30, n_features=4, t_horizon=24, n_gen=6,
                          cnn_filters=ba.cnn_filters, lstm_hidden=ba.lstm_hidden, dropout=0.0)
bl.load_state_dict(bl_ckpt['model_state']); bl.eval()

loader = DataLoader(test_ds, batch_size=1, shuffle=False)
gnn_soi=[]; bl_soi=[]; n_done=0

with torch.no_grad():
    for i, data in enumerate(loader):
        if n_done >= 100: break
        c = int(data.contingency)
        if c == -1: continue

        nl = raw['X'][i,:,:,0]*sig0+mu0 if 'net_load_bus' not in raw \
             else raw['net_load_bus'][i]

        z_opt  = raw['z'][i].astype(int)
        z_gnn  = (gnn(data)>=0.0).numpy().astype(int)
        z_bl   = (bl(data) >=0.0).numpy().astype(int)

        c_opt = lp_cost(z_opt, nl, c, nominal_costs)
        c_gnn = lp_cost(z_gnn, nl, c, nominal_costs)
        c_bl  = lp_cost(z_bl,  nl, c, nominal_costs)

        if c_opt and c_gnn and c_opt > 0:
            gnn_soi.append((c_gnn - c_opt) / c_opt)
        if c_opt and c_bl and c_opt > 0:
            bl_soi.append((c_bl - c_opt) / c_opt)

        n_done += 1
        if n_done % 20 == 0:
            print(f'{n_done} samples | GNN SOI={np.mean(gnn_soi):.4f} | CNN SOI={np.mean(bl_soi):.4f}')

print(f'FINAL (100 N-1 samples):')
print(f'  GNN-UC   SOI = {np.mean(gnn_soi):.4f}')
print(f'  CNN-BiLSTM SOI = {np.mean(bl_soi):.4f}')
print(f'  Venkatesh PM1 SOI = 0.0600 (their Case 2)')
