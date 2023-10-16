from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
import torch
from pymoo.core.problem import ElementwiseProblem
import numpy as np
import argparse
from pymoo.indicators.igd import IGD
from pymoo.problems import get_problem
from model import ParetoSetModel
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from matplotlib import pyplot
from pymoo.indicators.hv import HV
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from functions_evaluation import fast_non_dominated_sort
import random
import copy
from pymoo.util.ref_dirs import get_reference_directions
import heapq
from pymoo.algorithms.moo.nsga3 import NSGA3
import math

palette = pyplot.get_cmap('Set3')

def set_seed(seed):
    """for reproducibility
    :param seed:
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def das_dennis_recursion(ref_dirs, ref_dir, n_partitions, beta, depth):
    if depth == len(ref_dir) - 1:
        ref_dir[depth] = beta / (1.0 * n_partitions)
        ref_dirs.append(ref_dir[None, :])
    else:
        for i in range(beta + 1):
            ref_dir[depth] = 1.0 * i / (1.0 * n_partitions)
            das_dennis_recursion(ref_dirs, np.copy(ref_dir), n_partitions, beta - i, depth + 1)

def das_dennis(n_partitions, n_dim):
    if n_partitions == 0:
        return np.full((1, n_dim), 1 / n_dim)
    else:
        ref_dirs = []
        ref_dir = np.full(n_dim, np.nan)
        das_dennis_recursion(ref_dirs, ref_dir, n_partitions, n_partitions, 0)
        return np.concatenate(ref_dirs, axis=0)
def select_MinPts(data,k):
    k_dist = []
    for i in range(data.shape[0]):
        dist = (((data[i] - data)**2).sum(axis=1)**0.5)
        dist.sort()
        k_dist.append(dist[k])
    return np.array(k_dist)

class DatasetFromNumPy(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
    def __len__(self):
        return len(self.data)  # 返回长度

    def __getitem__(self, idx):
        # 这里可以对数据进行处理,比如讲字符数值化
        features = self.data[idx]  # 索引到第idx行的数据
        label = self.label[idx]  # 最后一项为指标数据
        return features, label  # 返回特征和指标


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--pop_size', '-ps', default=55, help="The population size of EAs.")
    parser.add_argument('--n_var', '-var', default=10, type=int, help="The size of decision variables.")
    parser.add_argument('--n_obj', '-obj', default=3, help="The size of objective variables.")
    parser.add_argument('--n_gen', '-n', default=200, type=int, help="The genertion of EAs.")
    parser.add_argument('--lr', '-l', default=1e-4, help="learning rate")
    parser.add_argument('--seed', '-s', default=2023, type=int, help="Seed")
    parser.add_argument('--problem_name', '-pn', default='dtlz2', type=str, help='Test problems.')
    parser.add_argument('--is_plot', default=False, help='Plot.?')
    parser.add_argument('--archive', default=0, type=int, help='whether archive')
    parser.add_argument('--mode', default=0, type=int, help='whether archive')
    parser.add_argument('--test', default=0, type=int, help='test mode, 0:fix 1:focus on 1st...',)
    args = parser.parse_args()
    set_seed(args.seed)
    device = 'cuda'
    # args.lr = 1e-3
    if args.problem_name in ['zdt1', 'zdt2', 'zdt3', 'dtlz4','convex_dtlz4']:
        args.lr = 1e-3
    else:
        args.lr = 1e-4
    print('args:', args)
    # {0: nothing, 1: Using better training solution instead suggested solution,
    # 2: 1+ delete strange solutions and compare suggested solution with strange solutions}
    ref_point = {'zdt1': [1.1, 1.1], 'zdt2': [1.1, 1.1], 'zdt3': [1, 1], 'dtlz1': [0.55, 0.55, 0.55], 'dtlz3':[1.1, 1.1, 1.1],
                 'dtlz2': [1.1, 1.1, 1.1], 'dtlz4': [1.1, 1.1, 1.1], 'dtlz5': [1.1, 1.1, 1.1], 'dtlz7': [1.1, 1.1, 6.6],
                 'convex_dtlz2':[1.1, 1.1,1.1], 'convex_dtlz4':[1.1, 1.1,1.1] }
    ref = ref_point[args.problem_name]
    mode = args.mode
    epochs = 500

    n_run = 3
    bs = 1
    pop_size = args.pop_size
    result = {'HV_list': [], 'meta_hv': [], 'igd_list': [], 'meta_igd_list': [], 'Cos': [], 'meta_cos': []}
    problem = get_problem(args.problem_name, n_var=args.n_var)
    n_obj = problem.n_obj
    hv = HV(ref_point=ref)
    for n in range(n_run):
        print(f'###########n_run:{n}#############')
        # Using common algorithm to test the performance of Pareto front approximation
        algorithm = NSGA2(pop_size=args.pop_size)
        # ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=9)

        # create the algorithm object
        # algorithm = NSGA3(pop_size=55,
        #                   ref_dirs=ref_dirs)
        res = minimize(problem,
                       algorithm,
                       ('n_gen', args.n_gen),
                       save_history=True,
                       verbose=False)
        # TODO Normzation for Objectives
        if args.archive:
            print('using all non-dominated solutions...')
            DV, OV = [], []
            for gen in res.history:
                if len(gen.opt.get("F")) < args.pop_size:
                    Comb_y = gen.opt.get("F")
                    Comb_x = gen.opt.get("X")
                else:
                    Comb_y, indices = np.unique(np.concatenate([gen.opt.get("F"), OV]), axis=0, return_index=True)
                    Comb_x = np.concatenate([gen.opt.get("X"), DV])[indices, :]
                rank, front = fast_non_dominated_sort(Comb_y)
                OV = Comb_y[front[0], :]
                DV = Comb_x[front[0], :]
        else:
            print('only using final solutions')
            rank, front = fast_non_dominated_sort(res.F)
            res.F = res.F[front[0], :]
            res.x = res.X[front[0], :]
            DV, OV = res.X, res.F

        z_min = OV.min(0)
        z_max = OV.max(0)

        true_label = torch.tensor(DV).cuda()
        Norm_Objectives = (OV - z_min) / (z_max-z_min)
        pref = Norm_Objectives / Norm_Objectives.sum(1).reshape(-1, 1)
        pref = torch.tensor(pref).to(device).float()
        IM_model = ParetoSetModel(problem.n_var, problem.n_obj, problem)
        IM_model.to(device)

        train_dataset = DatasetFromNumPy(pref, true_label)
        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
        # optimizer
        optimizer = torch.optim.Adam(IM_model.parameters(), lr=args.lr)
        for t_step in range(epochs):
            IM_model.train()
            for data in train_loader:
                x = IM_model(data[0])
                Loss =  ((x - data[1])**2).sum(1).mean()
                optimizer.zero_grad()
                Loss.backward()
                optimizer.step()

        IM_model.eval()
        with torch.no_grad():
            if n_obj == 2:
                pref = np.stack([np.linspace(0, 1, 66), 1 - np.linspace(0, 1, 66)]).T
                # pref = pref / np.linalg.norm(pref, axis=1).reshape(len(pref), 1)
                pref = torch.tensor(pref).to(device).float()
                # local search
                ls_pref = np.stack([np.linspace(0, 1, 66*10), 1 - np.linspace(0, 1, 66*10)]).T
                ls_pref = torch.tensor(ls_pref).to(device).float()
            if n_obj == 3:
                if args.test==0:
                    pref = torch.tensor(das_dennis(60, 3)).to(device).float()
                    # local search
                    ls_pref = torch.tensor(das_dennis(9*4, 3)).to(device).float()
                elif args.test==1:
                    pref = torch.tensor(np.random.dirichlet([5, 1, 1], 66)).to(device).float()
                    ls_pref = torch.tensor(np.random.dirichlet([5, 1, 1], 66*10)).to(device).float()
                elif args.test==2:
                    pref = torch.tensor(np.random.dirichlet([1, 5, 1], 66)).to(device).float()
                    ls_pref = torch.tensor(np.random.dirichlet([1, 5, 1], 66*10)).to(device).float()
                else:
                    pref = torch.tensor(np.random.dirichlet([1, 1, 5], 66)).to(device).float()
                    ls_pref = torch.tensor(np.random.dirichlet([1, 1, 5], 66*10)).to(device).float()
            global out
            out = {}
            # if mode==1:
            #     sol = IM_model(ls_pref)
            # else:
            sol = IM_model(pref)
            problem._evaluate(sol.detach().cpu().numpy(), out)
            generated_ps = sol.cpu().numpy()
            generated_pf = out['F']
            rank, front = fast_non_dominated_sort(generated_pf)
            generated_pf = generated_pf[front[0], :]
            tem_pf = []
            avg_cos = []
            meta_cos = []
            avg_igd = []
            meta_igd = []
            res_ea = []
            APF_V = (generated_pf - z_min) / (z_max - z_min)
            compared_solutions = (OV - z_min) / (z_max - z_min)
            for idx, pre in enumerate(pref.cpu().numpy()):
                num = np.dot(pre, APF_V.T)
                denom = np.linalg.norm(pre) * np.linalg.norm(APF_V, axis=1)
                angle = num / denom
                angle[np.isneginf(angle)] = 0
                index = np.argmax(angle)
                sug_max_cos = np.max(angle)
                sug_solution = generated_pf[index]


                num = np.dot(pre, compared_solutions.T)
                denom = np.linalg.norm(pre) * np.linalg.norm(compared_solutions, axis=1)
                angle = num / denom
                angle[np.isneginf(angle)] = 0
                index = np.argmax(angle)
                training_max_cos = np.max(angle)
                training_solution = OV[index]
                dominated_n = (sug_solution > training_solution).sum()
                is_dominated = dominated_n == n_obj
                res_ea.append(training_solution)
                if is_dominated:
                    tem_pf.append(training_solution.tolist())
                elif np.sum(sug_solution<=training_solution) == n_obj:
                    tem_pf.append(sug_solution.tolist())
                elif (training_max_cos > sug_max_cos):
                    tem_pf.append(training_solution.tolist())
                else:
                    tem_pf.append(sug_solution.tolist())
                if mode > 0:
                    Sol = (tem_pf[idx] - z_min) / (z_max - z_min)
                else:
                    Sol = (sug_solution - z_min) / (z_max - z_min)
                num = np.dot(Sol, pre)
                denom = np.linalg.norm(Sol) * np.linalg.norm(pre)
                Cos = num / denom
                if np.isnan(Cos):
                    Cos = 0
                avg_cos.append(Cos)
                Sol_ea = (training_solution-z_min) / (z_max-z_min)
                num = np.dot(Sol_ea, pre)
                denom = np.linalg.norm(Sol_ea) * np.linalg.norm(pre)
                Cos_ea = num / denom
                if np.isnan(Cos_ea):
                    Cos_ea = 0
                meta_cos.append(Cos_ea)
            if mode > 0:
                generated_pf = np.array(tem_pf)
            res_ea = np.array(res_ea)

            if args.problem_name in ['zdt1', 'zdt2']:
                pf = get_problem(args.problem_name).pareto_front(n_pareto_points=500)
            elif args.problem_name == 'zdt3':
                pf = get_problem(args.problem_name).pareto_front(n_points=500)
            elif args.problem_name in ['dtlz7', 'dtlz5']:
                pf = get_problem(args.problem_name).pareto_front()
            else:
                ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=100)
                pf = get_problem(args.problem_name).pareto_front(ref_dirs=ref_dirs)
            # else:
            #     pf = get_problem(args.problem_name).pareto_front()

            ind = IGD(pf)

            hv_value = hv(generated_pf)


            cp_of_im = math.degrees(math.acos(np.array(avg_cos).mean()))
            cp_of_meta = math.degrees(math.acos(np.array(meta_cos).mean()))
            cp_hv = hv_value* (90-cp_of_im) / 90
            cp_meta_hv = hv(res_ea) * (90-cp_of_meta) / 90
            print(f'cp:{cp_of_im}, meta_cp{cp_of_meta}')

            result['Cos'].append(np.array(avg_cos).mean())
            result['meta_cos'].append(np.array(meta_cos).mean())
            result['HV_list'].append(cp_hv)
            result['meta_hv'].append(cp_meta_hv)
            result["igd_list"].append(ind(generated_pf) * (90-cp_of_im) / 90)
            result["meta_igd_list"].append(ind(res_ea) * (90-cp_of_meta) / 90)

            print(result)
            if args.is_plot:
                if n_obj == 2:
                    fig = plt.figure()

                    for p in range(10):
                        leg = f'Pref. :{round(0.1*p, 1)}~{round(0.1*(p+1), 1)}'
                        pre_index = (0.1*p <= pref[:, 0]) & (pref[:, 0] <= 0.1*(p+1))
                        if p ==1:
                            color = '#d6ce93'
                        else:
                            color = palette(p)
                        plt.scatter(generated_pf[pre_index.cpu().numpy(), 0], generated_pf[pre_index.cpu().numpy(), 1], c=color, alpha=1, lw=1, label=leg, zorder=2)
                    # plt.scatter(res.F[:, 0], res.F[:, 1], c='black', alpha=1, lw=1, label='final solution', zorder=2)
                    plt.xlabel(r'$f_1(x)$', size=16)
                    plt.ylabel(r'$f_2(x)$', size=16)
                    plt.legend(loc='best', fontsize=10)
                    # plt.legend(loc='best')

                    plt.grid()
                    plt.show(block=True)
                if n_obj == 3:
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    fig = plt.figure()
                    ax.scatter(pf[:, 0], pf[:, 1], pf[:, 2], c='gray', alpha=0.1, lw=1, label='Pareto front')
                    # ax.scatter(res.F[:, 0], res.F[:, 1],res.F[:, 2], c='w', alpha=0.5, lw=1, s=45, label='final solutions', edgecolors='black')

                    for p in range(5):
                        leg = f'Pref. :{round(0.2*p, 1)}~{round(0.2*(p+1), 1)}'
                        pre_index = (0.2*p <= pref[:, 0]) & (pref[:, 0] <= 0.2*(p+1))
                        if p ==1:
                            color = '#d6ce93'
                        else:
                            color = palette(p)
                        ax.scatter(generated_pf[pre_index.cpu().numpy(), 0], generated_pf[pre_index.cpu().numpy(), 1], generated_pf[pre_index.cpu().numpy(), 2],
                                    c=color, alpha=1, lw=1, label=leg, s=35)
                    # max_lim = np.max(generated_pf, axis=0)
                    # min_lim = np.min(generated_pf, axis=0)
                    #
                    # ax.set_xlim(min_lim[0], max_lim[0])
                    # ax.set_ylim(max_lim[1], min_lim[1])
                    # ax.set_zlim(min_lim[2], max_lim[2])

                    ax.set_xlabel(r'$f_1(x)$', size=12)
                    ax.set_ylabel(r'$f_2(x)$', size=12)
                    ax.set_zlabel(r'$f_3(x)$', size=12)

                    ax.legend(loc='best', fontsize=14)
                    plt.show(block=True)
    for key in result.keys():
        if len(result[key]) > 0:
            result[key] = [round(np.array(result[key]).mean(), 4), round(np.array(result[key]).std(), 4)]
    print(f'mode:{mode}, {result}')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
