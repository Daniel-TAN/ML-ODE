'''
@ 2020-03-10
@ Henning
@ meta-learning ODE
'''

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import data_utils
import tqdm
from sklearn.metrics import roc_auc_score
import os
from visdom import Visdom
from basic_models import NNFOwithBayesianJumps
import preprocess
import random


def update(self, loss, weights):
    grads = torch.gradients(loss, list(weights.values()))
    gradients = dict(zip(weights.keys(), grads))
    new_weights = dict(
        zip(weights.keys(), [weights[key] - self.update_lr * gradients[key] for key in weights.keys()]))
    return new_weights

# waiting...
def collate(inputs):
    times_a = []
    time_ptr = []
    X = []
    M = []
    obs_idx = []
    cov = []
    labels = []
    batch_size = []
    return times_a, time_ptr, X, M, obs_idx, cov, labels, batch_size

def eval_score(model, params_dict, device, dl_test):
    '''

    :param model:
    :param params_dict:
    :param device:
    :param dl_test:
    :return: recall@k
    '''
    recall1 = 0.
    recall5 = 0.
    recall10 = 0.
    recall100 = 0.
    recall1000 = 0.
    recall10000 = 0.
    iter_cnt = 0
    poi_list = preprocess.POI_list_hash()
    random.shuffle(poi_list)
    poi_list = random.sample(poi_list,99)
    print(len(poi_list))
    poi_tmp_dict = {}


    with torch.no_grad():
        for i,b in enumerate(dl_test):
            times = b["times"]
            time_ptr = b["time_ptr"]
            X = b["X"].to(device)
            M = b["M"].to(device)
            obs_idx = b["obs_idx"]
            cov = b["cov"].to(device)
            labels = b["y"].to(device)
            batch_size = labels.size(0)

            if b["X_val"] is not None:
                X_val = b["X_val"].to(device)
                M_val = b["M_val"].to(device)
                times_val = b["times_val"]
                times_idx = b["index_val"]


            h0 = 0  # torch.zeros(labels.shape[0], params_dict["hidden_size"]).to(device)
            hT, loss, class_pred, t_vec, p_vec, h_vec, _, _ = model(times, time_ptr, X, M, obs_idx,
                                                                    delta_t=params_dict["delta_t"], T=params_dict["T"],
                                                                    cov=cov, return_path=True)
            t_vec = np.around(t_vec, str(params_dict["delta_t"])[::-1].find('.')).astype(
                np.float32)  # Round floating points error in the time vector.

            p_val = data_utils.extract_from_path(t_vec, p_vec, times_val,times_idx)
            m, v = torch.chunk(p_val, 2, dim=1)

            val_pred = m.numpy().tolist()
            x_val = X_val.numpy().tolist()
            # print(x_val)
            # print(val_pred)
            # print(x_val)
            for i in range(len(val_pred)):
                poi_tuple_obs = [x for x in x_val]
                poi_tuple_pre = [y for y in val_pred]
                # print(poi_tuple_obs)
                # print(poi_tuple_obs,poi_tuple_pre)
                for poi in poi_list:
                    # print(poi)
                    # print(poi)
                    # print(poi_tuple_obs[i])
                    poi_tmp_dict["".join(str(j) for  j in poi)]=\
                        (np.array(poi)* np.array(poi_tuple_pre[i])).sum()
                    # print((np.array(poi)*np.array(poi_tuple_obs[i])).sum())
                poi_tmp_dict["".join(str(j) for j in poi_tuple_obs[i])]= \
                        (np.array(poi_tuple_pre[i])*np.array(poi_tuple_obs[i])).sum()

                poi_list_pred = sorted(poi_tmp_dict.items(), key=lambda x:x[1],reverse=True)
                poi_list_pred = [x[0] for x in poi_list_pred]
                poi_tuple_obs_str = "".join(str(j) for j in poi_tuple_obs[i])
                iter_cnt +=1
                if poi_tuple_obs_str not in poi_list_pred:
                    print('??????????----------------')
                    # print(poi_tuple_obs_str)
                    # print(poi_list_pred)
                recall1 += poi_tuple_obs_str in poi_list_pred[:1]
                recall5 += poi_tuple_obs_str in poi_list_pred[:5]
                recall10 += poi_tuple_obs_str in poi_list_pred[:10]
                recall100 += poi_tuple_obs_str in poi_list_pred[:100]
                recall1000 += poi_tuple_obs_str in poi_list_pred[:1000]
                recall10000 += poi_tuple_obs_str in poi_list_pred[:10000]

        print("recall@1: ", recall1/iter_cnt)
        print("recall@5: ", recall5/iter_cnt)
        print("recall@10: ", recall10/iter_cnt)
        print("recall@100: ", recall100/iter_cnt)
        print("recall@1000: ", recall1000/iter_cnt)
        print("recall@10000: ", recall10000/iter_cnt)
    return [recall1/iter_cnt,recall5/iter_cnt,recall10/iter_cnt,recall100/iter_cnt,
            recall1000/iter_cnt,recall10000/iter_cnt]

    # for batch in tqdm.tqdm(, desc="validation"):
    #     batch_user, batch_td, batch_ld, batch_loc, batch_dst = batch
    #     if len(batch_loc) < 3:
    #         continue
    #     iter_cnt += 1
    #     batch_o, target = run(batch_user, batch_td, batch_ld, batch_loc, batch_dst, step=step)
    #
    #


def data_validation(params_dict, train_idx, val_idx, test_idx):

    csv_file_path = params_dict["csv_file_path"]
    csv_file_cov = params_dict["csv_file_cov"]  # none
    csv_file_tags = params_dict["csv_file_tags"]  # none

    if params_dict["lambda"] == 0:
        validation = True
        val_options = {"T_val": params_dict["T_val"], "max_val_samples": params_dict["max_val_samples"]}
    else:
        validation = False
        val_options = None

    data_train = data_utils.ODE_Dataset(csv_file=csv_file_path, label_file=csv_file_tags, cov_file=csv_file_cov,
                                        idx=train_idx)
    data_val = data_utils.ODE_Dataset(csv_file=csv_file_path, label_file=csv_file_tags,
                                      cov_file=csv_file_cov, idx=val_idx, validation=validation,
                                      val_options=val_options)
    data_test = data_utils.ODE_Dataset(csv_file=csv_file_path, label_file=csv_file_tags,
                                       cov_file=csv_file_cov, idx=test_idx, validation=validation,
                                       val_options=val_options)

    dl = DataLoader(dataset=data_train, collate_fn=data_utils.custom_collate_fn, shuffle=True, batch_size=500,
                    num_workers=2)
    dl_val = DataLoader(dataset=data_val, collate_fn=data_utils.custom_collate_fn, shuffle=True,
                        batch_size=len(val_idx))
    dl_test = DataLoader(dataset=data_test, collate_fn=data_utils.custom_collate_fn, shuffle=True,
                         batch_size=len(test_idx))
    return data_train, data_val, data_test,dl, dl_val, dl_test

def meta_ode(simulation_name, params_dict, device, train_idx, val_idx, test_idx, epoch_max=40):

    best_recall_score = {'@1':0.0,'@5':0.0,'@10':0.0,'@100':0.0,'@1000':0.0,'@10000':0.0,}
    data_train, data_val, data_test,dl, dl_val, dl_test = data_validation(params_dict, train_idx,val_idx,test_idx)
    params_dict["input_size"] = data_train.variable_num
    params_dict["cov_size"] = data_train.cov_dim

    nnfwobj = NNFOwithBayesianJumps(input_size=params_dict["input_size"],
                                      hidden_size=params_dict["hidden_size"],
                                      p_hidden=params_dict["p_hidden"],
                                      prep_hidden=params_dict["prep_hidden"],
                                      logvar=params_dict["logvar"], mixing=params_dict["mixing"],
                                      classification_hidden=params_dict["classification_hidden"],
                                      cov_size=params_dict["cov_size"],
                                      cov_hidden=params_dict["cov_hidden"],
                                      dropout_rate=params_dict["dropout_rate"],
                                      full_gru_ode=params_dict["full_gru_ode"],
                                      impute=params_dict["impute"])
    nnfwobj.to(device)
    optimizer = torch.optim.Adam(nnfwobj.parameters(), lr=params_dict["lr"], weight_decay=params_dict["weight_decay"])

    if params_dict["meta"] == True:
        '''
        Perform gradient descent for one task in the meta-batch. 
        '''
        total_losses = {}

        # for key in source_inputs.keys():
        #
        #     inputa = source_inputs[key]
        #     task_lossa = []
        #
        #     optimizer.zero_grad()
        #     times_a, time_ptr, X, M, obs_idx, cov, labels, batch_size = collate(inputa)
        #
        #     for i in range(0,params_dict['meta_num_updates']):
        #
        #         hT, loss, class_pred, mse_loss = nnfwobj(times_a, time_ptr, X, M, obs_idx,
        #                                                  delta_t=params_dict["delta_t"],
        #                                                 T=params_dict["T"], cov=cov)
        #         task_lossa.append(mse_loss)
        #     total_losses['key'] = task_lossa
        #
        #     # Performance & Optimization
        #
        #     total_loss_eva = torch.sum(total_losses[key] for key in total_losses.keys())/params_dict['meta_num_updates']
        #     '''
        #     meta_learning updates
        #     '''
        #     total_loss_eva.backward()
        #     optimizer.step()

        train_idx_1 = [i for i in range(10,20)]
        val_idx_1 = [i for i in range(210,220)]
        test_idx_1 = [i for i in range(310,320)]

        data_train_1, data_val_1, data_test_1, dl_1, dl_val_1, dl_test_1 = data_validation(params_dict, train_idx_1, val_idx_1,
                                                                               test_idx_1)
        # val_metric_prev = -1000
        for epoch in range(2):
            nnfwobj.train()
            total_train_loss = 0
            auc_total_train = 0
            tot_loglik_loss = 0
            for i, b in enumerate(tqdm.tqdm(dl_1)):
                optimizer.zero_grad()
                times = b["times"]
                time_ptr = b["time_ptr"]
                X = b["X"].to(device)
                M = b["M"].to(device)
                obs_idx = b["obs_idx"]
                cov = b["cov"].to(device)
                labels = b["y"].to(device)
                batch_size = labels.size(0)

                hT, loss, class_pred, mse_loss = nnfwobj(times, time_ptr, X, M, obs_idx, delta_t=params_dict["delta_t"],
                                                         T=params_dict["T"], cov=cov)

                total_loss = (loss) / batch_size

                total_train_loss += total_loss
                tot_loglik_loss += mse_loss

                # try:
                #     auc_total_train += roc_auc_score(labels.detach().cpu(), torch.sigmoid(class_pred).detach().cpu())
                # except ValueError:
                #     if params_dict["verbose"] >= 3:
                #         print("Single CLASS ! AUC is erroneous")
                #     pass

                total_loss.backward()
                optimizer.step()

            info = {'training_loss': total_train_loss.detach().cpu().numpy() / (i + 1),
                    'AUC_training': auc_total_train / (i + 1), "loglik_loss": tot_loglik_loss.detach().cpu().numpy()}
            # for tag, value in info.items():
            #     logger.scalar_summary(tag, value, epoch)
            print(f"NegLogLik Loss train meta : {tot_loglik_loss.detach().cpu().numpy()}")

    print("Start Training")

    viz = Visdom()
    viz.line([0.], [0.], win='train_loss_no_10-4', opts=dict(title='train-loss_no_10-4'))
    # viz.line([0.], [0.], win='test_loss_no_10-3', opts=dict(title='test-loss_no_10-3'))
    # viz.line([0.], [50.], win='train_loss_no>=50-3', opts=dict(title='train_loss_no>=50-3'))



    # viz.line([loss.item()], [globe_step], win='train_loss', update='append')
    val_metric_prev = -1000
    for epoch in range(epoch_max):
        nnfwobj.train()
        total_train_loss = 0
        auc_total_train = 0
        tot_loglik_loss = 0
        for i, b in enumerate(tqdm.tqdm(dl)):

            optimizer.zero_grad()
            times = b["times"]
            time_ptr = b["time_ptr"]
            X = b["X"].to(device)
            M = b["M"].to(device)
            obs_idx = b["obs_idx"]
            cov = b["cov"].to(device)
            labels = b["y"].to(device)
            batch_size = labels.size(0)

            h0 = 0
            hT, loss, class_pred, mse_loss = nnfwobj(times, time_ptr, X, M, obs_idx, delta_t=params_dict["delta_t"],
                                                     T=params_dict["T"], cov=cov)

            total_loss = (loss) / batch_size

            total_train_loss += total_loss
            tot_loglik_loss += mse_loss

            # try:
            #     auc_total_train += roc_auc_score(labels.detach().cpu(), torch.sigmoid(class_pred).detach().cpu())
            # except ValueError:
            #     if params_dict["verbose"] >= 3:
            #         print("Single CLASS ! AUC is erroneous")
            #     pass

            total_loss.backward()
            optimizer.step()

        info = {'training_loss': total_train_loss.detach().cpu().numpy() / (i + 1),
                'AUC_training': auc_total_train / (i + 1), "loglik_loss": tot_loglik_loss.detach().cpu().numpy()}
        # for tag, value in info.items():
        #     logger.scalar_summary(tag, value, epoch)
        print(f"NegLogLik Loss train : {tot_loglik_loss.detach().cpu().numpy()}")

        data_utils.adjust_learning_rate(optimizer, epoch, params_dict["lr"])

        with torch.no_grad():
            nnfwobj.eval()
            total_loss_val = 0
            auc_total_val = 0
            loss_val = 0
            mse_val = 0
            corr_val = 0
            num_obs = 0
            for i, b in enumerate(dl_val):
                times = b["times"]
                time_ptr = b["time_ptr"]
                X = b["X"].to(device)
                M = b["M"].to(device)
                obs_idx = b["obs_idx"]
                cov = b["cov"].to(device)
                labels = b["y"].to(device)
                batch_size = labels.size(0)

                if b["X_val"] is not None:
                    X_val = b["X_val"].to(device)
                    M_val = b["M_val"].to(device)
                    times_val = b["times_val"]
                    times_idx = b["index_val"]
                h0 = 0  # torch.zeros(labels.shape[0], params_dict["hidden_size"]).to(device)
                hT, loss, class_pred, t_vec, p_vec, h_vec, _, _ = nnfwobj(times, time_ptr, X, M, obs_idx,
                                                                          delta_t=params_dict["delta_t"],
                                                                          T=params_dict["T"], cov=cov, return_path=True)
                total_loss = (loss) / batch_size

                try:
                    auc_val = roc_auc_score(labels.cpu(), torch.sigmoid(class_pred).cpu())
                except ValueError:
                    auc_val = 0.5
                    if params_dict["verbose"] >= 3:
                        print("Only one class : AUC is erroneous")
                    pass

                if params_dict["lambda"] == 0:
                    t_vec = np.around(t_vec, str(params_dict["delta_t"])[::-1].find('.')).astype(
                        np.float32)  # Round floating points error in the time vector.
                    p_val = data_utils.extract_from_path(t_vec, p_vec, times_val, times_idx)
                    m, v = torch.chunk(p_val, 2, dim=1)
                    last_loss = (data_utils.log_lik_gaussian(X_val, m, v) * M_val).sum()
                    mse_loss = (torch.pow(X_val - m, 2) * M_val).sum()
                    corr_val_loss = data_utils.compute_corr(X_val, m, M_val)

                    loss_val += last_loss.cpu().numpy()
                    num_obs += M_val.sum().cpu().numpy()
                    mse_val += mse_loss.cpu().numpy()
                    corr_val += corr_val_loss.cpu().numpy()
                else:
                    num_obs = 1

                total_loss_val += total_loss.cpu().detach().numpy()
                auc_total_val += auc_val

            loss_val /= num_obs
            mse_val /= num_obs
            info = {'validation_loss': total_loss_val / (i + 1), 'AUC_validation': auc_total_val / (i + 1),
                    'loglik_loss': loss_val, 'validation_mse': mse_val, 'correlation_mean': np.nanmean(corr_val),
                    'correlation_max': np.nanmax(corr_val), 'correlation_min': np.nanmin(corr_val)}
            # for tag, value in info.items():
            #     logger.scalar_summary(tag, value, epoch)

            if params_dict["lambda"] == 0:
                val_metric = - loss_val
            else:
                val_metric = auc_total_val / (i + 1)

            # if val_metric > val_metric_prev:
            if epoch>=0:
                print(f"New highest validation metric reached ! : {val_metric}")
                print("Saving Model")

                # torch.save(nnfwobj.state_dict(), f"./../trained_models/{simulation_name}_MAX.pt")

                val_metric_prev = val_metric
                test_loglik, test_auc, test_mse, recall_score = test_evaluation(nnfwobj, params_dict, device, dl_test)

                if recall_score[0] > best_recall_score['@1']: best_recall_score['@1']=recall_score[0]
                if recall_score[1] > best_recall_score['@5']: best_recall_score['@5'] = recall_score[1]
                if recall_score[2] > best_recall_score['@10']: best_recall_score['@10'] = recall_score[2]
                if recall_score[3] > best_recall_score['@100']: best_recall_score['@100'] = recall_score[3]
                if recall_score[4] > best_recall_score['@1000']: best_recall_score['@1000'] = recall_score[4]
                if recall_score[5] > best_recall_score['@10000']: best_recall_score['@10000'] = recall_score[5]

                print(f"Test loglik loss at epoch {epoch} : {test_loglik}")
                print(f"Test AUC loss at epoch {epoch} : {test_auc}")
                print(f"Test MSE loss at epoch{epoch} : {test_mse}")
            # else:
                # if epoch % 10:
                    # torch.save(nnfwobj.state_dict(), f"./../trained_models/{simulation_name}.pt")

        print(f"Total validation loss at epoch {epoch}: {total_loss_val/(i+1)}")
        print(f"Validation AUC at epoch {epoch}: {auc_total_val/(i+1)}")
        print(
            f"Validation loss (loglik) at epoch {epoch}: {loss_val:.5f}. MSE : {mse_val:.5f}. Correlation : {np.nanmean(corr_val):.5f}. Num obs = {num_obs}")
        print([mse_val.item()], [epoch])
        viz.line([mse_val.item()], [epoch], win='train_loss_no_10-4', update='append')
        # viz.line([test_mse.item()], [epoch], win='test_loss_meta_10-1', update='append')
        # if epoch>=50:
            # viz.line([mse_val.item()], [epoch], win='train_loss_no>=50-3', update='append')
        print(best_recall_score)
    print(f"Finished training GRU-ODE for Climate. Saved in ./../trained_models/{simulation_name}")

    return (info, val_metric_prev, test_loglik, test_auc, test_mse)





def test_evaluation(model, params_dict, device, dl_test):

    #calculate recall@k
    recall_score = eval_score(model, params_dict, device, dl_test)
    # recall_acore = [1,2,3,4,5,6]

    with torch.no_grad():
        model.eval()
        total_loss_test = 0
        auc_total_test = 0
        loss_test = 0
        mse_test = 0
        corr_test = 0
        num_obs = 0
        # viz = Visdom()
        # viz.line([0.], [0.], win='train_loss_no', opts=dict(title='train-loss_no'))
        for i, b in enumerate(dl_test):
            times = b["times"]
            time_ptr = b["time_ptr"]
            X = b["X"].to(device)
            M = b["M"].to(device)
            obs_idx = b["obs_idx"]
            cov = b["cov"].to(device)
            labels = b["y"].to(device)
            batch_size = labels.size(0)

            if b["X_val"] is not None:
                X_val = b["X_val"].to(device)
                M_val = b["M_val"].to(device)
                times_val = b["times_val"]
                times_idx = b["index_val"]

            h0 = 0  # torch.zeros(labels.shape[0], params_dict["hidden_size"]).to(device)
            hT, loss, class_pred, t_vec, p_vec, h_vec, _, _ = model(times, time_ptr, X, M, obs_idx,
                                                                    delta_t=params_dict["delta_t"], T=params_dict["T"],
                                                                    cov=cov, return_path=True)
            total_loss = (loss) / batch_size

            try:
                auc_test = roc_auc_score(labels.cpu(), torch.sigmoid(class_pred).cpu())
            except ValueError:
                if params_dict["verbose"] >= 3:
                    print("Only one class. AUC is wrong")
                auc_test = 0
                pass

            if params_dict["lambda"] == 0:
                t_vec = np.around(t_vec, str(params_dict["delta_t"])[::-1].find('.')).astype(
                    np.float32)  # Round floating points error in the time vector.
                p_val = data_utils.extract_from_path(t_vec, p_vec, times_val, times_idx)
                m, v = torch.chunk(p_val, 2, dim=1)
                last_loss = (data_utils.log_lik_gaussian(X_val, m, v) * M_val).sum()
                mse_loss = (torch.pow(X_val - m, 2) * M_val).sum()
                corr_test_loss = data_utils.compute_corr(X_val, m, M_val)

                loss_test += last_loss.cpu().numpy()
                num_obs += M_val.sum().cpu().numpy()
                mse_test += mse_loss.cpu().numpy()
                corr_test += corr_test_loss.cpu().numpy()
            else:
                num_obs = 1

            total_loss_test += total_loss.cpu().detach().numpy()
            auc_total_test += auc_test

        loss_test /= num_obs
        mse_test /= num_obs
        auc_total_test /= (i + 1)

        return (loss_test, auc_total_test, mse_test,recall_score)


if __name__ == "__main__":
    simulation_name = "poi recommendation"
    device = torch.device("cpu")

    idx_tmp = [i for i in range(0,269)]
    idx = random.shuffle(idx_tmp)
    train_idx = idx_tmp[:169]
    val_idx = idx_tmp[169:219]
    test_idx = idx_tmp[219:]
    source_inputs = []

    # Model parameters.
    params_dict = dict()

    # params_dict["csv_file_path"] = "./data/small_chunked_sporadic.csv"
    params_dict["csv_file_path"] = "./data/cc_sporadic_150%150_40.csv"
    params_dict["csv_file_tags"] = None
    params_dict["csv_file_cov"] = None

    params_dict["meta"] = False
    params_dict['meta_num_updates'] = 1
    params_dict["hidden_size"] = 50
    params_dict["p_hidden"] = 25
    params_dict["prep_hidden"] = 10
    params_dict["logvar"] = True
    params_dict["mixing"] = 1e-4  # Weighting between KL loss and MSE loss.
    params_dict["delta_t"] = 0.1
    # params_dict["T"] = 200
    params_dict["T"] = (1379345002.0-1325431402.0)/3600/24*1.5
    params_dict["lambda"] = 0  # Weighting between classification and MSE loss.

    params_dict["classification_hidden"] = 2
    params_dict["cov_hidden"] = 50
    params_dict["weight_decay"] = 0.0001
    params_dict["dropout_rate"] = 0.2
    params_dict["lr"] = 0.02
    params_dict["full_gru_ode"] = True
    params_dict["no_cov"] = True
    params_dict["impute"] = False
    params_dict["verbose"] = 0  # from 0 to 3 (highest)
    params_dict["T_val"] = 150
    # params_dict["T_val"] = (1343627854.00-1325431402.0)/3600/24
    params_dict["max_val_samples"] = 3


    info, val_metric_prev, test_loglik, test_auc, test_mse = meta_ode(simulation_name=simulation_name,
                                                                          params_dict=params_dict,
                                                                          device=device,
                                                                          train_idx=train_idx,
                                                                          val_idx=val_idx,
                                                                          test_idx=test_idx,
                                                                          epoch_max=200
                                                                      )
