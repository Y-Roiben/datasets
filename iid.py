import numpy as np
from torch.utils.data import Dataset, DataLoader

from GTSRB import get_GTSRB


def Dirichlet_noniid_dir(dataset, num_users, alpha):
    global idx_groups
    np.random.seed(7)
    """
    dataset: training set of CIFAR
    """
    users_dict = {}
    users_target_freq = {}
    num_classes = 10
    min_size = 0
    max_size = 0
    # labels = np.array(dataset.targets)
    dataset_type = str(type(dataset))
    if "ConcatDataset" in dataset_type:
        labels = np.array(dataset.datasets[0].targets + dataset.datasets[1].targets)
    else:
        labels = np.array(dataset.targets)
    num_items = int(len(dataset) / num_users)

    while min_size < 10:
        idx_groups = [[] for _ in range(num_users)]
        # for each class in the dataset
        for k in range(num_classes):
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_users))
            # Balance
            proportions = np.array(
                [p * (len(user_idx) < num_items) for p, user_idx in zip(proportions, idx_groups)]
            )
            proportions = proportions / proportions.sum()
            # cumsum 数组元素的累积和
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_groups = [user_idx + idx.tolist() for user_idx, idx in zip(idx_groups, np.split(idx_k, proportions))]
            min_size = min([len(user_idx) for user_idx in idx_groups])
            max_size = max([len(user_idx) for user_idx in idx_groups])

    # np.random.shuffle(idx_groups)
    for i in range(num_users):
        np.random.shuffle(idx_groups[i])
        users_dict[i] = idx_groups[i]
        users_target_freq[i] = np.unique(labels[idx_groups[i]], return_counts=True)

    return users_dict, users_target_freq


def CINIC_IID(dataset, num_users):
    np.random.seed(7)
    """
    Sample I.I.D. client data from dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    np.random.seed(0)
    num_items = int(len(dataset) / num_users)  # 每个用户拥有多少数据
    dict_users, all_idxs = {i: np.array([], dtype='int64') for i in range(num_users)}, [i for i in range(len(dataset))]
    dict_labels = {i: np.array([], dtype='int64') for i in range(num_users)}
    dataset_type = str(type(dataset))
    if "ConcatDataset" in dataset_type:
        labels = np.array(dataset.datasets[0].targets + dataset.datasets[1].targets)
    else:
        labels = np.array(dataset.targets)
    idxs_labels = np.vstack((all_idxs, labels))
    for i in range(num_users):
        idx = np.random.choice(all_idxs, num_items, replace=False)
        dict_users[i] = idx  # 所有数据里随机选择数据
        target_list = idxs_labels[1, idx]
        dict_labels[i] = np.unique(target_list)
        all_idxs = list(set(all_idxs) - set(idx))  # 从所有数据中去除已经被分配的数据

    return dict_users, dict_labels


def CINIC_noniid(dataset, num_users, num_labels=2):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    np.random.seed(7)
    # 将全部的 data 分成 200 组, 每一组 300 个数据 sample (num_users = 100, 60000/100 = 200 x 300)
    # 根据 clients 的数量, 需要从 0-199 中采样两组对应 2 x 300 = 600 个数据 sample 作为 client 的索引
    num_shards = int(num_users * num_labels)
    num_imgs = int(len(dataset) / num_shards)
    idx_shard = [i for i in range(num_shards)]
    dict_idxs = {i: np.array([], dtype='int64') for i in range(num_users)}
    dict_labels = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    dataset_type = str(type(dataset))
    if "ConcatDataset" in dataset_type:
        labels = np.array(dataset.datasets[0].targets + dataset.datasets[1].targets)
    else:
        labels = np.array(dataset.targets)
    # labels = np.array(dataset.datasets[0].targets + dataset.datasets[1].targets)
    # labels = np.array(dataset.targets)
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels_sort = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels_sort[0, :]  # 按照标签从 0-9 的数据 sample 索引
    # divide and assign
    for i in range(num_users):
        #  0-199 组随机选择中的两组
        rand_set = set(np.random.choice(idx_shard, num_labels, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)  # 删除对应的 rand_set
        for rand in rand_set:  # 将随机选到的数据连接保存到相应客户端
            dict_idxs[i] = np.concatenate((dict_idxs[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
        target_list = idxs_labels[1, dict_idxs[i]]
        target, frequency = np.unique(target_list, return_counts=True)
        dict_labels[i] = (target, frequency)
    return dict_idxs, dict_labels


def GTSRB_IID(dataset, num_users):
    np.random.seed(9)
    num_items = int(len(dataset) / num_users)  # 每个用户拥有多少数据
    dict_users, all_idxs = {i: np.array([], dtype='int64') for i in range(num_users)}, [i for i in range(len(dataset))]
    for i in range(num_users):
        idx = np.random.choice(all_idxs, num_items, replace=False)
        dict_users[i] = idx  # 所有数据里随机选择数据
        all_idxs = list(set(all_idxs) - set(idx))  # 从所有数据中去除已经被分配的数据
    return dict_users


class DatasetSplit(Dataset):  # 从原数据集中提取用户拥有的图片
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


if __name__ == '__main__':
    trainset, testset = get_GTSRB()
    dict_users = GTSRB_IID(trainset, 10)
    A = DatasetSplit(trainset, dict_users[0])
    print(A)
