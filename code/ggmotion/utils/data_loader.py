import torch
from torch.utils.data import Dataset
import numpy as np
from utils import data_utils
import os


class H36motion3D(Dataset):
    exp = "h36m"
    actions = ["walking", "eating", "smoking", "discussion", "directions",
               "greeting", "phoning", "posing", "purchases", "sitting",
               "sittingdown", "takingphoto", "waiting", "walkingdog",
               "walkingtogether"]
    def __init__(self, path_to_data, actions='all', input_n=10, output_n=10,
                 split=0, scale=100, sample_rate=2, test_manner="all", aug=False, hop=-1):
        """
        param split: 0 train, 1 testing, 2 validation
        """

        self.path_to_data = path_to_data
        self.split = split
        self.input_n = input_n
        self.output_n = output_n
        self.aug = aug
        # 1, 6, 7, 8, 9
        subs = [[1, 6, 7, 8, 9], [5], [11]]

        acts = self.define_acts(actions)

        subjs = subs[split]
        all_seqs, dim_ignore, dim_used, group = data_utils.load_data_3d(path_to_data, subjs, acts,
                                                                        sample_rate, input_n, output_n, test_manner)
        all_seqs = all_seqs / scale

        self.all_seqs = all_seqs
        self.dim_used = dim_used

        all_seqs = all_seqs[:, :, dim_used]  # (S,T,N*3)
        all_seqs = all_seqs.transpose(0, 2, 1)  # (S,N*3,T)
        all_seqs = all_seqs.reshape(all_seqs.shape[0], -1, 3, all_seqs.shape[-1])  # (S,N,3,T)

        self.used_seqs = all_seqs.copy()
        self.group = data_utils.group_attr(group)

        spine = self.group["spine"][1:]
        head = self.group["head"]
        arm_l = self.group["arm_l"]
        arm_r = self.group["arm_r"]
        leg_l = self.group["leg_l"]
        leg_r = self.group["leg_r"]

        graph = np.concatenate([spine, head, arm_l, arm_r, leg_l, leg_r], axis=0)
        edges = data_utils.edge_attr(graph, all_seqs.shape[1], hop)

        self.edges = np.unique(edges, axis=0)
        self.parent = np.unique(graph, axis=0)

    def __len__(self):
        return np.shape(self.all_seqs)[0]

    def __getitem__(self, item):
        used_seqs = self.used_seqs[item]  # (N,3,T)
        all_seqs = self.all_seqs[item]  # (T,N*3)
        if self.split == 0 and self.aug:
            if np.random.rand(1)[0] > 0.5:
                idx = np.arange(all_seqs.shape[0] - 1, -1, -1)
                used_seqs = used_seqs[..., idx]
                all_seqs = all_seqs[idx]

        input_seq = used_seqs[..., :self.input_n]
        output_seq = used_seqs[..., self.input_n:]
        return input_seq, output_seq, all_seqs

    @staticmethod
    def define_acts(action):
        """
        Define the action list.
        Args
          action: String with the passed action. Could be "all"
        Returns
          actions: List of strings of actions
        Raises
          ValueError if the action is not included in H3.6M
        """
        if action in H36motion3D.actions:
            return [action]

        if action == "all":
            return H36motion3D.actions

        if action == "all_srnn":
            return ["walking", "eating", "smoking", "discussion"]

        raise ValueError("Unrecognized action: {}".format(action))


class CMU_Motion3D(Dataset):
    exp = "cmu"
    actions = ["basketball", "basketball_signal", "directing_traffic",
               "jumping", "running", "soccer", "walking", "washwindow"]

    def __init__(self, path_to_data, actions, input_n=10, output_n=10,
                 split=0, scale=100, sample_rate=2, test_manner="all", aug=False, hop=-1):

        self.path_to_data = path_to_data
        self.split = split
        self.input_n = input_n
        self.output_n = output_n
        self.aug = aug

        acts = self.define_acts(actions)

        if split == 0:
            path_to_data = os.path.join(path_to_data, 'train')
        else:
            path_to_data = os.path.join(path_to_data, 'test')

        all_seqs, dim_ignore, dim_used, group = data_utils.load_data_cmu_3d(path_to_data, acts, sample_rate, input_n,
                                                                     output_n, test_manner)

        all_seqs = all_seqs / scale
        self.all_seqs = all_seqs
        self.dim_used = dim_used

        all_seqs = all_seqs[:, :, dim_used]  # (S,T,N*3)
        all_seqs = all_seqs.transpose(0, 2, 1)  # (S,N*3,T)
        all_seqs = all_seqs.reshape(all_seqs.shape[0], -1, 3, all_seqs.shape[-1])  # (S,N,3,T)

        self.used_seqs = all_seqs.copy()
        self.group = data_utils.group_attr(group)

        spine = self.group["spine"][1:]
        head = self.group["head"]
        arm_l = self.group["arm_l"]
        arm_r = self.group["arm_r"]
        leg_l = self.group["leg_l"]
        leg_r = self.group["leg_r"]

        graph = np.concatenate([spine, head, arm_l, arm_r, leg_l, leg_r], axis=0)
        edges = data_utils.edge_attr(graph, all_seqs.shape[1], hop)

        self.edges = np.unique(edges, axis=0)
        self.parent = np.unique(graph, axis=0)

    def __len__(self):
        return np.shape(self.all_seqs)[0]

    def __getitem__(self, item):
        used_seqs = self.used_seqs[item]  # (N,3,T)
        all_seqs = self.all_seqs[item]  # (T,N*3)
        if self.split == 0 and self.aug:
            if np.random.rand(1)[0] > 0.5:
                idx = np.arange(all_seqs.shape[0] - 1, -1, -1)
                used_seqs = used_seqs[..., idx]
                all_seqs = all_seqs[idx]

        input_seq = used_seqs[..., :self.input_n]
        output_seq = used_seqs[..., self.input_n:]
        return input_seq, output_seq, all_seqs

    @staticmethod
    def define_acts(action):
        """
        Define the action list.
        Args
          action: String with the passed action. Could be "all"
        Returns
          actions: List of strings of actions
        Raises
          ValueError if the action is not included in CMU
        """
        if action in CMU_Motion3D.actions:
            return [action]

        if action == "all":
            return CMU_Motion3D.actions

        raise ValueError("Unrecognized action: {}".format(action))


class CMU_Motion3DDQPose(Dataset):
    exp = "cmu"
    actions = CMU_Motion3D.actions

    def __init__(self, path_to_data, actions, input_n=10, output_n=10,
                 split=0, scale=100, sample_rate=2, test_manner="all", aug=False, hop=-1):

        self.path_to_data = path_to_data
        self.split = split
        self.input_n = input_n
        self.output_n = output_n
        self.aug = aug

        acts = self.define_acts(actions)

        if split == 0:
            path_to_data = os.path.join(path_to_data, 'train')
        else:
            path_to_data = os.path.join(path_to_data, 'test')

        all_pos, all_dq, dim_ignore, dim_used, group = data_utils.load_data_cmu_3d_dqpose(
            path_to_data, acts, sample_rate, input_n, output_n, test_manner)

        all_pos = all_pos / scale
        self.all_seqs = all_pos
        self.dim_used = dim_used

        pos_seqs = all_pos[:, :, dim_used]  # (S,T,N*3)
        pos_seqs = pos_seqs.transpose(0, 2, 1)
        pos_seqs = pos_seqs.reshape(pos_seqs.shape[0], -1, 3, pos_seqs.shape[-1])

        dq_seqs = all_dq.reshape(all_dq.shape[0], all_dq.shape[1], -1, 8).copy()
        dq_seqs[..., 4:] = dq_seqs[..., 4:] / scale
        dq_seqs = dq_seqs.transpose(0, 2, 3, 1)  # (S,N,8,T)

        self.used_seqs = dq_seqs.copy()
        self.pos_seqs = pos_seqs.copy()
        self.group = data_utils.group_attr(group)

        spine = self.group["spine"][1:]
        head = self.group["head"]
        arm_l = self.group["arm_l"]
        arm_r = self.group["arm_r"]
        leg_l = self.group["leg_l"]
        leg_r = self.group["leg_r"]

        graph = np.concatenate([spine, head, arm_l, arm_r, leg_l, leg_r], axis=0)
        edges = data_utils.edge_attr(graph, pos_seqs.shape[1], hop)

        self.edges = np.unique(edges, axis=0)
        self.parent = np.unique(graph, axis=0)

    def __len__(self):
        return np.shape(self.all_seqs)[0]

    def __getitem__(self, item):
        used_seqs = self.used_seqs[item]  # (N,8,T)
        pos_seqs = self.pos_seqs[item]    # (N,3,T)
        all_seqs = self.all_seqs[item]    # (T,N_all*3)
        if self.split == 0 and self.aug:
            if np.random.rand(1)[0] > 0.5:
                idx = np.arange(all_seqs.shape[0] - 1, -1, -1)
                used_seqs = used_seqs[..., idx]
                pos_seqs = pos_seqs[..., idx]
                all_seqs = all_seqs[idx]

        input_seq = used_seqs[..., :self.input_n]
        output_pos = pos_seqs[..., self.input_n:]
        output_dq = used_seqs[..., self.input_n:]
        return input_seq, output_pos, all_seqs, output_dq

    @staticmethod
    def define_acts(action):
        return CMU_Motion3D.define_acts(action)


class Pose3dPW3D(Dataset):
    exp = "Pose3d"
    actions = ["all"]
    def __init__(self, path_to_data, actions, input_n=10, output_n=12, dct_n=15,
                 split=0, scale=100, sample_rate="const", test_manner="all", aug=False, hop=-1):
        """
        param split: 0 train, 1 testing, 2 validation
        """

        self.path_to_data = path_to_data
        self.split = split
        self.dct_n = dct_n
        self.input_n = input_n
        self.output_n = output_n
        self.aug = aug
        # Some baselines observe 50 frames. This loader keeps the same target
        # subsequences, then uses only the last 10 observed frames as input.

        acts = self.define_acts(actions)

        if split == 1:
            their_input_n = 50
        else:
            their_input_n = input_n

        if split == 0:
            self.data_path = os.path.join(path_to_data, 'train')
        elif split == 1:
            self.data_path = os.path.join(path_to_data, 'test')
        elif split == 2:
            self.data_path = os.path.join(path_to_data, 'validation')

        all_seqs, dim_ignore, dim_used, group = data_utils.load_data_3dpw_3d(self.data_path, their_input_n, output_n)
        all_seqs = all_seqs[:, their_input_n - input_n:, :]

        all_seqs = all_seqs / scale
        self.all_seqs = all_seqs
        self.dim_used = dim_used

        all_seqs = all_seqs[:, :, dim_used]  # (S,T,N*3)
        all_seqs = all_seqs.transpose(0, 2, 1)  # (S,N*3,T)
        all_seqs = all_seqs.reshape(all_seqs.shape[0], -1, 3, all_seqs.shape[-1])  # (S,N,3,T)

        self.used_seqs = all_seqs.copy()
        self.group = data_utils.group_attr(group)

        spine = self.group["spine"][1:]
        head = self.group["head"]
        arm_l = self.group["arm_l"]
        arm_r = self.group["arm_r"]
        leg_l = self.group["leg_l"]
        leg_r = self.group["leg_r"]

        graph = np.concatenate([spine, head, arm_l, arm_r, leg_l, leg_r], axis=0)
        edges = data_utils.edge_attr(graph, all_seqs.shape[1], hop)

        self.edges = np.unique(edges, axis=0)
        self.parent = np.unique(graph, axis=0)

    def __len__(self):
        return np.shape(self.all_seqs)[0]

    def __getitem__(self, item):
        used_seqs = self.used_seqs[item]  # (N,3,T)
        all_seqs = self.all_seqs[item]  # (T,N*3)
        if self.split == 0 and self.aug:
            if np.random.rand(1)[0] > 0.5:
                idx = np.arange(all_seqs.shape[0] - 1, -1, -1)
                used_seqs = used_seqs[..., idx]
                all_seqs = all_seqs[idx]

        input_seq = used_seqs[..., :self.input_n]
        output_seq = used_seqs[..., self.input_n:]
        return input_seq, output_seq, all_seqs

    @staticmethod
    def define_acts(action):
        """
        Define the action list.
        Args
          action: String with the passed action. Could be "all"
        Returns
          actions: List of strings of actions
        Raises
          ValueError if the action is not included in 3dPW
        """
        if action in Pose3dPW3D.actions:
            return [action]

        if action == "all":
            return Pose3dPW3D.actions

        raise ValueError("Unrecognized action: {}".format(action))
