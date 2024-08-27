import numpy as np
import random
from math import floor
import utils
'''Buffer for Composite Experience Replay DDPG algorithm. It is a child class of Replay Buffer.
'''

class ReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t,copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(np.array(reward,copy=False))
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(np.array([done]))
        return np.stack(obses_t), np.stack(actions), np.stack(rewards),np.stack(obses_tp1), np.stack(dones)

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)


class CompositeReplayBuffer(ReplayBuffer):
    def __init__(self, size):
        """Create Prioritized Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)
        See Also
        --------
        ReplayBuffer.__init__
        """
        super(CompositeReplayBuffer, self).__init__(size)

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum_T = utils.SumSegmentTree(it_capacity)
        self._it_min_T = utils.MinSegmentTree(it_capacity)
        self._it_sum_R=utils.SumSegmentTree(it_capacity)
        self._it_min_R=utils.MinSegmentTree(it_capacity)
        self._max_priority_T = 1.0
        self._max_priority_R=1.0

    def add(self, *args, **kwargs):
        """See ReplayBuffer.store_effect"""
        idx = self._next_idx
        super().add(*args, **kwargs)
        self._it_sum_T[idx] = self._max_priority_T
        self._it_min_T[idx] = self._max_priority_T
        self._it_sum_R[idx]=self._max_priority_R
        self._it_min_R[idx]=self._max_priority_R 

    def _sample_proportional(self, batch_size,T_or_R) :
        if T_or_R:
            res = []
            p_total = self._it_sum_T.sum(0, len(self._storage) - 1)
            every_range_len = p_total / 32
            for i in range(batch_size):
                mass = random.random() * every_range_len + i * every_range_len
                idx = self._it_sum_T.find_prefixsum_idx(mass)
                res.append(idx)
        else:
            res = []
            p_total = self._it_sum_R.sum(0, len(self._storage) - 1)
            every_range_len = p_total / 32
            for i in range(batch_size):
                mass = 30*random.random() * every_range_len + i * every_range_len
                idx = self._it_sum_R.find_prefixsum_idx(mass)
                res.append(idx)
        return res

    def sample(self, batch_size, beta,alpha):
        """Sample a batch of experiences.
        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """
        l_i=np.zeros((len(self._storage)),dtype=np.int32)
        w_i=np.zeros((len(self._storage)),dtype=np.float32)
        _lambda=0
        idxes_T=self._sample_proportional(floor((1-alpha)*batch_size),True)
        l_i[idxes_T]=1
        assert beta > 0
        weights = []
        p_min = self._it_min_T.min() / self._it_sum_T.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)
        for idx in idxes_T:
            p_sample = self._it_sum_T[idx] / self._it_sum_T.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            w_i[idx]=weight/max_weight
        idxes_R=[]
        p_min = self._it_min_R.min() / self._it_sum_R.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)
        while (_lambda + floor((1-alpha)*batch_size))<batch_size:
            idx=self._sample_proportional(1,False)
            idx=idx[0]
            if l_i[idx]==0:
                l_i[idx]=2
                p_sample = self._it_sum_R[idx] / self._it_sum_R.sum()
                weight = (p_sample * len(self._storage)) ** (-beta)
                w_i[idx]=weight/max_weight
                idxes_R.append(idx)
                _lambda+=1
            elif l_i[idx]==1:
                l_i[idx]=3
                p_sample = self._it_sum_R[idx] / self._it_sum_R.sum()
                weight = (p_sample * len(self._storage)) ** (-beta)
                w_r=weight/max_weight
                w_i[idx]+=w_r
        idxes=idxes_T+idxes_R
        encoded_sample=self._encode_sample(idxes)
        weights=w_i[idxes]
        return tuple(list(encoded_sample) + [weights, idxes])

    def update_priorities(self, idxes, priorities_T,priorities_R):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities_T)
        assert len(idxes) == len(priorities_R)
        for idx, priority_T,priority_R in zip(idxes, priorities_T,priorities_R):
            assert priority_T > 0
            assert priority_R > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum_T[idx] = priority_T
            self._it_min_R[idx] = priority_R

            self._max_priority_T = max(self._max_priority_T, priority_T)
            self._max_priority_R = max(self._max_priority_R, priority_R)

