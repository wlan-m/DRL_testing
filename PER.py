import numpy as np

class SumTree(object):
    data_pointer = 0

    # initialize the dtree with all nodes = 0 and the data with all values = 0
    def __init__(self, capacity):
        # the number of leaf nodes that contains experiences
        self.capacity = capacity

        # generate the tree with all nodes values = 0
        # remember that we are in a binary node (max 2 children per node), so 2x size of leaf (capacity) - 1 (root node)
        # parent nodes = capacity - 1
        # leaf nodes = capacity
        self.tree = np.zeros(2*capacity-1)

        # contains the experiences, so the size of data is capacity
        self.data = np.zeros(capacity, dtype=object)

    
    # define the function that will add our priority score in the sumtree leaf and add the experience in data
    def add(self, priority, data):
        # look at what index we want to put the experince
        tree_index = self.data_pointer + self.capacity - 1

        # updata the data frame
        self.data[self.data_pointer] = data

        # updata the leaf
        self.update(tree_index, priority)

        # increase the data_pointer
        self.data_pointer += 1

        # if we are above the capacity, we go back to the first index --> overwrite
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0
        

    # update the leaf priority score and propate the change through tree
    def update(self, tree_index, priority):
        # change = new priority score - old priority score
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        # propagate the change through the tree
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2      # // floor division
            self.tree[tree_index] += change

    
    # build a function to get a leaf from our tree. so we will build a function to get the leaf_index,
    # priority value of that leaf and experience associated with that leaf index
    def get_leaf(self, v):
        parent_index = 0

        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            # if we reach the bottom, end the search
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else: 
                # downward search, alway search for a higher priority node
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index
            
        data_index = leaf_index - self.capacity + 1

        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self):
        # returns the root node
        return self.tree[0]



# after finishing constructing the sumTree object, we bulid a memory object
class Memory(object):   
    # stored as (state, action, rewardd, next_state) in SumTree
    per_e = 0.01    # hyperparameter used to avoid some experiences to have 0 probability of being taken
    per_a = 0.6     # hyperparameter used to make a tradeoff bewtween taking only exp with high priority and sampling ranodmly
    per_b = 0.4     # importance sampling, from initial value increasing to 1
    per_b_increment_per_sampling = 0.001
    absolute_error_upper = 1. # clipped abs error

    def __init__(self, capacity):
        # making the tree
        self.tree = SumTree(capacity)
    

    # define a function to store a new experience in the ree
    # each new experience will have a score of max_priority (it will be then impreved when we use this exp to train our D3QN)
    def store(self, experience):
        # find the max priority
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])

        # if the max priority = 0 we cant put priority = 0 since this experience will never have a change to be selected
        # --> use a minimum priority
        if max_priority == 0:
            max_priority = self.absolute_error_upper
        
        # set the max priority for new priority
        self.tree.add(max_priority, experience)


    # now create a sample function, will is used to pick a batch from the tree memory, which will bw used to train our model.
    #  - first, sample a minibatch of n size, the range [0, priority_total] into priority ranges
    #  - then, a value is uniformly sampled from each range
    #  - the we search in the sumtree, for the experience where priority score correspond to sample values are retrieved from.
    def sample(self, n):
        # create a minibatch array that will contain the minibatch
        minibatch = []

        b_idx = np.empty((n,),dtype=np.int32)

        # calculate the priority segment
        # devide the range[0, ptotal] into n ranges.
        priority_segment = self.tree.total_priority / n     # priority segment

        for i in range(n):
            # a value is uniformly sampled from each range
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)

            # experience that correspond to each value is retrieved
            index, priority, data = self.tree.get_leaf(value)

            b_idx[i] = index

            minibatch.append([data[0], data[1], data[2], data[3], data[4]])

        return b_idx, minibatch


    # update the priorities on the tree
    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.per_e # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.per_a)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)
