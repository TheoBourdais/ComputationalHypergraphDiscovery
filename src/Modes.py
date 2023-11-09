import numpy as onp




class ModeContainer:
    '''
        This class is used to store the modes of Graph Discovery
    '''
    def __init__(
        self,
        matrices,
        matrices_types,
        matrices_names,
        interpolatory_list,
        variable_names,
        beta,
        clusters=None,
        level=None,
        used=None
    ) -> None:
        self.constant_mat = onp.ones(matrices[0].shape[-2:])
        self.matrices=matrices
        self.matrices_types=matrices_types
        assert len(list(variable_names))==len(set(list(variable_names)))
        self.names = variable_names
        self.name_to_index = {name: i for i, name in enumerate(self.names)}
        self.beta = beta
        self.level = level
        self.matrices_names = matrices_names
        self.interpolatory_list=interpolatory_list
        self.assign_clusters(clusters)
        if level is None:
            self.level = onp.ones_like(matrices_names)
        else:
            self.level = level
        if used is not None:
            self.used = used
        else:
            self.used = {name:True for name in self.names}
        self.used_list=onp.array([self.used[name] for name in self.names])

    @property
    def node_number(self):
        return len(self.active_clusters)
    
    @property
    def active_names(self):
        return self.names[self.used_list]
    
    def assign_clusters(self,clusters):
        '''clusters must be a partition of the array matrices_names'''
        if clusters is None:
            self.clusters=list(self.names[:,None])
        else:
            clusters_list=[name for cluster in clusters for name in cluster]
            assert len(clusters_list)==len(set(clusters_list))
            assert len(clusters_list)==len(self.names)
            assert set(clusters_list).issubset(set(list(self.names)))
            self.clusters=clusters
    
    @property
    def active_clusters(self):
        res=[]
        for cluster in self.clusters:
            if self.cluster_is_active(cluster):
                res.append(cluster)
        return res
    
    def cluster_is_active(self,cluster):
        cluster_usage=[self.used[name] for name in cluster]
        #assert all elements in cluster_usage are the same
        assert len(set(cluster_usage))==1, f'all elements in cluster {"/".join(cluster)} must be either active or inactive at the same time'
        return cluster_usage[0]

    def is_interpolatory(self,chosen_level=None):
        level=self.get_level(chosen_level)
        res=False
        for li,is_interpolatory_bool in zip(level,self.interpolatory_list):
            res=res or (is_interpolatory_bool and li==1)
        return res
    
    def get_index_of_name(self, target_name):
        try:
            assert self.used[target_name]
        except KeyError or AssertionError:
            raise Exception(f"{target_name} is not in the modes' list of active names {self}")
        return self.name_to_index[target_name]
    
    def get_cluster_of_node_name(self, target_name):
        for cluster in self.clusters:
            if target_name in cluster:
                return cluster
        raise Exception(f"{target_name} was not in the modes' clusters {self.clusters}")
    
    def get_cluster_by_name(self,cluster_name):
        for cluster in self.clusters:
            if cluster_name=="/".join(cluster):
                return cluster
        raise Exception(f"{cluster_name} was not in the modes' clusters {self.clusters}")
    

    def delete_node(self, index):
        return self.delete_node_by_name(self.names[index])

    def delete_node_by_name(self, target_name):
        return self.delete_cluster(self.get_cluster_of_node_name(target_name))
    
    def delete_cluster_by_name(self,cluster_name):
        return self.delete_cluster(self.get_cluster_by_name(cluster_name))
    
    def delete_cluster(self,cluster):
        new_used=self.used.copy()
        for name in cluster:
            assert self.used[name]
            new_used[name]=False
        return ModeContainer(
            matrices=self.matrices,
            matrices_types=self.matrices_types,
            matrices_names=self.matrices_names,
            interpolatory_list=self.interpolatory_list,
            variable_names=self.names,
            beta=self.beta,
            level=self.level,
            used=new_used,
            clusters=self.clusters
        )

    def get_level(self, chosen_level):
        if chosen_level is None:
            return self.level
        level=[]
        found=False
        for level_name in self.matrices_names:
            if not found:
                level.append(1)
            else:
                level.append(0)
            if level_name==chosen_level:
                found=True
        if not found:
            raise Exception(f"Level {chosen_level} is not in the list of levels {self.matrices_names}")
        return onp.array(level)


    def set_level(self, chosen_level):
        assert chosen_level is not None
        self.level = self.get_level(chosen_level)

    def prod_and_sum(matrix,mask):
        return onp.prod(onp.add(matrix,onp.ones_like(matrix),where=mask[:,None,None]),axis=0,where=mask[:,None,None])

    def sum_a_matrix(matrix,matrix_type,used):
        if matrix_type=='individual':
            return onp.sum(matrix,axis=0,where=used[:,None,None])
        if matrix_type=='pairwise':
            used_2D=used[:,None]*used[None,:]
            return onp.sum(matrix,axis=(0,1),where=used_2D[:,:,None,None])
        if matrix_type=='combinatorial':
            return ModeContainer.prod_and_sum(matrix,used)
        raise f"Unknown matrix type {matrix_type}"
    
    def sum_a_matrix_of_index(matrix,matrix_type,used,index):
        if matrix_type=='individual':
            return matrix[index]
        if matrix_type=='pairwise':
            mask=onp.zeros(matrix.shape[0], dtype=bool)
            mask[index]=True
            used_2D=mask[:,None]*used[None,:] + used[:,None]*mask[None,:]
            return onp.sum(matrix, axis=(0,1),where=used_2D[:,:,None,None])
        if matrix_type=='combinatorial':
            used_for_prod = used.copy()
            used_for_prod[index]=False
            return matrix[index]*ModeContainer.prod_and_sum(matrix,used_for_prod)
        raise f"Unknown matrix type {matrix_type}"
    
    def sum_a_matrix_of_indexes(matrix,matrix_type,used,indexes):
        mask=onp.zeros(matrix.shape[0], dtype=bool)
        mask[indexes]=True
        if matrix_type=='individual':
            return onp.sum(matrix, axis=0,where=mask[:,None,None])
        if matrix_type=='pairwise':
            used_2D=mask[:,None]*used[None,:] + used[:,None]*mask[None,:]
            return onp.sum(matrix, axis=(0,1),where=used_2D[:,:,None,None])
        if matrix_type=='combinatorial':
            used_for_prod = used.copy()
            used_for_prod[indexes]=False
            return (ModeContainer.prod_and_sum(matrix,mask)-1)*ModeContainer.prod_and_sum(matrix,used_for_prod)
        raise f"Unknown matrix type {matrix_type}"

    def get_K(self, chosen_level=None):
        coeff = self.beta * self.get_level(chosen_level)
        K=onp.zeros_like(self.constant_mat)
        K+=self.constant_mat
        for i, matrix in enumerate(self.matrices):
            if coeff[i]!=0:
                K+=coeff[i]*ModeContainer.sum_a_matrix(matrix,self.matrices_types[i],self.used_list)
        return K
    
    def get_K_of_name(self, name):
        return self.get_K_of_index(self.get_index_of_name(name))

    def get_K_of_index(self, index):
        assert self.used_list[index]
        coeff = self.beta * self.level
        res = onp.zeros_like(self.constant_mat)
        for i, matrix in enumerate(self.matrices):
            res+=coeff[i]*ModeContainer.sum_a_matrix_of_index(matrix,self.matrices_types[i],self.used_list,index)
        return res
    
    def get_K_of_cluster(self,cluster):
        if len(cluster)==1:
            return self.get_K_of_name(cluster[0])
        assert self.cluster_is_active(cluster)
        indexes=[self.get_index_of_name(name) for name in cluster]
        coeff = self.beta * self.level
        res = onp.zeros_like(self.constant_mat)
        for i, matrix in enumerate(self.matrices):
            res+=coeff[i]*ModeContainer.sum_a_matrix_of_indexes(matrix,self.matrices_types[i],self.used_list,indexes)
        return res

    def get_K_without_index(self, index):
        assert self.used_list[index]
        return self.delete_node(index).get_K()
    
    def get_K_without_name(self, name):
        assert self.used[name]
        return self.delete_node_by_name(name).get_K()
    
    def get_K_without_cluster(self,cluster):
        assert self.cluster_is_active(cluster)
        return self.delete_cluster(cluster).get_K()

    def __repr__(self) -> str:
        res=['/'.join(cluster) for cluster in self.active_clusters]
        return res.__repr__()
    
    def make_container(X,variable_names,*args):
        assert X.shape[0]==len(variable_names)
        matrices=[]
        matrices_types=[]
        matrices_names=[]
        interpolatory_list=[]
        beta=[]
        for arg in args:
            assert arg['type'] in ['individual','combinatorial','pairwise']
            built=arg.get('built',False)
            if built:
                if arg['type'] in ['individual','combinatorial']:
                    assert arg['matrix'].shape==(len(variable_names),X.shape[1],X.shape[1])
                if arg['type']=='pairwise':
                    assert arg['matrix'].shape==(len(variable_names),len(variable_names),X.shape[1],X.shape[1])
                matrices.append(arg['matrix'])
            else:
                matrix=ModeContainer.build_matrix(X,**arg)
                matrices.append(matrix)
            matrices_types.append(arg['type'])
            matrices_names.append(arg['name'])
            interpolatory_list.append(arg['interpolatory'])
            beta.append(arg['beta'])
            if arg['type']=='pairwise':
                #in the inner workings of the code, off-diagonal matrices are counted twice
                matrices[-1][onp.triu_indices(matrices[-1].shape[0], k = 1)]=0
        return ModeContainer(
            matrices=matrices,
            matrices_types=matrices_types,
            matrices_names=matrices_names,
            variable_names=variable_names,
            interpolatory_list=interpolatory_list,
            beta=onp.array(beta)
        )
    

    def build_matrix(X,**kwargs):
        default=kwargs.get('default',False)
        if default:
            which=kwargs['name']
            assert which in ['linear','quadratic','gaussian'], f"Unknown default matrix {which}"
            if which=='linear':
                assert kwargs['type']=='individual', "Linear kernel is only available for individual matrices"
                return onp.expand_dims(X, -1) * onp.expand_dims(X, 1)
            if which=='quadratic':
                assert kwargs['type']=='pairwise', "Quadratic kernel is only available for pairwise matrices"
                linear_mat = onp.expand_dims(X, -1) * onp.expand_dims(X, 1)
                quadratic_mat= onp.expand_dims(linear_mat, 0) * onp.expand_dims(linear_mat, 1)
                return quadratic_mat
            if which=='gaussian':
                assert kwargs['type']=='combinatorial', "Gaussian kernel is only available for combinatorial matrices"
                try:
                    l=kwargs['l']
                except:
                    raise Exception("You must specify the lengthscale of the gaussian kernel")
                assert len(X.shape)==2, "Gaussian kernel is only available for 1D data"
                diff_X = onp.tile(onp.expand_dims(X, -1), (1, 1, X.shape[1])) - onp.tile(
                    onp.expand_dims(X, 1), (1, X.shape[1], 1)
                )
                return onp.exp(-((diff_X / l) ** 2) / 2)
            return onp.ones((X.shape[0],X.shape[0]))
        scipy_kernel=kwargs.get('scipy_kernel',None)
        if scipy_kernel is not None:
            '''must behave like scikit-learn kernels'''
            res=[]
            if kwargs['type'] in ['individual','combinatorial']:
                for col in X:
                    if len(col.shape)>1:
                        res.append(scipy_kernel(col))
                    res.append(scipy_kernel(col.expand_dims(1)))
                return onp.stack(res,axis=0)
            if kwargs['type']=='pairwise':
                res=onp.zeros((X.shape[0],X.shape[0],X.shape[1],X.shape[1]))
                for i,col1 in enumerate(X):
                    for j,col2 in enumerate(X[:i+1]):
                        data=onp.stack([col1,col2],axis=1)
                        res[i,j,:,:]=scipy_kernel(data)
                return res
        
        raise Exception("You must either provide a default kernel or a scipy kernel")
            





