import torch
class Distribution(torch.Tensor):
    # Init the params of the distribution
    def init_distribution(self, dist_type, **kwargs):
        self.dist_type = dist_type
        self.dist_kwargs = kwargs
        if self.dist_type == 'normal':
            self.mean, self.var = kwargs['mean'], kwargs['var']
        elif self.dist_type == 'categorical':
            self.num_categories = kwargs['num_categories']

    def sample_(self):
        if self.dist_type == 'normal':
            self.normal_(self.mean, self.var)
        elif self.dist_type == 'categorical':
            self.random_(0, self.num_categories)
            # return self.variable

    # Silly hack: overwrite the to() method to wrap the new object
    # in a distribution as well
    def to(self, *args, **kwargs):
        new_obj = Distribution(self)
        new_obj.init_distribution(self.dist_type, **self.dist_kwargs)
        print("before: ", new_obj.data)
        # new_obj.data = super().to(*args, **kwargs)  # now sure it is correct or not
        new_obj.data.to(args[0])
        new_obj.data = new_obj.data.type(args[1])
        print("after: ", new_obj.data)
        return new_obj

G_batch_size = 2
dim_z = 2
device = 'cpu'
fp16 = False
z_var = 1
z_ = Distribution(torch.randn(G_batch_size, dim_z, requires_grad=False))
z_.init_distribution('normal', mean=0, var=z_var)
z_ = z_.to(device, torch.float16 if fp16 else torch.float32)

y_ = Distribution(torch.zeros(G_batch_size, requires_grad=False))
y_.init_distribution('categorical', num_categories=5)
y_ = y_.to(device, torch.int64)
