from __future__ import division
import torch
import torch.nn as nn
from torch.nn import init
import numbers
import torch.nn.functional as F
from .STAEformer import STAEformer


class NConv(nn.Module):
	def __init__(self):
		super(NConv, self).__init__()

	def forward(self, x, adj):
		x = torch.einsum('ncwl,vw->ncvl', (x, adj))
		return x.contiguous()
# class GCNLayer(nn.Module):
#     def __init__(self):
#         super(GCNLayer, self).__init__()
#         self.act = nn.LeakyReLU(negative_slope=0.2)
#     def forward(self, x, adj):
#         return torch.bmm(adj, x)

class DyNconv(nn.Module):
	def __init__(self):
		super(DyNconv, self).__init__()

	def forward(self, x, adj):
		x = torch.einsum('ncvl,nvwl->ncwl', (x, adj))
		return x.contiguous()


class Linear(nn.Module):
	def __init__(self, c_in, c_out, bias=True):
		super(Linear, self).__init__()
		self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=bias)

	def forward(self, x):
		return self.mlp(x)


class Prop(nn.Module):
	def __init__(self, c_in, c_out, gdep, dropout, alpha):
		super(Prop, self).__init__()
		self.nconv = NConv()
		self.mlp = Linear(c_in, c_out)
		self.gdep = gdep
		self.dropout = dropout
		self.alpha = alpha

	def forward(self, x, adj):
		adj = adj + torch.eye(adj.size(0)).to(x.device)
		d = adj.sum(1)
		h = x
		dv = d
		a = adj / dv.view(-1, 1)
		for i in range(self.gdep):
			h = self.alpha*x + (1-self.alpha)*self.nconv(h, a)
		ho = self.mlp(h)
		return ho


class MixProp(nn.Module):
	def __init__(self, c_in, c_out, gdep, dropout, alpha):
		super(MixProp, self).__init__()
		# self.nconv = nn.Sequential(GCNLayer())
		self.nconv = NConv()
		self.mlp = Linear((gdep+1)*c_in, c_out)
		self.gdep = gdep
		self.dropout = dropout
		self.alpha = alpha

	def forward(self, x, adj):
		adj = adj + torch.eye(adj.size(0)).to(x.device)
		d = adj.sum(1)
		h = x
		out = [h]
		a = adj / d.view(-1, 1)
		for i in range(self.gdep):
			h = self.alpha*x + (1-self.alpha)*self.nconv(h, a)
			out.append(h)
		ho = torch.cat(out, dim=1)
		ho = self.mlp(ho)
		return ho


class DyMixprop(nn.Module):
	def __init__(self, c_in, c_out, gdep, dropout, alpha):
		super(DyMixprop, self).__init__()
		self.nconv = DyNconv()
		self.mlp1 = Linear((gdep+1)*c_in, c_out)
		self.mlp2 = Linear((gdep+1)*c_in, c_out)

		self.gdep = gdep
		self.dropout = dropout
		self.alpha = alpha
		self.lin1 = Linear(c_in, c_in)
		self.lin2 = Linear(c_in, c_in)

	def forward(self, x):
		x1 = torch.tanh(self.lin1(x))
		x2 = torch.tanh(self.lin2(x))
		adj = self.nconv(x1.transpose(2, 1), x2)
		adj0 = torch.softmax(adj, dim=2)
		adj1 = torch.softmax(adj.transpose(2, 1), dim=2)

		h = x
		out = [h]
		for i in range(self.gdep):
			h = self.alpha*x + (1-self.alpha)*self.nconv(h, adj0)
			out.append(h)
		ho = torch.cat(out, dim=1)
		ho1 = self.mlp1(ho)

		h = x
		out = [h]
		for i in range(self.gdep):
			h = self.alpha * x + (1 - self.alpha) * self.nconv(h, adj1)
			out.append(h)
		ho = torch.cat(out, dim=1)
		ho2 = self.mlp2(ho)
		return ho1+ho2


class Dilated1D(nn.Module):
	def __init__(self, cin, cout, dilation_factor=2):
		super(Dilated1D, self).__init__()
		self.tconv = nn.ModuleList()
		self.kernel_set = [2, 3, 6, 7]
		self.tconv = nn.Conv2d(cin, cout, (1, 7), dilation=(1, dilation_factor))

	def forward(self, inputs):
		x = self.tconv(inputs)
		return x


class DilatedInception(nn.Module):
	def __init__(self, cin, cout, dilation_factor=2):
		super(DilatedInception, self).__init__()
		self.tconv = nn.ModuleList()
		self.kernel_set = [2, 3, 6, 7]
		cout = int(cout/len(self.kernel_set))
		for kern in self.kernel_set:
			self.tconv.append(nn.Conv2d(cin, cout, (1, kern), dilation=(1, dilation_factor)))

	def forward(self, input):
		x = []
		for i in range(len(self.kernel_set)):
			x.append(self.tconv[i](input))
		for i in range(len(self.kernel_set)):
			x[i] = x[i][..., -x[-1].size(3):]
		x = torch.cat(x, dim=1)
		return x


class GraphConstructor(nn.Module):
	def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
		super(GraphConstructor, self).__init__()
		self.nnodes = nnodes
		if static_feat is not None:
			xd = static_feat.shape[1]
			self.lin1 = nn.Linear(xd, dim)
			self.lin2 = nn.Linear(xd, dim)
		else:
			self.emb1 = nn.Embedding(nnodes, dim)
			self.emb2 = nn.Embedding(nnodes, dim)
			self.lin1 = nn.Linear(dim, dim)
			self.lin2 = nn.Linear(dim, dim)

		self.device = device
		self.k = k
		self.dim = dim
		self.alpha = alpha
		self.static_feat = static_feat

	def forward(self, idx):
		if self.static_feat is None:
			nodevec1 = self.emb1(idx)
			nodevec2 = self.emb2(idx)
		else:
			nodevec1 = self.static_feat[idx, :]
			nodevec2 = nodevec1

		nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
		nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

		a = torch.mm(nodevec1, nodevec2.transpose(1, 0))-torch.mm(nodevec2, nodevec1.transpose(1, 0))
		adj = F.relu(torch.tanh(self.alpha*a))
		mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
		mask.fill_(float('0'))
		s1, t1 = adj.topk(self.k, 1)
		mask.scatter_(1, t1, s1.fill_(1))
		adj = adj*mask
		return adj

	def fulla(self, idx):
		if self.static_feat is None:
			nodevec1 = self.emb1(idx)
			nodevec2 = self.emb2(idx)
		else:
			nodevec1 = self.static_feat[idx, :]
			nodevec2 = nodevec1

		nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
		nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

		a = torch.mm(nodevec1, nodevec2.transpose(1, 0))-torch.mm(nodevec2, nodevec1.transpose(1, 0))
		adj = F.relu(torch.tanh(self.alpha*a))
		return adj


class GraphGlobal(nn.Module):
	def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
		super(GraphGlobal, self).__init__()
		self.nnodes = nnodes
		self.A = nn.Parameter(torch.randn(nnodes, nnodes).to(device), requires_grad=True).to(device)

	def forward(self, idx):
		return F.relu(self.A)


class GraphUndirected(nn.Module):
	def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
		super(GraphUndirected, self).__init__()
		self.nnodes = nnodes
		if static_feat is not None:
			xd = static_feat.shape[1]
			self.lin1 = nn.Linear(xd, dim)
		else:
			self.emb1 = nn.Embedding(nnodes, dim)
			self.lin1 = nn.Linear(dim, dim)

		self.device = device
		self.k = k
		self.dim = dim
		self.alpha = alpha
		self.static_feat = static_feat

	def forward(self, idx):
		if self.static_feat is None:
			nodevec1 = self.emb1(idx)
			nodevec2 = self.emb1(idx)
		else:
			nodevec1 = self.static_feat[idx, :]
			nodevec2 = nodevec1

		nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
		nodevec2 = torch.tanh(self.alpha*self.lin1(nodevec2))

		a = torch.mm(nodevec1, nodevec2.transpose(1, 0))
		adj = F.relu(torch.tanh(self.alpha*a))
		mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
		mask.fill_(float('0'))
		s1, t1 = adj.topk(self.k, 1)
		mask.scatter_(1, t1, s1.fill_(1))
		adj = adj*mask
		return adj


class GraphDirected(nn.Module):
	def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
		super(GraphDirected, self).__init__()
		self.nnodes = nnodes
		if static_feat is not None:
			xd = static_feat.shape[1]
			self.lin1 = nn.Linear(xd, dim)
			self.lin2 = nn.Linear(xd, dim)
		else:
			self.emb1 = nn.Embedding(nnodes, dim)
			self.emb2 = nn.Embedding(nnodes, dim)
			self.lin1 = nn.Linear(dim, dim)
			self.lin2 = nn.Linear(dim, dim)

		self.device = device
		self.k = k
		self.dim = dim
		self.alpha = alpha
		self.static_feat = static_feat

	def forward(self, idx):
		if self.static_feat is None:
			nodevec1 = self.emb1(idx)
			nodevec2 = self.emb2(idx)
		else:
			nodevec1 = self.static_feat[idx, :]
			nodevec2 = nodevec1

		nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
		nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

		a = torch.mm(nodevec1, nodevec2.transpose(1, 0))
		adj = F.relu(torch.tanh(self.alpha*a))
		mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
		mask.fill_(float('0'))
		s1, t1 = adj.topk(self.k, 1)
		mask.scatter_(1, t1, s1.fill_(1))
		adj = adj*mask
		return adj


class LayerNorm(nn.Module):
	__constants__ = ['normalized_shape', 'weight', 'bias', 'eps', 'elementwise_affine']

	def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
		super(LayerNorm, self).__init__()
		if isinstance(normalized_shape, numbers.Integral):
			normalized_shape = (normalized_shape,)
		self.normalized_shape = tuple(normalized_shape)
		self.eps = eps
		self.elementwise_affine = elementwise_affine
		if self.elementwise_affine:
			self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
			self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
		else:
			self.register_parameter('weight', None)
			self.register_parameter('bias', None)
		self.reset_parameters()

	def reset_parameters(self):
		if self.elementwise_affine:
			init.ones_(self.weight)
			init.zeros_(self.bias)

	def forward(self, inputs, idx):
		if self.elementwise_affine:
			return F.layer_norm(inputs, tuple(inputs.shape[1:]),
								self.weight[:, idx, :], self.bias[:, idx, :], self.eps)
		else:
			return F.layer_norm(inputs, tuple(inputs.shape[1:]),
								self.weight, self.bias, self.eps)

	def extra_repr(self):
		return '{normalized_shape}, eps={eps}, ' \
			'elementwise_affine={elementwise_affine}'.format(**self.__dict__)



class Teacher(nn.Module):
	def __init__(self, args):
		super(Teacher, self).__init__()
		self.adj_mx = args.adj_mx
		self.num_nodes = args.num_nodes
		self.feature_dim = args.input_dim

		self.input_window = args.input_window
		self.output_window = args.output_window
		self.output_dim = args.output_dim
		self.device = args.device

		self.gcn_true = args.gcn_true
		self.buildA_true = args.buildA_true
		self.gcn_depth = args.gcn_depth
		self.dropout = args.dropout
		self.subgraph_size = args.subgraph_size
		self.node_dim = args.node_dim
		self.dilation_exponential = args.dilation_exponential

		self.conv_channels = args.conv_channels
		self.residual_channels = args.residual_channels
		self.skip_channels = args.skip_channels
		self.end_channels = args.end_channels

		self.layers = args.layers
		self.propalpha = args.propalpha
		self.tanhalpha = args.tanhalpha
		self.layer_norm_affline = args.layer_norm_affline

		self.use_curriculum_learning = args.use_curriculum_learning

		self.task_level = args.task_level
		self.idx = torch.arange(self.num_nodes).to(self.device)

		if self.adj_mx is None:
			self.predefined_A = None
		else:
			# Ensure dtype is explicit and move to target device
			_pre = torch.as_tensor(self.adj_mx, dtype=torch.float32)
			_pre = _pre - torch.eye(self.num_nodes, dtype=torch.float32)
			self.predefined_A = _pre.to(self.device)
		self.static_feat = None

		# transformer attention neural network
		self.encoder_layer = nn.TransformerEncoderLayer(d_model=12, nhead=4)
		self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)

		self.filter_convs = nn.ModuleList()
		self.gate_convs = nn.ModuleList()
		self.residual_convs = nn.ModuleList()
		self.skip_convs = nn.ModuleList()
		self.gconv1 = nn.ModuleList()
		self.gconv2 = nn.ModuleList()
		self.norm = nn.ModuleList()
		self.stu_mlp = nn.ModuleList()
		self.stu_mlp.append(nn.Sequential(nn.Linear(13,13),nn.Linear(13,13),nn.Linear(13,13)))
		self.stu_mlp.append(nn.Sequential(nn.Linear(7,7),nn.Linear(7,7),nn.Linear(7,7)))
		self.stu_mlp.append(nn.Sequential(nn.Linear(1,1),nn.Linear(1,1),nn.Linear(1,1)))
		self.start_conv = nn.Conv2d(in_channels=self.feature_dim,
									out_channels=self.residual_channels,
									kernel_size=(1, 1))
		self.gc = GraphConstructor(self.num_nodes, self.subgraph_size, self.node_dim,
								   self.device, alpha=self.tanhalpha, static_feat=self.static_feat)

		kernel_size = 7
		if self.dilation_exponential > 1:
			self.receptive_field = int(self.output_dim + (kernel_size-1) * (self.dilation_exponential**self.layers-1)
									   / (self.dilation_exponential - 1))
		else:
			self.receptive_field = self.layers * (kernel_size-1) + self.output_dim

		for i in range(1):
			if self.dilation_exponential > 1:
				rf_size_i = int(1 + i * (kernel_size-1) * (self.dilation_exponential**self.layers-1)
								/ (self.dilation_exponential - 1))
			else:
				rf_size_i = i * self.layers * (kernel_size - 1) + 1
			new_dilation = 1
			for j in range(1, self.layers+1):
				if self.dilation_exponential > 1:
					rf_size_j = int(rf_size_i + (kernel_size-1) * (self.dilation_exponential**j - 1)
									/ (self.dilation_exponential - 1))
				else:
					rf_size_j = rf_size_i+j*(kernel_size-1)

				self.filter_convs.append(DilatedInception(self.residual_channels,
														  self.conv_channels, dilation_factor=new_dilation))
				self.gate_convs.append(DilatedInception(self.residual_channels,
														self.conv_channels, dilation_factor=new_dilation))
				self.residual_convs.append(nn.Conv2d(in_channels=self.conv_channels,
													 out_channels=self.residual_channels, kernel_size=(1, 1)))

				if self.input_window > self.receptive_field:
					self.skip_convs.append(nn.Conv2d(in_channels=self.conv_channels, out_channels=self.skip_channels,
													 kernel_size=(1, self.input_window-rf_size_j+1)))
					# self.skip_convs.append(self.transformer_encoder)
				else:
					self.skip_convs.append(nn.Conv2d(in_channels=self.conv_channels, out_channels=self.skip_channels,
													 kernel_size=(1, self.receptive_field-rf_size_j+1)))
					# self.skip_convs.append(self.transformer_encoder)

				if self.gcn_true:
					self.gconv1.append(MixProp(self.conv_channels, self.residual_channels,
											   self.gcn_depth, self.dropout, self.propalpha))
					self.gconv2.append(MixProp(self.conv_channels, self.residual_channels,
											   self.gcn_depth, self.dropout, self.propalpha))

				if self.input_window > self.receptive_field:
					self.norm.append(LayerNorm((self.residual_channels, self.num_nodes,
												self.input_window - rf_size_j + 1),
											   elementwise_affine=self.layer_norm_affline))
				else:
					self.norm.append(LayerNorm((self.residual_channels, self.num_nodes,
												self.receptive_field - rf_size_j + 1),
											   elementwise_affine=self.layer_norm_affline))

				new_dilation *= self.dilation_exponential

		self.end_conv_1 = nn.Conv2d(in_channels=self.skip_channels,
									out_channels=self.end_channels, kernel_size=(1, 1), bias=True)
		self.end_conv_2 = nn.Conv2d(in_channels=self.end_channels,
									out_channels=self.output_window, kernel_size=(1, 1), bias=True)
		if self.input_window > self.receptive_field:
			self.skip0 = nn.Conv2d(in_channels=self.feature_dim,
								   out_channels=self.skip_channels,
								   kernel_size=(1, self.input_window), bias=True)
			self.skipE = nn.Conv2d(in_channels=self.residual_channels,
								   out_channels=self.skip_channels,
								   kernel_size=(1, self.input_window-self.receptive_field+1), bias=True)
		else:
			self.skip0 = nn.Conv2d(in_channels=self.feature_dim,
								   out_channels=self.skip_channels, kernel_size=(1, self.receptive_field), bias=True)
			self.skipE = nn.Conv2d(in_channels=self.residual_channels,
								   out_channels=self.skip_channels, kernel_size=(1, 1), bias=True)

		# self._logger.info('receptive_field: ' + str(self.receptive_field))
		

	
	def forward(self, source, idx=None, save_visualization_data=False):
		# inputs = batch['X']  # (batch_size, input_window, num_nodes, feature_dim)
		sout = []
		tout = []
		inputs = source
		inputs = inputs.transpose(1, 3)  # (batch_size, feature_dim, num_nodes, input_window) #64, 1, 170, 12
		
		assert inputs.size(3) == self.input_window, 'input sequence length not equal to preset sequence length'
		# inputs = inputs.view(-1, self.num_nodes, self.input_window)
		
		# #spatial transformer 
		# out = self.transformer_encoder(inputs)
		# # print(inputs.size())
		# inputs = out.view(-1, self.feature_dim, self.num_nodes, self.input_window)
		# print(inputs.size())
		# println()
		if self.input_window < self.receptive_field:
			inputs = nn.functional.pad(inputs, (self.receptive_field-self.input_window, 0, 0, 0))

		if self.gcn_true:
			if self.buildA_true:
				if idx is None:
					adp = self.gc(self.idx)
				else:
					adp = self.gc(idx)
			else:
				adp = self.predefined_A

		x = self.start_conv(inputs)
		skip = self.skip0(F.dropout(inputs, self.dropout, training=self.training))
		for i in range(self.layers):
			residual = x
			filters = self.filter_convs[i](x)
			filters = torch.tanh(filters)
			gate = self.gate_convs[i](x)
			gate = torch.sigmoid(gate)
			x = filters * gate
			x = F.dropout(x, self.dropout, training=self.training)
			tout.append(x)
			s = x
			s = self.skip_convs[i](s)
			skip = s + skip
			if self.gcn_true:
				# print("gcn in x:", x.size())
				x = self.gconv1[i](x, adp)+self.gconv2[i](x, adp.transpose(1, 0)) # in :64, 32, 170, 13, out: 64, 32, 170, 13 , 64, 32, 170, 7], 64, 32, 170, 1]
				# print("gcn out x:", x.size())
				# println()
			else:
				# x = self.residual_convs[i](x)
				x = self.stu_mlp[i](x)
				print("mlp out x:", x.size())

			x = x + residual[:, :, :, -x.size(3):]
			if idx is None:
				x = self.norm[i](x, self.idx)
			else:
				x = self.norm[i](x, idx)
			sout.append(x)
		skip = self.skipE(x) + skip
		x = F.relu(skip)
		x = F.relu(self.end_conv_1(x))
		x = self.end_conv_2(x)
		# x = nn.Linear(self.num_nodes, self.input_window).cuda()(x.view(-1, self.input_window, self.num_nodes))
		# x = self.transformer_encoder(x)
		# x = nn.Linear(self.input_window, self.num_nodes).cuda()(x.view(-1, self.input_window, self.input_window)).view(-1, self.input_window, self.num_nodes, self.feature_dim)
		# print("x.size():", x.size())
		# println()
		ttout = nn.Linear(1, 32).cuda()(nn.Linear(self.residual_channels, self.input_window).cuda()(tout[-1].transpose(1,3)).transpose(1,3))
		ssout = nn.Linear(1, 32).cuda()(nn.Linear(self.residual_channels, self.input_window).cuda()(sout[-1].transpose(1,3)).transpose(1,3))
		# print(ttout.size(), ssout.size())
		# println()
        # x.shape 为（Batch_size, output_window, num_nodes, 1 traffic feature）
        # ttout与ssout为 batch_size, output_window, nums_nodes, 32
		return x, ttout, ssout
class STMLP(nn.Module):
	def __init__(self, args):
		super(STMLP, self).__init__()

		self.adj_mx = args.adj_mx
		self.num_nodes = args.num_nodes
		self.feature_dim = args.input_dim

		self.input_window = args.input_window
		self.output_window = args.output_window
		self.output_dim = args.output_dim
		self.device = args.device

		self.gcn_true = args.gcn_true
		self.buildA_true = args.buildA_true
		self.gcn_depth = args.gcn_depth
		self.dropout = args.dropout
		self.subgraph_size = args.subgraph_size
		self.node_dim = args.node_dim
		self.dilation_exponential = args.dilation_exponential

		self.conv_channels = args.conv_channels
		self.residual_channels = args.residual_channels
		self.skip_channels = args.skip_channels
		self.end_channels = args.end_channels

		self.layers = args.layers
		self.propalpha = args.propalpha
		self.tanhalpha = args.tanhalpha
		self.layer_norm_affline = args.layer_norm_affline

		self.use_curriculum_learning = args.use_curriculum_learning

		self.task_level = args.task_level
		self.idx = torch.arange(self.num_nodes).to(self.device)

		if self.adj_mx is None:
			self.predefined_A = None
		else:
			# Ensure dtype is explicit and move to target device
			_pre = torch.as_tensor(self.adj_mx, dtype=torch.float32)
			_pre = _pre - torch.eye(self.num_nodes, dtype=torch.float32)
			self.predefined_A = _pre.to(self.device)
		self.static_feat = None

		# transformer attention neural network
		self.encoder_layer = nn.TransformerEncoderLayer(d_model=12, nhead=4)
		self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)

		self.filter_convs = nn.ModuleList()
		self.gate_convs = nn.ModuleList()
		self.residual_convs = nn.ModuleList()
		self.skip_convs = nn.ModuleList()
		self.gconv1 = nn.ModuleList()
		self.gconv2 = nn.ModuleList()
		self.norm = nn.ModuleList()
		self.stu_mlp = nn.ModuleList()
		# self.stu_mlp.append(nn.Linear(13, 13))
		# self.stu_mlp.append(nn.Linear(7, 7))
		# self.stu_mlp.append(nn.Linear(1, 1))
		self.stu_mlp.append(nn.Sequential(nn.Linear(13,13),nn.Linear(13,13),nn.Linear(13,13)))
		self.stu_mlp.append(nn.Sequential(nn.Linear(7,7),nn.Linear(7,7),nn.Linear(7,7)))
		self.stu_mlp.append(nn.Sequential(nn.Linear(1,1),nn.Linear(1,1),nn.Linear(1,1)))
		self.start_conv = nn.Conv2d(in_channels=self.feature_dim,
									out_channels=self.residual_channels,
									kernel_size=(1, 1))
		self.gc = GraphConstructor(self.num_nodes, self.subgraph_size, self.node_dim,
								   self.device, alpha=self.tanhalpha, static_feat=self.static_feat)

		kernel_size = 7
		if self.dilation_exponential > 1:
			self.receptive_field = int(self.output_dim + (kernel_size-1) * (self.dilation_exponential**self.layers-1)
									   / (self.dilation_exponential - 1))
		else:
			self.receptive_field = self.layers * (kernel_size-1) + self.output_dim

		for i in range(1):
			if self.dilation_exponential > 1:
				rf_size_i = int(1 + i * (kernel_size-1) * (self.dilation_exponential**self.layers-1)
								/ (self.dilation_exponential - 1))
			else:
				rf_size_i = i * self.layers * (kernel_size - 1) + 1
			new_dilation = 1
			for j in range(1, self.layers+1):
				if self.dilation_exponential > 1:
					rf_size_j = int(rf_size_i + (kernel_size-1) * (self.dilation_exponential**j - 1)
									/ (self.dilation_exponential - 1))
				else:
					rf_size_j = rf_size_i+j*(kernel_size-1)

				self.filter_convs.append(DilatedInception(self.residual_channels,
														  self.conv_channels, dilation_factor=new_dilation))
				self.gate_convs.append(DilatedInception(self.residual_channels,
														self.conv_channels, dilation_factor=new_dilation))
				self.residual_convs.append(nn.Conv2d(in_channels=self.conv_channels,
													 out_channels=self.residual_channels, kernel_size=(1, 1)))

				if self.input_window > self.receptive_field:
					self.skip_convs.append(nn.Conv2d(in_channels=self.conv_channels, out_channels=self.skip_channels,
													 kernel_size=(1, self.input_window-rf_size_j+1)))
					# self.skip_convs.append(self.transformer_encoder)
				else:
					self.skip_convs.append(nn.Conv2d(in_channels=self.conv_channels, out_channels=self.skip_channels,
													 kernel_size=(1, self.receptive_field-rf_size_j+1)))
					# self.skip_convs.append(self.transformer_encoder)

				if self.gcn_true:
					self.gconv1.append(MixProp(self.conv_channels, self.residual_channels,
											   self.gcn_depth, self.dropout, self.propalpha))
					self.gconv2.append(MixProp(self.conv_channels, self.residual_channels,
											   self.gcn_depth, self.dropout, self.propalpha))

				if self.input_window > self.receptive_field:
					self.norm.append(LayerNorm((self.residual_channels, self.num_nodes,
												self.input_window - rf_size_j + 1),
											   elementwise_affine=self.layer_norm_affline))
				else:
					self.norm.append(LayerNorm((self.residual_channels, self.num_nodes,
												self.receptive_field - rf_size_j + 1),
											   elementwise_affine=self.layer_norm_affline))

				new_dilation *= self.dilation_exponential

		self.end_conv_1 = nn.Conv2d(in_channels=self.skip_channels,
									out_channels=self.end_channels, kernel_size=(1, 1), bias=True)
		self.end_conv_2 = nn.Conv2d(in_channels=self.end_channels,
									out_channels=self.output_window, kernel_size=(1, 1), bias=True)
		if self.input_window > self.receptive_field:
			self.skip0 = nn.Conv2d(in_channels=self.feature_dim,
								   out_channels=self.skip_channels,
								   kernel_size=(1, self.input_window), bias=True)
			self.skipE = nn.Conv2d(in_channels=self.residual_channels,
								   out_channels=self.skip_channels,
								   kernel_size=(1, self.input_window-self.receptive_field+1), bias=True)
		else:
			self.skip0 = nn.Conv2d(in_channels=self.feature_dim,
								   out_channels=self.skip_channels, kernel_size=(1, self.receptive_field), bias=True)
			self.skipE = nn.Conv2d(in_channels=self.residual_channels,
								   out_channels=self.skip_channels, kernel_size=(1, 1), bias=True)
	def forward(self, source, idx=None):
		# inputs = batch['X']  # (batch_size, input_window, num_nodes, feature_dim)
		sout = []
		tout = []
		inputs = source
		inputs = inputs.transpose(1, 3)  # (batch_size, feature_dim, num_nodes, input_window) #64, 1, 170, 12
		
		assert inputs.size(3) == self.input_window, 'input sequence length not equal to preset sequence length'
		
		if self.input_window < self.receptive_field:
			inputs = nn.functional.pad(inputs, (self.receptive_field-self.input_window, 0, 0, 0))

		# if self.gcn_true:
		# 	if self.buildA_true:
		# 		if idx is None:
		# 			adp = self.gc(self.idx)
		# 		else:
		# 			adp = self.gc(idx)
		# 	else:
		# 		adp = self.predefined_A

		x = self.start_conv(inputs)
		skip = self.skip0(F.dropout(inputs, self.dropout, training=self.training))
		
		for i in range(self.layers):
			residual = x
			filters = self.filter_convs[i](x)
			filters = torch.tanh(filters)
			gate = self.gate_convs[i](x)
			gate = torch.sigmoid(gate)
			x = filters * gate
			x = F.dropout(x, self.dropout, training=self.training)
			tout.append(x)
			s = x
			s = self.skip_convs[i](s)
			skip = s + skip
		# 	if self.gcn_true:
		# 		# print("gcn in x:", x.size())
		# 		x = self.gconv1[i](x, adp)+self.gconv2[i](x, adp.transpose(1, 0)) # in :64, 32, 170, 13, out: 64, 32, 170, 13 , 64, 32, 170, 7], 64, 32, 170, 1]
		# 		# print("gcn out x:", x.size())
		# 		# println()
		# 	else:
		# 		# x = self.residual_convs[i](x)
			# print("start x:", x.size())
			x = self.stu_mlp[i](x)
			# print("middle x:", x.size())
			x = x + residual[:, :, :, -x.size(3):] 
			x = self.norm[i](x, self.idx)
			# print("mlp out x:", x.size())

			 
		# 	if idx is None:
		# 		x = self.norm[i](x, self.idx)
		# 	else:
		# 		x = self.norm[i](x, idx)
		# 	sout.append(x)
		skip = self.skipE(x) + skip
		x = F.relu(skip)
		x = F.relu(self.end_conv_1(x))
		x = self.end_conv_2(x)
		x_ = nn.Linear(1, 32).cuda()(x)  #final out:  64, 12, 170, 1
		# print(x.size())
		# println()
		return x, x_, x


class STAEformerTeacher(nn.Module):
	"""STAEformer包装器，兼容Teacher接口"""
	def __init__(self, args):
		super(STAEformerTeacher, self).__init__()
		self.args = args
		self.num_nodes = args.num_nodes
		self.input_window = args.input_window
		self.output_window = args.output_window
		self.input_dim = args.input_dim
		self.output_dim = args.output_dim
		self.device = args.device
		
		# 创建STAEformer模型
		self.staeformer = STAEformer(
			num_nodes=self.num_nodes,
			in_steps=self.input_window,
			out_steps=self.output_window,
			steps_per_day=288,  # 可以根据数据集调整
			input_dim=self.input_dim,
			output_dim=self.output_dim,
			input_embedding_dim=24,
			tod_embedding_dim=24,
			dow_embedding_dim=24,
			spatial_embedding_dim=0,
			adaptive_embedding_dim=80,
			feed_forward_dim=256,
			num_heads=4,
			num_layers=3,
			dropout=0.1,
			use_mixed_proj=True,
		)
		
		# 添加额外的输出层以匹配Teacher的输出格式
		# STAEformer的model_dim = 24+24+24+0+80 = 152
		self.staeformer_model_dim = 24 + 24 + 24 + 0 + 80  # 152
		self.target_residual_channels = 32  # 目标通道数，与原始Teacher保持一致
		
		# 设计更合理的特征映射：使用可学习的线性层进行降维
		# 第一步：从model_dim降维到residual_channels
		self.temporal_feature_proj = nn.Linear(self.staeformer_model_dim, self.target_residual_channels)
		self.spatiotemporal_feature_proj = nn.Linear(self.staeformer_model_dim, self.target_residual_channels)
		
		# 第二步：模拟原始Teacher的两步线性变换：residual_channels -> input_window -> 32
		self.aux_linear1_step1 = nn.Linear(self.target_residual_channels, self.input_window)
		self.aux_linear1_step2 = nn.Linear(1, 32)
		self.aux_linear2_step1 = nn.Linear(self.target_residual_channels, self.input_window) 
		self.aux_linear2_step2 = nn.Linear(1, 32)
		
	def forward(self, source, idx=None):
		# source: (batch_size, input_window, num_nodes, input_dim)
		# 确保输入格式正确
		if source.dim() == 4:
			# 输入已经是 (batch_size, input_window, num_nodes, input_dim) 格式
			x = source
		else:
			# 如果是其他格式，进行转换
			x = source.transpose(1, 3)  # 从 (batch_size, feature_dim, num_nodes, input_window) 转换
			x = x.transpose(1, 2)  # 转换为 (batch_size, input_window, num_nodes, feature_dim)
		
		# 确保输入维度正确
		assert x.size(1) == self.input_window, f'input sequence length {x.size(1)} not equal to preset sequence length {self.input_window}'
		assert x.size(2) == self.num_nodes, f'input nodes {x.size(2)} not equal to preset nodes {self.num_nodes}'
		
		# 通过STAEformer，获取主输出和中间特征
		# 将save_visualization_data参数传递给内部的STAEformer模型
		output, temporal_feature, spatiotemporal_feature = self.staeformer(x, True)
		# output: (batch_size, output_window, num_nodes, output_dim)
		# temporal_feature: (batch_size, input_window, num_nodes, model_dim) - 时间注意力特征
		# spatiotemporal_feature: (batch_size, input_window, num_nodes, model_dim) - 时空注意力特征
		
		# 将中间特征转换为与原始Teacher相同的格式
		# 原始格式：(batch_size, residual_channels, num_nodes, time_steps)
		
		# 处理时间特征 (模拟tout[-1])



		if temporal_feature is not None:
			# 使用可学习的投影层进行降维：(batch_size, input_window, num_nodes, model_dim) -> (batch_size, input_window, num_nodes, target_residual_channels)
			temporal_projected = self.temporal_feature_proj(temporal_feature)
			# 转换维度：(batch_size, input_window, num_nodes, target_residual_channels) -> (batch_size, target_residual_channels, num_nodes, input_window)
			temporal_feat = temporal_projected.permute(0, 3, 2, 1)
		else:
			# 如果没有时间特征，使用输出特征并进行适当处理
			# 创建一个与temporal_feature相同形状的占位符
			placeholder = torch.zeros(output.shape[0], self.input_window, output.shape[2], self.staeformer_model_dim, 
									device=output.device, dtype=output.dtype)
			temporal_projected = self.temporal_feature_proj(placeholder)
			temporal_feat = temporal_projected.permute(0, 3, 2, 1)
		
		# 处理时空特征 (模拟sout[-1])
		if spatiotemporal_feature is not None:
			# 使用可学习的投影层进行降维：(batch_size, input_window, num_nodes, model_dim) -> (batch_size, input_window, num_nodes, target_residual_channels)
			spatiotemporal_projected = self.spatiotemporal_feature_proj(spatiotemporal_feature)
			# 转换维度：(batch_size, input_window, num_nodes, target_residual_channels) -> (batch_size, target_residual_channels, num_nodes, input_window)
			spatiotemporal_feat = spatiotemporal_projected.permute(0, 3, 2, 1)
		else:
			# 如果没有时空特征，使用输出特征并进行适当处理
			# 创建一个与spatiotemporal_feature相同形状的占位符
			placeholder = torch.zeros(output.shape[0], self.input_window, output.shape[2], self.staeformer_model_dim, 
									device=output.device, dtype=output.dtype)
			spatiotemporal_projected = self.spatiotemporal_feature_proj(placeholder)
			spatiotemporal_feat = spatiotemporal_projected.permute(0, 3, 2, 1)
		#
		# # 模拟原始Teacher的辅助输出计算：
		# # ttout = nn.Linear(1, 32)(nn.Linear(residual_channels, input_window)(tout[-1].transpose(1,3)).transpose(1,3))
		# aux1_temp = self.aux_linear1_step1(temporal_feat.transpose(1, 3)).transpose(1, 3)  # 第一步变换
		# aux1 = self.aux_linear1_step2(aux1_temp.transpose(1, 3)).transpose(1, 3)  # 第二步变换
		ttout = temporal_feat.permute(0, 3, 2, 1)

		# aux2_temp = self.aux_linear2_step1(spatiotemporal_feat.transpose(1, 3)).transpose(1, 3)  # 第一步变换
		# aux2 = self.aux_linear2_step2(aux2_temp.transpose(1, 3)).transpose(1, 3)  # 第二步变换
		stout = spatiotemporal_feat.permute(0, 3, 2, 1)

		return output, ttout, stout