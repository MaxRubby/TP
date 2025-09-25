import torch
import math
import os
import time
import copy
import numpy as np
from lib.logger import get_logger
from lib.metrics import All_Metrics
import torch.nn.functional as F
from model.Teacher import Teacher as Network1
from model.Teacher import STAEformerTeacher

import torch.nn as nn 




class Trainer(object):
	def __init__(self, model, loss, optimizer, train_loader, val_loader, test_loader,
				 scaler, args, lr_scheduler=None):
		super(Trainer, self).__init__()
		self.model = model
		self.args = args
		self.loss = loss
		self.optimizer = optimizer
		self.train_loader = train_loader
		self.val_loader = val_loader
		self.test_loader = test_loader
		self.scaler = scaler
		self.args = args
		self.lr_scheduler = lr_scheduler
		self.train_per_epoch = len(train_loader)
		if val_loader != None:
			self.val_per_epoch = len(val_loader)
		self.best_path = os.path.join(self.args.log_dir, 'best_model.pth')
		self.best_pathS = os.path.join(self.args.log_dir, 'best_modelstudent.pth')
		self.best_path_ckpt = os.path.join(self.args.log_dir, 'ckpt_best_model')
		self.best_pathS_ckpt = os.path.join(self.args.log_dir, 'ckpt_best_modelstudent')
		self.loss_figure_path = os.path.join(self.args.log_dir, 'loss.png')
		#log
		if os.path.isdir(args.log_dir) == False and not args.debug:
			os.makedirs(args.log_dir, exist_ok=True)
		self.logger = get_logger(args.log_dir, name=args.model, debug=False)
		self.logger.info('Experiment log path in: {}'.format(args.log_dir))
		#if not args.debug:
		#self.logger.info("Argument: %r", args)
		# for arg, value in sorted(vars(args).items()):
		#     self.logger.info("Argument %s: %r", arg, value)
		if self.args.t is False: #如果教师模型不需要训练则直接加载
			self.tmodel = self.loadTeacher(self.args)
	# load teacher model
	def loadTeacher(self,args):
		# Build checkpoint path relative to this file to avoid CWD issues
		current_dir = os.path.dirname(os.path.realpath(__file__))
		ckpt_path = os.path.join(args.log_dir, 'best_model.pth')
		# ckpt_path = os.path.join(current_dir, 'SAVE', args.dataset, 'best_model.pth')
		self.logger.info("加载教师模型: {}".format(ckpt_path))
		# 选择使用STAEformerTeacher还是原始Teacher
		if args.t2:
			self.logger.info("教师模型为STAEFormer")
			tmodel = STAEformerTeacher(args).to(args.device)
		else:
			self.logger.info("教师模型为Network1")
			tmodel = Network1(args).to(args.device)
			
		# Map tensors to the active device (or CPU) to avoid CUDA index mismatch
		map_loc = torch.device(args.device) if torch.cuda.is_available() else torch.device('cpu')
		state_dict = torch.load(ckpt_path, map_location=map_loc)
		tmodel.load_state_dict(state_dict, strict=False)
		return tmodel
	def val_epoch(self, epoch, val_dataloader):
		self.model.eval()
		total_val_loss = 0

		with torch.no_grad():
			for batch_idx, (data, target) in enumerate(val_dataloader):
				data = data[..., :self.args.input_dim]
				label = target[..., :self.args.output_dim]
				output, _, _ = self.model(data)
				if self.args.real_value:
					label = self.scaler.inverse_transform(label)
				loss = self.loss(output.cuda(), label)
				#a whole batch of Metr_LA is filtered
				if not torch.isnan(loss):
					total_val_loss += loss.item()
		val_loss = total_val_loss / len(val_dataloader)
		self.logger.info('**********Val Epoch {}: average Loss: {:.6f}'.format(epoch, val_loss))
		return val_loss

	def train_epoch(self, epoch):
		self.model.train()
		total_loss = 0
		for batch_idx, (data, target) in enumerate(self.train_loader):
			data = data[..., :self.args.input_dim]
			label = target[..., :self.args.output_dim]  # (..., 1)
			self.optimizer.zero_grad()
			out, _, _ = self.model(data)

			if self.args.real_value:
				label = self.scaler.inverse_transform(label)

			loss = self.loss(out.cuda(), label)
			loss.backward()

			# add max grad clipping
			if self.args.grad_norm:
				torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
			self.optimizer.step()
			total_loss += loss.item()

			#log information
			if batch_idx % self.args.log_step == 0:
				self.logger.info('Train Epoch {}: {}/{} Loss: {:.6f}'.format(
					epoch, batch_idx, self.train_per_epoch, loss.item()))
		train_epoch_loss = total_loss/self.train_per_epoch
		self.logger.info('**********Train Epoch {}: averaged Loss: {:.6f}'.format(epoch, train_epoch_loss))

		#learning rate decay
		if self.args.lr_decay:
			self.lr_scheduler.step()
		return train_epoch_loss
	
	def loss_cls(self, x1, x2):
		temperature = 0.05
		x1 = F.normalize(x1, p=2, dim=-1)
		x2 = F.normalize(x2, p=2, dim=-1)
		weight = F.cosine_similarity(x1, x2, dim=-1)
		batch_size = x1.size()[0]
		# neg score
		out = torch.cat([x1, x2], dim=0)
		neg = torch.exp(torch.matmul(out, out.transpose(2,3).contiguous()) / temperature)
		
		pos = torch.exp(torch.sum(x1 * x2, dim=-1)*weight / temperature)
		# pos = torch.exp(torch.sum(x1 * x2, dim=-1) / temperature)
		pos = torch.cat([pos, pos], dim=0).sum(dim=1)

		Ng = neg.sum(dim=-1).sum(dim=1)
		
		loss = (- torch.log(pos / (pos + Ng) )).mean()
		
		return loss

	def loss_clt(self, x1, x2):
		temperature = 0.05
		x1 = F.normalize(x1, p=2, dim=-1)
		x2 = F.normalize(x2, p=2, dim=-1)
		weight = F.cosine_similarity(x1, x2, dim=-1)
		
		batch_size = x1.size()[0]
		# neg score
		out = torch.cat([x1, x2], dim=0)
		neg = torch.exp(torch.matmul(out, out.transpose(2,3).contiguous()) / temperature)
		
		pos = torch.exp(torch.sum(x1 * x2, dim=-1)*weight / temperature)
		# pos = torch.exp(torch.sum(x1 * x2, dim=-1) / temperature)
		pos = torch.cat([pos, pos], dim=0).sum(dim=-1)

		Ng = neg.sum(dim=-1).sum(dim=-1)
		
		loss = (- torch.log(pos / (pos + Ng) )).mean()
		
		return loss
	def train_epochs(self, epoch):
		self.tmodel.eval()  #--------
		self.model.train()
		total_loss = 0
		for batch_idx, (data, target) in enumerate(self.train_loader):
			data = data[..., :self.args.input_dim]
			label = target[..., :self.args.output_dim]  # (..., 1)
			self.optimizer.zero_grad()
			out, out_,_ = self.model(data)
			gout, tout,sout= self.tmodel(data)
			# print("student out:", out.size())
			# print("tout:", tout.size())
			# print("sout:", sout.size())
			# println()
			if self.args.real_value:
				label = self.scaler.inverse_transform(label)


			loss1 = self.loss(out.cuda(), label)

			tcl = self.loss_clt(out_, tout)
			
			scl = self.loss_cls(out_, sout)
			
			#kl divergence
			kl_loss = nn.KLDivLoss(reduction="batchmean",log_target=True).cuda()
			gout = F.log_softmax(gout).cuda()
			# sgcn = F.log_softmax(sout).cuda()
			mlp_emb_ = F.log_softmax(out).cuda()
			# weight = mlp_emb_*gout
			tkloss = kl_loss(mlp_emb_.cuda().float(),gout.cuda().float())
			# print("scl:", scl, tcl, tkloss)
			# skloss = kl_loss(mlp_emb_.cuda().float(),sgcn.cuda().float())
			
			# print("loss:", loss1, tkloss, skloss, tcl, scl)
			# loss = loss1 + 10*tkloss + 0.5*tcl + 0.5*scl
			# loss = loss1 + 10*tkloss + 1*scl + 1*tcl
			loss = loss1 + 10*tkloss + 1*scl
			loss.backward()

			# add max grad clipping
			if self.args.grad_norm:
				torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
			self.optimizer.step()
			total_loss += loss.item()

			#log information
			if batch_idx % self.args.log_step == 0:
				self.logger.info('Train Epoch {}: {}/{} Loss: {:.6f}'.format(
					epoch, batch_idx, self.train_per_epoch, loss.item()))
		train_epoch_loss = total_loss/self.train_per_epoch
		self.logger.info('**********Train Epoch {}: averaged Loss: {:.6f}'.format(epoch, train_epoch_loss))

		#learning rate decay
		if self.args.lr_decay:
			self.lr_scheduler.step()
		return train_epoch_loss

	def train(self):
		best_model = None
		best_loss = float('inf')
		not_improved_count = 0
		train_loss_list = []
		val_loss_list = []
		start_time = time.time()
		for epoch in range(1, self.args.epochs + 1):
			epoch_time = time.time()
			train_epoch_loss = self.train_epoch(epoch)
			print(time.time()-epoch_time)
			#exit()
			if self.val_loader == None:
				val_dataloader = self.test_loader
			else:
				val_dataloader = self.val_loader
			val_epoch_loss = self.val_epoch(epoch, val_dataloader)

			#print('LR:', self.optimizer.param_groups[0]['lr'])
			train_loss_list.append(train_epoch_loss)
			val_loss_list.append(val_epoch_loss)
			if train_epoch_loss > 1e6:
				self.logger.warning('Gradient explosion detected. Ending...')
				break
			#if self.val_loader == None:
			#val_epoch_loss = train_epoch_loss
			if val_epoch_loss < best_loss:
				best_loss = val_epoch_loss
				not_improved_count = 0
				best_state = True
			else:
				not_improved_count += 1
				best_state = False
			# early stop
			if self.args.early_stop:
				if not_improved_count == self.args.early_stop_patience:
					self.logger.info("Validation performance didn\'t improve for {} epochs. "
									"Training stops.".format(self.args.early_stop_patience))
					break
			# save the best state
			if best_state == True:
				self.logger.info('*********************************Current best model saved!')
				best_model = copy.deepcopy(self.model.state_dict())
				# save every best epoch
				torch.save(best_model, self.best_path)
				self.save_checkpoint(self.best_path_ckpt + "_" + str(epoch) + ".pth" )
				not_improved_count = 0
				self.logger.info("Saving current best model to " + self.best_path)
			



		training_time = time.time() - start_time
		self.logger.info("Total training time: {:.4f}min, best loss: {:.6f}".format((training_time / 60), best_loss))

		#save the best model to file
		# if not self.args.debug:
		if self.args.debug:
			torch.save(best_model, self.best_path)
			self.logger.info("Saving current best model to " + self.best_path)

		#test
		self.model.load_state_dict(best_model)
		#self.val_epoch(self.args.epochs, self.test_loader)
		self.test(self.model, self.args, self.test_loader, self.scaler, self.logger)


	def save_checkpoint(self, path):
		state = {
			'state_dict': self.model.state_dict(),
			'optimizer': self.optimizer.state_dict(),
			'config': self.args
		}
		torch.save(state, path)
		self.logger.info("Saving current best model to " + self.best_path)


	def trainS(self):
		best_model = None
		best_loss = float('inf')
		not_improved_count = 0
		train_loss_list = []
		val_loss_list = []
		start_time = time.time()
		for epoch in range(1, self.args.epochs + 1):
			epoch_time = time.time()
			train_epoch_loss = self.train_epochs(epoch)
			print(time.time()-epoch_time)
			#exit()
			if self.val_loader == None:
				val_dataloader = self.test_loader
			else:
				val_dataloader = self.val_loader
			val_epoch_loss = self.val_epoch(epoch, val_dataloader)

			#print('LR:', self.optimizer.param_groups[0]['lr'])
			train_loss_list.append(train_epoch_loss)
			val_loss_list.append(val_epoch_loss)
			if train_epoch_loss > 1e6:
				self.logger.warning('Gradient explosion detected. Ending...')
				break
			#if self.val_loader == None:
			#val_epoch_loss = train_epoch_loss
			if val_epoch_loss < best_loss:
				best_loss = val_epoch_loss
				not_improved_count = 0
				best_state = True
			else:
				not_improved_count += 1
				best_state = False
			# early stop
			if self.args.early_stop:
				if not_improved_count == self.args.early_stop_patience:
					self.logger.info("Validation performance didn\'t improve for {} epochs. "
									"Training stops.".format(self.args.early_stop_patience))
					break
			# save the best state
			if best_state == True:
				self.logger.info('*********************************Current best model saved to ' + self.best_pathS)
				best_model = copy.deepcopy(self.model.state_dict())
				self.save_checkpoint(self.best_pathS_ckpt + "_" + str(epoch) + ".pth")
				torch.save(best_model, self.best_pathS)
		training_time = time.time() - start_time
		self.logger.info("Total training time: {:.4f}min, best loss: {:.6f}".format((training_time / 60), best_loss))

		#save the best model to file
		# if not self.args.debug:
		if self.args.debug:
			torch.save(best_model, self.best_pathS)
			self.logger.info("Saving current best model to " + self.best_pathS)




		#test
		self.model.load_state_dict(best_model)
		#self.val_epoch(self.args.epochs, self.test_loader)
		self.test(self.model, self.args, self.test_loader, self.scaler, self.logger)

		
	def save_checkpoint(self, path):
		state = {
			'state_dict': self.model.state_dict(),
			'optimizer': self.optimizer.state_dict(),
			'config': self.args
		}
		torch.save(state, path)
		self.logger.info("Saving current best model to " + self.best_path)

	@staticmethod
	def test(model, args, data_loader, scaler, logger, path=None):
		if path != None:
			check_point = torch.load(path)
			state_dict = check_point['state_dict']
			args = check_point['config']
			model.load_state_dict(state_dict)
			model.to(args.device)
		model.eval()
		y_pred = []
		y_true = []
		y_emb = []
		start_time = time.time()
		with torch.no_grad():
			for batch_idx, (data, target) in enumerate(data_loader):
				data = data[..., :args.input_dim]
				label = target[..., :args.output_dim]
				output, vectors, _ = model(data)
				y_true.append(label)
				y_pred.append(output)
				y_emb.append(vectors)
		y_true = scaler.inverse_transform(torch.cat(y_true, dim=0))
		if args.real_value:
			y_pred = torch.cat(y_pred, dim=0)
		else:
			y_pred = scaler.inverse_transform(torch.cat(y_pred, dim=0))
		y_emb = torch.cat(y_emb, dim=0)
		np.save('./{}_true.npy'.format(args.dataset), y_true.cpu().numpy())
		np.save('./{}_pred.npy'.format(args.dataset), y_pred.cpu().numpy())
		np.save('./{}_emb.npy'.format(args.dataset), y_emb.cpu().numpy())
		for t in range(y_true.shape[1]):
			mae, rmse, mape, _, _ = All_Metrics(y_pred[:, t, ...], y_true[:, t, ...],
												args.mae_thresh, args.mape_thresh)
			logger.info("Horizon {:02d}, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(
				t + 1, mae, rmse, mape*100))
		mae, rmse, mape, _, _ = All_Metrics(y_pred, y_true, args.mae_thresh, args.mape_thresh)
		time_p = time.time()- start_time
		logger.info("Average Horizon, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%, Test Time:{:.7f}".format(
					mae, rmse, mape*100, time_p))

	def train_teacher(self):
		"""
		专门用于训练教师模型的方法
		"""
		self.logger.info("Starting Teacher Model Training...")
		best_model = None
		best_loss = float('inf')
		not_improved_count = 0
		train_loss_list = []
		val_loss_list = []
		start_time = time.time()
		
		for epoch in range(1, self.args.epochs + 1):
			epoch_time = time.time()
			train_epoch_loss = self.train_teacher_epoch(epoch)
			print(f"Teacher training epoch {epoch} time: {time.time()-epoch_time:.2f}s")
			
			if self.val_loader == None:
				val_dataloader = self.test_loader
			else:
				val_dataloader = self.val_loader
			val_epoch_loss = self.val_teacher_epoch(epoch, val_dataloader)

			train_loss_list.append(train_epoch_loss)
			val_loss_list.append(val_epoch_loss)
			
			if train_epoch_loss > 1e6:
				self.logger.warning('Gradient explosion detected. Ending teacher training...')
				break
				
			if val_epoch_loss < best_loss:
				best_loss = val_epoch_loss
				not_improved_count = 0
				best_state = True
			else:
				not_improved_count += 1
				best_state = False
				
			# early stop
			if self.args.early_stop:
				if not_improved_count == self.args.early_stop_patience:
					self.logger.info("Teacher validation performance didn't improve for {} epochs. "
									"Training stops.".format(self.args.early_stop_patience))
					break
					
			# save the best state
			if best_state == True:
				self.logger.info('*********************************Current best teacher model saved!')
				best_model = copy.deepcopy(self.model.state_dict())
				# 保存教师模型到best_model.pth，供后续知识蒸馏使用
				torch.save(best_model, self.best_path)
				self.save_teacher_checkpoint(self.best_path_ckpt + "_teacher_" + str(epoch) + ".pth")
				not_improved_count = 0
				self.logger.info("Saving current best teacher model to " + self.best_path)

		training_time = time.time() - start_time
		self.logger.info("Teacher training time: {:.4f}min, best loss: {:.6f}".format((training_time / 60), best_loss))

		# 保存最终的教师模型
		if self.args.debug:
			torch.save(best_model, self.best_path)
			self.logger.info("Saving final teacher model to " + self.best_path)

		# 测试教师模型
		self.model.load_state_dict(best_model)
		self.logger.info("Testing teacher model...")
		self.test(self.model, self.args, self.test_loader, self.scaler, self.logger)

	def train_teacher_epoch(self, epoch):
		"""
		教师模型的单个epoch训练
		"""
		self.model.train()
		total_loss = 0
		for batch_idx, (data, target) in enumerate(self.train_loader):

			data = data[..., :3] #staeformer
			label = target[..., :self.args.output_dim]
			self.optimizer.zero_grad()
			out, _, _ = self.model(data)

			if self.args.real_value:
				label = self.scaler.inverse_transform(label)

			loss = self.loss(out.cuda(), label)
			loss.backward()

			# add max grad clipping
			if self.args.grad_norm:
				torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
			self.optimizer.step()
			total_loss += loss.item()

			#log information
			if batch_idx % self.args.log_step == 0:
				self.logger.info('Teacher Train Epoch {}: {}/{} Loss: {:.6f}'.format(
					epoch, batch_idx, self.train_per_epoch, loss.item()))
		train_epoch_loss = total_loss/self.train_per_epoch
		self.logger.info('**********Teacher Train Epoch {}: averaged Loss: {:.6f}'.format(epoch, train_epoch_loss))

		#learning rate decay
		if self.args.lr_decay:
			self.lr_scheduler.step()
		return train_epoch_loss

	def val_teacher_epoch(self, epoch, val_dataloader):
		"""
		教师模型的验证
		"""
		self.model.eval()
		total_val_loss = 0

		with torch.no_grad():
			for batch_idx, (data, target) in enumerate(val_dataloader):
				data = data[..., :self.args.input_dim]
				label = target[..., :self.args.output_dim]
				output, _, _ = self.model(data)
				if self.args.real_value:
					label = self.scaler.inverse_transform(label)
				loss = self.loss(output.cuda(), label)
				if not torch.isnan(loss):
					total_val_loss += loss.item()
		val_loss = total_val_loss / len(val_dataloader)
		self.logger.info('**********Teacher Val Epoch {}: average Loss: {:.6f}'.format(epoch, val_loss))
		return val_loss

	def save_teacher_checkpoint(self, path):
		"""
		保存教师模型检查点
		"""
		state = {
			'state_dict': self.model.state_dict(),
			'optimizer': self.optimizer.state_dict(),
			'config': self.args
		}
		torch.save(state, path)
		self.logger.info("Saving teacher checkpoint to " + path)

	@staticmethod
	def _compute_sampling_threshold(global_step, k):
		"""
		Computes the sampling probability for scheduled sampling using inverse sigmoid.
		:param global_step:
		:param k:
		:return:
		"""
		return k / (k + math.exp(global_step / k))