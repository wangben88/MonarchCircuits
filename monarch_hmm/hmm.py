import math
import functools
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
import torch.distributed as dist


def matmul_a_logb(A, B):
    bd = len(B.shape) - 2
    B_max = torch.amax(B, dim=bd, keepdim=True)
    B = B - B_max
    B.exp_()
    C = torch.matmul(A, B)
    C.log_()
    C.add_(B_max)

    return C

def matmul_a_logb_inplace(A, B):
    bd = len(B.shape) - 2
    B_max = torch.amax(B, dim=bd, keepdim=True)
    B.sub_(B_max)
    B.exp_()
    torch.matmul(A, B, out=B)
    B.log_()
    B.add_(B_max)

    return B


def ib_ib_bj_to_ij(pf, pp, cp):
    ll = torch.amax(cp, dim=-1)
    pp = torch.exp(pp - ll[None, :])
    cp = torch.exp(cp - ll[:, None])

    ratio = pf / pp
    ratio[pp == 0.0] = 0.0
    ef = torch.matmul(ratio, cp)

    return ef


def hib_hib_hbj_to_hij(pf, pp, cp):
    ll = torch.amax(cp, dim=-1)
    pp = torch.exp(pp - ll[:, None, :])
    cp = torch.exp(cp - ll[:, :, None])

    ratio = pf / pp
    ratio.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
    ef = torch.matmul(ratio, cp)

    return ef


def hib_hib_hbj_to_hij_add_(pf, pp, cp, ef):
    ll = torch.amax(cp, dim=-1)
    pp = torch.exp(pp - ll[:, None, :])
    cp = torch.exp(cp - ll[:, :, None])

    ratio = pf / pp
    ratio.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
    torch.baddbmm(ef, ratio, cp, out=ef)


def compute_ef(weights, cp, pp, pf):
    weights_exp = torch.exp(weights).unsqueeze(-1)
    ef = weights_exp * torch.exp(cp - pp) * pf
    ef.nan_to_num_(nan=0.0)
    return ef


class ProdLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        return x + y


    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, pf):
        return pf, pf


class InputLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weights, input_ids):
        hidden_states, _ = weights.shape
        outputs = weights[
            torch.arange(0, hidden_states, device=weights.device)[:, None],
            input_ids[None, :]].contiguous() # hidden_states * batch_size
        ctx.save_for_backward(weights, input_ids)
        return outputs


    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, pf):
        weights, input_ids = ctx.saved_tensors
        hidden_states, _ = weights.shape
        input_ids = input_ids[None, :].expand(hidden_states, -1)
        weights.grad.scatter_add_(1, input_ids, pf.view(hidden_states, -1))
        return None, None


class LinearLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weights, inputs):
        outputs = matmul_a_logb(weights, inputs)
        ctx.save_for_backward(weights, inputs, outputs)
        return outputs


    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, pf):
        weights, cp, pp = ctx.saved_tensors

        ef = ib_ib_bj_to_ij(
            pf,
            pp,
            torch.permute(cp, (1, 0)).contiguous())
        ef *= weights

        pf = torch.permute(pf, (1, 0))
        pp = torch.permute(pp, (1, 0)) # batch_size * hidden_states
        cp = torch.permute(cp, (1, 0)) # batch_size * hidden_states

        pp_max = torch.amax(pp, dim=1, keepdim=True) # batch_size * 1
        pp_ = torch.exp(pp - pp_max)
        ratio = pf / pp_
        ratio[pp_ == 0.0] = 0.0
        cf = torch.matmul(ratio, weights) * torch.exp(cp - pp_max)

        cf = torch.permute(cf, (1, 0)).contiguous()

        return ef, cf


class MonarchLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weights_exp, inputs):
        outputs = matmul_a_logb(weights_exp, inputs)
        ctx.save_for_backward(weights_exp, inputs, outputs)
        return outputs


    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, pf):
        weights_exp, cp, pp = ctx.saved_tensors

        ef = hib_hib_hbj_to_hij(pf, pp,
            torch.transpose(cp, 1, 2).contiguous()) * weights_exp

        pp_max = torch.amax(pp, dim=1, keepdim=True) # block_num * 1 * batch_size
        ratio = pf / torch.exp(pp - pp_max)
        ratio.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        cf = torch.matmul(
            torch.transpose(weights_exp, 1, 2).contiguous(),
            ratio) * torch.exp(cp - pp_max)

        return ef, cf


class MonarchMatrix(torch.nn.Module):
    def __init__(self, hidden_states, block_size, requires_grad=True):
        super().__init__()

        if type(block_size) is int:
            num_layers = int(math.log(hidden_states) / math.log(block_size))
            assert block_size ** num_layers == hidden_states

            weights_exp = [torch.softmax(torch.randn(hidden_states // block_size, block_size, block_size), dim=-1)
                for _ in range(0, num_layers)]
        else:
            assert functools.reduce(lambda x,y: x * y, block_size) == hidden_states
            weights_exp = [torch.softmax(torch.randn(hidden_states // bs, bs, bs), dim=-1)
                for bs in block_size]

        self.weights_exp = nn.ParameterList([nn.Parameter(w, requires_grad=requires_grad)
            for w in weights_exp])
        self.block_size = block_size


    def forward(self, x):
        hidden_states, batch_size = x.shape

        for w in reversed(self.weights_exp):
            x = x.view(-1, self.block_size, batch_size)
            x = MonarchLayer.apply(w, x) # _ * block_size * batch_size
            x = torch.transpose(x, 0, 1).contiguous() # righr-shift
        x = x.view(hidden_states, batch_size)

        return x


class InputMonarchFused(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_ids, beta, x, x_half, *alpha_exp):
        saved_tensors = []

        ############################# input layer #############################
        input_prob = beta[
            torch.arange(0, beta.shape[0], device=beta.device)[:, None],
            input_ids[None, :]].contiguous() # h * b
        saved_tensors.extend([input_ids, beta])

        ############################# monarch transform #############################
        saved_tensors.append(x_half)
        x = input_prob.view(x.shape) + x
        for w in reversed(alpha_exp):
            x = x.view(-1, w.shape[-1], x.shape[-1])
            x = matmul_a_logb(w, x)
            x = torch.transpose(x, 0, 1).contiguous()
            x_half = x.to(dtype=torch.float16)
            saved_tensors.extend([w, x_half])

        saved_tensors = tuple(saved_tensors)
        ctx.save_for_backward(*saved_tensors)

        return x, x_half


    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, pf, _):
        saved_tensors = ctx.saved_tensors
        input_ids, beta, saved_tensors = saved_tensors[0], saved_tensors[1], saved_tensors[2:]  # x w x w x

        hidden_states = beta.shape[0]
        batch_size = pf.shape[-1]

        ############################# recompute input layer #############################
        input_prob = beta[
            torch.arange(0, beta.shape[0], device=beta.device)[:, None],
            input_ids[None, :]].contiguous() # h * b

        efs = []
        for i in range(len(saved_tensors)-1, 1, -2):
            pp, w, cp = saved_tensors[i], saved_tensors[i-1], saved_tensors[i-2]
            pp, cp = pp.to(w.dtype), cp.to(w.dtype)

            cp_shape = cp.shape # bug fix here
            cp = cp.view(-1, w.shape[-1], cp.shape[-1]) # bug fix here
            
            pf = torch.transpose(pf, 0, 1)
            pp = torch.transpose(pp, 0, 1)
            if i - 2 == 0:
                cp = input_prob.view(cp.shape) + cp

            # compute ef
            hib_hib_hbj_to_hij_add_(pf, pp,
                torch.transpose(cp, 1, 2), w.grad)
            efs.append(None)

            # compute cf
            pp_max = torch.amax(pp, dim=1, keepdim=True) # block_num * 1 * batch_size
            ratio = pf / torch.exp(pp - pp_max)
            ratio.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
            cf = torch.matmul(
                torch.transpose(w, 1, 2),
                ratio) * torch.exp(cp - pp_max)
            cf = cf.view(cp_shape)
            pf = cf

        efs = tuple(efs)

        ############################# backward for input layer #############################
        input_ids = input_ids[None, :].expand(hidden_states, -1)
        beta.grad.scatter_add_(1, input_ids, cf.view(hidden_states, -1))

        return None, None, cf, None, *efs


class MonarchHMM(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(self, hidden_states, block_size, vocab_size, eos_token_id):
        super().__init__()

        alpha_exp = MonarchMatrix(hidden_states, block_size)
        beta = torch.log_softmax(torch.randn(hidden_states, vocab_size), dim=-1)
        gamma_exp = torch.softmax(torch.randn(1, hidden_states), dim=-1)

        ################# INITIALIZATION FOR TEXT8 #################
        if type(block_size) is int:
            beta = beta.view(-1, block_size, vocab_size)
        else:
            beta = beta.view(-1, block_size[-1], vocab_size)
        beta[:, :-1, eos_token_id] = -1e10
        beta = beta.view(-1, vocab_size)
        beta = torch.log_softmax(beta, dim=-1)

        w = alpha_exp.weights_exp[-1]
        w.data[:, -1, -1] = 1e-10
        w.data.div_(torch.sum(w.data, dim=-1, keepdim=True))
        ################# INITIALIZATION FOR TEXT8 #################

        self.alpha_exp = alpha_exp
        self.beta = nn.Parameter(beta, requires_grad=True)
        self.gamma_exp = nn.Parameter(gamma_exp, requires_grad=True)

        self.hidden_states = hidden_states
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id


    def zero_grad(self):
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.grad is None:
                param.grad = torch.zeros_like(param.data, device=param.device)
                param.node_flow = torch.zeros(param.data.shape[:-1] + (1,), device=param.device)
            else:
                param.grad.fill_(0.0)


    def step(self, lr=1.0, neginf=-1e10, eps=1e-10):
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            with torch.no_grad():
                torch.distributed.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                if name.find('alpha') != -1:
                    param.grad.mul_(param)

                param.grad.mul_(lr)

                if name.find('exp') == -1:
                    param.data.exp_()
                param.data.mul_((1.0 - lr) * param.node_flow)

                param.data.add_(param.grad)
                param.node_flow = torch.sum(param.data, dim=-1, keepdim=True)

                param.data.div_(param.node_flow)
                param.data.clamp_(min=eps)
                if name.find('exp') == -1:
                    param.data.log_()
                    param.data.nan_to_num_(nan=neginf, neginf=neginf)
                    param.data = torch.log_softmax(param.data, dim=-1)


    def topdown(self, seq_len, scale=1.0):
        loss = self.forward(seq_len)
        loss = loss.sum()
        loss.backward()
        for name, param in self.named_parameters():
            if param.requires_grad:
                param.grad.mul_(scale)


    def forward(self, input_ids):
        weights_exp = tuple(w for w in self.alpha_exp.weights_exp)
        hidden_states, vocab_size = self.hidden_states, self.vocab_size

        if type(input_ids) is int: # for topdown only
            batch_size, seq_len = 1, input_ids
        else: # regular forward
            batch_size, seq_len = input_ids.shape

        x = torch.zeros((hidden_states, batch_size), device=self.beta.device)
        x = x.view(-1, weights_exp[-1].shape[1], batch_size)
        x_half = x.to(dtype=torch.float16)
        for t in range(seq_len-1, -1, -1):
            if t != 0:
                x, x_half = InputMonarchFused.apply(input_ids[:, t], self.beta, x, x_half, *weights_exp)
            else:
                input_prob = InputLayer.apply(self.beta, input_ids[:, t]).view(x.shape)
                x = ProdLayer.apply(input_prob, x)

        x = x.view(hidden_states, -1)
        x = LinearLayer.apply(self.gamma_exp, x)

        return x


class HMM(nn.Module, PyTorchModelHubMixin):
    def __init__(self, hidden_states: int, vocab_size: int, eos_token_id: int):
        super().__init__()

        alpha_exp = torch.softmax(torch.randn(hidden_states, hidden_states), dim=1)
        beta = torch.log_softmax(torch.randn(hidden_states, vocab_size), dim=1)
        gamma_exp = torch.softmax(torch.randn(1, hidden_states), dim=0)

        ################# INITIALIZATION FOR TEXT8 #################
        beta[-1, eos_token_id] = 1e10
        beta[:-1, eos_token_id] = -1e10
        beta = torch.log_softmax(beta, dim=-1)
        alpha_exp[-1, -1] = 1e-10
        ################# INITIALIZATION FOR TEXT8 #################

        self.alpha_exp = nn.Parameter(alpha_exp, requires_grad=True)
        self.beta = nn.Parameter(beta, requires_grad=True)
        self.gamma_exp = nn.Parameter(gamma_exp, requires_grad=True)

        self.hidden_states = hidden_states
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id


    def zero_grad(self):
        for name, param in self.named_parameters():
            if param.grad is None:
                param.grad = torch.zeros_like(param.data, device=param.device)
                param.node_flow = torch.zeros(param.data.shape[:-1] + (1,), device=param.device)
            else:
                param.grad.fill_(0.0)


    def step(self, lr=1.0, neginf=-1e10, eps=1e-10):
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            with torch.no_grad():
                torch.distributed.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad.mul_(lr)

                if name.find('exp') == -1:
                    param.data.exp_()
                param.data.mul_((1.0 - lr) * param.node_flow)

                param.data.add_(param.grad)
                param.node_flow = torch.sum(param.data, dim=-1, keepdim=True)

                param.data.div_(param.node_flow)
                param.data.clamp_(min=eps)
                if name.find('exp') == -1:
                    param.data.log_()
                    param.data.nan_to_num_(nan=neginf, neginf=neginf)


    # bottom-up circuit pass
    def forward(self, input_ids):
        hidden_states = self.hidden_states
        batch_size, seq_len = input_ids.shape

        y = torch.zeros((hidden_states, batch_size), device=self.alpha_exp.device)
        for t in range(seq_len-1, -1, -1):
            if t != seq_len - 1:
                y = LinearLayer.apply(self.alpha_exp, y)
            input_prob = InputLayer.apply(self.beta, input_ids[:, t])
            y = ProdLayer.apply(input_prob, y)
        y = LinearLayer.apply(self.gamma_exp, y)

        return y