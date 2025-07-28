import argparse
import functools
from hmm import *


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    
    arg_parser.add_argument('--model_paths', nargs="+", default=[], required=True)
    arg_parser.add_argument('--output_path', type=str, required=True)

    args = arg_parser.parse_args()

    block_size = tuple()
    hmm_models = []
    for model_path in args.model_paths:
        hmm_models.append(HMM.from_pretrained(model_path))
        block_size += (hmm_models[-1].hidden_states,)

    hidden_states = functools.reduce(lambda x,y: x * y, block_size)
    monarch_model = MonarchHMM(
        hidden_states, block_size, hmm_models[0].vocab_size, hmm_models[0].eos_token_id)

    betas = []
    for i, hmm_model in enumerate(hmm_models):
        beta_shape = (1,) * i + (hmm_model.beta.shape[0],) + (1,) * (len(block_size) - 1 - i) + (hmm_model.beta.shape[-1],)
        betas.append(hmm_model.beta.view(beta_shape))

    beta = betas[0]
    for beta_i in betas[1:]:
        beta = beta + beta_i
    beta = beta.view(-1, beta.shape[-1]).contiguous()    
    beta = torch.log_softmax(beta, dim=-1)
    monarch_model.beta.data.copy_(beta)

    for i, hmm_model in enumerate(hmm_models):
        monarch_model.alpha_exp.weights_exp[i].data.copy_(hmm_model.alpha_exp[None, :, :])    

    gammas = []
    for i, hmm_model in enumerate(hmm_models):
        gamma_shape = (1,) * i + (hmm_model.gamma_exp.shape[1],) + (1,) * (len(block_size) - 1 - i)
        gammas.append(torch.log(hmm_model.gamma_exp.view(gamma_shape)))        

    gamma = gammas[0]
    for gamma_i in gammas[1:]:
        gamma = gamma + gamma_i
    gamma = gamma.view(1, -1).contiguous()
    gamma_exp = torch.softmax(gamma, dim=-1)
    monarch_model.gamma_exp.data.copy_(gamma_exp)

    block_size_str = '_'.join(str(x) for x in block_size)
    print(f'saving {args.output_path}')
    monarch_model.save_pretrained(args.output_path)