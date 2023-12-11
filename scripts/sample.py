import os
import pathlib
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from nocap.model import NoCAP
from nocap.dataset import NoCAP_DS

torch.set_printoptions(linewidth=200, sci_mode=False)

def clip_context(past, current, max_ctx):
    if past is None:
        return None
    
    key, value = past

    assert len(current.shape) == 3, 'Expected current to be shape (batch_size, sequence_length, dim)'
    num_current = current.shape[1]
    if key.shape[-2] <= (max_ctx - num_current):
        return [key, value]
    
    overflow = key.shape[-2] - (max_ctx - num_current)
    assert overflow > 0

    key = key[..., overflow:, :]
    value = value[..., overflow:, :]
    return [key, value]


def sample(model, current, num_samples, dataset):
    past = None
    current = torch.as_tensor(current, dtype=torch.float32).unsqueeze(0)
    #current_joint_encoding = current[0, -1, 3:].clone()
    # position_ids = torch.arange(0, current.size(-2), dtype=torch.long, device=current.device)
    # position_ids = position_ids.unsqueeze(0).expand(current.size(0), -1)
    start_len = current.shape[1]
    position_ids = None
    print(f'Prompt size: {current.shape}')

    reference = dataset[0]
    reference = torch.as_tensor(reference, dtype=torch.float32).unsqueeze(0)
    output = [current.clone()[0]]
    x2mus = []
    mu2mus = []
    with torch.inference_mode():
        for t in range(start_len, num_samples):
            if t % 100 == 0:
                print(f'Sampling joint pose: ({t}/{num_samples})')

            ref_t = t - 1
            x2mu = (current[0] - reference[0, ref_t]).abs().sum().item()
            x2mus.append(x2mu)

            past = clip_context(past, current, model.n_ctx)
            mu, sigma, new_past = model(current, position_ids, past=past)

            # Sample new pose
            #dist = torch.distributions.Normal(loc=mu[0, -1], scale=sigma[0, -1])
            #new_pose = dist.sample().view(1, 1, -1)
            new_pose = mu[0, -1].view(1, 1, -1)

            # Increment joint encoding
            # new_joint_encoding = torch.zeros(21).to(dtype=torch.float32, device=mu.device)
            # joint_num = torch.argwhere(current_joint_encoding).squeeze()
            # new_joint_encoding[(joint_num+1) % 21] = 1.
            # new_pose = torch.cat((mu[0, -1], new_joint_encoding), dim=-1)
            # new_pose = new_pose.view(1, 1, -1)

            output.append(new_pose[0])
            current = new_pose
            #current_joint_encoding = new_joint_encoding
            past = new_past

            # position_ids = torch.Tensor([[(t + start_len - 1) % model.n_ctx]]).to(dtype=torch.long, device=current.device)

    output = torch.vstack(output)
    mu2mus = (output[1:] - output[:-1]).abs().sum(-1).numpy()

    #plt.plot(range(len(x2mus)), x2mus, label='x2mu')
    plt.plot(range(len(mu2mus)), mu2mus, label='mu2mu')
    plt.legend()
    plt.savefig('deltas.png')
    output = output.view(-1, 63)
    return output.numpy()


def sample_disc(model, current, num_samples, dataset):
    past = None
    current = torch.as_tensor(current).unsqueeze(0)
    #current_joint_encoding = current[0, -1, 3:].clone()
    # position_ids = torch.arange(0, current.size(-2), dtype=torch.long, device=current.device)
    # position_ids = position_ids.unsqueeze(0).expand(current.size(0), -1)
    start_len = current.shape[1]
    position_ids = None
    print(f'Prompt size: {current.shape}')

    output = [current.clone()[0]]
    with torch.inference_mode():
        for t in range(start_len, num_samples):
            if t % 100 == 0:
                print(f'Sampling joint pose: ({t}/{num_samples})')

            past = clip_context(past, current, model.n_ctx)
            logits, new_past = model(current, position_ids, past=past)
            logits = logits[0, -1]

            token = torch.argmax(logits)

            #dist = F.softmax(logits, dim=-1)
            #token = torch.multinomial(dist, num_samples=1)

            output.append(token)
            current = token.view(1, 1, -1)
            past = new_past

            # position_ids = torch.Tensor([[(t + start_len - 1) % model.n_ctx]]).to(dtype=torch.long, device=current.device)

    output_3d = []
    output = torch.vstack(output).numpy().squeeze()

    import pdb; pdb.set_trace()

    for out in output:
        output_3d.append(dataset.codebook[out])
    output_3d = np.vstack(output_3d).reshape(-1, 63)
    return output_3d


def run(args):
    ckpt = torch.load(args.ckpt_path, map_location='cpu')

    config = ckpt['model_config']
    model_state_dict = ckpt['model_state_dict']
    model = NoCAP(config)
    model.load_state_dict(model_state_dict)
    model.eval()

    dataset = NoCAP_DS(args.data_path, disc=True)
    num_prompt = dataset.num_joints
    print(dataset[0].shape, num_prompt)
    current = dataset[0][:num_prompt]
    num_samples = int(args.seconds * dataset.motion_freq * dataset.num_joints)
    #num_samples = dataset.num_joints * 30
    output = sample_disc(model, current, num_samples, dataset)

    ckpt_dir = args.ckpt_path.split('/')[:-2]
    ckpt_name = args.ckpt_path.split('/')[-1][:-len('.pt')]
    output_dir = os.path.join(*ckpt_dir, 'samples')
    pathlib.Path(output_dir).mkdir(exist_ok=True, parents=True)
    output_path = os.path.join(output_dir, f'seconds={args.seconds}_{ckpt_name}')
    save_dict = {
        'pose_body': output,
        'betas': dataset.betas,
        'root_orient': dataset.root_orient
    }
    np.savez(output_path, **save_dict)

    print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', required=True)
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--seconds', type=int, default=10)
    args = parser.parse_args()
    run(args)