import os
import pathlib
import argparse

import numpy as np
import torch

from nocap.model import NoCAP
from nocap.dataset import NoCAP_DS


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


def sample(model, current, num_samples):
    past = None
    current = torch.as_tensor(current, dtype=torch.float32).unsqueeze(0)
    output = [current.clone()[0]]
    with torch.inference_mode():
        for t in range(1, num_samples + 1):
            if t % 100 == 0:
                print(f'Sampling joint pose: ({t}/{num_samples})')

            past = clip_context(past, current, model.n_ctx)
            mu, sigma, new_past = model(current, past=past)
            dist = torch.distributions.Normal(loc=mu[0, -1], scale=sigma[0, -1])
            new_pose = dist.sample().view(1, 1, -1)

            output.append(new_pose[0])
            current = new_pose
            past = new_past
    
    output = torch.vstack(output).view(-1, 63)
    return output.numpy()


def run(args):
    ckpt = torch.load(args.ckpt_path, map_location='cpu')

    config = ckpt['model_config']
    model_state_dict = ckpt['model_state_dict']
    model = NoCAP(config)
    model.load_state_dict(model_state_dict)
    model.eval()

    dataset = NoCAP_DS(args.data_path)
    num_prompt = dataset.num_joints * 10
    current = dataset[0, :num_prompt]
    num_samples = int(args.seconds * dataset.motion_freq * dataset.num_joints)
    output = sample(model, current, num_samples)

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
    parser.add_argument('--seconds', default=10)
    args = parser.parse_args()
    run(args)