import os
import openai
import tiktoken
import ipdb
st = ipdb.set_trace
from tqdm import tqdm
import glob
import json
import numpy as np
import argparse
azure = True
if azure:
    openai.api_type = "azure"
    openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT") 
    openai.api_version = "2023-05-15"
    openai.api_key = os.getenv("AZURE_OPENAI_KEY")
else:
    openai.api_key = os.getenv("OPENAI_API_KEY")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default=f"./dataset/gpt_embeddings", help="where to save output")
    parser.add_argument('--embeddings_to_get', default=['planning', 'replanning'], type=str, nargs='+', help="planning, replanning, custom")
    args = parser.parse_args()

    if "planning" in args.embeddings_to_get: # embeddings for initial planning examples
        files_iterate = glob.glob("examples/*.txt")
        filenames = []
        embeddings = np.zeros((len(files_iterate), 1536), dtype=np.float64)
        for f_i in range(len(files_iterate)):
            
            file = files_iterate[f_i]
            with open(file) as f:
                prompt = f.read()
            # ust keep dialogue
            prompt = prompt.split('\n')[0].split('dialogue: ')[-1]
            print(prompt)

            messages = [
            {"role": "user", "content": prompt},
            ]
            embedding = openai.Embedding.create(
                        engine="text-embedding-ada-002",
                        input=prompt,
                        )['data'][0]['embedding']
            embedding = np.asarray(embedding)
            embeddings[f_i] = embedding

            # file_ = file.split('/')[-1]
            file_ = os.path.join('prompt', file)
            filenames.append(file_)

        embedding_dir = os.path.join(args.root_dir, 'embeddings.npy')
        np.save(embedding_dir, embeddings)
        file_order = os.path.join(args.root_dir, 'file_order.txt')
        with open(file_order, 'w') as fp:
            fp.write("\n".join(str(item) for item in filenames))
    if "replanning" in args.embeddings_to_get: # embeddings for re-planning examples
        files_iterate = glob.glob("examples/examples_errors/*.txt")
        filenames = []
        embeddings = np.zeros((len(files_iterate), 1536), dtype=np.float64)
        for f_i in range(len(files_iterate)):
            
            file = files_iterate[f_i]
            with open(file) as f:
                prompt = f.read()

            prompt = prompt.split('\nInput dialogue:')[0]
            
            print(prompt)

            messages = [
            {"role": "user", "content": prompt},
            ]
            embedding = openai.Embedding.create(
                        engine="text-embedding-ada-002",
                        input=prompt,
                        )['data'][0]['embedding']
            embedding = np.asarray(embedding)
            embeddings[f_i] = embedding

            file_ = file.split('/')[-1]
            filenames.append(file_)

        embedding_dir = os.path.join(args.root_dir, 'embeddings_replanning.npy')
        np.save(embedding_dir, embeddings)
        file_order = os.path.join(args.root_dir, 'file_order_replanning.txt')
        with open(file_order, 'w') as fp:
            fp.write("\n".join(str(item) for item in filenames))

    if "custom" in args.embeddings_to_get: # embeddings for custom
        files_iterate = glob.glob("examples/*.txt")
        files_iterate += glob.glob("examples/examples_custom/*.txt")
        filenames = []
        embeddings = np.zeros((len(files_iterate), 1536), dtype=np.float64)
        for f_i in range(len(files_iterate)):
            
            file = files_iterate[f_i]
            with open(file) as f:
                prompt = f.read()
            # ust keep dialogue
            prompt = prompt.split('\n')[0].split('dialogue: ')[-1]
            print(prompt)

            messages = [
            {"role": "user", "content": prompt},
            ]
            embedding = openai.Embedding.create(
                        engine="text-embedding-ada-002",
                        input=prompt,
                        )['data'][0]['embedding']
            embedding = np.asarray(embedding)
            embeddings[f_i] = embedding

            file_ = os.path.join('prompt', file)
            filenames.append(file_)

        embedding_dir = os.path.join(args.root_dir, 'embeddings_custom.npy')
        np.save(embedding_dir, embeddings)
        file_order = os.path.join(args.root_dir, 'file_order_custom.txt')
        with open(file_order, 'w') as fp:
            fp.write("\n".join(str(item) for item in filenames))