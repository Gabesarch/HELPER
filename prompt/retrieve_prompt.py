import os
import openai
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


def encode_prompts(root_dir):
    embeddings = np.zeros((4, 1536), dtype=np.float64)
    filenames = []

    # Teach
    with open('prompt_retrieval/prompt_combined_teach.txt', 'r') as file:
        file_contents = file.read()
    print(file_contents)
    embedding = openai.Embedding.create(
        engine="text-embedding-ada-002",
        input=file_contents,
    )['data'][0]['embedding']
    embedding = np.asarray(embedding)
    # embedding_dir = os.path.join(root_dir, 'embeddings_teach.npy')
    # np.save(embedding_dir, embedding)
    embeddings[0] = embedding
    filenames.append(os.path.join('prompt_retrieval', 'prompt_combined_teach_template.txt'))

    # Alfred
    with open('prompt_retrieval/prompt_combined_alfred.txt', 'r') as file:
        file_contents = file.read()
    print(file_contents)
    embedding = openai.Embedding.create(
        engine="text-embedding-ada-002",
        input=file_contents,
    )['data'][0]['embedding']
    embedding = np.asarray(embedding)
    # embedding_dir = os.path.join(root_dir, 'embeddings_alfred.npy')
    # np.save(embedding_dir, embedding)
    embeddings[1] = embedding
    filenames.append(os.path.join('prompt_retrieval', 'prompt_combined_alfred_template.txt'))

    # Dialfred
    with open('prompt_retrieval/prompt_combined_dialfred.txt', 'r') as file:
        file_contents = file.read()
    print(file_contents)
    embedding = openai.Embedding.create(
        engine="text-embedding-ada-002",
        input=file_contents,
    )['data'][0]['embedding']
    embedding = np.asarray(embedding)
    # embedding_dir = os.path.join(root_dir, 'embeddings_dialfred.npy')
    # np.save(embedding_dir, embedding)
    embeddings[2] = embedding
    filenames.append(os.path.join('prompt_retrieval', 'prompt_combined_dialfred_template.txt'))

    # Tidy
    with open('prompt_retrieval/prompt_combined_tidy.txt', 'r') as file:
        file_contents = file.read()
    print(file_contents)
    embedding = openai.Embedding.create(
        engine="text-embedding-ada-002",
        input=file_contents,
    )['data'][0]['embedding']
    embedding = np.asarray(embedding)
    # embedding_dir = os.path.join(root_dir, 'embeddings_tidy.npy')
    # np.save(embedding_dir, embedding)
    embeddings[3] = embedding
    filenames.append(os.path.join('prompt_retrieval', 'prompt_combined_tidy_template.txt'))

    embedding_dir = os.path.join(args.root_dir, 'embeddings_prompts.npy')
    np.save(embedding_dir, embeddings)
    file_order = os.path.join(args.root_dir, 'file_order_prompts.txt')
    with open(file_order, 'w') as fp:
        fp.write("\n".join(str(item) for item in filenames))

    return embeddings

def find_distance(task, embeddings):

    # Create embedding for the incoming task
    embedding = openai.Embedding.create(
        engine="text-embedding-ada-002",
        input=task,
    )['data'][0]['embedding']
    embedding = np.asarray(embedding)

    #Find distance with all three and pick with min distance

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default=f"./dataset/gpt_embeddings", help="where to save output")
    parser.add_argument("--is_embedding", type=bool, default=False, help="Do embeddings already exist")
    parser.add_argument('--embeddings_to_get', default=['planning', 'replanning'], type=str, nargs='+',
                        help="planning, replanning, custom")
    args = parser.parse_args()
    encode_prompts(args.root_dir)

    # if not args.is_embedding:
    #     embeddings = encode_prompts(args.root_dir)
    # else:
    #     embeddings = np.zeros(3)
    #     embeddings[0] = np.load(args.root_dir + 'embeddings_teach.npy')
    #     embeddings[1] = np.load(args.root_dir + 'embeddings_alfred.npy')
    #     embeddings[2] = np.load(args.root_dir + 'embeddings_dialfred.npy')


    # # TODO: Get tasks from file or during runtime
    # for task in tasks:
    #     find_distance(task, embeddings)
