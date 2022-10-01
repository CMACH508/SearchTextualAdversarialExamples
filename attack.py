import argparse
import copy
import datetime
import json
import os
import random
import time
from itertools import chain

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, GPT2LMHeadModel, GPT2TokenizerFast

from attack import CandidateSelectionModule
from attack.utils import Vocab, SynonymQuery
from attack.utils import nltk_tokenize, nltk_detokenize, recover_word_case
from misc.mcts import State, search
from misc.utils import preprocess_data, cal_ppl
from victim_model import HuggingFaceModelWrapper, PyTorchModelWrapper, RNNModelForSequenceClassification
from victim_model import TransformerTokenizer, GloveTokenizer


def preprocess_candidate_selection_module_train_data(path, sim_threshold, top_k, ppl_proportion=0.9):
    with open(path, 'r', encoding='utf8') as f:
        data = json.load(f)
    preprocessed_data = []
    for d in data:
        tokens = d['tokens']
        label = d['label']
        if isinstance(tokens[0], str):  # single text input
            tokens = [tokens, []]
        candidates = []
        for can in d['candidates']:
            index = can['index']
            synonyms = can['synonyms']
            sem_sim = can['sem_sim']
            pred_prob = can['pred_prob']
            text_ppl = can['ppl']
            if isinstance(index, int):
                index = [0, index]
            # filter synonyms
            selected_synonyms = [(syn, sim, ap, ppl) for syn, sim, ap, ppl in
                                 zip(synonyms, sem_sim, pred_prob, text_ppl) if sim >= sim_threshold]
            selected_synonyms.sort(key=lambda x: x[3])
            selected_synonyms = selected_synonyms[:int(len(selected_synonyms) * ppl_proportion)]
            selected_synonyms.sort(key=lambda x: x[2])
            selected_synonyms = selected_synonyms[:top_k]
            if selected_synonyms:
                candidates.append({
                    'index': index,
                    'synonyms': selected_synonyms
                })
        if candidates:
            preprocessed_data.append({
                'tokens': tokens,
                'label': label,
                'candidates': candidates
            })
    return preprocessed_data


def preprocess_word_ranking_module_train_data(path, sim_threshold, ppl_proportion=0.9):
    with open(path, 'r', encoding='utf8') as f:
        data = json.load(f)
    preprocessed_data = []
    for d in data:
        tokens = d['tokens']
        prob = d['prob']
        label = d['label']
        if isinstance(tokens[0], str):
            tokens = [tokens, []]
        candidates = []
        for can in d['candidates']:
            index = can['index']
            if isinstance(index, int):
                index = [0, index]
            saliency = can['saliency']
            sem_sim = can['sem_sim']
            pred_prob = can['pred_prob']
            text_ppl = can['ppl']
            # filter synonyms
            selected_synonyms = [(sim, ap, ppl) for sim, ap, ppl in zip(sem_sim, pred_prob, text_ppl)
                                 if sim >= sim_threshold]
            selected_synonyms.sort(key=lambda x: x[2])
            selected_synonyms = selected_synonyms[:int(len(selected_synonyms) * ppl_proportion)]
            attack_effect = max([(prob - ap) * sim for sim, ap, _ in selected_synonyms] + [1e-10])
            candidates.append({
                'index': index,
                'saliency': saliency,
                'attack_effect': attack_effect
            })
        preprocessed_data.append({
            'tokens': tokens,
            'prob': prob,
            'label': label,
            'candidates': candidates
        })
    return preprocessed_data


def train_candidate_selection_module(args):
    print(args)
    train_data = preprocess_candidate_selection_module_train_data(args.train_data_path, args.sim_threshold, args.top_k,
                                                                  args.ppl_proportion)
    if args.num_samples > 0:
        train_data = train_data[:args.num_samples]
    print(f'training data size {len(train_data)}')

    vocab = Vocab.load(args.vocab_path)
    model_config = {
        'encoder_path': args.encoder_path,
        'hidden_dim': args.hidden_dim,
        'num_classes': args.num_classes,
        'act_dim': len(vocab)
    }
    print(model_config)
    device = torch.device(args.device)
    candidate_selection_module = CandidateSelectionModule(model_config).to(device)
    candidate_selection_module.train()

    optimizer = torch.optim.Adam(candidate_selection_module.parameters(), lr=1e-5)
    loss = torch.nn.KLDivLoss(reduction='batchmean')

    batch_size = args.batch_size
    tot_cost, tot_samples = 0, 0

    for epoch in range(args.num_epochs):
        for idx in range(0, len(train_data), batch_size):
            batch_tokens = [d['tokens'] for d in train_data[idx:(idx + batch_size)]]
            batch_indices = [[can['index'] for can in d['candidates']]
                             for d in train_data[idx:(idx + batch_size)]]
            batch_labels = [[d['label']] * len(d['candidates'])
                            for d in train_data[idx:(idx + batch_size)]]
            batch_labels = list(chain(*batch_labels))
            obs = batch_tokens, batch_indices, batch_labels
            pred_prob = candidate_selection_module.forward_for_lm(obs)

            batch_prob = []
            for d in train_data[idx:(idx + batch_size)]:
                for can in d['candidates']:
                    p = torch.zeros(len(vocab))
                    synonyms = can['synonyms']
                    syn_indices = vocab.word2id([syn[0] for syn in synonyms])
                    syn_indices = torch.tensor(syn_indices)
                    p[syn_indices] = 1
                    p /= p.sum()
                    batch_prob.append(p)
            batch_prob = torch.stack(batch_prob).to(device)
            cost = loss(torch.log(pred_prob), batch_prob)
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            tot_cost += cost.item() * pred_prob.shape[0]
            tot_samples += pred_prob.shape[0]
        ctime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"{ctime} epoch {epoch}, cost {tot_cost / tot_samples:.4f}")
        tot_cost, tot_samples = 0, 0

    train_args = {
        'encoder_path': args.encoder_path,
        'sim_threshold': args.sim_threshold,
        'top_k': args.top_k,
        'hidden_dim': args.hidden_dim,
        'num_classes': args.num_classes,
        'num_epochs': args.num_epochs,
        'num_samples': args.num_samples,
        'batch_size': args.batch_size
    }
    if not os.path.exists(args.module_path):
        os.makedirs(args.module_path)
    with open(os.path.join(args.module_path, "train_args.json"), 'w', encoding='utf8') as f:
        json.dump(train_args, f, indent=4)
    torch.save(candidate_selection_module.state_dict(), os.path.join(args.module_path, "pytorch_model.bin"))


def word_ranking(text_info, victim_model, candidate_selection_module, synonym_query, vocab, top_k, result):
    text = text_info['text']
    label = text_info['label']
    tokens = text_info['tokens']
    tags = text_info['tags']
    indices = text_info['indices']
    orig_prob = text_info['orig_prob']

    # Calculate saliency
    batch_texts = []
    for i, j in indices:
        tmp = tokens[i][j]
        tokens[i][j] = '[UNK]'
        if isinstance(text, str):
            cur_text = nltk_detokenize(tokens[i])
        else:
            cur_text = copy.copy(text)
            cur_text[i] = nltk_detokenize(tokens[i])
        batch_texts.append(cur_text)
        tokens[i][j] = tmp
    prob = torch.softmax(torch.from_numpy(victim_model(batch_texts)), dim=-1)
    result['num_model_queries'] += len(batch_texts)
    saliency = prob[:, label]
    saliency = torch.softmax(saliency, dim=-1)

    # Calculate attack effect
    attack_effect = []
    obs = [tokens], [indices], [label] * len(indices)
    with torch.no_grad():
        candidate_prob = candidate_selection_module.forward_for_lm(obs)
    candidate_prob = candidate_prob.cpu().numpy()
    for (i, j), can_prob in zip(indices, candidate_prob):
        tmp = tokens[i][j]
        attack_texts = []
        synonyms = synonym_query(tokens[i][j], tags[i][j])
        synonym_index = np.array(vocab.word2id(synonyms))
        can_prob[synonym_index] += 1
        select_num = min(top_k, len(synonyms))
        synonym_index = np.argsort(can_prob)[-1:-(select_num + 1):-1].tolist()
        synonyms = vocab.id2word(synonym_index)
        for syn in synonyms:
            tokens[i][j] = syn
            if isinstance(text, str):
                cur_text = nltk_detokenize(tokens[i])
            else:
                cur_text = copy.copy(text)
                cur_text[i] = nltk_detokenize(tokens[i])
            attack_texts.append(cur_text)
        attack_pred = victim_model(attack_texts)
        result['num_model_queries'] += len(attack_texts)
        attack_prob = torch.softmax(torch.from_numpy(attack_pred), dim=-1)[:, label]
        attack_effect.append(orig_prob - torch.min(attack_prob))
        tokens[i][j] = tmp
    attack_effect = torch.tensor(attack_effect)
    h = saliency * attack_effect
    idx = torch.argsort(h).tolist()[::-1]
    indices = [indices[i] for i in idx]
    return indices


def evaluate_beam(d, candidate_selection_module, victim_model, sent_encoder, vocab, synonym_query,
                  params):
    text = d['text']
    label = d['label']
    tokens = d['tokens']
    tags = d['tags']
    indices = d['indices']
    text = text if text[1] else text[0]
    indices = [(0, i) for i in indices[0]] + [(1, i) for i in indices[1]]

    start = time.time()

    result = {
        'orig_text': text,
        'atk_text': text,
        'orig_tokens': tokens,
        'atk_tokens': tokens,
        'sem_sim': -1,
        'num_model_queries': 0,
        'consuming_time': 0.0
    }

    if not indices:
        return result

    orig_pred = victim_model([text])
    result['num_model_queries'] += 1
    orig_prob = torch.softmax(torch.from_numpy(orig_pred), dim=-1)
    orig_prob = orig_prob[0, label].item()

    orig_embed = sent_encoder.encode([text])

    text_info = {
        'text': text,
        'label': label,
        'tokens': tokens,
        'tags': tags,
        'indices': indices,
        'orig_prob': orig_prob
    }
    use_candidate_selection_module = True
    if use_candidate_selection_module:
        indices = word_ranking(text_info, victim_model, candidate_selection_module, synonym_query, vocab,
                               params['wr_top_k'], result)

    beam = [[copy.deepcopy(tokens), orig_prob]]

    use_candidate_selection_module = True
    for i, j in indices:
        new_beam = []
        for atk_tokens, atk_prob in beam:
            obs = [atk_tokens], [[i, j]], [label]
            if use_candidate_selection_module:
                with torch.no_grad():
                    prob = candidate_selection_module(obs)
            else:
                prob = torch.ones(1, len(vocab)) / len(vocab)
            prob = prob.squeeze(0)
            prob = prob.cpu().numpy()

            synonyms = synonym_query(atk_tokens[i][j], tags[i][j]) + [atk_tokens[i][j]]
            synonym_index = np.array(vocab.word2id(synonyms))
            prob[synonym_index] += 1
            select_num = min(params['top_k'], len(synonyms))
            synonym_index = np.argsort(prob)[-1:-(select_num + 1):-1].tolist()
            synonyms = vocab.id2word(synonym_index)

            batch_texts = []
            for syn in synonyms:
                atk_tokens[i][j] = recover_word_case(syn, tokens[i][j])
                if atk_tokens[1]:
                    batch_texts.append([nltk_detokenize(atk_tokens[0]), nltk_detokenize(atk_tokens[1])])
                else:
                    batch_texts.append(nltk_detokenize(atk_tokens[0]))
            atk_tokens[i][j] = tokens[i][j]

            embed = sent_encoder.encode(batch_texts)
            sem_sim = cosine_similarity(orig_embed, embed).reshape(-1)
            if np.max(sem_sim) < params['sim_threshold']:
                new_beam.append([atk_tokens, atk_prob])
                continue
            sem_sim = sem_sim.tolist()
            tmp = [(atk_text, syn, sim) for atk_text, syn, sim in zip(batch_texts, synonyms, sem_sim)
                   if sim >= params['sim_threshold']]
            batch_texts, synonyms, sem_sim = zip(*tmp)
            batch_texts = list(batch_texts)

            pred = victim_model(batch_texts)
            result['num_model_queries'] += len(batch_texts)
            prob = torch.softmax(torch.from_numpy(pred), dim=-1)[:, label]

            if torch.min(prob) >= atk_prob:
                new_beam.append([atk_tokens, atk_prob])
                continue
            for idx in torch.argsort(prob).tolist():
                current_tokens = copy.deepcopy(atk_tokens)
                current_tokens[i][j] = recover_word_case(synonyms[idx], tokens[i][j])
                new_beam.append([current_tokens, prob[idx]])
                if np.argmax(pred[idx]) != label:
                    result['atk_text'] = batch_texts[idx]
                    result['atk_tokens'] = current_tokens
                    result['sem_sim'] = sem_sim[idx]
                    break
            if result['atk_text'] != result['orig_text']:
                break
        if result['atk_text'] != result['orig_text']:
            break
        beam += new_beam
        beam.sort(key=lambda x: x[1])
        beam = beam[:params['beam_size']]
    result['consuming_time'] = time.time() - start

    return result


def evaluate_mcts(d, candidate_selection_module, victim_model, sent_encoder, vocab,
                  synonym_query, params):
    orig_text = d['text']
    orig_tokens = d['tokens']
    label = d['label']
    tags = d['tags']
    indices = d['indices']
    orig_text = orig_text if orig_text[1] else orig_text[0]
    indices = [(0, i) for i in indices[0]] + [(1, i) for i in indices[1]]

    start = time.time()

    result = {
        'orig_text': orig_text,
        'atk_text': orig_text,
        'orig_tokens': orig_tokens,
        'atk_tokens': orig_tokens,
        'sem_sim': -1,
        'num_model_queries': 0,
        'consuming_time': 0.0
    }

    if not indices:
        return result

    orig_pred = victim_model([orig_text])
    result['num_model_queries'] += 1
    orig_prob = torch.softmax(torch.from_numpy(orig_pred), dim=-1)[0]

    orig_embed = sent_encoder.encode([orig_text])[0]

    text_info = {
        'text': orig_text,
        'label': label,
        'tokens': orig_tokens,
        'tags': tags,
        'indices': indices,
        'orig_prob': orig_prob[label]
    }
    use_candidate_selection_module = True
    if use_candidate_selection_module:
        indices = word_ranking(text_info, victim_model, candidate_selection_module, synonym_query, vocab,
                               params['wr_top_k'], result)

    use_candidate_selection_module = True

    class TextState(State):

        token_cache = {
            (tuple(orig_tokens[0]), tuple(orig_tokens[1])): (orig_text, tuple(orig_prob), 1.0)
        }
        text_state_cache = {}

        def __init__(self, tokens, token_idx):
            self.tokens = tokens
            self.idx = token_idx
            assert (self.tokens, self.idx) not in TextState.text_state_cache

            if self.tokens not in TextState.token_cache:
                self.text = (nltk_detokenize(self.tokens[0]), nltk_detokenize(self.tokens[1])) if self.tokens[1] else \
                    nltk_detokenize(self.tokens[0])

                text_embed = sent_encoder.encode([self.text])[0]
                self.sem_sim = cosine_similarity([orig_embed], [text_embed]).item()

                self.text_prob = None
                if self.is_valid():
                    self.text_prob = victim_model([self.text])
                    self.text_prob = tuple(torch.softmax(torch.from_numpy(self.text_prob), dim=-1)[0])
                    result['num_model_queries'] += 1

                TextState.token_cache[self.tokens] = (self.text, self.text_prob, self.sem_sim)
            else:
                self.text, self.text_prob, self.sem_sim = TextState.token_cache[self.tokens]

            current_synonyms, prior_prob = (), ()
            if self.is_valid() and self.idx < len(indices):
                i, j = indices[self.idx]
                current_synonyms = tuple(synonym_query(self.tokens[i][j], tags[i][j]))

                obs = [self.tokens], [[i, j]], [label]
                if use_candidate_selection_module:
                    with torch.no_grad():
                        prior_prob = candidate_selection_module(obs)
                else:
                    prior_prob = torch.ones(1, len(vocab)) / len(vocab)
                prior_prob = prior_prob.squeeze(0).cpu().numpy()

                synonym_index = vocab.word2id(current_synonyms)
                synonym_index.sort(key=lambda x: prior_prob[x])
                select_num = min(15, len(current_synonyms))
                synonym_index = synonym_index[-select_num:]
                current_synonyms = vocab.id2word(synonym_index)
                if tokens[i][j] not in current_synonyms:
                    current_synonyms.append(tokens[i][j])
                current_synonyms = tuple(current_synonyms)

                prior_prob = prior_prob[np.array(vocab.word2id(current_synonyms))]
                prior_prob /= np.sum(prior_prob)
                prior_prob = tuple(prior_prob)
                current_synonyms = tuple(recover_word_case(s, tokens[i][j]) for s in current_synonyms)

            super(TextState, self).__init__(current_synonyms, prior_prob)
            TextState.text_state_cache[(self.tokens, self.idx)] = self

        def next_state(self, act):
            new_tokens = [list(self.tokens[0]), list(self.tokens[1])]
            i, j = indices[self.idx]
            new_tokens[i][j] = act
            new_tokens = (tuple(new_tokens[0]), tuple(new_tokens[1]))
            new_idx = self.idx + 1
            if (new_tokens, new_idx) in TextState.text_state_cache:
                return TextState.text_state_cache[(new_tokens, new_idx)]
            return TextState(new_tokens, new_idx)

        def value(self):
            assert np.argmax(state.text_prob) == label
            return (1 - self.text_prob[label]) ** 2 * self.sem_sim ** 2

        def is_valid(self):
            return self.sem_sim >= params['sim_threshold']

        def is_terminal(self):
            return np.argmax(self.text_prob) != label or self.idx == len(indices)

        def rollout(self):
            st = self
            tries = 0
            while not st.is_terminal():
                max_idx = np.argsort(st.prior_prob)[-params['top_k']:].tolist()
                random.shuffle(max_idx)
                if (len(st.actions) - 1) not in max_idx:
                    max_idx.append(len(st.actions) - 1)
                valid_new_state_exists = False
                for i in max_idx:
                    new_state = st.next_state(st.actions[i])
                    if new_state.is_valid():
                        valid_new_state_exists = True
                        st = new_state
                        break
                assert valid_new_state_exists
                tries += 1
            success = np.argmax(st.text_prob).item() != label
            reward = -(st.text_prob[label] - 1 / len(st.text_prob)) ** 2
            reward = [reward, 1][success]
            return reward, success, st

    current_tokens = copy.deepcopy(orig_tokens)
    for idx in range(len(indices)):
        tmp = (tuple(current_tokens[0]), tuple(current_tokens[1]))
        if (tmp, idx) in TextState.text_state_cache:
            state = TextState.text_state_cache[(tmp, idx)]
        else:
            state = TextState(tmp, idx)
        assert state.is_valid()
        if state.sem_sim >= params['sim_threshold'] and np.argmax(state.text_prob) != label:
            result['atk_text'] = state.text
            result['sem_sim'] = state.sem_sim
            break
        act, rollout_state = search(state, params['search_budget'], params['exploration_coefficient'],
                                    params['state_value_coefficient'])
        assert (act is None) + (rollout_state is None) == 1
        if rollout_state is not None:
            result['atk_text'] = rollout_state.text
            result['atk_tokens'] = rollout_state.tokens
            result['sem_sim'] = rollout_state.sem_sim
            break
        i, j = indices[idx]
        current_tokens[i][j] = act
    result['consuming_time'] = time.time() - start

    return result


def test(args):
    print(args)
    vocab = Vocab.load(args.vocab_path)

    device = torch.device(args.device)
    synonym_query = SynonymQuery.load(args.synonym_query_path)
    sent_encoder = SentenceTransformer(args.sent_encoder_path, device=args.device)

    pretrained_path = args.victim_model_path
    if pretrained_path[-1] == '/':
        pretrained_path = pretrained_path[:-1]
    if os.path.split(pretrained_path)[-1].startswith('bert'):
        bert_tokenizer = TransformerTokenizer(tokenizer_path=pretrained_path)
        victim_model = AutoModelForSequenceClassification.from_pretrained(pretrained_path).to(device)
        victim_model = HuggingFaceModelWrapper(victim_model, bert_tokenizer, batch_size=64)
    elif os.path.split(pretrained_path)[-1].startswith('lstm'):
        glove_tokenizer = GloveTokenizer(tokenizer_path=pretrained_path)
        victim_model = RNNModelForSequenceClassification.from_pretrained(pretrained_path).to(device)
        victim_model = PyTorchModelWrapper(victim_model, glove_tokenizer, batch_size=512)
    else:
        raise ValueError('Invalid victim model path')

    candidate_selection_module_config = {
        'encoder_path': args.encoder_path,
        'hidden_dim': args.hidden_dim,
        'num_classes': args.num_classes,
        'act_dim': len(vocab)
    }
    candidate_selection_module = CandidateSelectionModule(candidate_selection_module_config).to(device)
    candidate_selection_module.load_state_dict(torch.load(args.candidate_selection_module_path, map_location='cpu'))
    candidate_selection_module.eval()

    test_data = preprocess_data(args.dataset_path, nltk_tokenize, synonym_query, 'test')
    print(f'testing data size {len(test_data)}')

    gpt_model = GPT2LMHeadModel.from_pretrained(args.gpt_path).to(device)
    gpt_tokenizer = GPT2TokenizerFast.from_pretrained(args.gpt_path)

    if args.search_method == 'beam_search':
        params = {
            'wr_top_k': args.wr_top_k,
            'top_k': args.top_k,
            'sim_threshold': args.sim_threshold,
            'beam_size': args.beam_size
        }
        results = [evaluate_beam(d, candidate_selection_module, victim_model, sent_encoder,
                                 vocab, synonym_query, params)
                   for d in tqdm(test_data)]
    elif args.search_method == 'mcts':
        params = {
            'wr_top_k': args.wr_top_k,
            'top_k': args.top_k,
            'sim_threshold': args.sim_threshold,
            'search_budget': args.mcts_search_budget,
            'exploration_coefficient': args.mcts_exploration_coefficient,
            'state_value_coefficient': args.mcts_state_value_coefficient
        }
        results = [evaluate_mcts(d, candidate_selection_module, victim_model, sent_encoder,
                                 vocab, synonym_query, params)
                   for d in tqdm(test_data)]
    else:
        raise ValueError('Invalid search method')
    assert len(results) == len(test_data)
    for r in results:
        r['ppl'] = 0.0
        if r['sem_sim'] == -1:
            continue
        orig_tokens = r['orig_tokens'][0] + r['orig_tokens'][1]
        atk_tokens = r['atk_tokens'][0] + r['atk_tokens'][1]
        r['ppl'] = cal_ppl(r['atk_text'], gpt_model, gpt_tokenizer)

    num_success = sum(r['sem_sim'] != -1 for r in results)
    asr = num_success / len(test_data)
    sem_sim = sum(r['sem_sim'] for r in results if r['sem_sim'] != -1) / num_success  # only for successful samples
    num_model_queries = sum(r['num_model_queries'] for r in results) / len(results)
    ppl = sum(r['ppl'] for r in results if r['sem_sim'] != -1) / num_success
    consuming_time = sum(r['consuming_time'] for r in results) / len(results)

    if args.search_method == 'beam_search':
        with open(os.path.join(args.output_path, 'beam_results.txt'), 'a', encoding='utf8') as f:
            print(f"wr_top_k {args.wr_top_k}, top_k {args.top_k}, sem_thr {args.sim_threshold:.1f}, "
                  f"beam size {args.beam_size}, "
                  f"asr {asr:.3f}, model query {num_model_queries:.0f}, "
                  f"sim {sem_sim:.3f}, ppl {ppl:.1f}, "
                  f"consuming_time {consuming_time:.3f} sec", file=f)
    elif args.search_method == 'mcts':
        with open(os.path.join(args.output_path, 'mcts_results.txt'), 'a', encoding='utf8') as f:
            print(f"wr_top_k {args.wr_top_k}, top_k {args.top_k}, sem_thr {args.sim_threshold:.1f}, "
                  f"budget {args.mcts_search_budget}, alpha {args.mcts_exploration_coefficient: .1f}, "
                  f"lambda {args.mcts_state_value_coefficient: .1f}, "
                  f"asr {asr:.3f}, model query {num_model_queries:.0f}, "
                  f"sim {sem_sim:.3f}, ppl {ppl:.1f}, "
                  f"consuming_time {consuming_time:.3f} sec", file=f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=0)
    subparsers = parser.add_subparsers()

    parser_train_attack = subparsers.add_parser('train_candidate_selection_module', help='candidate selection module')
    parser_train_attack.add_argument('--device', type=str, default='cuda:0')
    parser_train_attack.add_argument('--encoder_path', type=str, default='resources/encoder_models/bert')
    parser_train_attack.add_argument('--vocab_path', type=str, default='resources/hownet/vocab.pkl')
    parser_train_attack.add_argument('--train_data_path', type=str, default='resources/datasets/bert_train/qqp.json')
    parser_train_attack.add_argument('--module_path', type=str,
                                     default='results/prediction/bert/qqp/bert/candidate_selection_module')
    parser_train_attack.add_argument('--sim_threshold', type=float, default=0.95)
    parser_train_attack.add_argument('--top_k', type=int, default=1)
    parser_train_attack.add_argument('--ppl_proportion', type=float, default=0.9)
    parser_train_attack.add_argument('--hidden_dim', type=int, default=128)
    parser_train_attack.add_argument('--num_classes', type=int, default=2)
    parser_train_attack.add_argument('--num_epochs', type=int, default=5)
    parser_train_attack.add_argument('--num_samples', type=int, default=-1)
    parser_train_attack.add_argument('--batch_size', type=int, default=5)
    parser_train_attack.set_defaults(func=train_candidate_selection_module)

    parser_test = subparsers.add_parser('test', help='testing')
    parser_test.add_argument('--device', type=str, default='cuda:0')
    parser_test.add_argument('--encoder_path', type=str, default='resources/encoder_models/bert')
    parser_test.add_argument('--sent_encoder_path', type=str, default='stsb-mpnet-base-v2')
    parser_test.add_argument('--vocab_path', type=str, default='resources/hownet/vocab.pkl')
    parser_test.add_argument('--synonym_query_path', type=str, default='resources/hownet/synonyms.pkl')
    parser_test.add_argument('--dataset_path', type=str, default='resources/datasets/bert_original/qqp.json')
    parser_test.add_argument('--gpt_path', type=str, default='gpt2')
    parser_test.add_argument('--hidden_dim', type=int, default=128)
    parser_test.add_argument('--num_classes', type=int, default=2)
    parser_test.add_argument('--dropout', type=float, default=0.5)
    parser_test.add_argument('--sim_threshold', type=float, default=0.9)
    parser_test.add_argument('--beam_size', type=int, default=4)
    parser_test.add_argument('--wr_top_k', type=int, default=10)
    parser_test.add_argument('--top_k', type=int, default=10)
    parser_test.add_argument('--mcts_search_budget', type=int, default=100)
    parser_test.add_argument('--mcts_exploration_coefficient', type=float, default=1.0)
    parser_test.add_argument('--mcts_state_value_coefficient', type=float, default=0.5)
    parser_test.add_argument('--search_method', type=str, default='beam_search')
    parser_test.add_argument('--victim_model_path', type=str, default='resources/victim_models/bert-qqp')
    parser_test.add_argument('--candidate_selection_module_path', type=str,
                             default='results/prediction/bert/qqp/bert/candidate_selection_module')
    parser_test.add_argument('--output_path', type=str, default='results/prediction/bert/qqp/bert')
    parser_test.set_defaults(func=test)

    args = parser.parse_args()
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    args.func(args)


if __name__ == '__main__':
    main()
