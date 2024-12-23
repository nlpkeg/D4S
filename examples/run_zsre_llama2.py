import os.path
import sys
import random

sys.path.append('..')
import json
import random
from easyeditor import (
    FTHyperParams,
    IKEHyperParams,
    KNHyperParams,
    MEMITHyperParams,
    ROMEHyperParams,
    LoRAHyperParams,
    MENDHyperParams,
    SERACHparams,
    PMETHyperParams
)
from easyeditor import BaseEditor
from easyeditor.models.ike import encode_ike_facts
from sentence_transformers import SentenceTransformer
from easyeditor import ZsreDataset

import argparse

if __name__ == "__main__":
    model_name = "gpt-j-6B"
    #gpt-j-6B
    parser = argparse.ArgumentParser()
    parser.add_argument('--editing_method', default="MEMIT", type=str)
    parser.add_argument('--hparams_dir', default=f"/home/easyedit/hparams/MEMIT/{model_name}.yaml", type=str)#gpt-j-6B
    parser.add_argument('--data_dir', default="/home/easyedit/examples/data/zsre/", type=str)
    parser.add_argument('--ds_size', default=None, type=int)
    parser.add_argument('--metrics_save_dir', default='./output', type=str)
    args = parser.parse_args()
    print(model_name, args.editing_method)

    if args.editing_method == 'FT':
        editing_hparams = FTHyperParams
    elif args.editing_method == 'IKE':
        editing_hparams = IKEHyperParams
    elif args.editing_method == 'KN':
        editing_hparams = KNHyperParams
    elif args.editing_method == 'MEMIT':
        editing_hparams = MEMITHyperParams # Done
    elif args.editing_method == 'ROME':
        editing_hparams = ROMEHyperParams # Done
    elif args.editing_method == 'LoRA':
        editing_hparams = LoRAHyperParams
    elif args.editing_method == 'PMET':
        editing_hparams = PMETHyperParams # Done
    else:
        raise NotImplementedError

    if "zsre" in args.data_dir:
        print("zsre")
        test_data = json.load(open(os.path.join(args.data_dir, 'zsre_mend_train_10000.json'), 'r', encoding='utf-8'))[: 2000]
        # pre_tem = test_data[:500]
        # post_tem = test_data[500:]
        # test_data = post_tem + pre_tem

        # random.shuffle(test_data)

        # with open(f"{args.data_dir}zsre_mend_eval_portability_gpt4_2.json", "w", encoding="utf-8") as f:
        #    json.dump(test_data, f, indent=4, ensure_ascii=False)
        # f.close()
        # if args.ds_size is not None:
        # test_data = random.sample(test_data, args.ds_size)

        prompts = [test_data_['src'] for test_data_ in test_data]
        rephrase_prompts = [edit_data_['rephrase'] for edit_data_ in test_data]
        target_new = [edit_data_['alt'] for edit_data_ in test_data]
        locality_prompts = [edit_data_['loc'] for edit_data_ in test_data]
        locality_ans = [edit_data_['loc_ans'] for edit_data_ in test_data]
        # portability_prompts = [edit_data_['portability']['New Question'] for edit_data_ in test_data]
        # portability_ans = [edit_data_['portability']['New Answer'] for edit_data_ in test_data]
    else:
        test_data = json.load(open(os.path.join(args.data_dir, 'counterfact-edit.json'), 'r', encoding='utf-8'))[:6]
        prompts = [test_data_['prompt'] for test_data_ in test_data]
        rephrase_prompts = [edit_data_['rephrase_prompt'] for edit_data_ in test_data]
        target_new = [edit_data_['target_new'] for edit_data_ in test_data]
        locality_prompts = [edit_data_['locality_promp'] for edit_data_ in test_data]
        locality_ans = [edit_data_['locality_ground_truth'] for edit_data_ in test_data]

    locality_inputs = {
        'neighborhood': {
            'prompt': locality_prompts,
            'ground_truth': locality_ans
        },
    }

    subject = [edit_data_['subject'] for edit_data_ in test_data]
    hparams = editing_hparams.from_hparams(args.hparams_dir)

    if args.editing_method == 'IKE':
        train_data_path = os.path.join(args.data_dir, 'zsre_mend_train_10000.json')
        train_ds = ZsreDataset(train_data_path)
        sentence_model = SentenceTransformer(hparams.sentence_model_name).to(f'cuda:{hparams.device}')
        encode_ike_facts(sentence_model, train_ds, hparams)
    else:
        train_ds = None

    editor = BaseEditor.from_hparams(hparams)

    if "llama" in model_name.lower():
        if args.editing_method != "ROME":
            te = ['model.layers.{}.mlp.down_proj.weight']
        else:
            te = ['model.layers.{}.mlp.down_proj.weight']
    else:
        if args.editing_method != "ROME":
            te = ['transformer.h.{}.mlp.fc_out.weight']
        else:
            te = ['transformer.h.{}.mlp.fc_out.weight']

    metrics, edited_model, _, nom, bm = editor.edit(
        prompts=prompts,
        rephrase_prompts=rephrase_prompts,
        target_new=target_new,
        subject=subject,
        train_ds=train_ds,
        locality_inputs=locality_inputs,
        nom_monitor_name=te,
        keep_original_weight=True
    )
    """
    ['model.layers.4.mlp.down_proj.weight', 'model.layers.5.mlp.down_proj.weight',
                          'model.layers.6.mlp.down_proj.weight', 'model.layers.7.mlp.down_proj.weight',
                          'model.layers.8.mlp.down_proj.weight']
    """
    """
    ['transformer.h.3.mlp.fc_out.weight', 'transformer.h.4.mlp.fc_out.weight',
                          'transformer.h.5.mlp.fc_out.weight', 'transformer.h.6.mlp.fc_out.weight',
                          'transformer.h.7.mlp.fc_out.weight', 'transformer.h.8.mlp.fc_out.weight']
    """
    with open(f"{args.editing_method}_result_zsre_{model_name}_fn.json", "w", encoding="utf-8") as f:
        json.dump(nom, f, indent=4, ensure_ascii=False)
    f.close()
    # print(nom)
    json.dump(metrics, open(f'{args.editing_method}_results_{model_name}_fn.json', 'w'), indent=4)
    json.dump(bm, open(f'{args.editing_method}_batch_metric_{model_name}_fn.json', 'w'))
