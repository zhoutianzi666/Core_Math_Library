import json
scale_map_file = "/zhoukangkang/2023-06-06minigpt/PaddleNLP/llm/bigscience-bloomz-7b1-mt_sft_bs2_fp16_DP_quant/checkpoints/checkpoints_ptq/weight_scales.json"
#scale_map_file = "/zhoukangkang/2023-06-06minigpt/PaddleNLP/llm/meta-llama-Llama-2-7b_lora_bs2_fp16_DP_quant/checkpoints/checkpoints_ptq/weight_scales.json"
with open(scale_map_file) as json_file:
    scale_map_dict = json.load(json_file)
    print(len(scale_map_dict["bloom.h.0.self_attention.query_key_value.weight_quanter"]))
    print(len(scale_map_dict["bloom.h.0.mlp.dense_4h_to_h.weight_quanter"]))
