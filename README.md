# nlp_nmt_project
测试说明
inference.py使用方法
# 1) 翻译单句（最快）
python inference.py --model_type rnn --model_path checkpoints/rnn/rnn_attn-additive_tf-1.0_bs-128_100k.pt --sentence "但是即使是官方活动也带有政治色彩 。"

# 2) 批量翻译 + BLEU
python inference.py --model_type rnn --model_path ccheckpoints/rnn/rnn_attn-additive_tf-1.0_bs-128_100k.pt \
  --input data/processed/test.jsonl --output outputs/preds_transformer.jsonl --decode beam --beam_size 4 --bleu

# 3) T5
python inference.py --model_type t5 --model_path checkpoints/t5 --sentence "但是即使是官方活动也带有政治色彩 。"

# 所有训练好的模型都在checkpoints文件夹，可以选不同的模型测试