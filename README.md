# nlp_nmt_project
inference.py使用方法
# 1) 翻译单句（最快）
python inference.py --model_type rnn --model_path checkpoints/rnn/rnn_attn-additive_tf-1.0_bs-128_100k.pt --sentence "但是即使是官方活动也带有政治色彩 。"

# 2) 批量翻译 + BLEU
python inference.py --model_type rnn --model_path ccheckpoints/rnn/rnn_attn-additive_tf-1.0_bs-128_100k.pt \
  --input data/processed/test.jsonl --output outputs/preds_transformer.jsonl --decode beam --beam_size 4 --bleu

# 3) T5
python inference.py --model_type t5 --model_path checkpoints/t5 --sentence "但是即使是官方活动也带有政治色彩 。"

# 注意
所有训练好的模型都在checkpoints文件夹，可以选不同的模型测试。但是由于模型太大，无法全部上传到github，仅传了一个示例模型.
全部模型可以从以下链接中下载（ https://pan.baidu.com/s/1OCtE0v0QWYVNCosQKr6lsw 提取码: dxv3），下载好的模型放入checkpoints文件夹即可。