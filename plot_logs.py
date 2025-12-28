import pandas as pd
import matplotlib.pyplot as plt
import os

#rnn三种注意力机制
files = {
    "dot": "logs/rnn_runs_attn_dot_tf-1.0.csv",
    "general": "logs/rnn_runs_attn_general_tf-1.0.csv",
    "additive": "logs/rnn_runs_attn_additive_tf-1.0.csv",
}

# 为每种 attention 固定一种颜色
color_map = {
    "dot": "tab:blue",
    "general": "tab:orange",
    "additive": "tab:green",
}
label_map = {
    "dot": "dot-product",
    "general": "multiplicative",
    "additive": "additive",
}

os.makedirs("plots", exist_ok=True)

plt.figure(figsize=(8, 6))

for attn, path in files.items():
    df = pd.read_csv(path)
    color = color_map[attn]
    label = label_map[attn]

    # validation loss（实线）
    plt.plot(
        df["epoch"],
        df["valid_loss"],
        color=color,
        linestyle="-",
        linewidth=2,
        label=f"{label} (valid)",
    )

    # training loss（虚线）
    plt.plot(
        df["epoch"],
        df["train_loss"],
        color=color,
        linestyle="--",
        linewidth=2,
         label=f"{label} (train)",
    )

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("RNN Training vs Validation Loss with Different Attention Mechanisms")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/rnn_attention_train_valid_loss.png", dpi=300)
plt.show()

#rnn的训练策略teacher forcing对比
files = {
    "teacher forcing": "logs/rnn_runs_attn_additive_tf-1.0.csv",
    "free running": "logs/rnn_runs_attn_additive_tf-0.0.csv",
}

# 为每种 attention 固定一种颜色
color_map = {
    "teacher forcing": "tab:blue",
    "free running": "tab:orange",
}
label_map = {
    "teacher forcing": "dot-product",
    "free running": "multiplicative",
}

os.makedirs("plots", exist_ok=True)

plt.figure(figsize=(8, 6))

for attn, path in files.items():
    df = pd.read_csv(path)
    color = color_map[attn]
    label = label_map[attn]

    # validation loss（实线）
    plt.plot(
        df["epoch"],
        df["valid_loss"],
        color=color,
        linestyle="-",
        linewidth=2,
        label=f"{label} (valid)",
    )

    # training loss（虚线）
    plt.plot(
        df["epoch"],
        df["train_loss"],
        color=color,
        linestyle="--",
        linewidth=2,
         label=f"{label} (train)",
    )

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("RNN Training vs Validation Loss with Different Training Policy")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/rnn_training policy_train_valid_loss.png", dpi=300)
plt.show()

#transformer的position embedding schemes对比
files = {
    "absolute": "logs/tranformer_absolute_greedy_epoch20.csv",
    "relative": "logs/tranformer_relative_greedy_epoch20.csv",
}

# 为每种 attention 固定一种颜色
color_map = {
    "absolute": "tab:blue",
    "relative": "tab:orange",
}


os.makedirs("plots", exist_ok=True)

plt.figure(figsize=(8, 6))

for pos_type, path in files.items():
    df = pd.read_csv(path)
    color = color_map[pos_type]

    # validation loss（实线）
    plt.plot(
        df["epoch"],
        df["valid_loss"],
        color=color,
        linestyle="-",
        linewidth=2,
        label=f"{pos_type} (valid)",
    )

    # training loss（虚线）
    plt.plot(
        df["epoch"],
        df["train_loss"],
        color=color,
        linestyle="--",
        linewidth=2,
         label=f"{pos_type} (train)",
    )

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Transformer Training vs Validation Loss with Different position embedding schemes")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/Transformer Training vs Validation Loss with Different position embedding schemes.png", dpi=300)
plt.show()

#transformer的normalization methods对比
files = {
    "layernorm": "logs/tranformer_relative_greedy_epoch20.csv",
    "rmsnorm": "logs/tranformer_relative_rmsnorm_greedy_epoch20.csv",
}

# 为每种 attention 固定一种颜色
color_map = {
    "layernorm": "tab:blue",
    "rmsnorm": "tab:orange",
}


os.makedirs("plots", exist_ok=True)

plt.figure(figsize=(8, 6))

for norm_type, path in files.items():
    df = pd.read_csv(path)
    color = color_map[norm_type]

    # validation loss（实线）
    plt.plot(
        df["epoch"],
        df["valid_loss"],
        color=color,
        linestyle="-",
        linewidth=2,
        label=f"{norm_type} (valid)",
    )

    # training loss（虚线）
    plt.plot(
        df["epoch"],
        df["train_loss"],
        color=color,
        linestyle="--",
        linewidth=2,
         label=f"{norm_type} (train)",
    )

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Transformer Training vs Validation Loss with Different normalization methods ")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/Transformer Training vs Validation Loss with Different normalization methods.png", dpi=300)
plt.show()

#图5 transformer的batchsize对比
files = {
    "batch_size256": "logs/tranformer_relative_greedy_epoch20.csv",
    "batch_size128": "logs/tranformer_relative_batchsize128_greedy_epoch20.csv",
}

# 为每种 attention 固定一种颜色
color_map = {
    "batch_size256": "tab:blue",
    "batch_size128": "tab:orange",
}


os.makedirs("plots", exist_ok=True)

plt.figure(figsize=(8, 6))

for batch_size, path in files.items():
    df = pd.read_csv(path)
    color = color_map[batch_size]

    # validation loss（实线）
    plt.plot(
        df["epoch"],
        df["valid_loss"],
        color=color,
        linestyle="-",
        linewidth=2,
        label=f"{batch_size} (valid)",
    )

    # training loss（虚线）
    plt.plot(
        df["epoch"],
        df["train_loss"],
        color=color,
        linestyle="--",
        linewidth=2,
         label=f"{batch_size} (train)",
    )

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Transformer Training vs Validation Loss with Different batch size ")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/Transformer Training vs Validation Loss with Different batch size.png", dpi=300)
plt.show()

#图6 transformer的lr对比
files = {
    "lr0.0003": "logs/tranformer_relative_greedy_epoch20.csv",
    "lr0.001": "logs/tranformer_relative_lr0.001_greedy_epoch20.csv",
}

# 为每种 attention 固定一种颜色
color_map = {
    "lr0.0003": "tab:blue",
    "lr0.001": "tab:orange",
}


os.makedirs("plots", exist_ok=True)

plt.figure(figsize=(8, 6))

for lr, path in files.items():
    df = pd.read_csv(path)
    color = color_map[lr]

    # validation loss（实线）
    plt.plot(
        df["epoch"],
        df["valid_loss"],
        color=color,
        linestyle="-",
        linewidth=2,
        label=f"{lr} (valid)",
    )

    # training loss（虚线）
    plt.plot(
        df["epoch"],
        df["train_loss"],
        color=color,
        linestyle="--",
        linewidth=2,
         label=f"{lr} (train)",
    )

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Transformer Training vs Validation Loss with Different lr ")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/Transformer Training vs Validation Loss with Different lr.png", dpi=300)
plt.show()

# 图7：Transformer model scale 对比
# small: d_model=128, n_layers=2, d_ff=512
# large: d_model=256, n_layers=4, d_ff=1024
files = {
    "Small model (d=128, L=2, ff=512)": "logs/tranformer_relative_smallmodelscales_greedy_epoch20.csv",
    "Large model (d=256, L=4, ff=1024)": "logs/tranformer_relative_greedy_epoch20.csv",
}

# 固定颜色
color_map = {
    "Small model (d=128, L=2, ff=512)": "tab:orange",
    "Large model (d=256, L=4, ff=1024)": "tab:blue",
}

os.makedirs("plots", exist_ok=True)

plt.figure(figsize=(8, 6))

for model_name, path in files.items():
    df = pd.read_csv(path)
    color = color_map[model_name]

    # validation loss（实线）
    plt.plot(
        df["epoch"],
        df["valid_loss"],
        color=color,
        linestyle="-",
        linewidth=2,
        label=f"{model_name} (valid)",
    )

    # training loss（虚线）
    plt.plot(
        df["epoch"],
        df["train_loss"],
        color=color,
        linestyle="--",
        linewidth=2,
        label=f"{model_name} (train)",
    )

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Transformer Training vs Validation Loss with Different Model Scales")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/transformer_model_scale_comparison.png", dpi=300)
plt.show()