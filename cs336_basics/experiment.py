from pathlib import Path
import json
import matplotlib.pyplot as plt

def parse_training_log(
    log_path: Path,
) -> tuple[
    list[dict[str, int | float]],
    list[dict[str, int | float]],
]:
    train_records: list[dict[str, int | float]] = []
    val_records: list[dict[str, int | float]] = []

    with log_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if line.startswith("step=") or line.startswith("validation_step="):
                try:
                    fields = {}
                    for item in line.split():
                        key, value = item.split("=", 1)
                        fields[key] = value

                    if line.startswith("step="):
                        train_records.append({
                            "step": int(fields["step"]),
                            "loss": float(fields["train_loss"]),
                            "learning_rate": float(fields["learning_rate"]),
                            "elapsed_seconds": float(fields["elapsed_seconds"]),
                        })

                    if line.startswith("validation_step="):
                        val_records.append({
                            "step": int(fields["validation_step"]),
                            "loss": float(fields["validation_loss"]),
                        })
                except (KeyError, ValueError) as exc:
                    raise ValueError(f"第 {line_number} 行指标格式错误") from exc

    return train_records, val_records

def save_experiment_metrics(
    log_path: Path,
    output_path: Path,
) -> Path:
    train_records, validation_records = parse_training_log(log_path)

    payload = {
        "schema_version": 1,
        "source_log": str(log_path),
        "train_records": train_records,
        "validation_records": validation_records,
    }
    
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    )

    with output_path.open("x", encoding="utf-8") as f:
        f.write(serialized)

    return output_path

def plot_loss_curves(
    metrics_path: Path,
    output_path: Path,
) -> Path:
    with metrics_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
        train_records = payload["train_records"]
        validation_records = payload["validation_records"]

    if not train_records or not validation_records:
        raise ValueError("训练记录和验证记录均不能为空")

    train_steps = [record["step"] for record in train_records]
    train_losses = [record["loss"] for record in train_records]
    val_steps = [record["step"] for record in validation_records]
    val_losses = [record["loss"] for record in validation_records]

    fig, ax = plt.subplots(figsize=(8, 5))
    try:
        ax.plot(train_steps, train_losses, label="Train", marker="o")
        ax.plot(val_steps, val_losses, label="Validation", marker="o")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Loss")
        ax.set_title("Smoke run: training and validation loss")
        ax.legend()
        ax.grid(alpha=0.3)
        with output_path.open("xb") as f:
            fig.savefig(f, format="png", dpi=150, bbox_inches="tight")
    finally:
        plt.close(fig)
    return output_path