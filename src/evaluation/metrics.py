import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.models.causal_tft.tft_baseline import TFTBaseline
from src.models.ct import CT


def forecast_tft_values(
    model: TFTBaseline, dataloader: DataLoader, max_seq_length: int
):

    projection_length = (
        model.projection_horizon if isinstance(model, CT) else model.projection_length
    )
    # format y_true values
    y_true = np.zeros((len(dataloader.dataset), projection_length, 1))
    for i, tau in enumerate(dataloader.dataset.data["future_past_split"]):
        tau = int(tau)
        if isinstance(model, CT):
            y_true[i] = dataloader.dataset.data["outputs"][
                i, tau - 1 : tau + projection_length - 1
            ]
        else:
            y_true[i] = torch.tensor(
                dataloader.dataset.data["outputs"][i, tau : tau + projection_length]
            )

    if isinstance(model, CT):
        print("DEVICE ", model.device)
        predictions = model.get_autoregressive_predictions(dataloader.dataset)
    else:
        # compute predictions from model
        predictions = []
        for batch in tqdm(dataloader):
            predicted_outputs = []
            for key in [
                "vitals",
                "static_features",
                "current_treatments",
                "outputs",
                "prev_treatments",
                "prev_outputs",
                "active_entries",
            ]:
                batch[key] = batch[key].to(model.device)
            outputs_scaled = model.forecast(batch).cpu()

            for i in range(batch["vitals"].shape[0]):
                split = int(batch["future_past_split"][i])
                if split + projection_length < (max_seq_length + 1):
                    predicted_outputs.append(
                        outputs_scaled[i, split : split + projection_length, :]
                    )
            predicted_outputs = torch.stack(predicted_outputs)
            predictions.append(predicted_outputs)
        predictions = torch.concat(predictions).numpy()

    return predictions, y_true
