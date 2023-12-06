from pathlib import Path

import torch
import torch.nn as nn

from . import hybrid, transformer, vit


class Model(nn.Module):
    def __init__(self, encoder, decoder, args):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.args = args

    def data_parallel(self, x: torch.Tensor, device_ids, output_device=None, **kwargs):
        if not device_ids or len(device_ids) == 1:
            return self(x, **kwargs)
        if output_device is None:
            output_device = device_ids[0]
        replicas = nn.parallel.replicate(self, device_ids)
        inputs = nn.parallel.scatter(
            x, device_ids
        )  # Slices tensors into approximately equal chunks and distributes them across given GPUs.
        kwargs = nn.parallel.scatter(
            kwargs, device_ids
        )  # Duplicates references to objects that are not tensors.
        replicas = replicas[: len(inputs)]
        kwargs = kwargs[: len(inputs)]
        outputs = nn.parallel.parallel_apply(replicas, inputs, kwargs)
        return nn.parallel.gather(outputs, output_device).mean()

    def forward(self, x: torch.Tensor, tgt_seq: torch.Tensor, **kwargs):
        encoded = self.encoder(x)
        out = self.decoder(tgt_seq, context=encoded, **kwargs)
        return out

    @torch.no_grad()
    def generate(self, x: torch.Tensor, temperature: float = 0.25):
        context = self.encoder(x)
        root_dir = Path(__file__).resolve().parent.parent.parent
        save_dir = root_dir / "models"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_onnx_path = save_dir / "encoder.onnx"

        if not Path(save_onnx_path).exists():
            torch.onnx.export(
                self.encoder,
                x,
                save_onnx_path,
                export_params=True,
                opset_version=11,
                verbose=False,
                input_names=["input"],
                output_names=["output"],
                do_constant_folding=True,
                dynamic_axes={
                    "input": {0: "batch", 2: "height", 3: "width"},
                    "output": {0: "batch", 1: "context1", 2: "context2"},
                },
            )

            import numpy as np
            import onnxruntime

            ort_session = onnxruntime.InferenceSession(save_onnx_path)

            input_name = ort_session.get_inputs()[0].name
            ort_inputs = {input_name: x.cpu().numpy()}
            ort_outs = ort_session.run(None, ort_inputs)

            # compare ONNX Runtime and PyTorch results
            np.testing.assert_allclose(
                context.cpu().numpy(),
                ort_outs[0],
                rtol=1e-3,
                atol=1e-5,
            )
            print(
                "Exported model has been tested with ONNXRuntime, and the result looks good!"
            )
            print(f"ONNX Model has been saved {save_onnx_path}")

        return self.decoder.generate(
            (torch.LongTensor([self.args.bos_token] * len(x))[:, None]).to(x.device),
            self.args.max_seq_len,
            eos_token=self.args.eos_token,
            context=context,
            temperature=temperature,
        )


def get_model(args):
    if args.encoder_structure.lower() == "vit":
        encoder = vit.get_encoder(args)
    elif args.encoder_structure.lower() == "hybrid":
        encoder = hybrid.get_encoder(args)
    else:
        raise NotImplementedError(
            'Encoder structure "%s" not supported.' % args.encoder_structure
        )
    decoder = transformer.get_decoder(args)
    encoder.to(args.device)
    decoder.to(args.device)
    model = Model(encoder, decoder, args)
    if args.wandb:
        import wandb

        wandb.watch(model)

    return model
