from pathlib import Path

import torch
import torch.nn.functional as F

from x_transformers import Decoder, TransformerWrapper
from x_transformers.autoregressive_wrapper import AutoregressiveWrapper, top_k, top_p


class CustomARWrapper(AutoregressiveWrapper):
    def __init__(self, *args, **kwargs):
        super(CustomARWrapper, self).__init__(*args, **kwargs)

    @torch.no_grad()
    def generate(
        self,
        start_tokens,
        seq_len=256,
        eos_token=None,
        temperature=1.0,
        filter_logits_fn=top_k,
        filter_thres=0.9,
        **kwargs,
    ):
        device = start_tokens.device
        was_training = self.net.training
        num_dims = len(start_tokens.shape)

        if num_dims == 1:
            start_tokens = start_tokens[None, :]

        b, t = start_tokens.shape

        self.net.eval()
        out = start_tokens
        mask = kwargs.pop("mask", None)
        if mask is None:
            mask = torch.full_like(out, True, dtype=torch.bool, device=out.device)

        for _ in range(seq_len):
            x = out[:, -self.max_seq_len :]
            mask = mask[:, -self.max_seq_len :]
            # print('arw:',out.shape)
            logits = self.net(x, mask=mask, **kwargs)

            context = kwargs["context"]

            root_dir = Path(__file__).resolve().parent.parent.parent
            save_dir = root_dir / "models"
            save_dir.mkdir(parents=True, exist_ok=True)
            save_onnx_path = save_dir / "decoder.onnx"

            if not Path(save_onnx_path).exists():
                torch.onnx.export(
                    self.net,
                    (x, {"context": context, "mask": mask}),
                    save_onnx_path,
                    export_params=True,
                    opset_version=11,
                    verbose=False,
                    input_names=["input"],
                    output_names=["output"],
                    do_constant_folding=True,
                    dynamic_axes={
                        "input": {0: "batch", 2: "height", 3: "width"},
                        "output": {0: "batch", 1: "output_seq", 2: "token_size"},
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

            logits = logits[:, -1, :]

            if filter_logits_fn in {top_k, top_p}:
                filtered_logits = filter_logits_fn(logits, thres=filter_thres)
                probs = F.softmax(filtered_logits / temperature, dim=-1)

            sample = torch.multinomial(probs, 1)

            out = torch.cat((out, sample), dim=-1)
            mask = F.pad(mask, (0, 1), value=True)

            if (
                eos_token is not None
                and (torch.cumsum(out == eos_token, 1)[:, -1] >= 1).all()
            ):
                break

        out = out[:, t:]

        if num_dims == 1:
            out = out.squeeze(0)

        self.net.train(was_training)
        return out


def get_decoder(args):
    return CustomARWrapper(
        TransformerWrapper(
            num_tokens=args.num_tokens,
            max_seq_len=args.max_seq_len,
            attn_layers=Decoder(
                dim=args.dim,
                depth=args.num_layers,
                heads=args.heads,
                **args.decoder_args,
            ),
        ),
        pad_value=args.pad_token,
    )
