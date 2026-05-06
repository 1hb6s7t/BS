# Model Assets

Large model weights are intentionally not committed to GitHub because several
files exceed GitHub's 100 MB single-file limit.

Expected local locations:

- `CRA/cra.ckpt`
- `ckpt/pretrained_generator_epoch100000.ckpt`
- `ckpt/generator_epoch11_batch56358.ckpt`
- `ckpt/discriminator_epoch11_batch56358.ckpt`
- `srgan/src/vgg19/vgg19.ckpt`

The application can still run without these files by using the `classic` or
`auto` backend:

```powershell
python run.py --demo --backend classic --output_dir output_demo
```

For deep CRA/SRGAN inference, install a compatible MindSpore environment and
place the checkpoint files in the paths above, or pass explicit checkpoint
paths through the CLI/GUI.
