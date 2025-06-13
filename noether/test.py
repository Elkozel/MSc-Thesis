from datasets.UFW22H_local import UFW22HL


d = UFW22HL("/data/datasets/UWF22", batch_size=2)
d.prepare_data()

# splits/transforms
d.setup(stage="fit")
f = None
for batch in d.train_dataloader():
    f = batch
    break