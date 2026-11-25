"""Compare four representations on exactly the same random-motion AL5D frames."""
import csv
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import yaml

from exp_run_config import Config
from flow import setup_flow, flow_entry, run_flow, display_flow_report
from demonstration.demopack import import_demopack
from demonstration.demonstration import Demonstration
from robot.al5d.position import RobotPosition
from .conv_vae_neo import ConvVAENeo, ConvVAENeoLoss, DemonstrationImageDataset, seed_everything
from .vae_gan import VAEGAN
from .vae_gan_training import VAEGANTrainer
from .vq_vae import VQVAE, VQVAELoss


FAMILY = "vae_encoding_comparison"
MODELS = {
    "old": ("Old VAE", "sensorprocessing_conv_vae", "sp_vae_128"),
    "neo": ("VAE-Neo", "sensorprocessing_conv_vae_neo", "sp_vae_neo_128_256px"),
    "gan": ("VAE-GAN", "sensorprocessing_vae_gan", "sp_vae_gan_128_256px"),
    "vq": ("VQ-VAE", "sensorprocessing_vq_vae", "sp_vq_vae_128_256px"),
}


def split_demos(names):
    n = len(names)
    train, val = int(.6*n), int(.8*n)
    groups = dict(train=names[:train], validation=names[train:val], test=names[val:])
    if not all(groups.values()):
        raise ValueError("Need independent training, validation and test demonstrations")
    return {n: n for n in names}, groups


def run_comparison(flow_name="VAE_AL5D_Random", demopack="random-both-cameras-video",
                   camera="dev2", epochs=6, frame_stride=20, batch_size=8, flows_path=None,
                   patience=0, min_relative_gain=.005):
    config = Config()
    config.set_exprun_path(config.exprun_path_internal)
    families = [FAMILY, "demonstration", "robot_al5d"] + [m[1] for m in MODELS.values()]
    expruns, results, notebooks = setup_flow(flow_name, families, flows_path=flows_path)
    groups = import_demopack(Path(config["demopacks_path"]).expanduser()/demopack, split_demos)
    settings = dict(demopack=demopack, camera=camera, epochs=epochs, batch_size=batch_size,
                    patience=patience, min_relative_gain=min_relative_gain,
                    frame_stride=frame_stride, results_root=str(results/FAMILY))
    for split in ("train","validation","test"):
        settings[split] = [[demopack, demo, camera] for demo in groups[split]]
    queue = [("prepare","Prepare_VAE_Comparison")]
    queue += [(name,"Train_VAE_Comparison") for name in MODELS]
    queue.append(("compare","Compare_VAE_Encodings"))
    entries = []
    for run, notebook in queue:
        recipe = dict(settings, model=run)
        recipe["input-to-notebook"] = [f"sensorprocessing/{notebook}.ipynb"]
        (expruns/FAMILY/f"{run}.yaml").write_text(yaml.safe_dump(recipe))
        entries.append(flow_entry(run,FAMILY,run,0,"version"))
    execution = run_flow(entries, expruns, results, notebooks)
    final = results/FAMILY/"compare"
    display_flow_report(entries, results, notebooks, final, execution=execution)
    return final


def prepare(exp):
    out = Path(exp["data_dir"])
    pack, demo, camera = exp["train"][0]
    source = Demonstration(Config().get_experiment("demonstration",pack),demo)
    if source.metadata["stored_as_images"]:
        height,width = cv2.imread(str(source.get_image_path(0,camera))).shape[:2]
    else:
        cap = cv2.VideoCapture(str(source.get_video_path(camera)))
        try:
            ok, frame = cap.read()
            if not ok:
                raise OSError(f"Cannot read {demo}/{camera}")
            height,width = frame.shape[:2]
        finally:
            cap.release()
    robot = Config().get_experiment("robot_al5d","position_controller_00")
    for split in ("train","validation","test"):
        ds = DemonstrationImageDataset(dict(exp.values, image_size=[height,width]),split)
        try:
            frames, targets, keys = [], [], []
            for i,(source_id,t) in enumerate(ds.samples):
                demonstration, view = ds.sources[source_id]
                frames.append((255*ds[i]).round().byte())
                pose = demonstration.get_action(t,"rc-position-target",robot)
                targets.append(pose.to_normalized_vector(robot))
                keys.append((demonstration.demo,view,t))
            torch.save(dict(images=torch.stack(frames), targets=torch.tensor(np.array(targets)),
                            keys=keys), out/f"{split}.pt")
        finally:
            ds.close()
    (out/"native_size.json").write_text(json.dumps([height,width]))


def load_split(exp, split):
    return torch.load(Path(exp["results_root"])/"prepare"/f"{split}.pt",weights_only=True)


def model_settings(exp, kind):
    _, family, run = MODELS[kind]
    base = Config().get_experiment(family,run,create_data_dir=False)
    size = json.loads((Path(exp["results_root"])/"prepare"/"native_size.json").read_text())
    return dict(base.values, image_size=[64,64] if kind=="old" else size, latent_size=exp["latent_size"])


def build(kind, settings):
    if kind == "old":
        sys.path.insert(0,str(Path(Config()["conv_vae"]["code_dir"]).expanduser()))
        from model.model import VanillaVAE
        return VanillaVAE(in_channels=3, latent_dims=settings["latent_size"])
    return {"neo": ConvVAENeo, "vq": VQVAE, "gan": VAEGAN}[kind](settings)


def input_images(raw, kind, device):
    x = raw.to(device=device, dtype=torch.float32)/255
    if kind=="old":
        return F.interpolate(x,size=(64,64),mode="bilinear",align_corners=False,antialias=True)
    return x


def decode_latent(model, x, kind):
    if kind=="old":
        mu,_,_ = model.encode(x)
        return model.decode(mu),mu
    z = model.encode(x)
    return model.decode(z),z


def native_output(output, size):
    if output.shape[-2:] != size:
        return F.interpolate(output, size=size, mode="bilinear", align_corners=False)
    return output


def train(exp):
    kind = exp["model"]
    settings = model_settings(exp,kind)
    seed_everything(exp["random_seed"])
    device = Config().runtime["device"]
    model = build(kind,settings).to(device)
    if kind=="gan":
        trainer = VAEGANTrainer(model,settings)
    elif kind=="old":
        from model.loss import elbo_loss
        optimizer = torch.optim.Adam(model.parameters(),lr=.005,amsgrad=True)
    else:
        criterion = (ConvVAENeoLoss if kind=="neo" else VQVAELoss)(settings)
        optimizer = torch.optim.Adam(model.parameters(),lr=settings["learning_rate"],
                                     weight_decay=settings["weight_decay"])
    loader = DataLoader(load_split(exp,"train")["images"], batch_size=exp["batch_size"],
                        shuffle=True,generator=torch.Generator().manual_seed(exp["random_seed"]))
    validation = load_split(exp,"validation")["images"]
    out = Path(exp["data_dir"])
    history, best = [], math.inf
    reference_mse = math.inf
    last_gain = 0
    reason = "epoch limit"
    for epoch in range(1,exp["epochs"]+1):
        model.train()
        totals = {}
        for raw in loader:
            x = input_images(raw,kind,device)
            if kind=="gan":
                logs = trainer.batch(x)
            else:
                optimizer.zero_grad(set_to_none=True)
                output = model(x)
                if kind=="old":
                    rec, mu, logvar = output
                    loss = elbo_loss(rec,x,mu,logvar)
                    logs = {"objective":loss.item()/len(x)}
                elif kind=="neo":
                    parts = criterion.components(output,x)
                    loss = parts["loss"]
                    logs = {k:v.item() for k,v in parts.items()}
                    logs["objective"] = logs.pop("loss")
                else:
                    rec, z, q, _ = output
                    loss = criterion(output,x)
                    logs = dict(objective=loss.item(), reconstruction=F.mse_loss(rec,x).item(),
                                codebook=F.mse_loss(z.detach(),q).item(),
                                commitment=F.mse_loss(z,q.detach()).item())
                loss.backward()
                if kind!="old":
                    torch.nn.utils.clip_grad_norm_(model.parameters(),settings["grad_clip_norm"])
                optimizer.step()
            for key,val in logs.items():
                totals[key] = totals.get(key,0.) + val*len(x)
        model.eval()
        squared_error = 0.
        with torch.no_grad():
            for raw in validation.split(exp["batch_size"]):
                rec,_ = decode_latent(model,input_images(raw,kind,device),kind)
                rec = native_output(rec,raw.shape[-2:])
                squared_error += F.mse_loss(rec,raw.to(device)/255,reduction="sum").item()
        mse = squared_error/validation.numel()
        row = dict(epoch=epoch,validation_mse=mse,**{k:v/len(loader.dataset) for k,v in totals.items()})
        if not np.isfinite(list(row.values())).all():
            raise FloatingPointError(f"{kind} produced nonfinite metrics at epoch {epoch}")
        history.append(row)
        if mse < best:
            best = mse
            torch.save(dict(epoch=epoch,weights=model.state_dict()),out/"best.pt")
        with (out/"history.csv").open("w" if epoch == 1 else "a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=row)
            if epoch == 1:
                writer.writeheader()
            writer.writerow(row)
        if mse < reference_mse * (1 - exp["min_relative_gain"]):
            reference_mse = mse
            last_gain = epoch
        if exp["patience"] and epoch - last_gain >= exp["patience"]:
            reason = "validation plateau"
            break
    (out/"stopping.json").write_text(json.dumps(dict(
        reason=reason, epochs_run=epoch, best_validation_mse=best, last_gain_epoch=last_gain
    ), indent=2))
    return history[-1]


def write_csv(path, rows):
    with path.open("w",newline="") as handle:
        writer = csv.DictWriter(handle,fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def evaluate(model, kind, data, batch_size, device, examples=()):
    latents, error, absolute, pictures = [], [], [], []
    for start in range(0,len(data),batch_size):
        raw = data[start:start+batch_size]
        rec,z = decode_latent(model,input_images(raw,kind,device),kind)
        rec = native_output(rec, raw.shape[-2:])
        delta = (rec - raw.to(device)/255).flatten(1)
        latents.append(z.cpu())
        error.extend(delta.square().mean(1).cpu().tolist())
        absolute.extend(delta.abs().mean(1).cpu().tolist())
        for i in examples:
            if start <= i < start+len(raw):
                pictures.append(rec[i-start].cpu())
    return torch.cat(latents).double().numpy(), np.array(error), np.array(absolute), pictures


def position_probe(features, targets):
    """Choose ridge strength on validation; fit scaling and coefficients on train only."""
    train, val, test = features
    mean, scale = train.mean(0), train.std(0)
    scale[scale<1e-6] = 1
    x, v, t = [(a-mean)/scale for a in (train,val,test)]
    y, yv, yt = targets
    intercept = y.mean(0)
    u,s,vt = np.linalg.svd(x,full_matrices=False)
    best = math.inf
    for alpha in (.01, .1, 1., 10., 100., 1000.):
        coef = (vt.T*(s/(s*s+alpha))) @ (u.T@(y-intercept))
        loss = np.square(v@coef+intercept-yv).mean()
        if loss < best:
            best, weight, selected = loss, coef, alpha
    return t@weight+intercept-yt, selected


def compare(exp):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    root, out = Path(exp["results_root"]), Path(exp["data_dir"])
    device = Config().runtime["device"]
    train_data, val_data, test = [load_split(exp,s) for s in ("train","validation","test")]
    keys = test["keys"]
    # Two preselected frames from each held-out demonstration (one-third and two-thirds).
    picks = []
    for demo in dict.fromkeys(k[0] for k in keys):
        indices = [i for i,k in enumerate(keys) if k[0]==demo]
        picks.extend(sorted({indices[len(indices)//3],indices[2*len(indices)//3]}))
    originals = [test["images"][i].float()/255 for i in picks]
    summary, per_frame, probe_rows, histories = [], [], [], {}
    panels = {"Original":originals}
    targets = [s["targets"].numpy() for s in (train_data,val_data,test)]
    for kind,(label,_,_) in MODELS.items():
        model = build(kind,model_settings(exp,kind)).to(device)
        checkpoint = torch.load(root/kind/"best.pt",weights_only=True,map_location=device)
        model.load_state_dict(checkpoint["weights"])
        model.eval()
        zs = [evaluate(model,kind,d["images"],exp["batch_size"],device)[0] for d in (train_data,val_data)]
        z,mse,mae,panels[label] = evaluate(model,kind,test["images"],exp["batch_size"],device,picks)
        residual,alpha = position_probe([*zs,z],targets)
        probe_rows.append(dict(model=label,**dict(zip(RobotPosition.FIELDS,np.sqrt((residual**2).mean(0))))))
        with (root/kind/"history.csv").open() as f:
            histories[kind] = list(csv.DictReader(f))
        stopping_file = root/kind/"stopping.json"
        stop = (json.loads(stopping_file.read_text()) if stopping_file.exists() else
                dict(epochs_run=len(histories[kind]), reason="fixed budget"))
        size = model_settings(exp,kind)["image_size"]
        summary.append(dict(model=label,height=size[0],width=size[1],selected_epoch=checkpoint["epoch"],
                            epochs_run=stop["epochs_run"],stop_reason=stop["reason"],
                            mse=mse.mean(),mae=mae.mean(),psnr_db=-10*np.log10(mse.mean()),
                            pose_rmse=np.sqrt((residual**2).mean()),ridge_alpha=alpha))
        for i,(demo,cam,t) in enumerate(keys):
            per_frame.append(dict(model=label,demonstration=demo,camera=cam,frame=t,
                                  mse=mse[i],mae=mae[i],pose_mse=(residual[i]**2).mean()))
    write_csv(out/"scores.csv",summary)
    write_csv(out/"per_frame.csv",per_frame)
    write_csv(out/"pose_by_field.csv",probe_rows)
    for errors in (False, True):
        fig, axes = plt.subplots(len(picks),len(panels),squeeze=False,
                                 figsize=(12,2.55*len(picks)),layout="constrained")
        for col,(label,pics) in enumerate(panels.items()):
            axes[0,col].set_title(label + (" (64×64)" if label=="Old VAE" else ""))
            for row,(img,idx) in enumerate(zip(pics,picks)):
                ax = axes[row,col]
                if errors and col:
                    im = ax.imshow((img-originals[row]).abs().mean(0),cmap="inferno",norm=Normalize(0,.25))
                else:
                    ax.imshow(img.clamp(0,1).permute(1,2,0))
                ax.set_xticks([])
                ax.set_yticks([])
                if col==0:
                    ax.set_ylabel(f"{keys[idx][0][-8:]}\nframe {keys[idx][2]}")
        if errors:
            fig.colorbar(im,ax=axes[:,1:].ravel().tolist(),label="Mean absolute RGB error",shrink=.7,extend="max")
        fig.savefig(out/("errors.png" if errors else "reconstructions.png"),dpi=150)
        plt.close(fig)
    fig,axes = plt.subplots(2,3,figsize=(13,7),layout="constrained")
    for ax,(kind,(label,_,_)) in zip(axes.flat,MODELS.items()):
        rows = histories[kind]
        for key in (("encoder","generator","discriminator") if kind=="gan" else ("objective",)):
            ax.plot([int(r["epoch"]) for r in rows],[float(r[key]) for r in rows],label=key)
        ax.set(title=label,xlabel="Epoch",ylabel="Native training loss")
        ax.legend()
        axes.flat[4].plot([int(r["epoch"]) for r in rows],[float(r["validation_mse"]) for r in rows],label=label)
    axes.flat[4].set(title="Common validation metric",ylabel="Native-grid RGB MSE",xlabel="Epoch")
    axes.flat[4].legend()
    axes.flat[5].axis("off")
    fig.savefig(out/"losses.png",dpi=150)
    plt.close(fig)
    baseline = np.sqrt(np.square(targets[2]-targets[0].mean(0)).mean())
    lines = ["# Random-motion AL5D encoding comparison","",
             f"Pack `{exp['demopack']}`, camera `{exp['camera']}`, latent width {exp['latent_size']}. "
             f"Training/validation/test frames: {len(targets[0])}/{len(targets[1])}/{len(targets[2])}. "
             f"Sampled every {exp['frame_stride']} frames; epoch limit {exp['epochs']}. "
             "Checkpoints minimize validation RGB MSE. All metrics use held-out test demonstrations.",
             "", "| Model | Input | Selected epoch | RGB MSE | PSNR dB | Pose RMSE |",
             "|---|---|---:|---:|---:|---:|"]
    for row in summary:
        lines.append("| {model} | {height}×{width} | {selected_epoch} | {mse:.5f} | {psnr_db:.2f} | {pose_rmse:.4f} |".format(**row))
    if exp.get("patience", 0):
        lines += ["", f"Patience is {exp['patience']} epochs without a cumulative "
                  f"{100*exp['min_relative_gain']:g}% improvement in validation RGB MSE."]
    lines += [f"- {r['model']}: {r['epochs_run']} epochs ({r['stop_reason']})." for r in summary]
    lines += ["", "The pose probe predicts six normalized AL5D commanded position fields from frozen latents. "
              f"Predicting the training mean gives RMSE {baseline:.4f}. Pixel scores use RGB [0,1] on the native grid, "
              "with the legacy output bilinearly upsampled. Gaussian models decode the posterior mean; VQ decodes quantized embeddings. "
              "The display clips RGB to [0,1]; scores use the decoder output.",
              "", "## Objectives", "",
              "- Old VAE: the legacy package's summed pixel squared error plus summed Gaussian KL (reported per image).",
              "- VAE-Neo: mean pixel squared error + 0.001 × mean Gaussian KL.",
              "- VAE-GAN: encoder uses discriminator-feature MSE + 0.001 × KL; generator uses feature MSE + 0.1 × adversarial BCE. "
              "The discriminator learns real versus reconstructed and prior-sampled images. Pixel loss weight is zero.",
              "- VQ-VAE: mean pixel squared error + codebook MSE + 0.25 × commitment MSE.",
              "", "The native objective scales differ. Compare reconstruction on the common RGB metric and encoded pose on the probe.",
              "", "![Matched reconstructions](reconstructions.png)", "", "![Absolute errors](errors.png)", "", "![Loss histories](losses.png)"]
    (out/"report.md").write_text("\n".join(lines)+"\n")
    return summary
