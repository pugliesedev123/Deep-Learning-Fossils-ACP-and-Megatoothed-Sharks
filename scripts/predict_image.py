#!/usr/bin/env python3
import argparse
import csv
import json
import os
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


# Check supported image types.
def is_image(filename):
    extensions = (".png", ".jpg", ".jpeg")
    return filename.lower().endswith(extensions)


# Normalize a class name so comparisons are consistent.
# Example: "taxon-otodus_megalodon" -> "otodus_megalodon"
def norm_class_name(name: str) -> str:
    n = (name or "").strip().lower()
    if n.startswith("taxon-"):
        n = n.replace("taxon-", "", 1)
    return n


def load_embedder_from_classifier(model_path_str, classifier_state, device):
    # Pick architecture and set identity at the correct location.
    if "vgg16" in model_path_str:
        embedder = models.vgg16(weights=None)
        embedder.classifier[6] = nn.Identity()      # 4096-d features
        drop_prefixes = ("classifier.6.",)
    elif "densenet121" in model_path_str:
        embedder = models.densenet121(weights=None)
        embedder.classifier = nn.Identity()         # 1024-d features
        drop_prefixes = ("classifier.",)
    elif "resnet18" in model_path_str:
        embedder = models.resnet18(weights=None)
        embedder.fc = nn.Identity()                 # 512-d features
        drop_prefixes = ("fc.",)
    elif "resnet34" in model_path_str:
        embedder = models.resnet34(weights=None)
        embedder.fc = nn.Identity()                 # 512-d features
        drop_prefixes = ("fc.",)
    elif "resnet50" in model_path_str:
        embedder = models.resnet50(weights=None)
        embedder.fc = nn.Identity()                 # 2048-d features
        drop_prefixes = ("fc.",)

    # unwrap and strip DDP prefix
    state = classifier_state.get("state_dict", classifier_state)
    state = {k.replace("module.", ""): v for k, v in state.items()}

    # drop the head weights for this architecture
    keep = {k: v for k, v in state.items() if not any(k.startswith(p) for p in drop_prefixes)}
    embedder.load_state_dict(keep, strict=False)
    return embedder.to(device).eval()


def main():

    # Command-Line Arguments
    parser = argparse.ArgumentParser(description="Predict fossil classes for images in a folder (recursively).")
    parser.add_argument("--example-dir", required=True, help="Path to folder containing example images (processed recursively).")
    parser.add_argument("--console-print", action='store_true', help="Print extra details to console")
    parser.add_argument("--top-predictions", type=int, default=3, help="How many top guesses to record for each image.")
    parser.add_argument("--neighbors", type=int, default=3, help="How many closest training images to record.")
    parser.add_argument("--model-path", default="models/fossil_resnet18.pt", help="Path to the trained model weights file.")
    parser.add_argument("--class-names", default="models/class_names.json", help="Path to class_names.json file.")
    parser.add_argument("--output-dir", default="output", help="Folder where the CSV will be saved.")
    args = parser.parse_args()

    # Pick GPU if available, else CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Open class names .json file.
    with open(args.class_names, "r") as f:
        class_names = json.load(f)
    if isinstance(class_names, dict):
        try:
            class_names = [class_names[str(i)] for i in range(len(class_names))]
        except Exception:
            class_names = list(class_names.values())

    # Build the classifier model shape to match training.
    if (args.model_path == 'models/fossil_resnet18.pt'):
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(class_names))
    elif (args.model_path == 'models/fossil_resnet34.pt'):
        model = models.resnet34(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(class_names))
    elif (args.model_path == 'models/fossil_resnet50.pt'):
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(class_names))
    elif (args.model_path == 'models/fossil_vgg16.pt'):
        model = models.vgg16(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, len(class_names))
    elif (args.model_path == 'models/fossil_densenet121.pt'):
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, len(class_names))

    # Load checkpoint once
    ckpt = torch.load(args.model_path, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}

    # Classifier: DO NOT drop the head
    missing, unexpected = model.load_state_dict(state, strict=False)

    # Make sure the classifier runs in eval mode on the device.
    model = model.to(device).eval()

    # Embedder: use a copy with the head removed.
    state_for_embedder = dict(state)

    # Select model-dependent state for embedder.
    if "resnet" in args.model_path:
        state_for_embedder.pop("fc.weight", None); state_for_embedder.pop("fc.bias", None)
    elif "vgg16" in args.model_path:
        state_for_embedder.pop("classifier.6.weight", None); state_for_embedder.pop("classifier.6.bias", None)
    elif "densenet121" in args.model_path:
        state_for_embedder.pop("classifier.weight", None); state_for_embedder.pop("classifier.bias", None)

    embedder = load_embedder_from_classifier(args.model_path, state_for_embedder, device)

    # Define Model-dependent ImageNet Mean and Standard Deviation.
    if "resnet" in args.model_path or "densenet121" in args.model_path:
        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD = [0.229, 0.224, 0.225]
    elif "vgg16" in args.model_path:
        IMAGENET_MEAN = [0.48235, 0.45882, 0.40784]
        IMAGENET_STD = [0.00392156862745098, 0.00392156862745098, 0.00392156862745098]
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),    # Set shorter side to 256px,
        transforms.CenterCrop(224),       # cut a 224x224 square from center.
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    # Build a file name for the CSV that includes the input folder name and a timestamp.
    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_name = os.path.basename(os.path.abspath(args.example_dir)) or "examples"

    model_tag = args.model_path.replace("models/fossil_", "").replace(".pt", "")
    csv_path = os.path.join(args.output_dir, f"predictions_{model_tag}_{base_name}_{ts}.csv")
    metrics_csv_path = os.path.join(args.output_dir, f"metrics_{model_tag}_{base_name}_{ts}.csv")

    header = [
        "filename",
        "parent_folder",
    ]

     # Columns for top class predictions.
    for i in range(1, args.top_predictions + 1):
        header += [
            f"class_prediction_{i}",
            f"class_prediction_{i}_confidence_percentage",
            f"class_prediction_{i}_IsAccurate"
        ]

    rows = []

    # Store true/pred class indices so we can compute TP/FP/TN/FN later.
    # This is based on top-1 predictions only.
    y_true = []
    y_pred = []

    # Map normalized class name -> index.
    class_to_index = {norm_class_name(name): i for i, name in enumerate(class_names)}

    # Check input folder location.
    if not os.path.isdir(args.example_dir):
        raise FileNotFoundError(f"Input folder not found: {args.example_dir}")
    
    # With gradient calculation disabled, walk directories and run predictions.
    with torch.no_grad():
        for root, _, files in os.walk(args.example_dir):
            parent_folder = os.path.basename(root)

            # Keep only image files
            image_files = [f for f in files if is_image(f)]
            if not image_files:
                continue

            # Process each image in sorted order for stable output
            for file in sorted(image_files):
                img_path = os.path.join(root, file)
                try:
                    img = Image.open(img_path).convert("RGB")  # ensure 3 channels
                except Exception as e:
                    print(f"Skipping {img_path}: failed to open ({e})")
                    continue

                # Prepare the image for the model
                x = transform(img).unsqueeze(0).to(device)  # add batch dimension

                # Classify the image,
                logits = model(x)                   # return raw scores for each class,
                probs = F.softmax(logits, dim=1)    # convert scores into probabilities.

                # Pick the top K classes
                num_classes = probs.shape[1]
                k = min(args.top_predictions, num_classes)

                if args.top_predictions > num_classes and args.console_print:
                    print(f"[WARN] Requested top-predictions={args.top_predictions} but model has only {num_classes} classes. Using k={k}.")

                top_probs, top_indices = torch.topk(probs, k)
                top_probs = top_probs[0].cpu().numpy()
                top_indices = top_indices[0].cpu().numpy()


                # Save top-1 prediction for metrics.
                # True label comes from folder name.
                true_class = norm_class_name(parent_folder)
                pred_class = norm_class_name(class_names[top_indices[0]])

                # Only score images if the folder name matches a known class.
                if true_class in class_to_index and pred_class in class_to_index:
                    y_true.append(class_to_index[true_class])
                    y_pred.append(class_to_index[pred_class])

                if(args.console_print):
                    rel_path_for_print = os.path.relpath(img_path, args.example_dir)
                    print(f"\n{rel_path_for_print}:")

                # Start the CSV row with file name and parent folder name.
                row = [file, parent_folder]

                # Add the top K class predictions to the row and print them,
                for i in range(len(top_indices)):
                    
                    predicted_class = class_names[top_indices[i]]   # map the index to a class name,
                    confidence_pct = float(top_probs[i] * 100.0)    # return the confidence score as a percent.
                    class_accurate = ""


                    # Check class prediction according to folder name, should turn this into an argument.
                    if predicted_class.replace("taxon-","") == (parent_folder.replace("taxon-","") or parent_folder):
                        class_accurate = "Yes"
                    else:
                        class_accurate = "No"

                    # Return predictions and confidence scores to the console if --console-print is specified.
                    if(args.console_print):
                        print(f"{i+1}. {predicted_class} ({confidence_pct:.2f}% confidence)")
                    row += [predicted_class, f"{confidence_pct:.2f}", class_accurate]
                rows.append(row)


    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    # Generate key metrics per class using TP / FP / TN / FN.
    # Metrics are based on top-1 predictions only.
    metrics_header = [
        "class_name",
        "support",
        "TP",
        "FP",
        "TN",
        "FN",
        "ACC",
        "PRC",
        "TPR",
        "FPR",
    ]

    metrics_rows = []

    N = len(y_true)
    n_classes = len(class_names)

    # If no images could be scored, still create the file with header.
    if N > 0:
        for c in range(n_classes):

            # Count TP/FP/FN for this class.
            TP = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
            FP = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
            FN = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)

            # Everything else is TN.
            TN = N - TP - FP - FN

            # Support = number of true examples for this class.
            support = sum(1 for t in y_true if t == c)

            # ACC = (TP + TN) / (TP + FP + TN + FN)
            denom_acc = (TP + FP + TN + FN)
            ACC = (TP + TN) / denom_acc if denom_acc > 0 else 0.0

            # PRC = TP / (TP + FP)
            denom_prc = (TP + FP)
            PRC = TP / denom_prc if denom_prc > 0 else 0.0

            # TPR = TP / (TP + FN)
            denom_tpr = (TP + FN)
            TPR = TP / denom_tpr if denom_tpr > 0 else 0.0

            # FPR = FP / (FP + TN)
            denom_fpr = (FP + TN)
            FPR = FP / denom_fpr if denom_fpr > 0 else 0.0

            metrics_rows.append([
                class_names[c],
                support,
                TP,
                FP,
                TN,
                FN,
                f"{ACC:.6f}",
                f"{PRC:.6f}",
                f"{TPR:.6f}",
                f"{FPR:.6f}",
            ])

    with open(metrics_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(metrics_header)
        writer.writerows(metrics_rows)

    print(f"\nSaved predictions to: {csv_path}")
    print(f"\nSaved metrics to: {metrics_csv_path}")


if __name__ == "__main__":
    main()
