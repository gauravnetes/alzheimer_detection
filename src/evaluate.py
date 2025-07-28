import torch
import torch.nn.functional as F
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import argparse
from tqdm import tqdm
from itertools import cycle
from . import config
from . import model as model_def
from . import dataset

# --- PREDICTION & EVALUATION LOGIC ---

def get_predictions(model, data_loader):
    """Gets predictions and probabilities for a single model without TTA."""
    model.eval()
    predictions, real_values, probabilities = [], [], []
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Predicting"):
            inputs = inputs.to(config.DEVICE)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(probs, 1)
            
            predictions.extend(preds.cpu().numpy())
            real_values.extend(labels.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
            
    return real_values, predictions, np.array(probabilities)

def plot_detailed_metrics(y_true, y_pred, y_probs, class_names, save_path_prefix):
    """Creates and saves comprehensive evaluation plots."""
    print("Generating detailed evaluation plots...")
    # 1. Detailed Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    annotations = np.array([f"{count}\n({percent:.1%})" for count, percent in zip(cm.flatten(), cm_percent.flatten())]).reshape(cm.shape)
    
    sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual'); plt.xlabel('Predicted'); plt.title('Confusion Matrix (Count & Percentage)')
    plt.savefig(f"{save_path_prefix}_confusion_matrix.png", dpi=300)
    plt.close()

    # 2. ROC Curves (One-vs-Rest)
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
    plt.figure(figsize=(10, 8))
    colors = cycle(['blue', 'red', 'green', 'orange', 'purple'])
    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2, label=f'ROC curve for {class_name} (AUC = {roc_auc:.3f})')
        
    plt.plot([0, 1], [0, 1], 'k--', lw=2); plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('Multi-class ROC Curves')
    plt.legend(loc="lower right"); plt.grid(True)
    plt.savefig(f"{save_path_prefix}_roc_curves.png", dpi=300)
    plt.close()
    print(f"Evaluation plots saved to reports folder with prefix: {os.path.basename(save_path_prefix)}")

# --- MAIN EVALUATION WORKFLOW ---

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument("model_path", type=str, help="Path to the saved .pth model file or ensemble_info.pt file.")
    args = parser.parse_args()
    
    eval_cfg = config.EVALUATION_CONFIG
    ens_cfg = config.ENSEMBLE_CONFIG
    
    # --- Data Loading ---
    _, test_loader, _, test_dataset = dataset.create_dataloaders()
    class_names = test_dataset.classes
    
    # --- Determine if it's an Ensemble run ---
    is_ensemble = 'ensemble_info' in args.model_path and ens_cfg.get('use_ensemble', False)
    
    if is_ensemble:
        print("--- Starting Ensemble Evaluation ---")
        ensemble_info = torch.load(args.model_path)
        all_model_probs = []
        for model_info in tqdm(ensemble_info['models'], desc="Evaluating models for ensemble"):
            model = model_def.create_model(model_info['name'])
            model.load_state_dict(torch.load(model_info['path'], map_location=config.DEVICE)['model_state_dict'])
            y_true, _, y_probs = get_predictions(model, test_loader)
            all_model_probs.append(y_probs)
        
        # Average probabilities for ensembling
        final_probs = np.mean(all_model_probs, axis=0)
        y_pred = np.argmax(final_probs, axis=1)
        model_name = "ensemble"
        
    else:
        print("--- Starting Single Model Evaluation ---")
        cfg = config.TRAINING_CONFIG
        model = model_def.create_model(cfg['model_name'])
        checkpoint = torch.load(args.model_path, map_location=config.DEVICE)
        model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
        print(f"Loaded model from {args.model_path}")
        best_acc = checkpoint.get('best_val_acc') # Get the value, will be None if not found
        if best_acc:
            print(f"Original best training accuracy: {best_acc:.4f}")
        else:
            print("Original best training accuracy: Not found in this model file.")
        
        # TTA is not yet implemented in a config-driven way in dataset.py, so we skip for now
        # Add TTA logic here if you create a TTA-enabled dataset/loader
        y_true, y_pred, final_probs = get_predictions(model, test_loader)
        model_name = cfg['model_name']
        
    # --- Reporting ---
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nOverall Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print("\n" + "="*60 + "\nDETAILED CLASSIFICATION REPORT\n" + "="*60)
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)
    
    # --- Save Results & Plots ---
    os.makedirs("reports", exist_ok=True)
    run_name = os.path.basename(args.model_path).replace('.pth', '').replace('.pt', '')
    
    # Save text report
    with open(f"reports/{run_name}_report.txt", "w") as f:
        f.write(f"Evaluation report for model: {args.model_path}\n\n")
        f.write(f"Overall Accuracy: {accuracy:.4f}\n\n")
        f.write(report)
        
    # Save JSON summary
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    with open(f"reports/{run_name}_summary.json", 'w') as f:
        json.dump(report_dict, f, indent=2)
        
    plot_detailed_metrics(y_true, y_pred, final_probs, class_names, save_path_prefix=f"reports/{run_name}")

if __name__ == '__main__':
    main()