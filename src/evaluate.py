import torch 
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt 
from . import config
from . import model as model_def
from . import dataset
import argparse

def get_predictions(model, data_loader): 
    
    model.eval()
    
    # predictions 
    predictions = []
    real_vals = []
    
    with torch.no_grad(): 
        for inputs, labels in data_loader: 
            
            inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            predictions.extend(predicted.cpu().numpy())
            real_vals.extend(labels.cpu().numpy())

    return predictions, real_vals

def main(): 
    print("Starting Evaluation...")
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument("model_path", type=str, help="Path to the saved .pth model file.")
    args = parser.parse_args()
    
    _, test_loader, _, test_dataset = dataset.create_dataloaders()
    class_names = test_dataset.classes
    
    
    model = model_def.create_model()
    print(f"Loading model from: {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=config.DEVICE))
    model.to(config.DEVICE)

    
    # get predictions 
    y_pred, y_true = get_predictions(model, test_loader)
    
    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names
    )
    
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    
    plt.savefig("reports/confusion_matrix.png")
    print("\nConfusion matrix plot saved to reports/confusion_matrix.png")
    
    plt.show()
    
    
if __name__ == '__main__': 
    main()
    